#include "link.h"

#include <Arduino.h>
#include <WiFi.h>
#include <errno.h>
#include <esp_wifi.h>
#include <fcntl.h>
#include <lwip/sockets.h>

namespace {

constexpr uint16_t LINK_PORT = 5000;   /* TCP: handshake + barriers        */
constexpr uint16_t DATA_PORT = 5001;   /* UDP: bulk K/V payload            */
constexpr int      MAX_SEGS  = 12;
constexpr int      IO_CHUNK  = 1460;
constexpr int      UDP_PAYLOAD = 1400;
constexpr int      MAX_PKTS  = 64;
constexpr int      HDR_BYTES = 8;

/* Datagram header: 'T','M', type, epoch, u16 index, u16 len.
 * type 0 = data, 1 = NAK (payload: two u32 missing bitmaps), 2 = ACK. */
constexpr uint8_t  PKT_DATA = 0, PKT_NAK = 1, PKT_ACK = 2, PKT_BAR = 3;

WiFiServer  g_server(LINK_PORT);
WiFiClient  g_client;
char        g_info[96] = "down";
char        g_err[64]  = "";
int         g_fd = -1;        /* TCP control socket */
int         g_ufd = -1;       /* UDP data socket    */
sockaddr_in g_peer{};
uint32_t    g_tx_total_stat = 0, g_rx_total_stat = 0;
uint8_t     g_epoch = 0;
int         g_role = 0;
uint32_t    g_udp_retx = 0, g_udp_lost = 0;
void      (*g_hb)(void) = nullptr;   /* keep the USB CDC awake while blocked */
uint32_t    g_hb_last = 0;

inline void beat(void) {
    if (g_hb && millis() - g_hb_last >= 100) { g_hb_last = millis(); g_hb(); }
}

/* one direction of an armed exchange */
struct Stream {
    TmIoSeg  seg[MAX_SEGS];
    int      n;
    int      idx;
    int      off;
    uint32_t done;
    uint32_t total;
};

Stream   g_tx, g_rx;
volatile uint32_t g_tx_ready = 0;      /* payload bytes the shard has filled  */
volatile bool     g_active   = false;
volatile bool     g_failed   = false;
TaskHandle_t      g_task     = nullptr;
SemaphoreHandle_t g_go       = nullptr;

void stream_load(Stream& s, const TmIoSeg* seg, int n) {
    s.n = (n > MAX_SEGS) ? MAX_SEGS : n;
    s.idx = 0; s.off = 0; s.done = 0; s.total = 0;
    for (int i = 0; i < s.n; i++) { s.seg[i] = seg[i]; s.total += (uint32_t)seg[i].len; }
}

/* Move up to `limit` more bytes on this stream. Returns bytes moved, or -1. */
int stream_pump(Stream& s, bool sending, uint32_t limit) {
    if (s.idx >= s.n || s.done >= limit) return 0;
    int want = s.seg[s.idx].len - s.off;
    if (want > IO_CHUNK) want = IO_CHUNK;
    if ((uint32_t)want > limit - s.done) want = (int)(limit - s.done);
    if (want <= 0) return 0;

    uint8_t* p = s.seg[s.idx].p + s.off;
    const int n = sending ? ::send(g_fd, p, (size_t)want, 0)
                          : ::recv(g_fd, p, (size_t)want, 0);
    if (n > 0) {
        s.off += n; s.done += (uint32_t)n;
        if (s.off == s.seg[s.idx].len) { s.idx++; s.off = 0; }
        return n;
    }
    if (n == 0 && !sending) { snprintf(g_err, sizeof g_err, "peer closed"); return -1; }
    if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
        snprintf(g_err, sizeof g_err, "%s errno=%d", sending ? "send" : "recv", errno);
        return -1;
    }
    return 0;
}


/* ---- byte-offset view over a segment list (the payload is not contiguous) ---- */
void stream_copy(Stream& s, uint32_t off, uint8_t* buf, int len, bool into_stream) {
    int i = 0;
    uint32_t base = 0;
    while (i < s.n && base + (uint32_t)s.seg[i].len <= off) { base += (uint32_t)s.seg[i].len; i++; }
    int done = 0;
    while (done < len && i < s.n) {
        const int in_seg = (int)(off + (uint32_t)done - base);
        int take = s.seg[i].len - in_seg;
        if (take > len - done) take = len - done;
        if (into_stream) memcpy(s.seg[i].p + in_seg, buf + done, (size_t)take);
        else             memcpy(buf + done, s.seg[i].p + in_seg, (size_t)take);
        done += take;
        if (in_seg + take == s.seg[i].len) { base += (uint32_t)s.seg[i].len; i++; }
    }
}

inline void bit_set(uint32_t* m, int k)      { m[k >> 5] |= 1u << (k & 31); }
inline bool bit_get(const uint32_t* m, int k){ return (m[k >> 5] >> (k & 31)) & 1u; }

/* Reliable symmetric UDP swap of the armed streams.
 *
 * TCP is unusable for the bulk payload here: the Arduino framework ships lwIP
 * with a 5744 B window and no way to raise it, which caps a single connection
 * at ~30-80 KB/s against a 32 KB per-layer swap. UDP carries the payload in
 * 1400 B datagrams; the receiver NAKs any gaps it still has every 12 ms and
 * the sender retransmits just those, which on a one-hop dedicated AP link
 * costs nothing in the common (lossless) case.
 *
 * g_rx.done is published as the *contiguous* prefix received, so the shard's
 * per-head tm_link_wait_rx() thresholds stay meaningful under reordering.
 */
bool udp_exchange(void) {
    const int ntx = (int)((g_tx.total + UDP_PAYLOAD - 1) / UDP_PAYLOAD);
    const int nrx = (int)((g_rx.total + UDP_PAYLOAD - 1) / UDP_PAYLOAD);
    if (ntx > MAX_PKTS || nrx > MAX_PKTS) {
        snprintf(g_err, sizeof g_err, "payload too large"); return false;
    }
    const uint8_t epoch = ++g_epoch;   /* reset per forward by tm_link_new_forward */

    uint32_t sent[2] = {0, 0}, got[2] = {0, 0}, resend[2] = {0, 0};
    int n_got = 0;
    bool peer_acked = false;
    uint8_t pkt[HDR_BYTES + UDP_PAYLOAD];
    uint32_t last_ctrl = millis();
    uint32_t last_data = millis();
    const uint32_t t0 = millis();

    auto pkt_len = [&](int k, uint32_t total) {
        const uint32_t off = (uint32_t)k * UDP_PAYLOAD;
        const uint32_t rem = total - off;
        return (int)(rem > UDP_PAYLOAD ? UDP_PAYLOAD : rem);
    };

    while (!(n_got == nrx && peer_acked)) {
        if (millis() - t0 > 20000) {
            snprintf(g_err, sizeof g_err, "udp stall rx=%d/%d ack=%d",
                     n_got, nrx, (int)peer_acked);
            return false;
        }
        bool moved = false;

        /* send releasable data packets (fresh ones first, then NAKed ones) */
        for (int burst = 0; burst < 8; burst++) {
            int k = -1;
            for (int i = 0; i < ntx; i++)
                if (!bit_get(sent, i) &&
                    (uint32_t)i * UDP_PAYLOAD + (uint32_t)pkt_len(i, g_tx.total) <= g_tx_ready) { k = i; break; }
            if (k < 0)
                for (int i = 0; i < ntx; i++) if (bit_get(resend, i)) { k = i; break; }
            if (k < 0) break;

            const int len = pkt_len(k, g_tx.total);
            pkt[0] = 'T'; pkt[1] = 'M'; pkt[2] = PKT_DATA; pkt[3] = epoch;
            pkt[4] = (uint8_t)(k >> 8); pkt[5] = (uint8_t)k;
            pkt[6] = (uint8_t)(len >> 8); pkt[7] = (uint8_t)len;
            stream_copy(g_tx, (uint32_t)k * UDP_PAYLOAD, pkt + HDR_BYTES, len, false);
            const int n = ::sendto(g_ufd, pkt, (size_t)(HDR_BYTES + len), 0,
                                   (sockaddr*)&g_peer, sizeof g_peer);
            if (n < 0) break;                       /* buffers full: retry later */
            if (bit_get(resend, k)) { resend[k >> 5] &= ~(1u << (k & 31)); g_udp_retx++; }
            bit_set(sent, k);
            g_tx.done += (uint32_t)len;
            moved = true;
        }

        /* drain everything the peer has sent us */
        for (;;) {
            const int n = ::recv(g_ufd, pkt, sizeof pkt, 0);
            if (n < HDR_BYTES) break;
            if (pkt[0] != 'T' || pkt[1] != 'M' || pkt[3] != epoch) continue;
            if (pkt[2] == PKT_DATA) {
                const int k = (pkt[4] << 8) | pkt[5];
                const int len = (pkt[6] << 8) | pkt[7];
                if (k >= nrx || len != n - HDR_BYTES) continue;
                if (!bit_get(got, k)) {
                    stream_copy(g_rx, (uint32_t)k * UDP_PAYLOAD, pkt + HDR_BYTES, len, true);
                    bit_set(got, k);
                    n_got++;
                    /* publish the contiguous prefix for tm_link_wait_rx */
                    uint32_t done = 0;
                    for (int i = 0; i < nrx && bit_get(got, i); i++)
                        done += (uint32_t)pkt_len(i, g_rx.total);
                    g_rx.done = done;
                    last_data = millis();
                }
                moved = true;
            } else if (pkt[2] == PKT_NAK && n >= HDR_BYTES + 8) {
                uint32_t m[2];
                memcpy(m, pkt + HDR_BYTES, 8);
                for (int i = 0; i < ntx; i++)
                    if (bit_get(m, i) && bit_get(sent, i)) bit_set(resend, i);
                moved = true;
            } else if (pkt[2] == PKT_ACK) {
                peer_acked = true;
                moved = true;
            }
        }

        /* every 12 ms: NAK what is still missing, or ACK once complete */
        if (millis() - last_ctrl >= 12) {
            last_ctrl = millis();
            pkt[0] = 'T'; pkt[1] = 'M'; pkt[3] = epoch;
            pkt[4] = pkt[5] = pkt[6] = pkt[7] = 0;
            if (n_got == nrx) {
                pkt[2] = PKT_ACK;
                ::sendto(g_ufd, pkt, HDR_BYTES, 0, (sockaddr*)&g_peer, sizeof g_peer);
            } else if (millis() - last_data < 25) {
                /* the stream is still arriving: NAKing now would just ask for
                 * datagrams that are already in flight */
            } else {
                uint32_t miss[2] = {0, 0};
                int nmiss = 0;
                for (int i = 0; i < nrx; i++)
                    if (!bit_get(got, i)) { bit_set(miss, i); nmiss++; }
                g_udp_lost += (uint32_t)nmiss;
                pkt[2] = PKT_NAK;
                memcpy(pkt + HDR_BYTES, miss, 8);
                ::sendto(g_ufd, pkt, HDR_BYTES + 8, 0, (sockaddr*)&g_peer, sizeof g_peer);
            }
        }
        if (!moved) { beat(); vTaskDelay(1); }
    }
    /* a few extra ACKs so a lost final ACK does not strand the peer */
    pkt[0] = 'T'; pkt[1] = 'M'; pkt[2] = PKT_ACK; pkt[3] = epoch;
    for (int i = 0; i < 3; i++)
        ::sendto(g_ufd, pkt, HDR_BYTES, 0, (sockaddr*)&g_peer, sizeof g_peer);
    return true;
}

void link_task(void*) {
    for (;;) {
        xSemaphoreTake(g_go, portMAX_DELAY);
        if (g_ufd >= 0 && g_tx.total > 64) {
            if (!udp_exchange()) g_failed = true;
        } else {
            /* small control payloads (the barriers) stay on TCP */
            uint32_t idle = 0;
            while (g_tx.done < g_tx.total || g_rx.done < g_rx.total) {
                int moved = 0;
                for (int i = 0; i < 8; i++) {
                    const int a = stream_pump(g_tx, true, g_tx_ready);
                    const int b = stream_pump(g_rx, false, g_rx.total);
                    if (a < 0 || b < 0) { g_failed = true; break; }
                    if (a == 0 && b == 0) break;
                    moved += a + b;
                }
                if (g_failed) break;
                if (moved) { idle = 0; continue; }
                if (++idle > 30000) {
                    snprintf(g_err, sizeof g_err, "tcp stall rx=%u/%u",
                             (unsigned)g_rx.done, (unsigned)g_rx.total);
                    g_failed = true; break;
                }
                vTaskDelay(1);
            }
        }
        g_tx_total_stat += g_tx.done;
        g_rx_total_stat += g_rx.done;
        g_active = false;
    }
}

void udp_open(IPAddress peer_ip) {
    g_ufd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (g_ufd < 0) return;
    const int flags = fcntl(g_ufd, F_GETFL, 0);
    fcntl(g_ufd, F_SETFL, flags | O_NONBLOCK);
    sockaddr_in me{};
    me.sin_family = AF_INET;
    me.sin_port = htons(DATA_PORT);
    me.sin_addr.s_addr = INADDR_ANY;
    if (::bind(g_ufd, (sockaddr*)&me, sizeof me) < 0) { ::close(g_ufd); g_ufd = -1; return; }
    g_peer.sin_family = AF_INET;
    g_peer.sin_port = htons(DATA_PORT);
    g_peer.sin_addr.s_addr = (uint32_t)peer_ip;
    ::connect(g_ufd, (sockaddr*)&g_peer, sizeof g_peer);   /* filter to the peer */
}

void adopt(WiFiClient& c) {
    g_fd = c.fd();
    const int flags = fcntl(g_fd, F_GETFL, 0);
    fcntl(g_fd, F_SETFL, flags | O_NONBLOCK);
    int one = 1;
    setsockopt(g_fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    if (!g_task) {
        g_go = xSemaphoreCreateBinary();
        /* above loopTask (1) so the socket is serviced during a forward, well
         * below the WiFi/lwIP tasks */
        xTaskCreate(link_task, "tmlink", 4096, nullptr, 3, &g_task);
    }
}

}  // namespace

bool tm_link_up(void) { return g_fd >= 0 && g_client && g_client.connected(); }
const char* tm_link_info(void)  { return g_info; }
const char* tm_link_error(void) { return g_err; }
uint32_t tm_link_bytes_tx(void) { return g_tx_total_stat; }
uint32_t tm_link_bytes_rx(void) { return g_rx_total_stat; }
void tm_link_reset_stats(void)  { g_tx_total_stat = 0; g_rx_total_stat = 0;
                                  g_udp_retx = 0; g_udp_lost = 0; }
uint32_t tm_link_retx(void) { return g_udp_retx; }

bool tm_link_begin(int role, const char* ssid, const char* pass,
                   uint32_t timeout_ms) {
    const uint32_t t0 = millis();
    g_role = role;

    if (role == 0) {
        WiFi.mode(WIFI_AP);
        if (!WiFi.softAP(ssid, pass, /*channel=*/6, /*hidden=*/0,
                         /*max_connection=*/1)) {
            snprintf(g_info, sizeof g_info, "softAP failed");
            return false;
        }
        WiFi.setSleep(false);          /* modem sleep parks the radio between
                                        * beacons and costs ~1 beacon (100 ms)
                                        * of latency per stalled transfer */
        esp_wifi_set_ps(WIFI_PS_NONE);
        g_server.begin();
        g_server.setNoDelay(true);
        while (millis() - t0 < timeout_ms) {
            g_client = g_server.available();
            if (g_client) {
                g_client.setNoDelay(true);
                adopt(g_client);
                udp_open(g_client.remoteIP());
                snprintf(g_info, sizeof g_info, "ap %s peer %s",
                         WiFi.softAPIP().toString().c_str(),
                         g_client.remoteIP().toString().c_str());
                return true;
            }
            delay(20);
        }
        snprintf(g_info, sizeof g_info, "no peer joined %s", ssid);
        return false;
    }

    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, pass);
    while (WiFi.status() != WL_CONNECTED) {
        if (millis() - t0 > timeout_ms) {
            snprintf(g_info, sizeof g_info, "join %s failed", ssid);
            return false;
        }
        delay(50);
    }
    WiFi.setSleep(false);
    esp_wifi_set_ps(WIFI_PS_NONE);
    const IPAddress gw = WiFi.gatewayIP();
    while (millis() - t0 < timeout_ms) {
        if (g_client.connect(gw, LINK_PORT, 3000)) {
            g_client.setNoDelay(true);
            adopt(g_client);
            udp_open(gw);
            snprintf(g_info, sizeof g_info, "sta %s peer %s rssi %d",
                     WiFi.localIP().toString().c_str(), gw.toString().c_str(),
                     (int)WiFi.RSSI());
            return true;
        }
        delay(100);
    }
    snprintf(g_info, sizeof g_info, "connect %s:%u failed",
             gw.toString().c_str(), (unsigned)LINK_PORT);
    return false;
}

void tm_link_begin_exchange(const TmIoSeg* tx, int ntx,
                            const TmIoSeg* rx, int nrx) {
    stream_load(g_tx, tx, ntx);
    stream_load(g_rx, rx, nrx);
    g_tx_ready = 0;
    g_failed = false;
    g_active = true;
    xSemaphoreGive(g_go);
}

void tm_link_tx_ready(uint32_t cumulative_bytes) {
    g_tx_ready = cumulative_bytes;
}

bool tm_link_wait_rx(uint32_t cumulative_bytes, uint32_t timeout_ms) {
    const uint32_t t0 = millis();
    while (g_rx.done < cumulative_bytes) {
        if (g_failed) return false;
        if (millis() - t0 > timeout_ms) {
            snprintf(g_err, sizeof g_err, "wait_rx %u/%u",
                     (unsigned)g_rx.done, (unsigned)cumulative_bytes);
            return false;
        }
        beat();
        vTaskDelay(1);
    }
    return true;
}

bool tm_link_end_exchange(uint32_t timeout_ms) {
    const uint32_t t0 = millis();
    while (g_active) {
        if (millis() - t0 > timeout_ms) {
            snprintf(g_err, sizeof g_err, "end_exchange timeout");
            return false;
        }
        beat();
        vTaskDelay(1);
    }
    return !g_failed;
}

bool tm_link_exchange(const TmIoSeg* tx, int ntx,
                      const TmIoSeg* rx, int nrx, uint32_t timeout_ms) {
    if (!tm_link_up()) { snprintf(g_err, sizeof g_err, "down"); return false; }
    uint32_t total = 0;
    for (int i = 0; i < ntx; i++) total += (uint32_t)tx[i].len;
    tm_link_begin_exchange(tx, ntx, rx, nrx);
    tm_link_tx_ready(total);
    return tm_link_end_exchange(timeout_ms);
}

/* Idempotent UDP rendezvous.
 *
 * Both sides resend their tag until they see the peer's, and echo back any tag
 * they are not currently waiting on. That makes the barrier tolerant of lost
 * datagrams and of one board arriving a whole forward ahead of the other -
 * unlike the TCP version it replaced, which desynchronised permanently if a
 * run was interrupted mid-stream. */
bool tm_link_barrier(uint8_t tag, uint32_t timeout_ms) {
    if (g_ufd < 0) { snprintf(g_err, sizeof g_err, "no udp"); return false; }
    uint8_t pkt[16] = {'T', 'M', PKT_BAR, tag, 0, 0, 0, 0};
    const uint32_t t0 = millis();
    uint32_t last_send = 0;

    for (;;) {
        if (millis() - last_send >= 10) {
            last_send = millis();
            pkt[2] = PKT_BAR; pkt[3] = tag;
            pkt[4] = g_epoch;      /* node 0 publishes the epoch base */
            ::sendto(g_ufd, pkt, 8, 0, (sockaddr*)&g_peer, sizeof g_peer);
        }
        uint8_t in[64];
        for (;;) {
            const int n = ::recv(g_ufd, in, sizeof in, 0);
            if (n < 8) break;
            if (in[0] != 'T' || in[1] != 'M' || in[2] != PKT_BAR) continue;
            if (in[3] == tag) {
                /* node 1 takes node 0's epoch base, so the two boards always
                 * number the same exchange the same way even if a previous
                 * forward was interrupted part-way through */
                if (g_role == 1) g_epoch = in[4];
                /* seal it for the peer, then go */
                for (int i = 0; i < 3; i++)
                    ::sendto(g_ufd, pkt, 8, 0, (sockaddr*)&g_peer, sizeof g_peer);
                return true;
            }
            /* a tag we are not waiting on: echo it so the peer can finish */
            uint8_t echo[8] = {'T', 'M', PKT_BAR, in[3], 0, 0, 0, 0};
            ::sendto(g_ufd, echo, 8, 0, (sockaddr*)&g_peer, sizeof g_peer);
        }
        if (millis() - t0 > timeout_ms) {
            snprintf(g_err, sizeof g_err, "barrier %02x timeout", tag);
            return false;
        }
        beat();
        vTaskDelay(1);
    }
}

/* Start-of-forward reset: epochs are per forward, so an aborted run cannot
 * leave the two boards numbering their exchanges differently. */
/* Epochs run free rather than restarting at 0 each forward: a restart lets a
 * stale ACK from the previous forward's last layer be mistaken for this one's,
 * which silently ends a transfer the peer has not finished receiving. Node 0
 * jumps clear of anything still in flight and the start barrier hands the new
 * base to node 1. */
void tm_link_new_forward(void) { if (g_role == 0) g_epoch += 8; }

void tm_link_set_heartbeat(void (*fn)(void)) { g_hb = fn; }

uint64_t tm_link_bandwidth(uint8_t* tx_buf, uint8_t* rx_buf, int bytes, int n) {
    if (!tm_link_barrier(0x33, 20000)) return 0;
    const int64_t t0 = esp_timer_get_time();
    for (int i = 0; i < n; i++) {
        TmIoSeg t{tx_buf, bytes}, r{rx_buf, bytes};
        if (!tm_link_exchange(&t, 1, &r, 1, 30000)) return 0;
    }
    return (uint64_t)(esp_timer_get_time() - t0);
}
