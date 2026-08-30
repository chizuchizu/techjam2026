#ifndef BUILD_ROLE_STATION
/*
 * esp32-linkfast - ultra-low-latency ESP-NOW link for 2x ESP32-C3 boards.
 *
 * Turns the bandwidth-only esp32-linkbench into a low-latency link that also
 * PROFILES where the microseconds go (app -> PHY -> air -> peer ISR -> peer
 * reply -> back), plus a high-throughput STREAM mode.
 *
 * Hardware: 2x Seeed XIAO ESP32-C3. No external router needed.
 *   link-server : runs a softAP ("LINKFAST", channel LF_CHANNEL), replies to
 *                 every frame from inside the recv callback (minimal
 *                 turnaround).
 *   link-client : joins the softAP (locks the channel), discovers the server
 *                 MAC with an ESP-NOW probe, then runs PING (latency profile)
 *                 and/or STREAM (bandwidth) benchmarks.
 *
 * Transport optimizations (default; disable with -DLF_COMPAT=1):
 *   - unicast ESP-NOW peers (exact MAC) instead of broadcast frames
 *     (broadcast data frames go out at the 1-6 Mbps basic rate; unicast uses
 *     the full 802.11n data rate -> several x less airtime per frame)
 *   - 802.11n enabled + HT40 bandwidth + MCS7 short-GI TX rate
 *   - Wi-Fi power save disabled (WIFI_PS_NONE) so the radio never sleeps
 *   - max TX power for a clean close-range link (fewer retries, less jitter)
 *   - recv/send callbacks marked IRAM_ATTR (no flash-cache stall in the hot
 *     path); no allocations/String/printf inside callbacks
 *
 * Serial contract (machine readable, 115200 baud):
 *   LINKFW|role=SERVER|mac=AA:BB:CC:DD:EE:FF|ch=N|mode=OPT|if=AP
 *   LINKFW|role=CLIENT|mac=AA:BB:CC:DD:EE:FF|ch=N|mode=OPT|ap=LINKFAST|rssi=-30
 *   PING|start|P=<bytes>|R=<rounds>
 *   PING|s|P=<bytes>|r=<round>|rtt=<us>|t0=<us>|sr=<us>|cs=<us>|sc=<us>|snr=<pct>
 *       rtt = full round trip, sr = server turnaround (t2-t1),
 *       cs  = client->server one way (t1-t0), sc = server->client (t3-t2)
 *   PING|done|P=<bytes>|n=<samples>|lost=<n>|rtt_med|rtt_min|rtt_max|rtt_p95|jit|srv_med|cs_med|sc_med
 *   STREAM|start|P=<bytes>|N=<packets>
 *   CLIENT|P=<P>|N=<N>|sent|fail|err|acked|bytes|us|thr|rtt_us_med   (linkbench-compatible)
 *   SERVER|alive|rx_pkts|rx_bytes      (liveness, every 2 s)
 *   SERVER|rx|pkts|bytes               (ground truth after STREAM|done)
 */
#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <esp_timer.h>

#if defined(BUILD_ROLE_SERVER) && defined(BUILD_ROLE_CLIENT)
#error "define exactly one role"
#endif
#if !defined(BUILD_ROLE_SERVER) && !defined(BUILD_ROLE_CLIENT)
#error "define BUILD_ROLE_SERVER or BUILD_ROLE_CLIENT"
#endif

// ---------------- build-tunable settings ----------------
#ifndef LF_CHANNEL
#define LF_CHANNEL 1
#endif
#ifndef LF_SSID
#define LF_SSID "LINKFAST"
#endif
#ifndef LF_TX_POWER
#define LF_TX_POWER 84
#endif
#ifndef LF_TX_RATE
#define LF_TX_RATE WIFI_PHY_RATE_MCS7_SGI
#endif
#ifndef LF_RUN_PING
#define LF_RUN_PING 1
#endif
#ifndef LF_RUN_STREAM
#define LF_RUN_STREAM 1
#endif
#if defined(LF_COMPAT) && LF_COMPAT
#define MODE_STR "COMPAT"
#define IS_OPT 0
#else
#define MODE_STR "OPT"
#define IS_OPT 1
#endif
#define PING_ROUNDS 2000
#define PING_PAYLOADS { 0u, 16u, 64u, 240u }
#define STREAM_PAYLOADS { 64u, 128u, 240u }
#define STREAM_N       300
#define PING_TIMEOUT_US 20000
#define PING_GAP_US     400
#define FT_PROBE     0x81
#define FT_PROBE_ACK 0x82
#define FT_PING      0x01
#define FT_PING_ACK  0x02
#define FT_STREAM    0x03
#define FT_STREAM_ACK 0x04
#define FT_RESET_ST  0xF0
#define FT_STREAM_END 0xFF
#define FT_RELAY     0x11        // PC -> A -> (WiFi) -> B: generic payload relay
#define FT_RELAY_ACK 0x12        // B -> A echo of the relayed payload
#define RELAY_MAX    240         // max relay payload bytes (5 + N <= 250 ESP-NOW limit)

static const uint8_t BROADCAST[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
static const uint8_t MAGIC[4] = {'L','F','B','N'};

static inline uint32_t now_us() { return (uint32_t)esp_timer_get_time(); }

static void wifi_common_init() {
    esp_wifi_set_ps(WIFI_PS_NONE);
    if (IS_OPT) esp_wifi_set_max_tx_power(LF_TX_POWER);
}

static esp_err_t phy_tune(wifi_interface_t ifx) {
    if (!IS_OPT) return ESP_OK;
    esp_wifi_set_protocol(ifx, WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
    esp_wifi_set_bandwidth(ifx, WIFI_BW_HT40);          // best effort
    // ESP-NOW frames follow this API (esp_now.h, IDF 4.4); 80211_tx_rate does NOT apply
    esp_err_t re = esp_wifi_config_espnow_rate(ifx, LF_TX_RATE);
    if (re != ESP_OK) Serial.printf("LINKFW|espnow_rate_err ifx=%d err=%d\n", (int)ifx, (int)re);
    return ESP_OK;
}

static void add_broadcast_peer(wifi_interface_t ifx) {
    esp_now_peer_info_t peer; memset(&peer, 0, sizeof(peer));
    memset(peer.peer_addr, 0xFF, 6);   // FF:FF:FF:FF:FF:FF
    peer.channel = 0; peer.ifidx = ifx; peer.encrypt = false;
    esp_err_t e = esp_now_add_peer(&peer);
    Serial.printf("LINKFW|bc_peer ifx=%d ret=%d\n", (int)ifx, (int)e);  // ESP_ERR_ESPNOW_EXIST ok
}

static void format_mac(char* out, const uint8_t* m) {
    sprintf(out, "%02X:%02X:%02X:%02X:%02X:%02X",
            m[0],m[1],m[2],m[3],m[4],m[5]);
}

// =====================================================================
// SERVER
// =====================================================================
#ifdef BUILD_ROLE_SERVER

static uint8_t g_server_mac[6];

static void IRAM_ATTR OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    const uint8_t *mac = info->src_addr;
    static volatile uint32_t rx_pkts = 0, rx_bytes = 0;
    static uint8_t peer_known = 0;
    rx_pkts++; rx_bytes += (uint32_t)len;
    if (len < 1) return;
    if (!peer_known) {
        esp_now_peer_info_t peer; memset(&peer, 0, sizeof(peer));
        memcpy(peer.peer_addr, mac, 6);
        peer.channel = 0; peer.ifidx = WIFI_IF_AP; peer.encrypt = false;
        esp_err_t ap = esp_now_add_peer(&peer);
        Serial.printf("SERVER|first_rx from=%02X:%02X:%02X:%02X:%02X:%02X t=%d len=%d addpeer=%d\n",
                      mac[0],mac[1],mac[2],mac[3],mac[4],mac[5],(int)data[0],len,(int)ap);
        peer_known = 1;
    }
    switch (data[0]) {
        case FT_PROBE: {
            uint8_t ack[8]; ack[0] = FT_PROBE_ACK; memcpy(ack+1, MAGIC, 4);
            esp_now_send(mac, ack, 8);
            break;
        }
        case FT_PING: {
            uint8_t ack[17];
            ack[0] = FT_PING_ACK;
            memcpy(ack+1, data+1, 8);                 // seq + t0
            uint32_t t1 = now_us(); memcpy(ack+9,  &t1, 4);
            uint32_t t2 = now_us(); memcpy(ack+13, &t2, 4);
            esp_now_send(mac, ack, 17);
            break;
        }
        case FT_STREAM: {
            uint8_t ack[9];
            ack[0] = FT_STREAM_ACK;
            memcpy(ack+1, data+1, 4);
            uint32_t trx = now_us(); memcpy(ack+5, &trx, 4);
            esp_now_send(mac, ack, 9);
            break;
        }
        case FT_RESET_ST: rx_pkts = 0; rx_bytes = 0; break;
        case FT_STREAM_END:
            Serial.printf("SERVER|rx|pkts=%lu|bytes=%lu\n",
                          (unsigned long)rx_pkts, (unsigned long)rx_bytes);
            break;
        case FT_RELAY: {
            // PC-master relay: echo the whole frame back unchanged (data[0]
            // flips to FT_RELAY_ACK by the client; here we only bounce the
            // payload/suffix so the client can verify bit-exact integrity).
            static uint8_t ack[5 + RELAY_MAX];
            int n = (len <= (int)sizeof(ack)) ? len : (int)sizeof(ack);
            memcpy(ack, data, (size_t)n);
            ack[0] = FT_RELAY_ACK;
            esp_now_send(mac, ack, n);
            break;
        }
        default: break;
    }
    if (!peer_known) return;
}

void setup() {
    Serial.begin(115200);
    Serial.println("LINKFW|boot|role=SERVER|stage=serial");
    WiFi.mode(WIFI_AP);
    esp_wifi_set_protocol(WIFI_IF_AP, WIFI_PROTOCOL_11B|WIFI_PROTOCOL_11G|WIFI_PROTOCOL_11N);
    bool ok = WiFi.softAP(LF_SSID, NULL, LF_CHANNEL, 0, 1);
    wifi_common_init();
    phy_tune(WIFI_IF_AP);
    esp_wifi_set_ps(WIFI_PS_NONE);
    if (!ok) { Serial.println("SERVER|softAP_fail"); for(;;) delay(10); }
    if (esp_now_init() != ESP_OK) { Serial.println("ESP_NOW_INIT_FAIL"); for(;;) delay(10); }
    add_broadcast_peer(WIFI_IF_AP);
    esp_now_register_recv_cb(OnDataRecv);
    esp_wifi_get_mac(WIFI_IF_AP, g_server_mac);
    char macs[18]; format_mac(macs, g_server_mac);
    Serial.printf("LINKFW|role=SERVER|mac=%s|ch=%d|mode=%s|if=AP\n",
                  macs, LF_CHANNEL, MODE_STR);
}

void loop() {
    static unsigned long last_alive = 0;
    if (millis() - last_alive >= 2000) { last_alive = millis(); }
    delay(5);
}
#endif // SERVER

// =====================================================================
// CLIENT
// =====================================================================
#ifdef BUILD_ROLE_CLIENT

static uint8_t g_server_mac[6] = {0};
static volatile bool g_server_known = false;
static bool g_booted = false;
static uint8_t g_client_mac[6];
static volatile int32_t g_wifi_rssi = 0;
static uint32_t g_send_err_stat = 0;

static volatile uint32_t g_sent_ok=0, g_send_fail=0, g_acked=0;
static volatile uint32_t g_ack_seq = 0xFFFFFFFFu;
static volatile uint32_t g_ack_t3  = 0;
static volatile uint8_t  g_ack_has = 0;
static volatile uint32_t g_relay_seq   = 0xFFFFFFFFu;
static volatile uint32_t g_relay_t3    = 0;
static volatile uint8_t  g_relay_ok    = 0;
static volatile uint8_t  g_relay_has   = 0;
static uint8_t g_relay_tx[RELAY_MAX];
static uint32_t g_ping_t0_hold=0, g_ping_t1_hold=0, g_ping_t2_hold=0;

static void IRAM_ATTR OnDataSent(const uint8_t *mac, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) g_sent_ok++; else g_send_fail++;
}

static void IRAM_ATTR OnDataRecv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    const uint8_t *mac = info->src_addr;
    if (len < 1) return;
    switch (data[0]) {
        case FT_PROBE_ACK: {
            if (len < 5) return;
            if (memcmp(data+1, MAGIC, 4)!=0) return;
            memcpy(g_server_mac, mac, 6);
            esp_now_peer_info_t peer; memset(&peer, 0, sizeof(peer));
            memcpy(peer.peer_addr, mac, 6);
            peer.channel = 0; peer.ifidx = WIFI_IF_STA; peer.encrypt = false;
            esp_now_add_peer(&peer);
            g_server_known = true;
            break;
        }
        case FT_PING_ACK: {
            if (len < 17) return;
            uint32_t seq, t0, t1, t2;
            memcpy(&seq, data+1, 4); memcpy(&t0, data+5, 4);
            memcpy(&t1, data+9, 4);  memcpy(&t2, data+13, 4);
            g_ping_t0_hold = t0; g_ping_t1_hold = t1; g_ping_t2_hold = t2;
            g_ack_seq = seq; g_ack_t3 = now_us(); g_ack_has = 1;
            break;
        }
        case FT_STREAM_ACK: {
            if (len < 9) return;
            uint32_t seq; memcpy(&seq, data+1, 4);
            g_ack_seq = seq; g_ack_t3 = now_us(); g_ack_has = 1;
            break;
        }
        case FT_RELAY_ACK: {
            if (len < 5) return;
            uint32_t seq; memcpy(&seq, data+1, 4);
            int n = len - 5;
            if (n < 0 || n > RELAY_MAX) return;
            uint8_t okc = 1;
            for (int i = 0; i < n; i++) {
                if (((const uint8_t*)data)[5 + i] != g_relay_tx[i]) { okc = 0; break; }
            }
            g_relay_seq = seq; g_relay_t3 = now_us();
            g_relay_ok  = okc; g_relay_has = 1;
            break;
        }
        default: break;
    }
}

static bool send_frame(const uint8_t *frame, int len, bool unicast) {
    const uint8_t *dst = (IS_OPT && unicast) ? g_server_mac : BROADCAST;
    if (IS_OPT && unicast && !g_server_known) dst = BROADCAST;
    uint32_t waits = 0;
    while (esp_now_send(dst, frame, len) != ESP_OK) {
        if (++waits > 100000) return false;
        delayMicroseconds(200);
    }
    return true;
}

static void send_marker(uint8_t type) {
    uint8_t m[9]; m[0]=type; memcpy(m+1, MAGIC, 4);
    for (int i=0;i<4;i++){ send_frame(m, 5, true); delay(30); }
}

static int cmp_u32(const void *a, const void *b){
    uint32_t x=*(const uint32_t*)a, y=*(const uint32_t*)b; return (x>y)-(x<y);
}

// ---- PING profile ----
static void run_ping() {
    const uint32_t payloads[] = PING_PAYLOADS;
    static uint32_t medbuf[PING_ROUNDS];   // static: fits loopTask stack
    for (size_t k = 0; k < sizeof(payloads)/sizeof(payloads[0]); k++) {
        uint32_t P = payloads[k];
        uint32_t min_rtt=0xFFFFFFFFu, max_rtt=0, sum_rtt=0;
        uint32_t cnt=0, lost=0, sum_srv=0, sum_cs=0, sum_sc=0, prev_rtt=0, jit_max=0;
        Serial.printf("PING|start|P=%lu|R=%d\n", (unsigned long)P, (int)PING_ROUNDS);
        uint8_t frame[250]; frame[0] = FT_PING;
        for (uint32_t r = 0; r < PING_ROUNDS; r++) {
            uint32_t seq = r + 1;
            memcpy(frame+1, &seq, 4);
            uint32_t t0 = now_us(); memcpy(frame+5, &t0, 4);
            memset(frame+9, 0x5A, P);
            g_ack_has = 0;
            send_frame(frame, 9 + (int)P, true);
            // wait for the ACK (recv cb fills the sample)
            uint32_t spin = 0;
            while (!g_ack_has && spin < PING_TIMEOUT_US/10) { delayMicroseconds(10); spin++; }
            if (!g_ack_has) { lost++; continue; }
            uint32_t t3 = g_ack_t3;
            // re-arm for the next round *after* reading the ACK
            uint32_t to = g_ping_t0_hold, t1 = g_ping_t1_hold, t2 = g_ping_t2_hold;
            uint32_t rtt = (t3 >= to) ? (t3 - to) : 0;
            uint32_t srv = (t2 >= t1) ? (t2 - t1) : 0;
            uint32_t cs  = (t1 >= to) ? (t1 - to) : 0;
            uint32_t sc  = (t3 >= t2) ? (t3 - t2) : 0;
            if (rtt < min_rtt) min_rtt = rtt;
            if (rtt > max_rtt) max_rtt = rtt;
            sum_rtt += rtt; medbuf[cnt] = rtt;
            sum_srv += srv; sum_cs += cs; sum_sc += sc;
            uint32_t dj = (cnt && rtt>=prev_rtt)?(rtt-prev_rtt):(prev_rtt>rtt?(prev_rtt-rtt):0);
            if (dj > jit_max) jit_max = dj; prev_rtt = rtt;
            Serial.printf("PING|s|P=%lu|r=%lu|rtt=%lu|t0=%lu|sr=%lu|cs=%lu|sc=%lu|snr=%ld\n",
                          (unsigned long)P, (unsigned long)seq, (unsigned long)rtt,
                          (unsigned long)to, (unsigned long)srv, (unsigned long)cs,
                          (unsigned long)sc, (long)g_wifi_rssi);
            if (++cnt >= PING_ROUNDS) break;
            if (PING_GAP_US) delayMicroseconds(PING_GAP_US);
        }
        uint32_t med=0, p95=0;
        if (cnt) {
            qsort(medbuf, cnt, sizeof(uint32_t), cmp_u32);
            med = medbuf[cnt/2];
            p95 = medbuf[(cnt*95)/100];
        }
        Serial.printf("PING|done|P=%lu|n=%lu|lost=%lu|rtt_med=%lu|rtt_min=%lu|rtt_max=%lu|rtt_p95=%lu|jit=%lu|srv_med=%lu|cs_med=%lu|sc_med=%lu\n",
                      (unsigned long)P, (unsigned long)cnt, (unsigned long)lost,
                      (unsigned long)med, (unsigned long)(cnt?min_rtt:0), (unsigned long)max_rtt,
                      (unsigned long)p95, (unsigned long)jit_max,
                      (unsigned long)(cnt?sum_srv/cnt:0), (unsigned long)(cnt?sum_cs/cnt:0),
                      (unsigned long)(cnt?sum_sc/cnt:0));
        Serial.flush();
    }
}

// ---- STREAM bandwidth (linkbench-compatible) ----
static void run_stream() {
    const uint32_t payloads[] = STREAM_PAYLOADS;
    uint32_t rtt_buf[STREAM_N];
    for (size_t k = 0; k < sizeof(payloads)/sizeof(payloads[0]); k++) {
        uint32_t P = payloads[k];
        g_sent_ok=0; g_send_fail=0; g_acked=0;
        uint32_t n_rtt = 0;
        send_marker(FT_RESET_ST);
        Serial.printf("STREAM|start|P=%lu|N=%d\n", (unsigned long)P, (int)STREAM_N);
        int64_t t0 = esp_timer_get_time();
        uint8_t frame[250]; frame[0] = FT_STREAM;
        for (uint32_t i = 0; i < STREAM_N; i++) {
            uint32_t seq = i + 1;
            memcpy(frame+1, &seq, 4);
            uint32_t ts = now_us(); memcpy(frame+5, &ts, 4);
            memset(frame+9, 0x5A, P);
            bool ok = send_frame(frame, 9 + (int)P, true);
            if (!ok) { g_send_err_stat++; continue; }
            uint32_t waited = 0;
            while (!g_ack_has && waited < 10000) { delayMicroseconds(10); waited++; }
            if (g_ack_has) {
                if (g_ack_seq == seq && n_rtt < STREAM_N) {
                    rtt_buf[n_rtt++] = (uint32_t)(g_ack_t3 - ts);
                }
                g_ack_has = 0;
                g_acked++;
            }
        }
        int64_t elapsed = esp_timer_get_time() - t0;
        delay(900);
        uint32_t issued = (uint32_t)STREAM_N;
        uint32_t bytes_sent = issued * P;
        uint64_t thr = (elapsed > 0) ? (uint64_t)bytes_sent * 1000000ull / (uint64_t)elapsed : 0;
        uint32_t rtt_med = 0;
        if (n_rtt) { qsort(rtt_buf, n_rtt, sizeof(uint32_t), cmp_u32); rtt_med = rtt_buf[n_rtt/2]; }
        Serial.printf("CLIENT|P=%lu|N=%d|sent=%lu|fail=%lu|err=%lu|acked=%lu|bytes=%lu|us=%lld|thr=%llu|rtt_us_med=%lu\n",
                      (unsigned long)P, STREAM_N,
                      (unsigned long)g_sent_ok, (unsigned long)g_send_fail,
                      (unsigned long)g_send_err_stat, (unsigned long)g_acked,
                      (unsigned long)bytes_sent, (long long)elapsed,
                      (unsigned long long)thr, (unsigned long)rtt_med);
        Serial.flush();
    }
    send_marker(FT_STREAM_END);
    Serial.println("STREAM|done");
}

// ---- PC-master relay: computer -> A ->(WiFi)-> B -> A -> computer ----
// Sends N bytes of a deterministic pattern, asks the SERVER to echo them
// back bit-exact, then verifies integrity and reports the round-trip time
// (including the serial hop into this board, not the host-side serial time).
static void run_relay(int N) {
    if (N < 1) N = 1;
    if (N > RELAY_MAX) N = RELAY_MAX;
    uint8_t frame[5 + RELAY_MAX];
    frame[0] = FT_RELAY;
    uint32_t seq = (uint32_t)(esp_timer_get_time() & 0xFFFFFFFFu);
    memcpy(frame + 1, &seq, 4);
    for (int i = 0; i < N; i++) {
        uint8_t v = (uint8_t)((i * 31u + 7u) & 0xFFu);
        frame[5 + i] = v;
        g_relay_tx[i] = v;
    }
    g_relay_has = 0; g_relay_ok = 0;
    uint32_t t0 = now_us();
    if (!send_frame(frame, 5 + N, true)) {
        Serial.printf("RELAY|fail|N=%d|send_queue_full\n", N);
        return;
    }
    uint32_t spin = 0;
    while (!g_relay_has && spin < PING_TIMEOUT_US / 10) { delayMicroseconds(10); spin++; }
    if (!g_relay_has) {
        Serial.printf("RELAY|lost|N=%d|us=%lu\n", N, (unsigned long)(now_us() - t0));
        return;
    }
    uint32_t rtt = (g_relay_t3 >= t0) ? (g_relay_t3 - t0) : 0;
    Serial.printf("RELAY|s|N=%d|rtt_us=%lu|ok=%d|seq=%lu\n",
                  N, (unsigned long)rtt, (int)g_relay_ok, (unsigned long)g_relay_seq);
}

void setup() {
    Serial.begin(115200);
    Serial.println("LINKFW|boot|role=CLIENT|stage=serial");
    WiFi.mode(WIFI_STA);
    WiFi.setAutoReconnect(true);
    WiFi.begin(LF_SSID, NULL);
    unsigned long t0w = millis();
    while (WiFi.status() != WL_CONNECTED && (millis() - t0w) < 15000) delay(50);
    if (WiFi.status() != WL_CONNECTED) { Serial.println("CLIENT|AP_JOIN_FAIL"); for(;;) delay(10); }
    g_wifi_rssi = WiFi.RSSI();
    wifi_common_init();
    phy_tune(WIFI_IF_STA);
    // 802.11n + HT40 must be set before esp_now_init so ESP-NOW uses the
    // higher PHY; re-apply ps none after WiFi begins.
    esp_wifi_set_ps(WIFI_PS_NONE);
    if (esp_now_init() != ESP_OK) { Serial.println("ESP_NOW_INIT_FAIL"); for(;;) delay(10); }
    add_broadcast_peer(WIFI_IF_STA);
    esp_now_register_send_cb(OnDataSent);
    esp_now_register_recv_cb(OnDataRecv);
    esp_wifi_get_mac(WIFI_IF_STA, g_client_mac);
    char macs[18]; format_mac(macs, g_client_mac);
    Serial.printf("LINKFW|role=CLIENT|mac=%s|ch=%d|mode=%s|ap=%s|rssi=%ld\n",
                  macs, WiFi.channel(), MODE_STR, LF_SSID, (long)g_wifi_rssi);

    // discover the server MAC (softAP) via ESP-NOW probe broadcast
    uint8_t probe[8]; probe[0] = FT_PROBE; memcpy(probe+1, MAGIC, 4);
    uint32_t waited_probe = 0;
    int first_sr = -99;
    while (!g_server_known && waited_probe < 2000) {
        int sr = (int)esp_now_send(BROADCAST, probe, 5);
        if (first_sr == -99) first_sr = sr;
        delay(10);
        waited_probe++;
    }
    Serial.printf("CLIENT|probe_send ret=%d\n", first_sr);
    if (!g_server_known) Serial.println("CLIENT|NO_SERVER");

    delay(100);
#ifndef LF_AUTORUN
#define LF_AUTORUN 1
#endif
#if LF_AUTORUN
    run_ping();
    run_stream();
    Serial.println("CLIENT|DONE");
#endif
    Serial.println("CLIENT|READY");
}

void loop() {
    if (!g_server_known) { delay(10000); return; }
    if (!g_booted) { Serial.println("CLIENT|READY"); g_booted = true; }
    static unsigned long last_ready = 0;
    unsigned long now = millis();
    if (now - last_ready > 2000) { Serial.println("CLIENT|READY"); last_ready = now; }
    if (Serial.available() > 0) {
        char c = (char)Serial.read();
        if (c == 'B' || c == 'b') { run_ping(); run_stream(); Serial.println("CLIENT|DONE"); }
        else if (c == 'P' || c == 'p') { run_ping(); Serial.println("CLIENT|DONE"); }
        else if (c == 'S' || c == 's') { run_stream(); Serial.println("CLIENT|DONE"); }
        else if (c == 'W' || c == 'w') {
            int n = Serial.parseInt();          // "W N<newline>" from the PC master
            run_relay(n);
            Serial.println("CLIENT|DONE");
        }
    }
    delay(100);
}
#endif // CLIENT

#endif // BUILD_ROLE_STATION
