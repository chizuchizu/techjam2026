/*
 * esp32-linkbench - ESP-NOW 2-node link bandwidth benchmark for XIAO ESP32-C3.
 *
 * Two boards pair automatically over ESP-NOW (WIFI_STA, no router needed).
 *   link-server  : receives payload frames, replies 8-byte ACK (seq + recv us).
 *   link-client  : pumps back-to-back frames of P bytes, counts send status
 *                  and ACKs, reports achieved useful throughput.
 *
 * Serial output is a machine-readable contract:
 *   LINKFW|role=SERVER|mac=AA:BB:CC:DD:EE:FF
 *   LINKFW|role=CLIENT|mac=AA:BB:CC:DD:EE:FF
 *   SERVER|alive|rx_pkts=N|rx_bytes=B          (every 2 s)
 *   CLIENT|P=P|N=N|sent=S|fail=F|err=E|acked=A|bytes=B|us=U|thr=T
 *   CLIENT|DONE
 */
#include <Arduino.h>
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>

#if defined(BUILD_ROLE_SERVER) && defined(BUILD_ROLE_CLIENT)
#error "define exactly one role"
#endif
#if !defined(BUILD_ROLE_SERVER) && !defined(BUILD_ROLE_CLIENT)
#error "define BUILD_ROLE_SERVER or BUILD_ROLE_CLIENT"
#endif

static const uint8_t BROADCAST[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
static const int N_PACKETS = 300;
static const int PAYLOADS[3] = {64, 128, 240};

static void wifi_espnow_init(uint8_t *mac_out) {
    WiFi.mode(WIFI_STA);
    WiFi.disconnect();
    esp_wifi_set_ps(WIFI_PS_NONE);
#ifdef LINK_RATE_54M
    esp_wifi_config_80211_tx_rate(WIFI_IF_STA, WIFI_PHY_RATE_54M);
#endif
    if (esp_now_init() != ESP_OK) {
        Serial.println("ESP_NOW_INIT_FAIL");
        for (;;) delay(10);
    }
    esp_read_mac(mac_out, ESP_MAC_WIFI_STA);
}

static void add_peer(const uint8_t *addr, int channel) {
    esp_now_peer_info_t peer;
    memset(&peer, 0, sizeof(peer));
    memcpy(peer.peer_addr, addr, 6);
    peer.channel = (int8_t)channel;
    peer.ifidx = WIFI_IF_STA;
    peer.encrypt = false;
    esp_now_add_peer(&peer);   // ESP_ERR_ESPNOW_EXIST is fine
}

#ifdef BUILD_ROLE_SERVER
static volatile uint32_t g_rx_pkts = 0, g_rx_bytes = 0;
static unsigned long g_last_alive = 0;

void OnDataRecv(const uint8_t *mac, const uint8_t *data, int len) {
    g_rx_pkts++;
    g_rx_bytes += (uint32_t)len;
    if (len < 8) return;                       // need seq for the ACK
    uint32_t seq;
    memcpy(&seq, data, 4);
    if (seq == 0xFFFFFFFFu) {                  // end-of-benchmark marker
        Serial.printf("SERVER|rx|pkts=%lu|bytes=%lu\n",
                      (unsigned long)g_rx_pkts, (unsigned long)g_rx_bytes);
        return;
    }
    if (seq == 0xFFFFFFFEu) {                  // run-start marker: reset counters
        g_rx_pkts = 0; g_rx_bytes = 0;
        return;
    }
    uint8_t ack[8];
    memcpy(ack, data, 4);                      // echo seq (u32 LE)
    uint32_t us = (uint32_t)esp_timer_get_time();
    memcpy(ack + 4, &us, 4);                   // server recv us (u32 LE)
    add_peer(mac, 1);
    esp_now_send(mac, ack, 8);
}

void setup() {
    Serial.begin(115200);
    uint8_t mac[6] = {0};
    wifi_espnow_init(mac);
    esp_now_register_recv_cb(OnDataRecv);
    Serial.printf("LINKFW|role=SERVER|mac=%02X:%02X:%02X:%02X:%02X:%02X\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
}

void loop() {
    if (millis() - g_last_alive >= 2000) {
        g_last_alive = millis();
        Serial.printf("SERVER|alive|rx_pkts=%lu|rx_bytes=%lu\n",
                      (unsigned long)g_rx_pkts, (unsigned long)g_rx_bytes);
    }
    delay(5);
}
#endif  // BUILD_ROLE_SERVER

#ifdef BUILD_ROLE_CLIENT
static volatile uint32_t g_sent_ok = 0, g_send_fail = 0, g_send_err = 0, g_acked = 0;
static uint32_t g_sent_us[N_PACKETS];            // client us per seq (for RTT)
static uint32_t g_rtt_buf[N_PACKETS];
static volatile uint32_t g_rtt_n = 0;

static int cmp_u32(const void *a, const void *b) {
    uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b;
    return (x > y) - (x < y);
}

void OnDataSent(const uint8_t *mac, esp_now_send_status_t status) {
    if (status == ESP_NOW_SEND_SUCCESS) g_sent_ok++;
    else g_send_fail++;
}

void OnDataRecv(const uint8_t *mac, const uint8_t *data, int len) {
    if (len < 8) return;
    g_acked++;                                   // server ACK is 8 bytes
    uint32_t seq;
    memcpy(&seq, data, 4);                       // echo seq (u32 LE)
    if (seq < N_PACKETS && g_sent_us[seq] != 0 && g_rtt_n < N_PACKETS) {
        g_rtt_buf[g_rtt_n++] = (uint32_t)esp_timer_get_time() - g_sent_us[seq];
    }
}

void setup() {
    Serial.begin(115200);
    uint8_t mac[6] = {0};
    wifi_espnow_init(mac);
    add_peer(BROADCAST, 1);
    esp_now_register_send_cb(OnDataSent);
    esp_now_register_recv_cb(OnDataRecv);
    Serial.printf("LINKFW|role=CLIENT|mac=%02X:%02X:%02X:%02X:%02X:%02X\n",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    delay(3000);                       // let both sides settle on channel 1

    uint8_t marker0[240];
    memset(marker0, 0xAB, sizeof(marker0));
    uint32_t rseq = 0xFFFFFFFEu;
    memcpy(marker0, &rseq, 4);
    for (int i = 0; i < 4; i++) { esp_now_send(BROADCAST, marker0, sizeof(marker0)); delay(20); }
    delay(100);

    uint8_t payload[240];
    memset(payload, 0xAB, sizeof(payload));
    for (int k = 0; k < 3; k++) {
        int P = PAYLOADS[k];
        g_sent_ok = 0; g_send_fail = 0; g_send_err = 0; g_acked = 0;
        int64_t t0 = esp_timer_get_time();
        for (int i = 0; i < N_PACKETS; i++) {
            uint32_t seq = (uint32_t)i;
            memcpy(payload, &seq, 4);
            uint32_t us = (uint32_t)esp_timer_get_time();
            g_sent_us[seq] = us;
            memcpy(payload + 4, &us, 4);
            // backpressure: retry until the MAC queue accepts (sustainable
            // rate); a rejected send means the queue is full, so the wait is
            // real delivered-frame pacing, not an artificial app delay.
            uint32_t wait_loops = 0;
            while (esp_now_send(BROADCAST, payload, P) != ESP_OK) {
                wait_loops++;
                delayMicroseconds(200);
                if (wait_loops > 50000) break;   // ~10 s hard cap
            }
            g_send_err += (wait_loops > 50000) ? 1 : 0;
        }
        int64_t elapsed = esp_timer_get_time() - t0;
        delay(900);                      // drain the ACK window
        uint32_t issued = (uint32_t)N_PACKETS - g_send_err;
        uint32_t bytes_sent = issued * (uint32_t)P;
        uint64_t thr = (elapsed > 0) ? (uint64_t)bytes_sent * 1000000ull / (uint64_t)elapsed : 0;
        uint32_t rtt_med = 0;
        if (g_rtt_n) {
            uint32_t tmp[N_PACKETS];
            memcpy(tmp, g_rtt_buf, g_rtt_n * sizeof(uint32_t));
            qsort(tmp, g_rtt_n, sizeof(uint32_t), cmp_u32);
            rtt_med = tmp[g_rtt_n / 2];
        }
        Serial.printf("CLIENT|P=%d|N=%d|sent=%lu|fail=%lu|err=%lu|acked=%lu|bytes=%lu|us=%lld|thr=%llu|rtt_us_med=%lu\n",
                      P, N_PACKETS, (unsigned long)g_sent_ok, (unsigned long)g_send_fail,
                      (unsigned long)g_send_err, (unsigned long)g_acked, (unsigned long)bytes_sent,
                      (long long)elapsed, (unsigned long long)thr, (unsigned long)rtt_med);
        Serial.flush();
    }
    Serial.println("CLIENT|DONE");
    // end-of-benchmark marker so the server reports ground-truth rx
    delay(200);
    uint8_t marker[240];
    memset(marker, 0xAB, sizeof(marker));
    uint32_t fseq = 0xFFFFFFFFu;
    memcpy(marker, &fseq, 4);
    for (int i = 0; i < 8; i++) {
        esp_now_send(BROADCAST, marker, sizeof(marker));
        delay(50);
    }
    Serial.println("CLIENT|MARKER_SENT");
}

void loop() { delay(10000); }
#endif  // BUILD_ROLE_CLIENT
