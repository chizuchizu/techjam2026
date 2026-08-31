// espnow_min.cpp -- minimal ESP-NOW sanity app (isolate platform vs fleet logic).
// Both boards: init ESP-NOW, register recv, broadcast "HI|<n>" every 300ms,
// log every TX and RX with board-local t_ms. AP board hosts LINKNET; STA joins.
#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>

#define TAG "LINKFW-M"

static uint8_t bc[6] = {0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
static uint32_t ntx = 0, nrx = 0, last = 0;

static void on_recv(const esp_now_recv_info_t *info, const uint8_t *data, int len) {
    if (nrx < 30) {
        nrx++;
        Serial.printf("%s|rx|n=%u|len=%d|t_ms=%u|src=%02X:%02X:%02X:%02X:%02X:%02X|b0=%02X\n",
                      TAG, (uint32_t)nrx, len, (uint32_t)millis(),
                      info->src_addr[0],info->src_addr[1],info->src_addr[2],
                      info->src_addr[3],info->src_addr[4],info->src_addr[5], data[0]);
    } else { nrx++; }
}
static void on_sent(const uint8_t *mac, esp_now_send_status_t st) {
    if (ntx <= 30) {
        Serial.printf("%s|txcb|n=%u|status=%s\n", TAG, (uint32_t)ntx,
                      st == ESP_NOW_SEND_SUCCESS ? "OK" : "FAIL");
    }
}

void setup() {
    Serial.begin(115200);
    delay(200);
#ifdef ESPNOW_MIN_AP
    WiFi.mode(WIFI_AP);
    WiFi.softAP("LINKNET", "linkfast123");
    Serial.printf("%s|init|ap|ch=%d\n", TAG, WiFi.channel());
#else
    WiFi.mode(WIFI_STA);
    WiFi.setSleep(false);
    WiFi.begin("LINKNET", "linkfast123");
    for (int i = 0; i < 60 && WiFi.status() != WL_CONNECTED; i++) { delay(200); }
    Serial.printf("%s|init|sta|connected=%d|ch=%d\n", TAG, WiFi.status() == WL_CONNECTED, WiFi.channel());
#endif
    esp_now_init();
    esp_now_register_recv_cb(on_recv);
    esp_now_register_send_cb(on_sent);
    esp_now_peer_info_t peer = {};
    memcpy(peer.peer_addr, bc, 6);
    peer.channel = 0;
#ifdef ESPNOW_MIN_AP
    peer.ifidx = (wifi_interface_t)WIFI_IF_AP;
#else
    peer.ifidx = (wifi_interface_t)WIFI_IF_STA;
#endif
    esp_now_add_peer(&peer);
    Serial.println("US-ESPNOW-MIN-READY");
}

void loop() {
    uint32_t nw = millis();
    if (nw - last >= 300) {
        last = nw;
        ntx++;
        if (ntx > 40) return;
        char o[24];
        int n = snprintf(o, sizeof(o), "HI|%u", (uint32_t)ntx);
        esp_err_t er = esp_now_send(bc, (const uint8_t*)o, (size_t)n);
        Serial.printf("%s|tx|n=%u|rc=%d|t_ms=%u\n", TAG, (uint32_t)ntx, (int)er, (uint32_t)nw);
    }
    delay(1);
}
