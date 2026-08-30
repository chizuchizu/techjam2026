/*
 * pc_master_uart_bridge.cpp - transparent WiFi/TCP <-> UART sidecar.
 *
 * This is deliberately a link-only image.  The paired radio-free compute C3
 * runs the esp32-uart-worker environment from the case-2 model project.  The
 * sidecar never buffers a complete tensor: TCP backpressure and the UART
 * hardware queues stream the existing M/R/T protocol in both directions.
 */
#include <Arduino.h>
#include <HardwareSerial.h>
#include <WiFi.h>

#if __has_include("../secrets.h")
#include "../secrets.h"
#define LB_HAS_WIFI_SECRETS 1
#else
#define LB_WIFI_SSID ""
#define LB_WIFI_PASSWORD ""
#define LB_HAS_WIFI_SECRETS 0
#endif

#ifndef LB_WIFI_SSID
#error "secrets.h must define LB_WIFI_SSID"
#endif
#ifndef LB_WIFI_PASSWORD
#error "secrets.h must define LB_WIFI_PASSWORD"
#endif
#ifndef LB_TCP_PORT
#define LB_TCP_PORT 5000
#endif
#ifndef LB_UART_BAUD
#define LB_UART_BAUD 2000000
#endif
#ifndef LB_UART_RX_PIN
#define LB_UART_RX_PIN D7
#endif
#ifndef LB_UART_TX_PIN
#define LB_UART_TX_PIN D6
#endif
#ifndef LB_FALLBACK_AP_PASSWORD
#define LB_FALLBACK_AP_PASSWORD "transformer"
#endif

static HardwareSerial g_worker_uart(1);
static WiFiServer g_server(LB_TCP_PORT);
static WiFiClient g_client;
static bool g_server_started = false;
static uint32_t g_next_wifi_retry_ms = 0;
static uint8_t g_tcp_to_uart[1024];
static uint8_t g_uart_to_tcp[1024];

static void start_server(void) {
    if (g_server_started) return;
    g_server.begin();
    g_server.setNoDelay(true);
    g_server_started = true;
    IPAddress ip = LB_HAS_WIFI_SECRETS ? WiFi.localIP() : WiFi.softAPIP();
    Serial.printf("BRIDGE|ready|ip=%s|port=%u|uart=%u|free=%u\n",
                  ip.toString().c_str(), (unsigned)LB_TCP_PORT,
                  (unsigned)LB_UART_BAUD, (unsigned)ESP.getFreeHeap());
}

static void start_fallback_ap(void) {
    uint64_t mac = ESP.getEfuseMac();
    char ssid[32];
    snprintf(ssid, sizeof ssid, "TM-BRIDGE-%04X", (unsigned)(mac & 0xffff));
    WiFi.mode(WIFI_AP);
    bool ok = WiFi.softAP(ssid, LB_FALLBACK_AP_PASSWORD, 1, false, 1);
    Serial.printf("BRIDGE|ap|ssid=%s|ok=%d|password=%s\n", ssid, ok ? 1 : 0,
                  LB_FALLBACK_AP_PASSWORD);
    if (ok) start_server();
}

static void service_wifi_state(void) {
#if LB_HAS_WIFI_SECRETS
    if (WiFi.status() == WL_CONNECTED) {
        start_server();
        return;
    }
    const uint32_t now = millis();
    if ((int32_t)(now - g_next_wifi_retry_ms) >= 0) {
        Serial.printf("BRIDGE|wifi_connecting|ssid=%s|free=%u\n", LB_WIFI_SSID,
                      (unsigned)ESP.getFreeHeap());
        WiFi.begin(LB_WIFI_SSID, LB_WIFI_PASSWORD);
        g_next_wifi_retry_ms = now + 15000;
    }
#endif
}

static void discard_worker_bytes(void) {
    while (g_worker_uart.available() > 0) {
        int n = g_worker_uart.available();
        if (n > (int)sizeof g_uart_to_tcp) n = sizeof g_uart_to_tcp;
        g_worker_uart.read(g_uart_to_tcp, (size_t)n);
    }
}

static void accept_client(void) {
    if (!g_server_started) return;
    if (g_client && !g_client.connected()) {
        g_client.stop();
        Serial.println("BRIDGE|tcp_disconnected");
    }
    if (g_client && g_client.connected()) return;

    /* Never leak the tail of an abandoned worker reply into a new stream. */
    discard_worker_bytes();
    WiFiClient incoming = g_server.available();
    if (incoming) {
        g_client = incoming;
        g_client.setNoDelay(true);
        Serial.printf("BRIDGE|tcp_connected|peer=%s\n",
                      g_client.remoteIP().toString().c_str());
    }
}

static void pump_tcp_to_uart(void) {
    int available = g_client.available();
    if (available <= 0) return;
    int room = g_worker_uart.availableForWrite();
    if (room <= 0) return;
    int n = available;
    if (n > room) n = room;
    if (n > (int)sizeof g_tcp_to_uart) n = sizeof g_tcp_to_uart;
    n = g_client.read(g_tcp_to_uart, (size_t)n);
    if (n > 0) g_worker_uart.write(g_tcp_to_uart, (size_t)n);
}

static void pump_uart_to_tcp(void) {
    int available = g_worker_uart.available();
    if (available <= 0) return;
    int n = available;
    if (n > (int)sizeof g_uart_to_tcp) n = sizeof g_uart_to_tcp;
    n = g_worker_uart.read(g_uart_to_tcp, (size_t)n);
    if (n <= 0) return;
    size_t sent = 0;
    while (sent < (size_t)n && g_client.connected()) {
        size_t wrote = g_client.write(g_uart_to_tcp + sent, (size_t)n - sent);
        if (wrote > 0) sent += wrote;
        else delay(1);
    }
}

void setup() {
    Serial.begin(115200);
    delay(200);

    g_worker_uart.setRxBufferSize(4096);
    g_worker_uart.setTxBufferSize(2048);
    g_worker_uart.begin(LB_UART_BAUD, SERIAL_8N1, LB_UART_RX_PIN, LB_UART_TX_PIN);
    Serial.printf("BRIDGE|boot|uart_rx=D7/GPIO%d|uart_tx=D6/GPIO%d|free=%u\n",
                  LB_UART_RX_PIN, LB_UART_TX_PIN, (unsigned)ESP.getFreeHeap());

#if LB_HAS_WIFI_SECRETS
    WiFi.mode(WIFI_STA);
    WiFi.begin(LB_WIFI_SSID, LB_WIFI_PASSWORD);
    g_next_wifi_retry_ms = millis() + 15000;
#else
    start_fallback_ap();
#endif
}

void loop() {
    service_wifi_state();
    accept_client();
    if (g_client && g_client.connected()) {
        pump_tcp_to_uart();
        pump_uart_to_tcp();
    } else {
        /* A compute that outlives a dropped TCP client may still finish and
         * emit 64 KB. Drain it so the next connection starts on a frame edge. */
        discard_worker_bytes();
        delay(1);
    }
}
