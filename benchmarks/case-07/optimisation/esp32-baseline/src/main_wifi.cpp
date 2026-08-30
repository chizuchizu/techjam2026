/*
 * Opt-in WiFi/TCP command endpoint for the complete case-7 forward.
 * The default serial firmware excludes this file. No model-sized transport
 * buffer is allocated: TCP reads and writes tm_input()/tm_output() directly.
 */
#include <Arduino.h>
#include <WiFi.h>
#include <esp_timer.h>

#include "model.h"
#include "kernels.h"

#if __has_include("../secrets.h")
#include "../secrets.h"
#define TM_HAS_WIFI_SECRETS 1
#else
#define TM_WIFI_SSID ""
#define TM_WIFI_PASSWORD ""
#define TM_HAS_WIFI_SECRETS 0
#warning "secrets.h is absent: esp32-wifi will run in USB-only mode"
#endif

#ifndef TM_WIFI_SSID
#error "secrets.h must define TM_WIFI_SSID"
#endif
#ifndef TM_WIFI_PASSWORD
#error "secrets.h must define TM_WIFI_PASSWORD"
#endif

extern const uint8_t _binary_weights_bin_start[] asm("_binary_weights_bin_start");
extern const uint8_t _binary_weights_bin_end[] asm("_binary_weights_bin_end");
extern const uint8_t _binary_weights_q12_bin_start[] asm("_binary_weights_q12_bin_start");
extern const uint8_t _binary_weights_q12_bin_end[] asm("_binary_weights_q12_bin_end");

static const uint16_t TM_TCP_PORT = 5000;
static const uint32_t TM_IO_IDLE_TIMEOUT_MS = 30000;

static const float* g_wf32 = (const float*)_binary_weights_bin_start;
static TMQ12Weights g_q12;
static bool g_q12_scanned = false;
static long g_forward_counter = 0;

static WiFiServer g_server(TM_TCP_PORT);
static WiFiClient g_client;
static bool g_server_started = false;
static uint32_t g_next_wifi_retry_ms = 0;
static Print* g_command_output = &Serial;

static void ensure_q12(void) {
    if (!g_q12_scanned) {
        tm_scan_q12(_binary_weights_q12_bin_start, &g_q12);
        g_q12_scanned = true;
    }
}

__attribute__((used)) void tm_prof_emit(const char* line) {
    g_command_output->print(line);
}

static void run_forward(void) {
    ensure_q12();
    tm_forward(tm_input(), tm_output(), g_wf32, &g_q12);
    g_forward_counter++;
}

static bool read_exact(Stream& io, uint8_t* dst, size_t size) {
    size_t got = 0;
    uint32_t last_progress = millis();
    while (got < size) {
        int available = io.available();
        if (available > 0) {
            size_t chunk = (size_t)available;
            if (chunk > size - got) chunk = size - got;
            size_t n = io.readBytes((char*)(dst + got), chunk);
            if (n > 0) {
                got += n;
                last_progress = millis();
                continue;
            }
        }
        if ((uint32_t)(millis() - last_progress) >= TM_IO_IDLE_TIMEOUT_MS)
            return false;
        delay(1);
    }
    return true;
}

static bool write_exact(Stream& io, const uint8_t* src, size_t size) {
    size_t sent = 0;
    uint32_t last_progress = millis();
    while (sent < size) {
        size_t chunk = size - sent;
        if (chunk > 1440) chunk = 1440;
        size_t n = io.write(src + sent, chunk);
        if (n > 0) {
            sent += n;
            last_progress = millis();
            continue;
        }
        if ((uint32_t)(millis() - last_progress) >= TM_IO_IDLE_TIMEOUT_MS)
            return false;
        delay(1);
    }
    return true;
}

static void process_command(Stream& io) {
    int raw = io.read();
    if (raw < 0) return;
    g_command_output = &io;

    switch ((uint8_t)raw) {
        case 'M':
            io.printf("TM %d %d %d\n", tm_get_mode(), TM_S, TM_D);
            break;

        case 'S':
            ensure_q12();
            io.printf("TM OK mode=%d\n", tm_get_mode());
            break;

        case 'R': {
            const size_t bytes = (size_t)TM_S * TM_D * sizeof(float);
            if (!read_exact(io, (uint8_t*)tm_input(), bytes)) {
                io.println("TM ERR input timeout");
                break;
            }
            const uint64_t t0 = esp_timer_get_time();
            run_forward();
            const uint64_t t1 = esp_timer_get_time();
            if (!write_exact(io, (const uint8_t*)tm_output(), bytes)) break;
            io.printf("END forward=%ld us=%lu\n", g_forward_counter,
                      (unsigned long)(t1 - t0));
            break;
        }

        case 'T': {
            uint8_t count = 3;
            if (!read_exact(io, &count, 1)) {
                io.println("TM ERR count timeout");
                break;
            }
            int n = (int)count;
            if (n < 1) n = 1;
            if (n > 9) n = 9;
            const size_t bytes = (size_t)TM_S * TM_D * sizeof(float);
            if (!read_exact(io, (uint8_t*)tm_input(), bytes)) {
                io.println("TM ERR input timeout");
                break;
            }
            io.printf("TM %d", tm_get_mode());
            for (int i = 0; i < 1 + n; i++) {
                const uint64_t t0 = esp_timer_get_time();
                run_forward();
                const uint64_t t1 = esp_timer_get_time();
                if (i > 0) io.printf(" %lu", (unsigned long)(t1 - t0));
            }
            io.println();
            break;
        }

        case 'P':
            tm_profile_dump();
            break;

        case 'X':
            io.println("TM reset");
            break;

        default:
            io.println("TM unknown cmd");
            break;
    }

    g_command_output = &Serial;
}

static void start_server_if_connected(void) {
    if (g_server_started || WiFi.status() != WL_CONNECTED) return;
    g_server.begin();
    g_server.setNoDelay(true);
    g_server_started = true;
    Serial.printf("TM WiFi ready ip=%s port=%u free=%u\n",
                  WiFi.localIP().toString().c_str(), (unsigned)TM_TCP_PORT,
                  (unsigned)ESP.getFreeHeap());
}

static void service_wifi(void) {
#if TM_HAS_WIFI_SECRETS
    start_server_if_connected();
    if (WiFi.status() != WL_CONNECTED) {
        const uint32_t now = millis();
        if ((int32_t)(now - g_next_wifi_retry_ms) >= 0) {
            Serial.printf("TM WiFi reconnecting free=%u\n",
                          (unsigned)ESP.getFreeHeap());
            WiFi.begin(TM_WIFI_SSID, TM_WIFI_PASSWORD);
            g_next_wifi_retry_ms = now + 15000;
        }
        return;
    }

    if (g_client && !g_client.connected()) g_client.stop();
    if (!g_client || !g_client.connected()) {
        WiFiClient incoming = g_server.available();
        if (incoming) {
            g_client = incoming;
            g_client.setNoDelay(true);
        }
    }
    if (g_client && g_client.connected() && g_client.available() > 0)
        process_command(g_client);
#endif
}

void setup() {
    Serial.begin(115200);
    Serial.setRxBufferSize(2048);
    delay(200);
    tm_set_mode(TM_MODE_DEFAULT);

    Serial.println("TM XIAO-ESP32C3 case7 WiFi ready");
    Serial.printf("TM weights f32=%u bytes q12=%u bytes\n",
                  (unsigned)(_binary_weights_bin_end - _binary_weights_bin_start),
                  (unsigned)(_binary_weights_q12_bin_end - _binary_weights_q12_bin_start));
    Serial.printf("TM heap before WiFi=%u\n", (unsigned)ESP.getFreeHeap());

#if TM_HAS_WIFI_SECRETS
    WiFi.mode(WIFI_STA);
    WiFi.begin(TM_WIFI_SSID, TM_WIFI_PASSWORD);
    g_next_wifi_retry_ms = millis() + 15000;
#else
    Serial.println("TM WiFi disabled: copy secrets.example.h to secrets.h");
#endif
}

void loop() {
    if (Serial.available() > 0) process_command(Serial);
    service_wifi();
    delay(1);
}
