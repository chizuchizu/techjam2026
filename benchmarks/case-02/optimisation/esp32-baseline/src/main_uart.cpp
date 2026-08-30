/*
 * main_uart.cpp - opt-in radio-free worker endpoint for a WiFi/UART sidecar.
 *
 * The full opt23 forward remains on this board.  A separate, link-only C3
 * terminates WiFi/TCP and transparently relays the byte stream over UART1:
 *
 *   sidecar TX (D6/GPIO21) -> worker RX (D7/GPIO20)
 *   sidecar RX (D7/GPIO20) <- worker TX (D6/GPIO21)
 *   sidecar GND             --- worker GND
 *
 * The command protocol is intentionally identical to the USB and tiled-TCP
 * endpoints, so run_batch_dp.py can address the sidecar as a normal TCP node.
 * No WiFi code is linked into this worker image.
 */
#include <Arduino.h>
#include <HardwareSerial.h>
#include <esp_timer.h>
#include <stdio.h>

#include "model.h"
#include "kernels.h"
#include "servo.h"

#ifndef TM_UART_BAUD
#define TM_UART_BAUD 2000000
#endif
#ifndef TM_UART_RX_PIN
#define TM_UART_RX_PIN D7
#endif
#ifndef TM_UART_TX_PIN
#define TM_UART_TX_PIN D6
#endif

extern const uint8_t _binary_weights_bin_start[] asm("_binary_weights_bin_start");
extern const uint8_t _binary_weights_bin_end[] asm("_binary_weights_bin_end");
extern const uint8_t _binary_weights_q12_bin_start[] asm("_binary_weights_q12_bin_start");
extern const uint8_t _binary_weights_q12_bin_end[] asm("_binary_weights_q12_bin_end");

static const float* g_wf32 = (const float*)_binary_weights_bin_start;
static TMQ12Weights g_q12;
static bool g_q12_scanned = false;
static long g_forward_counter = 0;

static void ensure_q12(void) {
    if (!g_q12_scanned) {
        tm_scan_q12(_binary_weights_q12_bin_start, &g_q12);
        g_q12_scanned = true;
    }
}

__attribute__((used)) void tm_prof_emit(const char* line) {
    Serial1.print(line);
}

static void run_forward(void) {
    ensure_q12();
    tm_servo_busy(1);
    tm_forward(tm_input(), tm_output(), g_wf32, &g_q12);
    tm_servo_busy(0);
    g_forward_counter++;
}

/* Abort a frame after an idle link rather than leaving the worker permanently
 * blocked when the TCP side of the bridge disappears mid-transfer. */
static bool read_exact(HardwareSerial& io, uint8_t* dst, size_t size,
                       uint32_t idle_timeout_ms = 10000) {
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
        if ((uint32_t)(millis() - last_progress) >= idle_timeout_ms) return false;
        delay(1);
    }
    return true;
}

static void process_command(HardwareSerial& io) {
    int raw = io.read();
    if (raw < 0) return;
    const uint8_t cmd = (uint8_t)raw;
    switch (cmd) {
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
            io.write((const uint8_t*)tm_output(), bytes);
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

        case 'K': {
            char line[160];
            tm_microbench(line, sizeof line);
            io.print(line);
            break;
        }

        case 'Q': {
            char line[220];
            tm_kbench2(line, sizeof line);
            io.print(line);
            break;
        }

        case 'X':
            io.println("TM reset");
            break;

        default:
            io.println("TM unknown cmd");
            break;
    }

}

void setup() {
    Serial.begin(115200);
    delay(200);
    tm_set_mode(TM_MODE_DEFAULT);

    /* These buffers are small relative to the model arena and provide enough
     * elasticity for a 2 Mbaud point-to-point link. */
    Serial1.setRxBufferSize(4096);
    Serial1.setTxBufferSize(2048);
    Serial1.begin(TM_UART_BAUD, SERIAL_8N1, TM_UART_RX_PIN, TM_UART_TX_PIN);

    Serial.println("TM XIAO-ESP32C3 opt23 UART worker ready");
    Serial.printf("TM UART baud=%u rx=D7/GPIO%d tx=D6/GPIO%d free=%u\n",
                  (unsigned)TM_UART_BAUD, TM_UART_RX_PIN, TM_UART_TX_PIN,
                  (unsigned)ESP.getFreeHeap());
    Serial.printf("TM weights f32=%u bytes q12=%u bytes\n",
                  (unsigned)(_binary_weights_bin_end - _binary_weights_bin_start),
                  (unsigned)(_binary_weights_q12_bin_end - _binary_weights_q12_bin_start));
}

void loop() {
    if (Serial1.available() > 0) process_command(Serial1);
    else delay(1);
}
