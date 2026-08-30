/*
 * main.cpp - ESP32-C3 firmware for the Tech Jam case-6 transformer baseline.
 *
 * Weight blobs are embedded in flash via board_build.embed_files
 * (symbols _binary_weights_bin_start/_end, _binary_weights_q12_bin_*).
 *
 * Serial protocol (native USB, 115200 baud):
 *   'M'         -> "TM <mode>\n"          (mode 0 EXACT / 1 FAST)
 *   'S'         -> "TM OK mode=<mode>\n"  (re-parse/on-demand init)
 *   'R' <65536 bytes float32 LE input> -> runs 1 forward,
 *                                          streams 65536 bytes output + "END\n"
 *   'T' <n>     -> warmup 1 + n timed forwards, prints "TM <mode> <us> ..."
 *   'X'         -> soft reset of static arena state (no-op)
 *
 * The C3 has no FPU: FAST mode is the actionable path (~2-4 s/forward),
 * EXACT mode (soft-float) is kept for A/B and reference checks.
 */
#include <Arduino.h>
#include <stdio.h>

#include "model.h"
#include "kernels.h"

/* embedded blobs from board_build.embed_files (project root) */
extern const uint8_t _binary_weights_bin_start[] asm("_binary_weights_bin_start");
extern const uint8_t _binary_weights_bin_end[]   asm("_binary_weights_bin_end");
extern const uint8_t _binary_weights_q12_bin_start[] asm("_binary_weights_q12_bin_start");
extern const uint8_t _binary_weights_q12_bin_end[]   asm("_binary_weights_q12_bin_end");

static const float* g_wf32 = (const float*)_binary_weights_bin_start;
static TMQ12Weights g_q12;
static bool g_q12_scanned = false;

static void ensure_q12(void) {
    if (!g_q12_scanned) {
        tm_scan_q12(_binary_weights_q12_bin_start, &g_q12);
        g_q12_scanned = true;
    }
}

static long g_forward_counter = 0;

/* device: route profile output to Serial (overrides weak model.c stub).
 * MUST have C linkage (declared in model.h's extern "C" block) so it
 * shadows the weak .C stub and not a mangled C++ sibling. */
__attribute__((used)) void tm_prof_emit(const char* line) {
    Serial.print(line);
}

static void run_forward(void) {
    ensure_q12();
    /* workspace lives in model.c: input = g_x, output = g_buf1 */
    tm_forward(tm_input(), tm_output(), g_wf32, &g_q12);
    g_forward_counter++;
}

static void read_input(void) {
    /* read 65536 bytes of float32 LE from USB serial into the model input */
    uint8_t* p = (uint8_t*)tm_input();
    size_t need = TM_S * TM_D * 4;
    size_t got = 0;
    while (got < need) {
        int n = Serial.readBytes((char*)(p + got), need - got);
        if (n <= 0) delay(1);
        got += (size_t)(n < 0 ? 0 : n);
    }
}

void setup() {
    Serial.begin(115200);
    /* Enlarge the native-USB CDC RX queue: the default (256 B) drops any
     * host->device burst larger than the app's drain quantum when the host
     * pushes the 64 KB 'R' input at USB speed. 8 KB + host-side 1 KB/20 ms
     * pacing makes full-frame delivery lossless (tools/device_test.py). */
    Serial.setRxBufferSize(8192);
    delay(200);
    tm_set_mode(TM_MODE_DEFAULT);
    Serial.println("TM XIAO-ESP32C3 case6 baseline ready");
    Serial.printf("TM weights f32=%u bytes q12=%u bytes\n",
                  (unsigned)(_binary_weights_bin_end - _binary_weights_bin_start),
                  (unsigned)(_binary_weights_q12_bin_end - _binary_weights_q12_bin_start));
}

void loop() {
    while (Serial.available() == 0) { delay(2); }
    uint8_t cmd = (uint8_t)Serial.read();

    switch (cmd) {
        case 'M': {
            Serial.printf("TM %d %d %ld\n", tm_get_mode(), TM_S, TM_D);
            break;
        }
        case 'S': {
            ensure_q12();
            Serial.printf("TM OK mode=%d\n", tm_get_mode());
            break;
        }
        case 'R': {
            read_input();
            unsigned long t0 = esp_timer_get_time();
            run_forward();
            unsigned long t1 = esp_timer_get_time();
            Serial.write((const uint8_t*)tm_output(), TM_S * TM_D * 4);
            Serial.printf("END forward=%ld us=%lu\n", g_forward_counter, (unsigned long)(t1 - t0));
            break;
        }
        case 'T': {
            int n = 3;
            /* optional count byte */
            if (Serial.available()) n = (int)Serial.read();
            if (n < 1) n = 1;
            if (n > 9) n = 9;
            /* load a fixed internal input (seed 0 precomputed? use zeros) */
            read_input();
            Serial.printf("TM %d", tm_get_mode());
            for (int i = 0; i < 1 + n; i++) {
                if (i == 1) { /* warmup */ }
                unsigned long t0 = esp_timer_get_time();
                run_forward();
                unsigned long t1 = esp_timer_get_time();
                if (i > 0) Serial.printf(" %lu", (unsigned long)(t1 - t0));
            }
            Serial.println();
            break;
        }
        case 'P': {
            tm_profile_dump();
            break;
        }
        case 'K': {
            char line[160]; tm_microbench(line, sizeof line);
            Serial.print(line);
            break;
        }
        case 'Q': {
            char line[220]; tm_kbench2(line, sizeof line);
            Serial.print(line);
            break;
        }
        case 'X': {
            Serial.println("TM reset");
            break;
        }
        default: {
            Serial.println("TM unknown cmd");
            break;
        }
    }
}
