/*
 * main.cpp - ESP32-C3 firmware for the runtime-parameterized transformer.
 *
 * One binary runs ANY geometry: the case is a plain tm_case struct that can be
 * changed at runtime.  This build embeds case-2 weights (S=128 D=128 H=4
 * F=128 L=4) plus one input/ref pair for a lossless on-device self-test.
 *
 * Serial protocol (native USB, 115200):
 *   'M'        -> toggle EXACT(0)/FAST(1), prints new mode
 *   'R'        -> read 65536 float32 LE bytes into the workspace input, run
 *                 one forward, stream 65536 output bytes + "END\n"
 *   'Z'        -> self-test against the embedded case-2 input_0/ref_0:
 *                 prints MAX_ABS, FAILS, gate, then the per-kernel profile
 *   'P'        -> reprint the last per-kernel profile
 */
#include <Arduino.h>
#include <stdio.h>
#include "esp_timer.h"

extern "C" {
#include "transformer.h"
}

extern const uint8_t _binary_weights_bin_start[]     asm("_binary_weights_bin_start");
extern const uint8_t _binary_weights_q12_bin_start[] asm("_binary_weights_q12_bin_start");
extern const uint8_t _binary_input_0_bin_start[]     asm("_binary_input_0_bin_start");
extern const uint8_t _binary_ref_0_bin_start[]       asm("_binary_ref_0_bin_start");

static tm_case g_cfg;
static const float* g_W    = (const float*)_binary_weights_bin_start;
static const void*  g_q12b = (const void*)_binary_weights_q12_bin_start;
static const float* g_xin  = (const float*)_binary_input_0_bin_start;
static const float* g_ref  = (const float*)_binary_ref_0_bin_start;

static const int16_t** g_q = NULL;
static float* g_wscl = NULL;

/* tm_workspace_size for the default case (1,128,128,4,128,4) = 256720 bytes.
 * static (not heap) so module loading / system heap state can never leave a
 * NULL arena. */
#define WS_MAX 257232  /* tm_workspace_size(1,128,128,4,128,4) */
static __attribute__((aligned(16))) uint8_t g_ws[WS_MAX];

static int64_t esp_now(void) { return (int64_t)esp_timer_get_time(); }

static void print_profile(void) {
    tm_profile p;
    tm_profile_get(&g_cfg, g_ws, &p);
    Serial.printf("prof norm1=%lld attn=%lld oproj=%lld res1=%lld\n",
                  (long long)p.norm1_us, (long long)p.attn_us,
                  (long long)p.oproj_us, (long long)p.res1_us);
    Serial.printf("prof norm2=%lld ffn1=%lld gelu=%lld ffn2=%lld res2=%lld final=%lld\n",
                  (long long)p.norm2_us, (long long)p.ffn1_us, (long long)p.gelu_us,
                  (long long)p.ffn2_us, (long long)p.res2_us, (long long)p.final_us);
    Serial.printf("prof qkv=%lld quant=%lld TOTAL=%lld us\n",
                  (long long)p.qkv_us, (long long)p.quant_us,
                  (long long)p.total_us);
}

static int run_one(const float* xin, float* yout) {
    return tm_run(&g_cfg, g_ws, g_W, g_q, g_wscl, xin, yout);
}

static void do_self_test(void) {
    float* y = (float*)tm_input_buf(&g_cfg, g_ws);
    int rc = run_one(g_xin, y);
    if (rc) { Serial.printf("tm_run rc=%d\n", rc); return; }
    const int n = g_cfg.S * g_cfg.D;
    double max_abs = 0.0, max_rel = 0.0;
    long fails = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)y[i] - (double)g_ref[i];
        if (d < 0) d = -d;
        double rd = (double)g_ref[i]; if (rd < 0) rd = -rd;
        if (d > max_abs) max_abs = d;
        if (rd > 0.0) { double r2 = d / rd; if (r2 > max_rel) max_rel = r2; }
        if (!(d <= 0.002 || d <= 0.02 * rd)) fails++;
    }
    Serial.printf("SELF mode=%s MAX_ABS=%.6e MAX_REL=%.6e FAILS=%ld/%d gate=%s\n",
                  g_cfg.mode == TM_FAST ? "FAST" : "EXACT",
                  max_abs, max_rel, fails, n, fails == 0 ? "PASS" : "FAIL");
    print_profile();
}

static void do_forward_stream(void) {
    float* in = (float*)tm_input_buf(&g_cfg, g_ws);
    size_t need = (size_t)g_cfg.S * g_cfg.D * 4;
    uint8_t* p = (uint8_t*)in;
    size_t got = 0;
    unsigned long t0 = millis();
    while (got < need) {
        int k = Serial.readBytes((char*)(p + got), need - got);
        if (k <= 0) { if (millis() - t0 > 30000) return; delay(1); }
        else { got += (size_t)k; t0 = millis(); }
    }
    float* y = in;
    int rc = run_one(in, y);
    if (rc) { Serial.printf("tm_run rc=%d\n", rc); return; }
    Serial.write((const uint8_t*)y, need);
    Serial.print("END\n");
}

void setup() {
    Serial.begin(115200);
    Serial.setRxBufferSize(8192);
    delay(200);

    g_cfg.B = 1; g_cfg.S = 128; g_cfg.D = 128; g_cfg.H = 4;
    g_cfg.F = 128; g_cfg.L = 4; g_cfg.causal = 1; g_cfg.mode = TM_FAST;

    int nm = tm_qmat_count(&g_cfg);
    g_q = (const int16_t**)calloc(nm, sizeof(int16_t*));
    g_wscl = (float*)calloc(nm, sizeof(float));
    tm_scan_q12(&g_cfg, g_q12b, g_q, g_wscl);

    size_t wsn = tm_workspace_size(&g_cfg);
    if (wsn > sizeof(g_ws)) {
        Serial.printf("ERROR: need %u workspace bytes, have %u\n",
                      (unsigned)wsn, (unsigned)sizeof(g_ws));
        return;
    }
    tm_set_clock(esp_now);
    tm_workspace_init(&g_cfg, g_ws);

    Serial.printf("PARAM transformer ready: S=%d D=%d H=%d F=%d L=%d mode=FAST ws=%u bytes\n",
                  g_cfg.S, g_cfg.D, g_cfg.H, g_cfg.F, g_cfg.L, (unsigned)wsn);
}

void loop() {
    if (Serial.available()) {
        int cmd = Serial.read();
        switch (cmd) {
            case 'M':
                g_cfg.mode = (g_cfg.mode == TM_FAST) ? TM_EXACT : TM_FAST;
                Serial.printf("mode=%s\n", g_cfg.mode == TM_FAST ? "FAST" : "EXACT");
                break;
            case 'R':
                do_forward_stream();
                break;
            case 'Z':
                do_self_test();
                break;
            case 'P':
                print_profile();
                break;
            default:
                break;
        }
    }
}
