/*
 * main.cpp - ESP32-C3 firmware for the runtime-parameterized transformer.
 *
 * ONE binary runs EVERY geometry: the case is a plain tm_case struct changed
 * at runtime ('C n').  Weights/input/ref blobs for the six implemented cases
 * are embedded in flash so on-device self-tests and per-kernel profiles can be
 * taken for all of them from a single image.
 *
 * Serial protocol (native USB, 115200):
 *   'C <2|7|9|10|11|12>'  select case (geometry + weight/ref blobs), prints it
 *   'M'                   toggle EXACT(0)/FAST(1)
 *   'Z'                   self-test the selected case: gate + per-kernel profile
 *   'P'                   reprint the last profile
 */
#include <Arduino.h>
#include <stdio.h>
#include "esp_timer.h"

extern "C" {
#include "transformer.h"
}

extern const uint8_t _binary_weights_bin_start[]       asm("_binary_weights_bin_start");
extern const uint8_t _binary_weights_q12_bin_start[]   asm("_binary_weights_q12_bin_start");
extern const uint8_t _binary_input_0_bin_start[]       asm("_binary_input_0_bin_start");
extern const uint8_t _binary_ref_0_bin_start[]         asm("_binary_ref_0_bin_start");
extern const uint8_t _binary_weights_07_bin_start[]    asm("_binary_weights_07_bin_start");
extern const uint8_t _binary_weights_07_q12_bin_start[] asm("_binary_weights_07_q12_bin_start");
extern const uint8_t _binary_input_07_bin_start[]      asm("_binary_input_07_bin_start");
extern const uint8_t _binary_ref_07_bin_start[]        asm("_binary_ref_07_bin_start");
extern const uint8_t _binary_input_12_bin_start[]      asm("_binary_input_12_bin_start");
extern const uint8_t _binary_ref_12_bin_start[]        asm("_binary_ref_12_bin_start");

typedef struct {
    const char* name;
    int S, D, H, F, L;
    const float* W;
    const void*  q12b;
    const float* xin;
    const float* ref;
} case_preset;

static const case_preset g_presets[] = {
    {"02", 128, 128,  4, 128, 4, (const float*)_binary_weights_bin_start,
     _binary_weights_q12_bin_start, (const float*)_binary_input_0_bin_start,
     (const float*)_binary_ref_0_bin_start},
    {"07", 128,  32,  4,  32, 4, (const float*)_binary_weights_07_bin_start,
     _binary_weights_07_q12_bin_start, (const float*)_binary_input_07_bin_start,
     (const float*)_binary_ref_07_bin_start},
    {"09", 128, 128,  1, 128, 4, (const float*)_binary_weights_bin_start,
     _binary_weights_q12_bin_start, (const float*)_binary_input_0_bin_start,
     (const float*)_binary_ref_0_bin_start},
    {"10", 128, 128,  2, 128, 4, (const float*)_binary_weights_bin_start,
     _binary_weights_q12_bin_start, (const float*)_binary_input_0_bin_start,
     (const float*)_binary_ref_0_bin_start},
    {"11", 128, 128, 16, 128, 4, (const float*)_binary_weights_bin_start,
     _binary_weights_q12_bin_start, (const float*)_binary_input_0_bin_start,
     (const float*)_binary_ref_0_bin_start},
    {"12",  32, 128,  4, 128, 4, (const float*)_binary_weights_bin_start,
     _binary_weights_q12_bin_start, (const float*)_binary_input_12_bin_start,
     (const float*)_binary_ref_12_bin_start},
};

static tm_case g_cfg;
static int g_sel = 0;
static bool g_valid = false;

static const int16_t** g_q = NULL;
static float* g_wscl = NULL;
static const case_preset* g_cp = NULL;

#define WS_MAX 257232   /* max that fits C3 SRAM beside the runtime; case 10 needs 282 KB, case 09 needs 331 KB */
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

static void do_self_test(void) {
    if (!g_valid) { Serial.println("select case first (C n)"); return; }
    float* y = (float*)tm_input_buf(&g_cfg, g_ws);
    int rc = tm_run(&g_cfg, g_ws, g_cp->W, g_q, g_wscl, g_cp->xin, y);
    if (rc) { Serial.printf("tm_run rc=%d\n", rc); return; }
    const int n = g_cfg.S * g_cfg.D;
    double max_abs = 0.0, max_rel = 0.0;
    long fails = 0;
    for (int i = 0; i < n; i++) {
        double d = (double)y[i] - (double)g_cp->ref[i];
        if (d < 0) d = -d;
        double rd = (double)g_cp->ref[i]; if (rd < 0) rd = -rd;
        if (d > max_abs) max_abs = d;
        if (rd > 0.0) { double r2 = d / rd; if (r2 > max_rel) max_rel = r2; }
        if (!(d <= 0.002 || d <= 0.02 * rd)) fails++;
    }
    Serial.printf("SELF case=%s mode=%s MAX_ABS=%.6e MAX_REL=%.6e FAILS=%ld/%d gate=%s\n",
                  g_cp->name, g_cfg.mode == TM_FAST ? "FAST" : "EXACT",
                  max_abs, max_rel, fails, n, fails == 0 ? "PASS" : "FAIL");
    print_profile();
}

static bool select_case(int id) {
    for (unsigned i = 0; i < sizeof(g_presets)/sizeof(g_presets[0]); i++) {
        if (atoi(g_presets[i].name) == id) {
            g_sel = i;
            g_cp = &g_presets[g_sel];
            int mode = g_valid ? g_cfg.mode : TM_FAST;
            g_cfg.B = 1; g_cfg.S = g_cp->S; g_cfg.D = g_cp->D;
            g_cfg.H = g_cp->H; g_cfg.F = g_cp->F; g_cfg.L = g_cp->L;
            g_cfg.causal = 1; g_cfg.mode = mode;
            size_t wsn = tm_workspace_size(&g_cfg);
            if (wsn > sizeof(g_ws)) {
                Serial.printf("ERROR case %s needs %u ws (have %u)\n",
                              g_cp->name, (unsigned)wsn, (unsigned)sizeof(g_ws));
                return false;
            }
            int nm = tm_qmat_count(&g_cfg);
            if (nm > 0) {
                free(g_q); free(g_wscl);
                g_q = (const int16_t**)calloc(nm, sizeof(int16_t*));
                g_wscl = (float*)calloc(nm, sizeof(float));
            }
            tm_scan_q12(&g_cfg, g_cp->q12b, g_q, g_wscl);
            tm_workspace_init(&g_cfg, g_ws);
            g_valid = true;
            Serial.printf("CASE %s: S=%d D=%d H=%d F=%d L=%d mode=%s ws=%u\n",
                          g_cp->name, g_cfg.S, g_cfg.D, g_cfg.H, g_cfg.F, g_cfg.L,
                          g_cfg.mode == TM_FAST ? "FAST" : "EXACT", (unsigned)wsn);
            return true;
        }
    }
    return false;
}

void setup() {
    Serial.begin(115200);
    Serial.setRxBufferSize(8192);
    delay(200);
    tm_set_clock(esp_now);
    select_case(2);
    Serial.printf("PARAM transformer ready: send 'C <2|7|9|10|11|12>', then 'Z'\n");
}

void loop() {
    if (Serial.available()) {
        int cmd = Serial.read();
        switch (cmd) {
            case 'C': {
                int id = Serial.parseInt();
                if (!select_case(id)) Serial.printf("unknown case id %d\n", id);
                break;
            }
            case 'M':
                g_cfg.mode = (g_cfg.mode == TM_FAST) ? TM_EXACT : TM_FAST;
                Serial.printf("mode=%s\n", g_cfg.mode == TM_FAST ? "FAST" : "EXACT");
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
