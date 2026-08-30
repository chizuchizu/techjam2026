/*
 * test_host.c - host driver for the parameterized transformer.
 *
 * Usage:
 *   test_host S D H F L MODE weights.bin weights_q12.bin input.bin ref.bin [-n N]
 *
 * MODE: 0 = exact fp32, 1 = fast Q15xQ12.
 * Prints max abs error, max rel error, gate pass/fail, and per-kernel profile.
 */
#include "transformer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static void* load_file(const char* path, size_t* out_n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    void* p = malloc((size_t)n);
    if (n > 0 && fread(p, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", path); exit(2); }
    fclose(f);
    *out_n = (size_t)n;
    return p;
}

int main(int argc, char** argv) {
    if (argc < 10) {
        fprintf(stderr, "usage: %s S D H F L MODE weights.bin q12.bin input.bin ref.bin [-n N]\n", argv[0]);
        return 2;
    }
    tm_case cfg;
    cfg.B = 1;
    cfg.S = atoi(argv[1]);
    cfg.D = atoi(argv[2]);
    cfg.H = atoi(argv[3]);
    cfg.F = atoi(argv[4]);
    cfg.L = atoi(argv[5]);
    cfg.mode = atoi(argv[6]);
    cfg.causal = 1;

    const char* wpath = argv[7];
    const char* qpath = argv[8];
    const char* xpath = argv[9];
    const char* rpath = argv[10];
    int reps = 1;
    for (int i = 11; i + 1 < argc; i++) {
        if (strcmp(argv[i], "-n") == 0) reps = atoi(argv[i+1]);
    }

    size_t wn = 0, qn = 0, xn = 0, rn = 0;
    float* W = (float*)load_file(wpath, &wn);
    void* qblob = load_file(qpath, &qn);
    float* x = (float*)load_file(xpath, &xn);
    float* ref = (float*)load_file(rpath, &rn);

    size_t expect_f = tm_w_total(&cfg) * sizeof(float);
    size_t expect_x = (size_t)cfg.S * cfg.D * sizeof(float);
    if (wn != expect_f) { fprintf(stderr, "weights size wrong: got %zu want %zu\n", wn, expect_f); return 3; }
    if (xn != expect_x || rn != expect_x) { fprintf(stderr, "input/ref size wrong\n"); return 3; }

    int nm = tm_qmat_count(&cfg);
    const int16_t** q = (const int16_t**)malloc((size_t)nm * sizeof(int16_t*));
    float* wscl = (float*)malloc((size_t)nm * sizeof(float));
    tm_scan_q12(&cfg, qblob, q, wscl);

    size_t wsn = tm_workspace_size(&cfg);
    void* ws = malloc(wsn);
    float* y = (float*)malloc(expect_x);
    if (!q || !wscl || !ws || !y) return 4;

    printf("# case B=%d S=%d D=%d H=%d F=%d L=%d causal=%d mode=%s\n",
           cfg.B, cfg.S, cfg.D, cfg.H, cfg.F, cfg.L, cfg.causal,
           cfg.mode == TM_FAST ? "FAST(Q15xQ12)" : "EXACT(fp32)");
    printf("# workspace %zu bytes; weights %zu bytes; reps %d\n",
           wsn, wn, reps);

    int rc = 0;
    for (int r = 0; r < reps; r++) {
        rc = tm_run(&cfg, ws, W, q, wscl, x, y);
        if (rc) { printf("tm_run returned %d\n", rc); return 5; }
    }

    double max_abs = 0.0, max_rel = 0.0;
    long fails = 0;
    const int n = cfg.S * cfg.D;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)y[i] - (double)ref[i]);
        double rd = fabs((double)ref[i]);
        if (d > max_abs) max_abs = d;
        if (rd > 0.0) { double r2 = d / rd; if (r2 > max_rel) max_rel = r2; }
        if (!(d <= 0.002 || d <= 0.02 * rd)) fails++;
    }
    printf("MAX_ABS %.6e  MAX_REL %.6e  FAILS %ld/%d  gate=%s\n",
           max_abs, max_rel, fails, n, fails == 0 ? "PASS" : "FAIL");
    if (reps > 1) {
        tm_profile prof;
        tm_profile_get(&cfg, ws, &prof);
        printf("AVG_TOTAL_US %.1f (over %d reps)\n", (double)prof.total_us / reps, reps);
    }
    tm_profile_dump(&cfg, ws);
    return (fails == 0) ? 0 : 1;
}
