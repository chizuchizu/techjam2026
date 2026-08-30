/*
 * host_test.c - validate the ESP32 kernels on the host against torch references.
 *
 * Build:   make            (or: cc -O2 -o host_test host_test.c ../src/kernels.c ../src/model.c -lm)
 * Run:     ./host_test [seeds...|all] [--exact] [--fast] [--quiet] [--bench]
 *
 * Reads (from the project root by default):
 *   weights.bin             flat fp32 weights (see tm_config.h)
 *   weights_q12.bin         Q12 blob for the FAST path
 *   testdata/input_<s>.bin  torch input  (16384 floats)
 *   testdata/ref_<s>.bin    torch output (16384 floats)
 *
 * For every seed it runs the C forward in the requested mode(s) and applies
 * the benchmark's real criterion: pass if |a-b| <= atol OR |a-b| <= rtol*|b|,
 * with atol=0.002, rtol=0.02. Prints per-seed + aggregate results.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../src/kernels.h"
#include "../src/model.h"

#define NBT (TM_S * TM_D)

static float* g_wf32 = NULL;
static TMQ12Weights g_q12;
static unsigned char* g_q12blob = NULL;

static float* load_bin(const char* base, const char* rel, size_t* out_nbytes) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", base, rel);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* buf = malloc((size_t)sz);
    if (!buf) { fprintf(stderr, "oom\n"); exit(2); }
    if (fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "short read\n"); exit(2); }
    fclose(f);
    if (out_nbytes) *out_nbytes = (size_t)sz;
    return (float*)buf;
}

/* benchmark gate, mirroring torch_transformer_benchmark.compare_outputs */
static long gate_failures(const float* got, const float* ref, double* max_abs) {
    long fails = 0;
    double ma = 0.0;
    for (int i = 0; i < NBT; i++) {
        double a = (double)got[i], b = (double)ref[i];
        double d = fabs(a - b);
        if (d > ma) ma = d;
        if (!(d <= 0.002 || d <= 0.02 * fabs(b))) fails++;
    }
    if (max_abs) *max_abs = ma;
    return fails;
}

static int run_seed(const char* base, int s, int mode, int bench, int reps) {
    char path[512];
    float* x = malloc(NBT * 4);
    float* ref = malloc(NBT * 4);
    float* out = malloc(NBT * 4);
    if (!x || !ref || !out) { fprintf(stderr, "oom\n"); exit(2); }

    snprintf(path, sizeof path, "%s/testdata/input_%d.bin", base, s);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", path); return -1; }
    if (fread(x, 4, NBT, f) != NBT) { fprintf(stderr, "short input\n"); exit(2); }
    fclose(f);
    snprintf(path, sizeof path, "%s/testdata/ref_%d.bin", base, s);
    f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "missing %s\n", path); return -1; }
    if (fread(ref, 4, NBT, f) != NBT) { fprintf(stderr, "short ref\n"); exit(2); }
    fclose(f);

    tm_set_mode(mode);
    int rc = 0;
    if (bench) {
        for (int i = 0; i < 3; i++) tm_forward(x, out, g_wf32, &g_q12);
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        for (int i = 0; i < reps; i++) tm_forward(x, out, g_wf32, &g_q12);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double secs = (double)(t1.tv_sec - t0.tv_sec) + (double)(t1.tv_nsec - t0.tv_nsec) / 1e9;
        printf("bench[%d] mode=%d reps=%d %.3f s/forward\n", s, mode, reps,
               secs / (double)reps);
        rc = 0;
    } else {
        tm_forward(x, out, g_wf32, &g_q12);
        double max_abs = 0.0;
        long fails = gate_failures(out, ref, &max_abs);
        printf("seed=%2d mode=%s fails=%5ld max_abs=%.4e %s\n",
               s, mode == TM_MODE_EXACT ? "EXACT" : "FAST",
               fails, max_abs, fails ? "FAIL" : "ok");
        rc = (int)(fails > 0);
    }
    free(x); free(ref); free(out);
    return rc;
}

int main(int argc, char** argv) {
    const char* base = "..";
    int do_exact = 0, do_fast = 1, bench = 0, quiet = 0, reps = 5;
    int seeds[64]; int nseeds = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--exact")) { do_exact = 1; do_fast = 0; }
        else if (!strcmp(argv[i], "--fast")) { do_fast = 1; }
        else if (!strcmp(argv[i], "--both")) { do_exact = 1; do_fast = 1; }
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (!strcmp(argv[i], "--quiet")) quiet = 1;
        else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "all")) { for (int s = 0; s < 25; s++) seeds[nseeds++] = s; }
        else { seeds[nseeds++] = atoi(argv[i]); }
    }
    if (nseeds == 0) { for (int s = 0; s < 5; s++) seeds[nseeds++] = s; }

    g_wf32 = load_bin(base, "weights.bin", NULL);
    g_q12blob = (unsigned char*)load_bin(base, "weights_q12.bin", NULL);
    tm_scan_q12((const void*)g_q12blob, &g_q12);

    printf("weights: %d floats; q12 blob loaded\n", TM_W_TOTAL);

    int total_fail = 0;
    for (int i = 0; i < nseeds; i++) {
        if (do_fast) total_fail += run_seed(base, seeds[i], TM_MODE_FAST, bench, reps);
        if (do_exact) total_fail += run_seed(base, seeds[i], TM_MODE_EXACT, bench, reps);
    }
    if (!quiet)
        printf("done: %d seed-runs, %d failed %s\n", nseeds * (do_fast + do_exact),
               total_fail, total_fail ? "==> FAIL" : "==> ALL PASS");
    return total_fail ? 1 : 0;
}
