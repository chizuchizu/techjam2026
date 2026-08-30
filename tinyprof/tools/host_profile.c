/*
 * host_profile.c - run the case-2 forward natively and emit a tinyprof capture.
 *
 * This is the hardware-free half of tinyprof. It links the *same* model.c and
 * kernels.c the firmware does, with TINYPROF_LIB enabled, so it exercises the
 * identical instrumentation and produces the identical TPROF| wire format that
 * a board emits over serial. tp_collect.py consumes either source.
 *
 * What it is good for: proving the whole pipeline end to end, comparing two
 * source revisions of the kernels, and getting per-op *ratios* while a board is
 * unavailable.
 *
 * What it is NOT: a substitute for a device capture. Absolute times come from a
 * 64-bit x86 host with an FPU and megabytes of cache, and the whole point of
 * this project is that the C3 has none of those. Every artifact this produces
 * is stamped `"device": "host"` and the report refuses to headline it as an
 * ESP32 result.
 *
 * Build/run: see tinyprof/tools/Makefile  (make host-profile)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "kernels.h"
#include "model.h"
#include "tinyprof.h"

#define NBT (TM_S * TM_D)

/* tinyprof emits here; model.c routes tp_emit -> tm_prof_emit, and this is the
 * host's override of that weak hook (device firmware overrides it with
 * Serial.print in exactly the same way). */
void tm_prof_emit(const char* line) { fputs(line, stdout); }

static void* load_bin(const char* path, size_t* n) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "tinyprof: cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void* b = malloc((size_t)sz);
    if (!b || fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        fprintf(stderr, "tinyprof: short read %s\n", path); exit(2);
    }
    fclose(f);
    if (n) *n = (size_t)sz;
    return b;
}

/* the benchmark's own criterion (torch_transformer_benchmark.compare_outputs) */
static long gate(const float* got, const float* ref, double* max_abs, double* max_rel) {
    long fails = 0; double ma = 0.0, mr = 0.0;
    for (int i = 0; i < NBT; i++) {
        double a = got[i], b = ref[i], d = fabs(a - b);
        double r = fabs(b) > 0.0 ? d / fabs(b) : (d > 0.0 ? 1e30 : 0.0);
        if (d > ma) ma = d;
        if (r > mr && r < 1e29) mr = r;
        if (!(d <= 0.002 || d <= 0.02 * fabs(b))) fails++;
    }
    if (max_abs) *max_abs = ma;
    if (max_rel) *max_rel = mr;
    return fails;
}

int main(int argc, char** argv) {
    const char* root = "..";
    int seed = 0, reps = 1, mode = TM_MODE_FAST;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--root") && i + 1 < argc)      root = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--exact"))                mode = TM_MODE_EXACT;
        else if (!strcmp(argv[i], "--fast"))                 mode = TM_MODE_FAST;
        else { fprintf(stderr,
            "usage: %s [--root DIR] [--seed N] [--reps N] [--fast|--exact]\n", argv[0]);
            return 2; }
    }
    if (reps < 1) reps = 1;

    char path[512];
    snprintf(path, sizeof path, "%s/weights.bin", root);
    float* wf32 = load_bin(path, NULL);
    snprintf(path, sizeof path, "%s/weights_q12.bin", root);
    unsigned char* q12blob = load_bin(path, NULL);
    TMQ12Weights q12;
    tm_scan_q12(q12blob, &q12);

    snprintf(path, sizeof path, "%s/testdata/input_%d.bin", root, seed);
    float* in = load_bin(path, NULL);
    snprintf(path, sizeof path, "%s/testdata/ref_%d.bin", root, seed);
    float* ref = load_bin(path, NULL);

    tm_set_mode(mode);

    /* One warm-up outside the profiler, mirroring the device 'T' command:
     * the first forward pays first-touch page faults here and flash-cache
     * misses there, and including it would misattribute that to the kernels. */
    memcpy(tm_input(), in, NBT * sizeof(float));
    tm_forward(tm_input(), tm_output(), wf32, &q12);

    tp_reset();
    uint64_t t0 = tp_now_us();
    for (int r = 0; r < reps; r++) {
        memcpy(tm_input(), in, NBT * sizeof(float));
        tm_forward(tm_input(), tm_output(), wf32, &q12);
    }
    uint64_t t1 = tp_now_us();

    double max_abs = 0.0, max_rel = 0.0;
    long fails = gate(tm_output(), ref, &max_abs, &max_rel);

    /* Provenance lines the collector folds into the artifact. `device=host` is
     * what makes the report label this capture correctly. */
    printf("TPROF|env|device=host|mode=%d|seed=%d|reps=%d\n", mode, seed, reps);
    printf("TPROF|gate|seed=%d|fails=%ld|max_abs=%.6e|max_rel=%.6e|atol=0.002|rtol=0.02\n",
           seed, fails, max_abs, max_rel);
    printf("TPROF|fwd|us=%llu|reps=%d|per_forward_us=%llu\n",
           (unsigned long long)(t1 - t0), reps,
           (unsigned long long)((t1 - t0) / (unsigned)reps));

    tp_dump();
    return fails == 0 ? 0 : 1;
}
