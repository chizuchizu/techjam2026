/*
 * shard_host_test.c - run the two-node sequence-sharded forward on the host.
 *
 * Both nodes live in one process and the per-layer K/V exchange is a memcpy,
 * so the numerics of the distributed path can be validated against the torch
 * references before any firmware is flashed. The node code is exactly the
 * firmware's (src/model_shard.c) and the driver order is exactly the
 * firmware's (pre -> exchange -> post per layer, then final).
 *
 * Build: make -C tools shard_host_test
 * Run:   ./tools/shard_host_test [seeds...|all] [--bench] [--reps N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "kernels.h"
#include "model_shard.h"

#define NBT (TM_S * TM_D)
#define BASE "../../../optimisation/esp32-baseline"

static float* g_wf32 = NULL;
static TMQ12Weights g_q12;

static void* load_bin(const char* rel) {
    char path[512];
    snprintf(path, sizeof path, "%s/%s", BASE, rel);
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    void* buf = malloc((size_t)sz);
    if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fprintf(stderr, "read %s\n", path); exit(2); }
    fclose(f);
    return buf;
}

/* The quantized LayerNorm output in kernels.c's shared Q15 scratch is per-node
 * state that has to survive the exchange (post() projects the queries from it).
 * On the boards each node owns its own copy; in this single-process driver the
 * two nodes take turns, so it is saved and restored around every switch. */
static int16_t a16_save[TMS_NODES][TMS_SLOC * TM_D];
static void ctx_save(int n)    { memcpy(a16_save[n], tm_gemm_a16(), sizeof a16_save[0]); }
static void ctx_restore(int n) { memcpy(tm_gemm_a16(), a16_save[n], sizeof a16_save[0]); }

/* the exchange the firmware performs over TCP, done in memory here */
static void exchange(TMShard* a, TMShard* b) {
    float   hdr[TMS_HDR_FLOATS], sk[TM_H];
    static int16_t kv[TMS_KV_ELEMS];
    memcpy(hdr, a->hdr_own, sizeof hdr);
    memcpy(sk,  a->sk_own,  sizeof sk);
    memcpy(kv,  a->big.kv.own, sizeof kv);
    memcpy(a->hdr_peer, b->hdr_own, sizeof hdr);
    memcpy(a->sk_peer,  b->sk_own,  sizeof sk);
    memcpy(a->big.kv.peer, b->big.kv.own, sizeof kv);
    memcpy(b->hdr_peer, hdr, sizeof hdr);
    memcpy(b->sk_peer,  sk,  sizeof sk);
    memcpy(b->big.kv.peer, kv, sizeof kv);
    /* the firmware moves exactly this much per layer, per direction */
    (void)TMS_CHUNK_BYTES;
}

/* scatter a [S,D] tensor into the two nodes' local row sets, and gather back */
static float g_rows[TMS_NODES][TMS_SLOC * TM_D];
static void scatter(const float* x, TMShard* a, TMShard* b) {
    for (int i = 0; i < TM_S; i++)
        memcpy(g_rows[i % TMS_NODES] + (size_t)(i / TMS_NODES) * TM_D,
               x + (size_t)i * TM_D, TM_D * sizeof(float));
    tm_shard_load(a, g_rows[0]);
    tm_shard_load(b, g_rows[1]);
}
static void gather(float* y, TMShard* a, TMShard* b) {
    for (int i = 0; i < TM_S; i++) {
        TMShard* n = (i % TMS_NODES == 0) ? a : b;
        memcpy(y + (size_t)i * TM_D,
               tm_shard_output(n) + (size_t)(i / TMS_NODES) * TM_D,
               TM_D * sizeof(float));
    }
}

static void cluster_forward(const float* x, float* y, TMShard* a, TMShard* b) {
    scatter(x, a, b);
    for (int l = 0; l < TM_L; l++) {
        tm_shard_layer_pre(a, l);  ctx_save(0);
        tm_shard_layer_pre(b, l);  ctx_save(1);
        exchange(a, b);
        ctx_restore(0); tm_shard_layer_post(a, l);
        ctx_restore(1); tm_shard_layer_post(b, l);
    }
    tm_shard_final(a);
    tm_shard_final(b);
    gather(y, a, b);
}

static long gate_failures(const float* got, const float* ref, double* max_abs) {
    long fails = 0; double ma = 0.0;
    for (int i = 0; i < NBT; i++) {
        double d = fabs((double)got[i] - (double)ref[i]);
        if (d > ma) ma = d;
        if (!(d <= 0.002 || d <= 0.02 * fabs((double)ref[i]))) fails++;
    }
    if (max_abs) *max_abs = ma;
    return fails;
}

int main(int argc, char** argv) {
    int seeds[64], nseeds = 0, bench = 0, reps = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "all")) { for (int s = 0; s < 25; s++) seeds[nseeds++] = s; }
        else if (!strcmp(argv[i], "--bench")) bench = 1;
        else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
        else seeds[nseeds++] = atoi(argv[i]);
    }
    if (!nseeds) for (int s = 0; s < 5; s++) seeds[nseeds++] = s;

    g_wf32 = (float*)load_bin("weights.bin");
    tm_scan_q12(load_bin("weights_q12.bin"), &g_q12);

    TMShard* a = (TMShard*)malloc(sizeof(TMShard));
    TMShard* b = (TMShard*)malloc(sizeof(TMShard));
    if (!a || !b) { fprintf(stderr, "oom\n"); return 2; }
    tm_shard_init(a, 0, g_wf32, &g_q12);
    tm_shard_init(b, 1, g_wf32, &g_q12);
    printf("shard state = %zu B/node (%d local rows)\n", sizeof(TMShard), TMS_SLOC);

    float* x = malloc(NBT * 4);
    float* ref = malloc(NBT * 4);
    float* out = malloc(NBT * 4);
    long total_fail = 0;
    char path[512];

    for (int i = 0; i < nseeds; i++) {
        int s = seeds[i];
        snprintf(path, sizeof path, "%s/testdata/input_%d.bin", BASE, s);
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "missing %s\n", path); return 2; }
        if (fread(x, 4, NBT, f) != NBT) return 2;
        fclose(f);
        snprintf(path, sizeof path, "%s/testdata/ref_%d.bin", BASE, s);
        f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "missing %s\n", path); return 2; }
        if (fread(ref, 4, NBT, f) != NBT) return 2;
        fclose(f);

        if (bench) {
            cluster_forward(x, out, a, b);
            struct timespec t0, t1;
            clock_gettime(CLOCK_MONOTONIC, &t0);
            for (int r = 0; r < reps; r++) cluster_forward(x, out, a, b);
            clock_gettime(CLOCK_MONOTONIC, &t1);
            double sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
            printf("bench[%d] %.4f s per cluster forward (both nodes, host)\n", s, sec / reps);
            continue;
        }
        cluster_forward(x, out, a, b);
        double max_abs = 0.0;
        long fails = gate_failures(out, ref, &max_abs);
        total_fail += fails;
        printf("seed=%2d fails=%5ld max_abs=%.4e %s\n", s, fails, max_abs,
               fails ? "FAIL" : "ok");
    }
    if (!bench)
        printf("done: %d seeds, %ld failing elements %s\n", nseeds, total_fail,
               total_fail ? "==> FAIL" : "==> ALL PASS");
    return total_fail ? 1 : 0;
}
