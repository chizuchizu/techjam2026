/*
 * transformer.c - runtime-parameterized Transformer forward (EXACT + FAST).
 *
 * See transformer.h for the public contract.  No malloc/free inside tm_run:
 * every scratch buffer is a span of the caller's workspace, whose size comes
 * from tm_workspace_size().  This keeps the function MCU-friendly (arena
 * allocated once) while every geometry constant is an ordinary runtime value
 * read from `tm_case`, so ONE binary serves cases 02, 06, 07, 09, 10, 11, 12,
 * 13 shape family, and any other valid B/S/D/H/F/L combination.
 */
#include "transformer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================ clock hook ============================ */

static int64_t tm_host_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000LL + (int64_t)ts.tv_nsec / 1000;
}

static int64_t (*g_now)(void) = tm_host_now_us;
void tm_set_clock(int64_t (*fn)(void)) { g_now = fn ? fn : tm_host_now_us; }
static int64_t now_us(void) { return g_now(); }

/* ----------------------- weight flat-buffer offsets ----------------------- */
static int64_t w_layer_floats(const tm_case* c) {
    int64_t D = c->D, F = c->F;
    return 2*D + 4*(D*D + D) + 2*D + (F*D + F) + (D*F + D);
}
size_t tm_w_layer(const tm_case* c) { return (size_t)w_layer_floats(c); }
size_t tm_w_total(const tm_case* c) {
    return (size_t)((int64_t)c->L * w_layer_floats(c) + 2LL*c->D);
}
int tm_qmat_count(const tm_case* c) { return c->L * 6; }

void tm_scan_q12(const tm_case* c, const void* blob,
                 const int16_t** q, float* wscale) {
    const uint8_t* p = (const uint8_t*)blob;
    const int nm = tm_qmat_count(c);
    for (int i = 0; i < nm; i++) {
        uint32_t cnt; float ws2;
        memcpy(&cnt, p, 4); p += 4;
        memcpy(&ws2, p, 4); p += 4;
        q[i] = (const int16_t*)p;
        wscale[i] = ws2;
        p += (size_t)cnt * 2;
    }
}

/* ============================ workspace layout ============================ */

typedef struct {
    int S, D, H, F, L, HD;
    float     *x, *buf1, *buf2;
    int16_t   *a16;
    int64_t   *acc;
    int16_t   *qh, *kh, *vh;
    int64_t   *score;
    int32_t   *p15;
    int64_t prof[12];
    int64_t total_us;
} ws_t;

enum { K_NORM1=0, K_ATTN=1, K_OPROJ=2, K_RES1=3,
       K_NORM2=4, K_FFN1=5, K_GELU=6, K_FFN2=7, K_RES2=8, K_FINAL=9,
       K_QKV=10, K_QUANT=11 };

static void* ws_span(void** p, size_t* rem, size_t need, size_t align) {
    if (align < 8) align = 8;
    uintptr_t a = (uintptr_t)*p;
    uintptr_t up = (a + align - 1) & ~((uintptr_t)(align - 1));
    size_t skip = (size_t)(up - a);
    if (skip > *rem) return NULL;
    *p = (void*)up;
    if (need > *rem - skip) return NULL;
    *rem -= skip;
    void* out = *p;
    *p = (char*)(*p) + need;
    *rem -= need;
    return out;
}

size_t tm_workspace_size(const tm_case* c) {
    size_t S = (size_t)c->S, D = (size_t)c->D, F = (size_t)c->F;
    size_t HD = (size_t)(c->D / c->H);
    size_t mx = (D > F) ? D : F;
    size_t need = sizeof(ws_t);
    need += S*D*4 * 3;      /* x, buf1, buf2 */
    need += S*mx*2;         /* a16   */
    need += mx*8;           /* acc64 (per-row scratch, overflow-proof) */
    need += S*HD*2 * 3;     /* qh, kh, vh */
    need += S*8;            /* score */
    need += S*4;            /* p15 */
    need += 512;            /* alignment slack */
    return need;
}

static ws_t* ws_setup(const tm_case* c, void* ws) {
    ws_t* w = (ws_t*)ws;
    size_t rem = tm_workspace_size(c);
    void* q = (char*)ws + sizeof(ws_t);
    rem -= sizeof(ws_t);
    memset(ws, 0, sizeof(ws_t));

    w->S = c->S; w->D = c->D; w->H = c->H; w->F = c->F; w->L = c->L;
    w->HD = c->D / c->H;

    size_t S = (size_t)c->S, D = (size_t)c->D, F = (size_t)c->F;
    size_t HD = (size_t)w->HD;
    size_t mx = (D > F) ? D : F;

    w->x    = (float*)   ws_span(&q, &rem, S*D*4,   16);
    w->buf1 = (float*)   ws_span(&q, &rem, S*D*4,   16);
    w->buf2 = (float*)   ws_span(&q, &rem, S*D*4,   16);
    w->a16  = (int16_t*) ws_span(&q, &rem, S*mx*2,  16);
    w->acc  = (int64_t*) ws_span(&q, &rem, mx*8,    16);
    w->qh   = (int16_t*) ws_span(&q, &rem, S*HD*2,   8);
    w->kh   = (int16_t*) ws_span(&q, &rem, S*HD*2,   8);
    w->vh   = (int16_t*) ws_span(&q, &rem, S*HD*2,   8);
    w->score= (int64_t*) ws_span(&q, &rem, S*8,      8);
    w->p15  = (int32_t*) ws_span(&q, &rem, S*4,      8);

    if (!w->x || !w->buf1 || !w->buf2 || !w->a16 || !w->acc ||
        !w->qh || !w->kh || !w->vh || !w->score || !w->p15)
        return NULL;
    return w;
}

void tm_workspace_init(const tm_case* c, void* ws) {
    ws_t* w = ws_setup(c, ws);
    if (w) memset(w->prof, 0, sizeof w->prof);
}

#define TB() do { int64_t _t0 = now_us(); (void)_t0;
#define TE(slot) { int64_t _d = now_us() - _t0; \
                    if (_d > 0) (w)->prof[slot] += _d; } } while (0)

/* ============================ LayerNorm ============================ */
/* torch nn.LayerNorm: mean, biased variance (divide by N), eps 1e-5, affine. */
static void layernorm(float* out, const float* in,
                      const float* g, const float* b, int S, int D) {
    for (int i = 0; i < S; i++) {
        const float* row = in + (size_t)i*D;
        float* orow = out + (size_t)i*D;
        double sum = 0.0, sq = 0.0;
        for (int d = 0; d < D; d++) { double v = row[d]; sum += v; sq += v*v; }
        double mean = sum / (double)D;
        double var = sq / (double)D - mean*mean;
        double inv = 1.0 / sqrt(var + 1e-5);
        for (int d = 0; d < D; d++)
            orow[d] = (float)(((double)row[d] - mean) * inv) * g[d] + b[d];
    }
}

/* LayerNorm -> Q15, dequant scale sa returned (value = q * sa). */
static float layernorm_q15(const float* in, const float* g, const float* b,
                           int S, int D, float* scratch, int16_t* out) {
    float amax = 0.0f;
    for (int i = 0; i < S; i++) {
        const float* row = in + (size_t)i*D;
        float* srow = scratch + (size_t)i*D;
        double sum = 0.0, sq = 0.0;
        for (int d = 0; d < D; d++) { double v = row[d]; sum += v; sq += v*v; }
        double mean = sum / (double)D;
        double var = sq / (double)D - mean*mean;
        double inv = 1.0 / sqrt(var + 1e-5);
        for (int d = 0; d < D; d++) {
            float v = (float)(((double)row[d] - mean) * inv) * g[d] + b[d];
            srow[d] = v;
            float a = v < 0.0f ? -v : v;
            if (a > amax) amax = a;
        }
    }
    float sa = (amax > 0.0f) ? amax / 32767.0f : 1.0f;
    float isa = (amax > 0.0f) ? 32767.0f / amax : 1.0f;
    for (int i = 0; i < S*D; i++) {
        float v = scratch[i] * isa;
        int q;
        if (v >= 0.0f) { q = (int)(v + 0.5f); if (q > 32767) q = 32767; }
        else           { q = (int)(v - 0.5f); if (q < -32767) q = -32767; }
        out[i] = (int16_t)q;
    }
    return sa;
}

/* Strided variant: in is rows x cols with row stride `stride`. */
static float quant_act_s(const float* in, int rows, int cols, int stride,
                         int16_t* out) {
    float amax = 0.0f;
    for (int i = 0; i < rows; i++) {
        const float* row = in + (size_t)i*stride;
        for (int d = 0; d < cols; d++) {
            float a = row[d] < 0.0f ? -row[d] : row[d];
            if (a > amax) amax = a;
        }
    }
    float sa = (amax > 0.0f) ? amax / 32767.0f : 1.0f;
    float isa = (amax > 0.0f) ? 32767.0f / amax : 1.0f;
    for (int i = 0; i < rows; i++) {
        const float* row = in + (size_t)i*stride;
        int16_t* orow = out + (size_t)i*cols;
        for (int d = 0; d < cols; d++) {
            float v = row[d] * isa;
            int q;
            if (v >= 0.0f) { q = (int)(v + 0.5f); if (q > 32767) q = 32767; }
            else           { q = (int)(v - 0.5f); if (q < -32767) q = -32767; }
            orow[d] = (int16_t)q;
        }
    }
    return sa;
}

/* Quantize a general fp32 activation to Q15; returns dequant scale sa. */
static float quant_act(const float* in, int n, int16_t* out) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = in[i] < 0.0f ? -in[i] : in[i];
        if (a > amax) amax = a;
    }
    float sa = (amax > 0.0f) ? amax / 32767.0f : 1.0f;
    float isa = (amax > 0.0f) ? 32767.0f / amax : 1.0f;
    for (int i = 0; i < n; i++) {
        float v = in[i] * isa;
        int q;
        if (v >= 0.0f) { q = (int)(v + 0.5f); if (q > 32767) q = 32767; }
        else           { q = (int)(v - 0.5f); if (q < -32767) q = -32767; }
        out[i] = (int16_t)q;
    }
    return sa;
}

static float exp_fast_nonpos(float y);  /* defined below */

/* ============================ GELU (exact erf) ============================ */
static float erf7(float x) {
    const float ax = x < 0.0f ? -x : x;
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    float y = 1.0f - (((((1.061405429f*t - 1.453152027f)*t) + 1.421413741f)*t
                       - 0.284496736f)*t + 0.254829592f)*t*exp_fast_nonpos(-ax*ax);
    return x < 0.0f ? -y : y;
}
static void gelu_exact(float* x, int n) {
    const float inv = 0.7071067811865475f;
    for (int i = 0; i < n; i++) x[i] = 0.5f*x[i]*(1.0f + erf7(x[i]*inv));
}

/* fast exp for non-positive y (softmax), rel error ~1e-4 */
static float exp_fast_nonpos(float y) {
    if (y < -87.0f) return 0.0f;
    float v = y * 1.4426950408889634f;
    float fl = floorf(v);
    int n = (int)fl;
    float f = v - fl;
    float p2 = 1.0f + f*(0.6931472f + f*(0.24022651f + f*(0.05550411f
                                  + f*0.00961813f)));
    return ldexpf(p2, n);
}

/* ============================ GEMMs ============================ */

/* C[M,N] = A[M,K] . W[N,K]^T + bias[N]; A row-major, W row-major [N,K]. */
static void gemm_f32(const float* A, const float* W, const float* bias,
                     float* C, int M, int K, int N, int strideA, int strideC) {
    for (int m = 0; m < M; m++) {
        const float* a = A + (size_t)m*strideA;
        float* c = C + (size_t)m*strideC;
        for (int n = 0; n < N; n++) {
            const float* w = W + (size_t)n*K;
            double acc = 0.0;
            for (int k = 0; k < K; k++) acc += (double)a[k]*w[k];
            c[n] = (float)acc + bias[n];
        }
    }
}

/* C[M,N] = Aq[M,K](Q15,scale sa) . Wq[N,K](Q12,scale ws) + bias[N];
 * dequant factor = sa*ws.  acc is scratch of M*N int32. */
static void gemm_q12(const int16_t* Aq, float sa, const int16_t* Wq,
                     float ws, const float* bias, float* C,
                     int M, int K, int N, int strideA, int strideC,
                     int64_t* acc) {
    const float deq = sa * ws;
    for (int m = 0; m < M; m++) {
        const int16_t* a = Aq + (size_t)m*strideA;
        float* c = C + (size_t)m*strideC;
        for (int n = 0; n < N; n++) {
            const int16_t* w = Wq + (size_t)n*K;
            /* CMSIS-NN style split accumulator: int32 inner blocks (fast on
             * RV32) folded into an int64 total so long K can never overflow.
             * Q15 act x Q12 wgt peaks at 6.7e7; an 8-term int32 block stays
             * below 5.4e8, far inside int32. */
            int64_t s = 0;
            int k = 0;
            for (; k + 8 <= K; k += 8) {
                int32_t sb = 0;
                for (int j = 0; j < 8; j++)
                    sb += (int32_t)a[k+j]*(int32_t)w[k+j];
                s += sb;
            }
            for (; k < K; k++) s += (int64_t)((int32_t)a[k]*(int32_t)w[k]);
            acc[n] = s;
        }
        for (int n = 0; n < N; n++) c[n] = (float)acc[n]*deq + bias[n];
    }
}


/* ============================ attention head ============================ */
/* qh/kh/vh are Q15 int16 with dequant scales sq/sk/sv.  Writes the head's
 * context into ctx at column offset head*HD.  score/p15 are ws scratch. */
static void attn_head(ws_t* w, int head, const int16_t* qh, float sq,
                      const int16_t* kh, float sk,
                      const int16_t* vh, float sv, float* ctx, int fast) {
    const int S = w->S, HD = w->HD, D = w->D;
    int64_t* score = w->score;
    int32_t* p15 = w->p15;
    const float gscale = sq * sk / sqrtf((float)HD);

    for (int i = 0; i < S; i++) {
        const int16_t* qi = qh + (size_t)i*HD;
        int64_t maxdot = INT64_MIN;
        for (int j = 0; j <= i; j++) {
            const int16_t* kj = kh + (size_t)j*HD;
            int64_t acc = 0;
            for (int d = 0; d < HD; d++) acc += (int64_t)((int32_t)qi[d]*(int32_t)kj[d]);
            score[j] = acc;
            if (acc > maxdot) maxdot = acc;
        }
        int32_t lsum = 0;
        for (int j = 0; j <= i; j++) {
            double sf = (double)(score[j] - maxdot) * (double)gscale;
            float p = fast ? exp_fast_nonpos((float)sf) : expf((float)sf);
            int32_t qq = (int32_t)(p*32768.0 + 0.5);
            p15[j] = qq;
            lsum += qq;
        }
        float* o = ctx + (size_t)i*D + (size_t)head*HD;
        const float f = (lsum > 0) ? sv / (float)lsum : 0.0f;
        for (int d = 0; d < HD; d++) {
            int64_t acc = 0;
            for (int j = 0; j <= i; j++)
                acc += (int64_t)((int32_t)p15[j] * (int32_t)vh[(size_t)j*HD + d]);
            o[d] = (float)acc * f;
        }
    }
}

/* ============================ main forward ============================ */

int tm_run(const tm_case* c, void* ws,
           const float* W, const int16_t* const* q, const float* wscl,
           const float* xin, float* yout) {
    if (!c || !ws || !W || !xin || !yout) return -1;
    if (c->S <= 0 || c->D <= 0 || c->H <= 0 || c->F <= 0 || c->L <= 0) return -2;
    if (c->D % c->H != 0) return -3;
    if (c->mode != TM_EXACT && c->mode != TM_FAST) return -4;
    if (c->mode == TM_FAST && (!q || !wscl)) return -5;

    ws_t* w = ws_setup(c, ws);
    if (!w) return -6;
    memset(w->prof, 0, sizeof w->prof);

    const int fast = (c->mode == TM_FAST);
    const int S = c->S, D = c->D, H = c->H, F = c->F, L = c->L, HD = w->HD;
    const int64_t lf = w_layer_floats(c);

    if (xin != w->x) memcpy(w->x, xin, (size_t)S*D*4);
    int64_t t_total0 = now_us();

    for (int l = 0; l < L; l++) {
        const float* WL = W + (int64_t)l*lf;
        const float* n1w = WL + 0;
        const float* n1b = WL + D;
        const float* qw  = WL + 2*D;
        const float* qb  = WL + 2*D + (int64_t)D*D;
        const float* kw  = WL + 3*D + (int64_t)D*D;
        const float* kb  = WL + 3*D + 2LL*D*D;
        const float* vw  = WL + 4*D + 2LL*D*D;
        const float* vb  = WL + 4*D + 3LL*D*D;
        const float* ow  = WL + 5*D + 3LL*D*D;
        const float* ob  = WL + 5*D + 4LL*D*D;
        const float* n2w = WL + 6*D + 4LL*D*D;
        const float* n2b = WL + 7*D + 4LL*D*D;
        const float* f1w = WL + 8*D + 4LL*D*D;
        const float* f1b = WL + 8*D + 4LL*D*D + (int64_t)F*D;
        const float* f2w = WL + 8*D + 4LL*D*D + (int64_t)F*D + F;
        const float* f2b = WL + 8*D + 4LL*D*D + (int64_t)F*D + F + (int64_t)D*F;

        /* ---- norm1 ---- */
        float sa1 = 32767.0f;
        TB(); {
            if (fast) sa1 = layernorm_q15(w->x, n1w, n1b, S, D, w->buf1, w->a16);
            else      layernorm(w->buf1, w->x, n1w, n1b, S, D);
        } TE(K_NORM1);

        /* ---- per-head QKV projections + attention, then output proj ---- */
        TB(); {
            for (int h = 0; h < H; h++) {
                float sq = 0.0f, sk = 0.0f, sv = 0.0f;
                if (fast) {
                    const int16_t* wq = q[l*6+0] + (size_t)h*HD*D;
                    const int16_t* wk = q[l*6+1] + (size_t)h*HD*D;
                    const int16_t* wv = q[l*6+2] + (size_t)h*HD*D;
                    gemm_q12(w->a16, sa1, wq, wscl[l*6+0],
                             qb + (size_t)h*HD, w->buf2 + (size_t)h*HD,
                             S, D, HD, D, D, w->acc);
                    sq = quant_act_s(w->buf2 + (size_t)h*HD, S, HD, D, w->qh);
                    gemm_q12(w->a16, sa1, wk, wscl[l*6+1],
                             kb + (size_t)h*HD, w->buf2 + (size_t)h*HD,
                             S, D, HD, D, D, w->acc);
                    sk = quant_act_s(w->buf2 + (size_t)h*HD, S, HD, D, w->kh);
                    gemm_q12(w->a16, sa1, wv, wscl[l*6+2],
                             vb + (size_t)h*HD, w->buf2 + (size_t)h*HD,
                             S, D, HD, D, D, w->acc);
                    sv = quant_act_s(w->buf2 + (size_t)h*HD, S, HD, D, w->vh);
                } else {
                    gemm_f32(w->buf1, qw + (size_t)h*HD*D, qb + (size_t)h*HD,
                             w->buf2 + (size_t)h*HD, S, D, HD, D, D);
                    sq = quant_act_s(w->buf2 + (size_t)h*HD, S, HD, D, w->qh);
                    gemm_f32(w->buf1, kw + (size_t)h*HD*D, kb + (size_t)h*HD,
                             w->buf2 + (size_t)h*HD, S, D, HD, D, D);
                    sk = quant_act_s(w->buf2 + (size_t)h*HD, S, HD, D, w->kh);
                    gemm_f32(w->buf1, vw + (size_t)h*HD*D, vb + (size_t)h*HD,
                             w->buf2 + (size_t)h*HD, S, D, HD, D, D);
                    sv = quant_act_s(w->buf2 + (size_t)h*HD, S, HD, D, w->vh);
                }
                { int64_t _ta = now_us();
                  attn_head(w, h, w->qh, sq, w->kh, sk, w->vh, sv, w->buf2, fast);
                  int64_t _d = now_us() - _ta;
                  if (_d > 0) w->prof[K_ATTN] += _d; }
            }
        } TE(K_QKV);

        /* ---- out projection ---- */
        TB(); {
            if (fast) {
                float so = quant_act(w->buf2, S*D, w->a16);
                gemm_q12(w->a16, so, q[l*6+3], wscl[l*6+3],
                         ob, w->buf1, S, D, D, D, D, w->acc);
            } else {
                gemm_f32(w->buf2, ow, ob, w->buf1, S, D, D, D, D);
            }
        } TE(K_OPROJ);

        /* ---- residual ---- */
        TB(); {
            for (int i2 = 0; i2 < S*D; i2++) w->x[i2] += w->buf1[i2];
        } TE(K_RES1);

        /* ---- norm2 ---- */
        float sa2 = 32767.0f;
        TB(); {
            if (fast) sa2 = layernorm_q15(w->x, n2w, n2b, S, D, w->buf1, w->a16);
            else      layernorm(w->buf1, w->x, n2w, n2b, S, D);
        } TE(K_NORM2);

        /* ---- FFN1 ---- */
        float sa3 = 32767.0f;
        TB(); {
            if (fast) {
                gemm_q12(w->a16, sa2, q[l*6+4], wscl[l*6+4],
                         f1b, w->buf2, S, D, F, D, F, w->acc);
            } else {
                gemm_f32(w->buf1, f1w, f1b, w->buf2, S, D, F, D, F);
            }
        } TE(K_FFN1);

        /* ---- GELU ---- */
        TB(); {
            gelu_exact(w->buf2, S*F);
        } TE(K_GELU);
        TB(); {
            sa3 = fast ? quant_act(w->buf2, S*F, w->a16) : 32767.0f;
        } TE(K_QUANT);

        /* ---- FFN2 ---- */
        TB(); {
            if (fast) {
                gemm_q12(w->a16, sa3, q[l*6+5], wscl[l*6+5],
                         f2b, w->buf1, S, F, D, F, D, w->acc);
            } else {
                gemm_f32(w->buf2, f2w, f2b, w->buf1, S, F, D, F, D);
            }
        } TE(K_FFN2);

        /* ---- residual ---- */
        TB(); {
            for (int i2 = 0; i2 < S*D; i2++) w->x[i2] += w->buf1[i2];
        } TE(K_RES2);
    }

    /* ---- final norm ---- */
    TB(); {
        layernorm(yout, w->x, W + (int64_t)L*lf, W + (int64_t)L*lf + D, S, D);
    } TE(K_FINAL);

    w->total_us = now_us() - t_total0;
    return 0;
}

/* ======================= direct buffer accessors ======================= */

void* tm_input_buf(const tm_case* c, void* ws) {
    ws_t* w = ws_setup(c, ws);
    return w ? w->x : NULL;
}

void* tm_output_buf(const tm_case* c, void* ws) {
    (void)c;
    ws_t* w = (ws_t*)ws;
    return w ? w->x : NULL;   /* final LayerNorm is written in place */
}

/* ============================ profile API ============================ */

static const char* const kname[12] = {
    "norm1", "attn", "oproj", "res1", "norm2", "ffn1",
    "gelu", "ffn2", "res2", "final", "qkv", "quant"
};

void tm_profile_get(const tm_case* c, const void* ws, tm_profile* out) {
    ws_t* w = (ws_t*)ws;
    if (!w || !out) return;
    memset(out, 0, sizeof *out);
    out->norm1_us = w->prof[K_NORM1];
    out->qkv_us   = w->prof[K_QKV];
    out->attn_us  = w->prof[K_ATTN];
    out->oproj_us = w->prof[K_OPROJ];
    out->res1_us  = w->prof[K_RES1];
    out->norm2_us = w->prof[K_NORM2];
    out->ffn1_us  = w->prof[K_FFN1];
    out->gelu_us  = w->prof[K_GELU];
    out->ffn2_us  = w->prof[K_FFN2];
    out->res2_us  = w->prof[K_RES2];
    out->final_us = w->prof[K_FINAL];
    out->quant_us = w->prof[K_QUANT];
    out->total_us = w->total_us;
    (void)c;
}

void tm_profile_dump(const tm_case* c, const void* ws) {
    ws_t* w = (ws_t*)ws;
    if (!w) return;
    const int64_t* p = w->prof;
    printf("== tm profile (us, per one forward) ==\n");
    for (int i = 0; i < 12; i++) {
        if (p[i] > 0)
            printf("  %-6s %12lld\n", kname[i], (long long)p[i]);
    }
    printf("  %-6s %12lld\n", "TOTAL", (long long)w->total_us);
    (void)c;
}
