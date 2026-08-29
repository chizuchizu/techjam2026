
"""quant_review.py - independent numpy verification of the strided per-head int16
attention design for the ESP32-C3 TinyStories transformer (case 2).

Implements, from scratch, exactly how the fixed firmware computes (see
src/model.c / src/kernels.c / src/tm_config.h / tools/export_case2.py for the
layouts and formulas; no C code is reused here):

  EXACT : fp32 GEMMs (fp32 weights) everywhere EXCEPT the 3 attention head
          projections which are quantized to int16 Q15 (per-head amax scale),
          attention math in fp32 with the online running-max softmax, exp = expf.
  FAST  : all 6 per-layer linear projections via Q15-activation x Q12-weight
          GEMM (tm_gemm_q12 semantics), attention exp = tm_exp_fast,
          LayerNorm/GELU in fp32 like the C.

Gate (same as manifest.json / compare.py): pass if |a-b| <= 0.002 OR
|a-b| <= 0.02*|b|, vs testdata/ref_<s>.bin (torch fp32).
"""
import argparse
import numpy as np

S, D, H, HD, FDIM, L = 128, 128, 4, 32, 128, 4
LN_EPS    = np.float32(1e-5)
ATTN_SC   = np.float32(0.1767766922712326)      # 1/sqrt(32), matches torch scale
INV_SQRT2 = np.float32(0.7071067690849304)
HALF      = np.float32(0.5)
QACT_MAX  = 32767.0
QWT_MAX   = 2047.0

ERF_C = np.array([0.9952985640,0.0414220320,-0.1635044520,0.3830335440,-0.5738591550,
                  0.4437286850,0.1318328530,-0.5317970510,0.2018017170,0.1784498840,
                  -0.0916074370,-0.0148286080], dtype=np.float32)

_EXPF = (np.float32(1.4426950), np.float32(0.69314718), np.float32(0.24022651),
         np.float32(0.055504110), np.float32(0.0096181287), np.float32(0.0013333558))

# ---- weight block offsets (mirror model.c woff()) ----
def woff(layer, blk):
    o = layer * 99584
    dd = D * D
    T = {0:0, 1:D, 2:2*D, 3:2*D+dd, 4:3*D+dd, 5:3*D+2*dd, 6:4*D+2*dd, 7:4*D+3*dd,
         8:5*D+3*dd, 9:5*D+4*dd, 10:6*D+4*dd, 11:7*D+4*dd, 12:8*D+4*dd,
         13:8*D+4*dd+FDIM*D, 14:8*D+4*dd+FDIM*D+FDIM, 15:8*D+4*dd+FDIM*D+FDIM+D*FDIM}
    return o + T[blk]

def load_weights(path):
    w = np.fromfile(path, dtype="<f4").astype(np.float32)
    assert w.size == L * 99584 + 2 * D, w.size
    return w

def load_q12(path):
    """Parse blob exactly as tm_scan_q12: per layer 6 mats {u32 count, f32 w_scale, i16 data]."""
    raw = open(path, "rb").read()
    p = 0
    ws_layers, q_layers = [], []
    for _ in range(L):
        ws_l, q_l = [], []
        for _ in range(6):
            count = int.from_bytes(raw[p:p+4], "little"); p += 4
            wscale = np.frombuffer(raw[p:p+4], dtype="<f4")[0]; p += 4
            ws_l.append(np.float32(wscale))
            q_l.append(np.frombuffer(raw[p:p+2*count], dtype="<i2").astype(np.int16))
            p += 2 * count
        ws_layers.append(ws_l); q_layers.append(q_l)
    assert p == len(raw), (p, len(raw))
    return ws_layers, q_layers

def _rint_clamp(v, amax, qmax):
    if amax == 0.0: amax = 1.0
    sa = np.float32(qmax / amax)
    q = np.rint(v * sa).astype(np.int32)          # llrintf: round-half-even
    return np.clip(q, -int(qmax), int(qmax)).astype(np.int16)

def quant_head(A):
    """per-head Q15 quant of an (S,HD) slice; returns (q16, scale=amax/32767)."""
    amax = float(np.abs(A).max())
    q = _rint_clamp(A, amax, QACT_MAX)
    scale = np.float32((1.0 if amax == 0.0 else amax) / QACT_MAX)
    return q, scale

def gemm_f32(A, Wq, bias, M, K, N):
    """fp32 GEMM, fp32 accumulation in k-order (matches tm_gemm_f32 and the C3 soft-float)."""
    acc = np.zeros((M, N), np.float32)
    for k in range(K):
        acc += A[:, k:k+1] * Wq[:, k][None, :]
    C = acc.copy()
    if bias is not None:
        C += bias[None, :].astype(np.float32)
    return C

def gemm_q12(A, Wq, w_scale, bias, M, K, N):
    """Q15(act) x Q12(wt) GEMM: per-call act amax, llrint+clamp; int acc in k-order;
    out = acc * (1/sa) * w_scale + bias. int64 acc (exact; verify no int32 wrap)."""
    amax = float(np.abs(A).max())
    if amax == 0.0: amax = 1.0
    sa = np.float32(QACT_MAX / amax)
    sa_inv = np.float32(1.0 / sa)
    q = _rint_clamp(A, amax, QACT_MAX).reshape(M, K).astype(np.int32)
    Wi = Wq.reshape(N, K).astype(np.int32)
    acc = np.zeros((M, N), np.int64)
    for k in range(K):
        acc += q[:, k:k+1] * Wi[:, k][None, :]
    g = np.float32(sa_inv * w_scale)
    C = acc.astype(np.float32) * g
    if bias is not None:
        C += bias[None, :].astype(np.float32)
    return C

def layernorm(X, gamma, beta):
    Sx, Dx = X.shape
    mean = np.zeros(Sx, np.float32)
    for k in range(Dx):
        mean += X[:, k]
    mean = mean / np.float32(Dx)
    var = np.zeros(Sx, np.float32)
    for k in range(Dx):
        t = X[:, k] - mean
        var += t * t
    var = var / np.float32(Dx)
    rstd = np.float32(1.0) / np.sqrt(var + LN_EPS)
    out = np.empty_like(X)
    for k in range(Dx):
        out[:, k] = (X[:, k] - mean) * rstd * gamma[k] + beta[k]
    return out

def _erf_poly(x):
    x = x.astype(np.float32)
    ax = np.abs(x)
    big = ax >= np.float32(4.0)
    t = HALF * ax - np.float32(1.0)
    p = np.full_like(x, ERF_C[11])
    for k in range(10, -1, -1):
        p = p * t + ERF_C[k]
    p = np.where(x < 0, -p, p)
    return np.where(big, np.where(x < 0, np.float32(-1.0), np.float32(1.0)), p)

def gelu(X):
    return HALF * X * (np.float32(1.0) + _erf_poly(X * INV_SQRT2))

def exp_fast(y):
    """tm_exp_fast for y<=0 (scalar or array), fp32 arithmetic."""
    y = y.astype(np.float32)
    arr = np.ndim(y) > 0
    if not arr:
        if y < np.float32(-40.0):
            return np.float32(0.0)
        LOG2E, C1, C2, C3, C4, C5 = _EXPF
        t = y * LOG2E
        n = int(t)
        if t < np.float32(n): n -= 1
        f = t - np.float32(n)
        p = C5
        p = p * f + C4; p = p * f + C3; p = p * f + C2; p = p * f + C1
        p = p * f + np.float32(1.0)
        return (p * ((np.uint32(n + 127) << np.uint32(23)).view(np.float32)))
    small = y < np.float32(-40.0)
    LOG2E, C1, C2, C3, C4, C5 = _EXPF
    t = y * LOG2E
    n = np.floor(t).astype(np.int64)
    f = t - n.astype(np.float32)
    p = np.full_like(y, C5)
    p = p * f + C4; p = p * f + C3; p = p * f + C2; p = p * f + C1
    p = p * f + np.float32(1.0)
    bits = ((n + 127).astype(np.uint64) << np.uint64(23)).astype(np.uint32)
    v = p * bits.view(np.float32)
    v[small] = 0.0
    return v

def exp_f32(y):
    return np.exp(y).astype(np.float32)

def attn_head(q16, k16, v16, sq, sk, sv, exp_fun):
    """Online-running-max causal attention for one head; ctx slice (S,HD) f32.
    Matches attn_head(): score per (i,j): sum_d ((q16*sq)*k16)*sk -> *ATTN_SC;
    running-max rescale; out = sum_j p_j * ((p*v16)*sv) / lsum."""
    Sx, HDx = q16.shape
    qf = q16.astype(np.float32) * sq
    kk = k16.astype(np.float32)
    vv = v16.astype(np.float32)
    s = np.zeros((Sx, Sx), np.float32)
    for d in range(HDx):
        s += qf[:, d:d+1] * kk[:, d][None, :] * sk
    s *= ATTN_SC
    ctx = np.zeros((Sx, HDx), np.float32)
    for i in range(Sx):
        m = np.float32(-np.inf)
        lsum = np.float32(0.0)
        acc = np.zeros(HDx, np.float32)
        for j in range(i + 1):
            sj = s[i, j]
            if sj > m:
                r = exp_fun(np.float32(m - sj))       # m old, sj new-max
                lsum = np.float32(lsum * r)
                acc = acc * r
                m = sj
            p = exp_fun(np.float32(sj - m))
            lsum = np.float32(lsum + p)
            acc += p * vv[j] * sv
        ctx[i, :] = acc / lsum
    return ctx

def forward(X, W, q12mats, fast=True):
    ws_layers, q_layers = q12mats
    X = X.astype(np.float32).copy()
    for l in range(L):
        B1 = layernorm(X, W[woff(l,0):woff(l,0)+D], W[woff(l,1):woff(l,1)+D])
        ctx = np.zeros((S, D), np.float32)
        for h in (1, 2, 3, 0):                      # head order in model.c
            if fast:
                B_, sq_ = quant_head(gemm_q12(B1, q_layers[l][0][h*HD*D:(h+1)*HD*D],
                        ws_layers[l][0], W[woff(l,3)+h*HD:woff(l,3)+(h+1)*HD], S, D, HD))
                K_, sk_ = quant_head(gemm_q12(B1, q_layers[l][1][h*HD*D:(h+1)*HD*D],
                        ws_layers[l][1], W[woff(l,5)+h*HD:woff(l,5)+(h+1)*HD], S, D, HD))
                V_, sv_ = quant_head(gemm_q12(B1, q_layers[l][2][h*HD*D:(h+1)*HD*D],
                        ws_layers[l][2], W[woff(l,7)+h*HD:woff(l,7)+(h+1)*HD], S, D, HD))
                ef = exp_fast
            else:
                Wq = W[woff(l,2):woff(l,2)+D*D].reshape(D, D)[h*HD:(h+1)*HD, :]
                B_, sq_ = quant_head(gemm_f32(B1, Wq, W[woff(l,3)+h*HD:woff(l,3)+(h+1)*HD], S, D, HD))
                Wk = W[woff(l,4):woff(l,4)+D*D].reshape(D, D)[h*HD:(h+1)*HD, :]
                K_, sk_ = quant_head(gemm_f32(B1, Wk, W[woff(l,5)+h*HD:woff(l,5)+(h+1)*HD], S, D, HD))
                Wv = W[woff(l,6):woff(l,6)+D*D].reshape(D, D)[h*HD:(h+1)*HD, :]
                V_, sv_ = quant_head(gemm_f32(B1, Wv, W[woff(l,7)+h*HD:woff(l,7)+(h+1)*HD], S, D, HD))
                ef = exp_f32
            at = attn_head(B_, K_, V_, sq_, sk_, sv_, ef)
            ctx[:, h*HD:(h+1)*HD] = at
        if fast:
            B1 = gemm_q12(ctx, q_layers[l][3], ws_layers[l][3],
                          W[woff(l,9):woff(l,9)+D], S, D, D)
        else:
            B1 = gemm_f32(ctx, W[woff(l,8):woff(l,8)+D*D].reshape(D, D),
                          W[woff(l,9):woff(l,9)+D], S, D, D)
        X = X + B1                                   # residual
        B1 = layernorm(X, W[woff(l,10):woff(l,10)+D], W[woff(l,11):woff(l,11)+D])
        if fast:
            F = gemm_q12(B1, q_layers[l][4], ws_layers[l][4],
                         W[woff(l,13):woff(l,13)+FDIM], S, D, FDIM)
        else:
            F = gemm_f32(B1, W[woff(l,12):woff(l,12)+FDIM*D].reshape(FDIM, D),
                         W[woff(l,13):woff(l,13)+FDIM], S, D, FDIM)
        F = gelu(F)
        if fast:
            B1 = gemm_q12(F, q_layers[l][5], ws_layers[l][5],
                          W[woff(l,15):woff(l,15)+D], S, FDIM, D)
        else:
            B1 = gemm_f32(F, W[woff(l,14):woff(l,14)+D*FDIM].reshape(D, FDIM),
                          W[woff(l,15):woff(l,15)+D], S, FDIM, D)
        X = X + B1
    return layernorm(X, W[L*99584:L*99584+D], W[L*99584+D:L*99584+2*D])

def gate_stats(y, ref, atol=0.002, rtol=0.02):
    y = y.astype(np.float64); ref = ref.astype(np.float64)
    d = np.abs(y - ref)
    ok = (d <= atol) | (d <= rtol * np.abs(ref))
    rel = d / np.maximum(np.abs(ref), 1e-30)
    return int((~ok).sum()), float(d.max()), float(rel.max()), int(d.size)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--mode", choices=["exact", "fast", "both"], default="both")
    ap.add_argument("--seeds", default=None, help="comma list or 'all' (25)")
    ap.add_argument("--check-int32", action="store_true")
    args = ap.parse_args()

    W = load_weights(f"{args.root}/weights.bin")
    q12 = load_q12(f"{args.root}/weights_q12.bin")
    if args.seeds is None or args.seeds == "all":
        seeds = list(range(25))
    else:
        seeds = [int(a) for a in args.seeds.split(",")]
    modes = ["exact", "fast"] if args.mode == "both" else [args.mode]
    for mode in modes:
        fast = mode == "fast"
        tot_fail = 0; maxabs = 0.0; maxrel = 0.0
        for s in seeds:
            X = np.fromfile(f"{args.root}/testdata/input_{s}.bin", dtype="<f4").reshape(S, D)
            R = np.fromfile(f"{args.root}/testdata/ref_{s}.bin", dtype="<f4").reshape(S, D)
            y = forward(X, W, q12, fast=fast)
            nf, ma, mr, n = gate_stats(y, R)
            tot_fail += nf; maxabs = max(maxabs, ma); maxrel = max(maxrel, mr)
            print(f"seed={s:2d} mode={mode.upper():5s} fails={nf:5d}/{n} max_abs={ma:.3e} max_rel={mr:.3e} {'ok' if nf==0 else 'FAIL'}")
        print(f"MODE {mode.upper()}: total fails={tot_fail}/{len(seeds)*16384}, max_abs={maxabs:.3e}, max_rel={maxrel:.3e} -> {'ALL PASS' if tot_fail==0 else 'FAIL'}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
