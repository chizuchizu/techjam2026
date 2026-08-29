# opt18: KB1 int32 bias-fold + hand-asm requant, core5 bias-fold epilogue, int32-limb QK (qk_v2)

**Result: 2.701 → 2.447 s/forward (17.2× vs fp32 baseline 42.15 s), gate-green.**

Three independent device-verified changes, committed as one unit (opt18). No new
static RAM (all scratch aliases `a16` / `g_buf2`); flash unchanged.

## 1. KB1 (Q/K/V head projection) epilogue → int32 bias-fold + hand-asm requant
The V-projection epilogue ran an int64/Q30 fixed-point path (soft-float __muldf3
etc. per element). Replaced with:
- **int32 bias fold**: precompute `BQ = round(bias/g)` once per GEMM call; the
  per-element add then happens in the int domain `c + BQ[j]` (range ±31M, exact
  in int32), and only one float op remains per element: `(float)(c+BQ[j]) * g`.
- **hand-asm requant** (`__riscv` only): `mul + mulh` accumulate, `+2^29`
  rounding, arithmetic `>>30`, clamp — the C fallback (host tests) is separate.

Device: **KB1 3.95 → 0.925 ms/call** (4×; ~48 ms/forward saved). Root cause of
cost: the old path did soft-float int64→float + two fp32 ops per element.

## 2. core5 GEMM bias fold (tm_gemm_core5)
Same trick in the core5 GEMM epilogue: `(float)(c + BQ[j]) * g` replaces
`c_float + bias[j]` — one fewer soft-float add per element (2 soft-float
libcalls → 1). Bias range ±31M still fits int32; verified exact vs old path.
Device: **C5CYC 13.18 M → 11.586 M cycles/call** (~12% fewer cycles; f2 + oproj,
the two core5 GEMMs, are ~160 ms/forward combined).

## 3. qk_v2: int64→int32 limb QK dot (causal attention)
Replaced the 64-bit QK accumulator with a two-limb int32 scheme:
`up = (uint32_t)L + (uint32_t)p; H += (up < L) + (p < 0 ? -1 : 0); L = up;`
(32-bit store of the low limb, 32-bit high-limb carry/borrow update — the whole
dot fits in 64 bits so the high limb is exact).

> **Bug caught during dev:** the first branchless version used
> `H += (up < L) - (up > L)`, which is WRONG when the low add carries for a
> *negative* `p` (it adds +1 instead of leaving H unchanged; a device dump showed
> target −467,491,856 vs buggy H=4/L=−467,491,856). Verified the corrected form
> against int64 on 2M random cases before shipping.

Device: attn_qk **142.5 → 129.6 µs per call** on the profile build (the
standalone QK microbench measured −28%). Causal triangular QK is now
memory/branch bound rather than int64-bound.

## Measurements (device, 160 MHz, 16 forwards)
```
KB0 (QKV head inner) n=384 avg_us=13432.8  <- unchanged (8-MAC asm, opt16)
KB1 (QKV head epilogue) n=384 avg_us=925.1  <- opt18 #1 (was 3950)
C5CYC (core5 f2/oproj)  n=64  avg_cyc=11585540, avg_us=72409.6  <- opt18 #2
attn_qk n=16384 avg_us=129.6; attn_exp avg_us=20.6; attn_pv avg_us=131.8
TOTAL ~ 2447382 us total_wall  (2.447 s/forward)
```
Per-seed forward (5 seeds, R command): 2443871/2443977/2443572/2443798/2443371 µs.

## Gate
- Host (`make -C tools test`): 50/50 PASS — FAST max_abs ≤ 9.8e-4, EXACT ≤ 6.8e-5.
- Device (5 seeds, torch gate atol=0.002 / rtol=0.02): 5/5 PASS, max_abs
  1.1065/1.0107/1.0414/1.2925/1.0145e-3.

## Cost profile after opt18 (per forward, from device TOTAL build)
```
norm1 33 ms · qkv(KB0+KB1) ~165 ms · attn_qk ~130 ms + exp 21 + attn_pv ~130 ms
oproj ~145 ms (2× C5CYC) · f1 ~54 ms · gelu ~31 ms · f2 ~176 ms (2× C5CYC)
final ~45 ms · norms+resid ~95 ms —— TOTAL 2.447 s
```
Dominant remaining costs: KB0 head-projection inner (~13.4 ms/call), the two
core5 GEMM families (f2/oproj), and the integer attention QK+PV passes.
