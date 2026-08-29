# opt14: integer GEMM amax scan + Q15 quant (tm_gemm_amax / tm_gemm_quantA_into)

Date: 2026-08. Applies on top of opt13.

## What
Replaced the fp32 `amax` scan and the per-element `__mulsf3`+`__fixsfsi`
quantize loop in the FAST GEMM input path with the same exact integer
bit-trick:
- `tm_gemm_amax`: integer bit-lex float max (no soft float).
- `tm_gemm_quantA_into`: `tm_f2q15` (int64 mul + shift + half-round + clamp).
Duplicate fp32 originals removed — kernels.c now has exactly one copy of each
helper.

## Results
- host 50/50 PASS, worst ~1.05e-3
- device 5/5 PASS, **3.205 s/fwd** (from 3.388)
- gelu stage avg 67,634 → 21,902 µs/call; f2 165,867 → 120,137 µs/call
