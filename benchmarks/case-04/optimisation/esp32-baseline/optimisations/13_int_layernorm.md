# opt13: integer-only LayerNorm pass (tm_bn_q15_int)

Date: 2026-08. Applies on top of opt9/9b's fused LN→Q15 (still 3.21 → 3.205
after opt14; standalone device ~3.388 → 3.21 s after both).

## What
Replaced the fp32 soft-float work in the LayerNorm pass with exact integer:
- amax scan: fp32 `fabsf` compare → **integer bit-lex max** of |bits| (one AND
  + one int compare per element, no soft-float at all).
- per-element `__mulsf3`+`__addsf3`+`__fixsfsi` → one exact 48-bit fixed-point
  conversion: `sc = Q/amax` split into 24-bit mantissa `sc_m` × 2^sc_e, then
  `q = round((x_mant·sc_m) << sh)` with a single int64 mul + shift + half-round
  + clamp.  Helpers `tm_f2q15(uint32_t b, int32_t sc_m, int sc_e, int32_t Q)`
  and `tm_split_scale(...)` are shared with opt14.

## Results
- host 50/50 gate PASS, worst 1.2698e-3
- device: 5/5 PASS, worst 1.088e-3
- norm1/norm2 slots dropped from ~16,950 µs to ~16,950 µs/call? (see opt14 for
  the combined final numbers)
