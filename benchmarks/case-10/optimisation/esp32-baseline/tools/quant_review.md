# quant-review — independent verification of the strided int16 attention design

Method: a from-scratch NumPy implementation of the ESP32 forward pass that
reproduces the intended fixed design — per-head Q15 int16 Q/K/V (per-head
amax scale = amax/32767, llrintf rounding, +/-32767 clamp), dequant-on-read
fp32 attention with the strided ctx layout (`ctx[i*128 + h*32 + j]`, no
aliasing), Q12 weight + Q15 activation GEMMs for the FAST path, matching the
C's GEMM/LayerNorm/GELU order.

Results (25 seeds, 409600 outputs each, gate max(0.002, 0.02*|ref|)):
  mode  max_abs   fails/409600   margin
  EXACT 7.49e-05  0              ~26x
  FAST  9.68e-04  0              ~2x

Verdict: the intended strided per-head int16 attention design PASSES the gate
in both modes. It also confirmed the pre-fix snapshot's corruption pattern
(heads 1-3 wrong only in rows 0..31; head 0 correct) and that the head-order
trick did not fix the real i*HD-vs-i*D stride mismatch.

Edge findings (non-blocking): amax==0 guard quantizes to all-zeros + scale 1,
which dequantizes as 0 (safe); llrintf vs llrint rounding matches kernels.c;
the large max_rel values are ref~=0 artifacts and are not failures.

run:  python3 tools/quant_review.py   (needs numpy; loads ./weights*.bin and testdata/)
