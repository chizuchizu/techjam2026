# Case 2 single-board optimisation — speaker notes

## Stage 1 — Integer attention: Opts 1–2

“The ESP32-C3 has no hardware floating-point unit, so the original attention code was extremely expensive. Attention alone took 30.09 seconds out of a 42.15-second forward. Opt 1 replaced the QK, softmax, and PV inner loops with fixed-point integer arithmetic. Opt 2 replaced the expensive exponential function with an integer lookup table. This reduced the complete forward to 13.70 seconds, giving a 3.08× speedup. The key message is that we first removed the biggest software-floating-point bottleneck.”

**Emphasise:** `42.15 s → 13.70 s` and `attention: 30.09 s → 1.64 s`.

---

## Stage 2 — Reduce memory traffic: Opts 3–8

“Once attention was faster, matrix multiplication and data movement became the main costs. Tiled GEMM calculated several outputs together, allowing weights to be reused instead of repeatedly read from flash. GELU was replaced with a scale-aware integer lookup table, and QKV quantisation was fused directly into the projection step. This removed unnecessary FP32 buffers, scans, and conversions. The forward time fell from 13.70 to 5.27 seconds, reaching an 8× speedup.”

**Emphasise:** tiled GEMM and fused QKV quantisation; they reduced both computation and memory traffic.

---

## Stage 3 — Fuse normalisation and quantisation: Opts 8b–14

“The next inefficiency was repeated movement of the same data. LayerNorm used to write FP32 output, scan it, and then quantise it in separate passes. We fused those operations so LayerNorm could calculate its scale and write Q15 output directly. Attention also began producing Q15 context directly for the output projection. Integer LayerNorm, integer PV, and integer quantisation removed more software-floating-point work. This reduced the forward from 5.27 to 3.205 seconds, or 13.1× faster than baseline.”

**Emphasise:** removing complete memory passes, not just speeding up individual instructions.

---

## Stage 4 — Tune the kernels to the processor: Opts 15–20

“At this stage, the algorithms were efficient, but some kernels were too large for the ESP32 register file. The resulting register spills caused extra stack-memory traffic. Opt 15 introduced a smaller register-fitting GEMM tile. Opt 16 added hand-scheduled RISC-V assembly, while Opt 17 emitted FFN1 directly as Q15. Opts 18 and 19 further improved bias handling, QK arithmetic, and loop unrolling. These changes reduced the time from about 3.2 to 2.386 seconds. Opt 20 confirmed that memory was already nearly full, so larger tiles or IRAM placement were not viable.”

**Emphasise:** a smaller tile can be faster when it avoids register spills.

---

## Stage 5 — Remove the final FP32 path: Opts 21–23

“The final major bottleneck was the residual connection between layers, which was still stored and processed as FP32. Opt 21 changed the FAST residual path to fixed-scale int32, allowing residual addition, GEMM epilogues, and normalisation to remain integer. Opt 22 combined this with the hand-written head GEMM assembly. Opt 23 corrected a second-column assembly bug and verified both columns against the C implementation. The final result was 1.996 seconds per forward: 21.1× faster than baseline, with all 25 device seeds passing the accuracy gate.”

**Emphasise:** `42.15 s → 1.996 s`, `21.1× faster`, and `25/25 device seeds passed`.

---

## Closing sentence

“We did not rely on one isolated trick. We removed the largest cost first, then reduced memory movement, fused conversions, tuned the kernels to the hardware, and finally converted the residual path. Together, those changes reduced the complete forward time by 95.3%.”
