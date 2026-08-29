# 20 — memory + compute profiling (opt20)

Device: Seeed XIAO ESP32-C3, 160 MHz RV32IMC, no FPU, 400 KB SRAM (313 KB usable
sequential DRAM), 4 MB flash. Build: opt19 (QK j-unroll-4 + PV 8-acc), 2.383–2.386 s/fwd.

## 1. Memory profile (static SRAM)
| buffer | type | size | role |
|---|---|---|---|
| g_x | float | 64 KB | float residual / layer input (norm source) |
| g_buf1 | float | 64 KB | norm/GEMM fp32 output staging; final-norm output |
| g_buf2 | float | 64 KB | per-head fp32 staging (re-used as g_ctxq q15 alias) |
| g_qh / g_kh / g_vh | int16 | 8 KB each (24 KB) | per-head Q15 Q/K/V projections |
| a16 | int16 | 32 KB | Q15 activation scratch (norm2/f1/f2 out, f1 in) |
| s1_/s2_ | int32/int64 | 512 B / 1 KB | integer LayerNorm stats |
| Fk_/Bk_ | int32 | 512 B | fused norm (gain/bias) fold |
| mx_ | int16 | 256 B | norm amax |
| g_exp_lut | int16 | 1 KB | attention exp LUT (513) |
| g_gelu_lut | int16 | 1 KB | integer GELU LUT |
| g_p15 / g_attn_score | int32/int64 | 512 B / 1 KB | attention PV p/row scores |
| + profiling/stat/stack/vec | — | ~180 KB | framework, heap, stack (libgcc, Arduino) |

- **DRAM segment used = 320,400 B of 320,784 B = 99.9% (only 384 B free).**
- Flash: text 221,926 B + embedded weights 2,430,940 B (weights.bin 1.59 MB +
  weights_q12.bin 0.79 MB) ≈ 2.65 MB image of the 4 MB flash.
- Consequence: no new static buffers are possible; IRAM placement of code is impossible
  (IRAM and DRAM alias the same 313 KB SRAM pool — verified opt19). Any further speed
  must come from re-timing existing memory streams, not new memory.

## 2. Memory-traffic profile (fast mode, per forward)
Flash weight reads (all XIP DROM through the 16 KB flash cache):
| kernel | calls/fwd | bytes/call | total |
|---|---|---|---|
| KB0 (QKV head q15) | 48 | 8 KB (HD×K=32×128×i16) | 384 KB |
| oproj core5 | 4 | 32 KB (D×D×i16) | 128 KB |
| f1 core5_q15 | 4 | 32 KB (F×D×i16) | 128 KB |
| f2 core5 | 4 | 32 KB (F×D×i16) | 128 KB |
| **total flash reads** | | | **768 KB/forward** |

SRAM traffic (activations): every layer reads+writes the 64 KB buffers several times
(a16 SQ15 write/read, g_x read/write, head buffers). Rough estimate 3–4 MB SRAM
traffic/forward; SRAM bandwidth is not the limit on this core (in-order issue is).

Key insight (from opt19 device microbench): the 512 B core5 j-tile working set stays in
the 16 KB flash cache, so flash streaming is NOT the core5 bottleneck (3.65 cyc/MAC
flash vs 3.32 SRAM). Instruction + soft-float latency dominate the non-GEMM parts.

## 3. Compute profile (opt19, per forward, ~2.386 s)
| region | ms/fwd | % | note |
|---|---|---|---|
| QKV (KB0 48×13.43 + KB1 48×0.93) | 690 | 28.9 | KB0 3.5 instr/MAC, 4.09 cyc/MAC |
| attention qk | 206 | 8.6 | 10→7.8 cyc/MAC after unroll; latency-bound |
| attention pv | 208 | 8.7 | 8-acc flat; mul-latency bound |
| attention exp | 34 | 1.4 | LUT |
| oproj core5 | 290 | 12.2 | ~5.8 cyc/MAC incl. float epilogue |
| f1 core5_q15 | 216 | 9.1 | integer epilogue |
| f2 core5 | 352 | 14.8 | incl. gelu + float epilogue |
| norms (norm1+norm2) | 136 | 5.7 | integer LN |
| gelu | 62 | 2.6 | LUT |
| res1+res2 | 86 | 3.6 | fp32 add (separate pass) |
| final norm | 90 | 3.8 | fp32 LN |
| **total** | **≈2386** | 100 | |

Biggest remaining lever: the **float epilogue of core5 (oproj + f2)** — 8 calls × 8 races
of 128×128 elements, each 2 soft-float libcalls (int32→float + float multiply) ≈ 140 ms/fwd
(see opt20-epi worker). Second: KB0 inner loop (~10%).
