# Parameterized Transformer (single C function, any case geometry)

One C99 function runs a Tech Jam case from a runtime `tm_case` struct
(any geometry whose weights + workspace fit in flash/SRAM):

```c
tm_case cfg = { .B=1, .S=128, .D=128, .H=4, .F=128, .L=4,
                .causal=1, .mode=TM_FAST };
size_t n = tm_workspace_size(&cfg);
void* ws = malloc(n);
int nm   = tm_qmat_count(&cfg);
const int16_t** q = malloc(nm * sizeof(*q));
float* wscale     = malloc(nm * sizeof(float));
tm_scan_q12(&cfg, q12_blob, q, wscale);
tm_run(&cfg, ws, weights_f32, q, wscale, x, y);   /* x,y: S*D floats */
```

No geometry is a `#define`, no malloc lives inside `tm_run`, and the caller's
workspace is sized once (`malloc`-once, run many times), so it is MCU-safe.
`transformer.h` documents the full API.

## Files

- `transformer.h` / `transformer.c` — the parameterized forward (EXACT + FAST).
- `test_host.c` — host CLI: run one forward, check the official gate, profile.
- `esp32/` — PlatformIO project that flashes the same `transformer.c` onto a
  Seeed XIAO ESP32-C3 and self-tests + profiles it on the real no-FPU core.
- `tools/gen_case.py` — regenerates `weights.bin`, `weights_q12.bin`, and a
  torch reference frame for any `S D H F L` (uses the vendored torch ref).

## Numeric modes

| Mode | GEMM | Attention | Softmax | GELU | Typical worst `abs_err` (host) |
|---|---|---|---|---|---|
| `TM_EXACT` | fp32 | int64 QK + int PV | fp32 | A&S erf (fast exp) | ~1.7-2.7e-4 |
| `TM_FAST` | Q15 act x Q12 wgt, int64+int32 block acc | int64 QK + int PV | fast exp (Q15) | A&S erf (fast exp) | ~7.0-8.8e-4 |

The FAST GEMM accumulator is the CMSIS-NN split pattern: 8-term int32 inner
blocks folded into an int64 total, so a full K=128 dot can never overflow even
if every activation saturates.

Both modes compile together; select per call with `cfg.mode`. `TM_FAST` is the
actionable path for the C3 (no FPU): only the attention softmax and epilogue
run in fp32, everything else is integer (the softmax is computed in fp32 with
fast pow2-exp, then emitted as Q15).

## Host validation (all 6 implemented geometries, single `test_host` binary)

Generated weights + torch reference for each case with
`tools/gen_case.py` (vendored torch ref), then ran one forward per case in both modes.
Official gate is `abs(u-r) <= 0.002 OR abs(u-r) <= 0.02*abs(r)`.

| Case | `(S,D,H,F,L)` | workspace | EXACT fails | FAST fails |
|---|---|---:|---:|---:|
| 07 | `(128,32,4,32,4)` | 66,000 B | 0/4096 | 0/4096 |
| 02 | `(128,128,4,128,4)` | 257,232 B | 0/16384 | 0/16384 |
| 09 | `(128,128,1,128,4)` | 330,960 B | 0/16384 | 0/16384 |
| 10 | `(128,128,2,128,4)` | 281,808 B | 0/16384 | 0/16384 |
| 11 | `(128,128,16,128,4)` | 238,800 B | 0/16384 | 0/16384 |
| 12 | `(32,128,4,128,4)` | 65,616 B | 0/4096 | 0/4096 |

A full pass = 0 failing elements across all 6 geometries and both modes.

## Host profile (case 02, `cc -O2`, Apple Silicon) — one forward

| slice | EXACT | FAST |
|---|---:|---:|
| norm1 | 68 us | 108 us |
| qkv | 10,353 us | 3,419 us |
| attention | 2,420 us | 2,322 us |
| out proj | 2,552 us | 353 us |
| ffn1 | 2,569 us | 343 us |
| gelu | 169 us | 161 us |
| ffn2 | 2,549 us | 332 us |
| **TOTAL** | **18,357 us** | **4,901 us** |

`TM_FAST` is ~3.7x faster on the host; the gap is far larger on the C3 where
fp32 emulation is soft-float.

## Device profile (case 02, Seeed XIAO ESP32-C3 @ 160 MHz, FAST) — one forward

Measured with `esp_timer_get_time()`; self-test = embedded `input_0` vs
`ref_0`, official gate, 0/16384 fails (`MAX_ABS 7.68e-4`).

| slice | us | share |
|---|---:|---:|
| norm1 | 658,442 | 5.1% |
| qkv (3 heads GEMM) | 6,019,483 | 47.0% |
| attention | 3,149,646 | 24.6% |
| out proj | 1,242,863 | 9.7% |
| gelu | 1,676,158 | 13.1% |
| ffn1 + ffn2 | 2,171,059 | 16.9% |
| **TOTAL** | **12,809,659 us (12.81 s)** | 100% |

> The generic single-binary kernels run ~12.8 s/forward on the C3. The tuned
> per-case firmware (fused GEMM, integer attention softmax, cached weight
> blocks) runs the same case in ~2.4 s. 64 such generic forwards would need
> ~819 s, past the 600 s full-batch cutoff, so the repository's benchmark
> submissions still use the per-case optimized firmware; this module is the
> portable correctness/profile reference and the "parameters in, response out"
> API.

## What fits where (workspace formula)

`ws = 3*S*D*4 (x,buf1,buf2) + 2*S*max(D,F) (a16) + max(D,F)*8 (acc)
     + 3*S*(D/H)*2 (qh,kh,vh) + small`

- `S<=128, D=128, H>=2` → <= ~281 KB, fits the C3's 400 KB SRAM.
- `H=1` (head_dim = D) → the 3 head stages cost `3*S*D*2`, pushing case 09 to
  ~330 KB: too big for a C3 without the fused-staging tricks of the tuned
  per-case firmware; fits an ESP32-S3 with PSRAM.
- Case 08 (`D=1024`) needs ~100 KB* but weights are 100 MB fp32 / 50 MB Q12 —
  far past the 4 MB flash. Impossible on a C3.
- Case 14 (`S=100000, D=1024`) is out of reach for any MCU (attention
  `H*S^2` alone is ~80 GB in int8).

## Regenerate + retest

```sh
# one-shot host test for any geometry
cc -O2 -std=c99 test_host.c transformer.c -o test_host -lm
./test_host S D H F L MODE weights.bin q12.bin input.bin ref.bin
```

## Online research (firecrawl subagents)

Four parallel subagents researched the design space; reports live in
`benchmarks/parameterized/research/` (and sources are linked inside each):
dynamic-shape MCU runtimes, no-FPU quantized transformer math, long-sequence /
large-dim attention bounds on the C3, and host+device profiling methodology.

## Key research takeaways wired into this code

- **Data-driven runtime** (TFLM tensor arena, llama2.c config struct): geometry
  lives in a runtime struct, kernels loop over `S/D/H/F/L`, one arena allocator.
- **Q15/Q12 integer GEMM with int32 accumulation** is the established no-FPU
  pattern; int64 is used only for attention QK/PV where the sums need it.
- **`esp_timer_get_time()`** (µs) for per-kernel timing; `esp_cpu_get_cycle_count()`
  (`rdcycle`) for finer microbenchmarks on RISC-V.
- **Memory verdict** (matches this repo's earlier per-case work): S=128/D=128
  is the practical envelope on the C3; S=1024 or D=1024 exceed flash/SRAM, and
  S=100000 is infeasible.
