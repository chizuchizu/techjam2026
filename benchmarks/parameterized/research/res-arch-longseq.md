# Memory-Efficient Attention for Long Sequences and Large Model Dims on the ESP32-C3 (400 KB SRAM)

**Research report** - produced with the Firecrawl skills (`firecrawl search` / `firecrawl scrape`) against live sources, 30 Aug 2026.
Covers: flash-attention derivatives, streaming attention, chunked/block-sparse attention, KV sharding, 16-bit fixed-point attention, PSRAM/SPI-flash streaming on MCUs, and concrete SRAM byte-counts for four shape families.

---

## 1. The hardware envelope (facts from the ESP32-C3 datasheet)

| Resource | Value on ESP32-C3 | Byte-noted |
|---|---|---|
| CPU | 32-bit RISC-V **RV32IMC**, single core, 4-stage in-order, **160 MHz** | no FPU (no F/D ext), no SIMD |
| On-chip SRAM | **400 KB** total, **16 KB normally reserved as flash cache** | ~**384 KB usable data RAM**, shared by code + data + stack |
| RTC SRAM | 8 KB (RTC FAST) | keeps data in deep sleep |
| ROM | 384 KB | boot code only |
| In-package flash | 4 MB (C3FH4/C3FN4) or 8 MB (C3FH8X) | weights live here |
| External memory | SPI / Dual / Quad / QPI **flash only**, mapped **read-only** through the 16 KB cache, max **16 MB**, 64 KB mapping blocks | **no PSRAM interface on the C3** |
| Flash speed | default 80 MHz (up to 120 MHz on request); quad I/O | ~40 MB/s sequential at 80 MHz; 32 B cache lines |

Source: ESP32-C3 Series Datasheet v2.4, documentation.espressif.com/esp32-c3_datasheet_en.html.

**Implications that decide every conclusion below.**
1. No FPU -> floating-point attention is software-emulated; an attention layer must be **fixed-point/integer** to be remotely practical.
2. No PSRAM on the C3 -> activations **cannot** be paged to external RAM; only **read-only weights** can be streamed from mapped flash.
3. The 400 KB budget is shared with app/heap/stack; reserve ~32-64 KB for runtime/stack when budgeting.
---

## 2. The attention memory model

Define, per transformer layer:

- `S` = sequence length, `D` = model (embedding) dimension, `H` = number of heads, `d = D/H` = head dim.
- `b` = bytes per element (`fp32 = 4`, `fp16/int16 = 2`, `int8 = 1`).

Naive (dense, materialized) attention memory per layer:

| Tensor | Elements | Bytes |
|---|---|---|
| Input activations `X` | `S*D` | `S*D*b` |
| `Q, K, V` (all heads) | `3*S*D` | `3*S*D*b` |
| Attention scores (all heads) | `H*S*S` | `H*S^2*b` |
| Probabilities `P` (all heads, optional) | `H*S*S` | `H*S^2*b` |
| Output `O` | `S*D` | `S*D*b` |
| **KV cache** (decode, per token, per layer) | accumulated `2*S*D` | `2*D*b` per new token (MHA); `2*G*d*b` with GQA (`G` = KV groups) |
| **Weights** per layer (attn `4*D^2` + FFN 4x `8*D^2`) | `12*D^2` | `12*D^2*b` |

The dominating term is `H*S^2*b` (quadratic in S) for the score/probability matrices, and `3*S*D*b` for QKV (linear in both S and D, but large when D is large).

Standard attention must write `S` and `P` (`S x S` each) to slow memory. The FlashAttention paper quantifies HBM traffic as `Omega(N*d + N^2)` for standard attention vs `O(N^2*d^2*M^-1)` for FlashAttention with on-chip SRAM budget `M` (arXiv:2205.14135). The platform-independent takeaway: **never materialize the S x S matrix.**

---

## 3. Technique survey (with sources)

### 3.1 Flash attention and derivatives

**FlashAttention** (Dao, Fu, Ermon 2022, arXiv:2205.14135) computes *exact* attention with tiling + online softmax rescaling + recomputation, avoiding storage/re-read of the `S x S` score matrix. Standard attention materializes `S` and `P` -> `O(N^2)` memory; FlashAttention keeps the working set in on-chip SRAM -> linear-in-S working memory. IO result: `O(N^2 d^2 M^-1)` HBM accesses vs `Omega(N d + N^2)` for standard attention.

**FlashAttention-2** (Dao 2023, arXiv:2307.08691; HazyResearch blog, hazyresearch.stanford.edu) keeps the same memory asymptotics, adds better parallelism/work partitioning and fewer non-matmul FLOPs (online-softmax payoff), ~2x over FA1, head dims up to 256.

**On the MCU.** A direct CUDA port is wrong for a cache-less MCU. The Tiny-Transformer-on-MCU paper (ETH Zurich/Bologna, arXiv:2404.02945): *"some of the optimizations FlashAttention uses are not beneficial for extreme edge platforms ... we favor output-stationary tiling instead of the block tiling proposed in [FlashAttention]."* Their **Depth-First Tiling (DFT)** tiles the two GEMMs and softmax together and never materializes the full attention map; it cuts attention-memory peak by up to **6.19x**. Per-head SRAM formula (8-bit I/O, P = head dim, S = sequence, x = Q-tile rows; `2*P*S` = one head of K and V):

```
Mem_DFT(x) = (2*P + S)*x + 2*P*S     bytes
```

This is the correct building block for the C3 (numerical examples in Section 4).

### 3.2 Streaming attention (window + attention sinks)

StreamingLLM (Xiao et al. 2023, arXiv:2309.17453) shows a pure sliding-window KV cache **collapses** once the first token is evicted (window-only PPL jumps >5000 vs about 5), because autoregressive models deposit excess softmax mass in the **first tokens ("attention sinks")**. Fix: keep **4 sink tokens** + the recent **W** tokens -> bounded O(W) KV cache, stable over **4 million tokens**, up to **22.2x** faster than sliding-window-with-recompute. Cache size is fixed (they use 1 K-2 K), so long sequences never grow SRAM.

**On the C3:** this is the realistic way to process a *long* token stream with a transformer head: memory is bounded by window `W`, not by total `S`. It is not dense attention over all of `S`.

### 3.3 Chunked / block-sparse attention

BigBird (Zaheer et al. 2020; HF explainer, huggingface.co/blog/big-bird) replaces the dense `S x S` pattern with block-granular components per query: **sliding window** (`3*block_size`), **global** (`2*block_size`), **random** (`num_random_blocks*block_size`). Memory per row drops from O(S) to O(w + g + r), so the score matrix is a thin band plus a few dense rows/columns. Longformer = sliding + global; Sparse Transformer = strided/factorized blocks. Block granularity is what makes it cache-friendly on an MCU: pick block size so `block_rows * (w+g+r) * b` fits available SRAM.

**On the C3:** block-sparse (window + a few global tokens) is the right approximation as `S` grows; on a scalar RV32 core without SIMD, sparse patterns only pay if they reduce actual FLOPs/memory traffic, not FLOP counts alone (the FlashAttention paper makes this exact point about approximate methods that ignore IO).

### 3.4 KV sharding / KV-cache compression

Per-token KV cache (per layer) in bytes = `2 * n_kv_heads * d_head * b`.

- **MHA**: `n_kv_heads = H` -> `2*D*b`.
- **GQA** (grouped-query attention): `H` query heads share `G` KV heads -> shrink by `H/G`. Llama-class uses G=8 for H=64 (~8x smaller; Ainslie et al. 2023; zeroentropy.dev/concepts/grouped-query-attention).
- **MQA** (multi-query attention): `G = 1` -> `2*d_head*b` (cheapest, some quality loss).

Stacks with KV **quantization** (int8 KV = half of fp16; 2-bit/4-bit group quantization goes further) and with GQA.

**On the C3:** a small model has H=4-8, so the KV cache is tiny (KBs; see tables). GQA/MQA matter here more as **bandwidth/data-reuse wins** (all query heads read one shared K/V) than as RAM savings. The MCU paper notes GQA/MQA kernels are trivial to derive but have *"yet to be successfully applied in the TinyML domain"* (arXiv:2404.02945).

### 3.5 16-bit (and 8-bit) fixed-point attention

The C3's RV32 **M** extension gives `MUL`/`MULH`: a `Q15 x Q15 -> Q30` product (via `MULH`) plus `int32` accumulation is the natural fixed-point attention datatype.

Recipe that fits the C3:

1. `Q,K,V` in `int8` (or `int16`) with power-of-two scale (hardware-friendly shifts, no division; the MCU paper uses exactly this in QuantLib, arXiv:2404.02945).
2. Scores `S = Q*K^T` accumulated in `int32`, scaled by `1/sqrt(d)` with a shift.
3. Softmax **in the integer domain**: IntAttention (arXiv:2511.21513) replaces the float `exp` with a **32-entry lookup table + clipping + direct integer normalization** ("IndexSoftmax"). Breaking the int dataflow with `dequant -> fp32 softmax -> requant` can cost **up to 65% of total attention latency** on edge CPUs, so LUT-integer softmax is the right choice on a no-FPU core.
4. `P*V` accumulated `int32 -> int8` with a shift.

**Overflow guard:** `int16 x int16` products are up to 2^30 each; summing over `d` head dims can overflow `int32` unless products are right-shifted during accumulation. Using int8 Q/K sidesteps it: `int8 x int8` products are <= 2^14 and a sum over d <= 128 is about 2^21, comfortably inside `int32`. This is a key reason TinyML transformers (arXiv:2404.02945, MCUFormer arXiv:2310.16898) run **int8** activations with **int16/int32** accumulators.

### 3.6 PSRAM / SPI-flash streaming on MCUs (and the C3 specifically)

- **C3 has no PSRAM.** The datasheet's only external memory is flash via SPI/Dual/Quad/QPI, mapped **read-only** through a 16 KB cache (64 KB blocks). Writing flash needs sector erase + page program (slow) and wears it - **activations cannot be swapped to flash**.
- **What flash streaming can do:** stream **weights** (read-only) layer-by-layer into SRAM as they are consumed ("weight-stationary"), the standard CMSIS-NN/MCU pattern. On the C3 this is mandatory: all activations stay on-chip.
- **What it cannot do:** hold `Q/K/V`, attention maps, KV caches, or gradients in external memory. ESP32 / ESP32-S3 (which *do* have quad-PSRAM, typically 2-8 MB at ~40-80 MB/s) can offload weights and some activations, but the C3 is not in that class.
- Proof point: MambaLite-Micro (arXiv:2509.05488) cuts peak RAM **83%** (1,352 KB -> 230 KB on an ESP32-S3) purely with operator fusion + lifetime-aware buffers. Fusion, not external RAM, is what makes MCU sequence models fit.
---

## 4. Concrete SRAM footprint - the four requested shapes

Convention: `D` = model/embedding dim, `d = D/H`, FFN 4x (so weights/layer = `12*D^2`). `b` in {4 (fp32), 2 (fp16/int16), 1 (int8)}. Values are **per transformer layer**. "Score (materialized)" is the `H*S*S` block; FlashAttention-style tiling removes it and leaves only a small tile.

### (a) `S = 128, D = 128, H = 4`  ->  `d = 32`

| Item | fp32 | fp16/int16 | int8 |
|---|---|---|---|
| Score map, all heads `H*S^2` | 256.0 KiB | 128.0 KiB | 64.0 KiB |
| QKV all heads `3*S*D` | 192.0 KiB | 96.0 KiB | 48.0 KiB |
| QKV one head `3*S*d` | 48.0 KiB | 24.0 KiB | 12.0 KiB |
| Input `X` `S*D` | 64.0 KiB | 32.0 KiB | 16.0 KiB |
| Output `O` `S*D` | 64.0 KiB | 32.0 KiB | 16.0 KiB |
| KV cache @ S (whole sequence) | 128.0 KiB | 64.0 KiB | 32.0 KiB |
| **Weights/layer** `12*D^2` | 768.0 KiB | 384.0 KiB | 192.0 KiB |

**Fits? Yes.** In int8, a head-serial, DFT-tiled pipeline peaks at about `X(16 KiB) + QKV(48 KiB) + score-tile(~8-16 KiB) + O(16 KiB) + runtime(32-64 KiB)` ~ **130-160 KiB**, well inside 384 KiB. Weights at 192 KiB/layer (int8) let a 4 MB flash hold ~20 layers. Even naive fp32 per-head processing peaks ~336 KiB (feasible, but fp32 is software-emulated and slow). Use **int8 + DFT (Section 3.1)**.

Explicit DFT numbers for (a), P=d=32, S=128, int8 (1 B/elem): `Mem_DFT(x) = (64 + 128)*x + 2*32*128 = 192*x + 8192 B`. x=2 rows -> 8.3 KiB; x=32 -> 14.0 KiB. The full one-head K+V is only 8 KiB (int8), so per-head attention is tiny.

### (b) `S = 1024, D = 128`  (case 13)  ->  assume `H = 4, d = 32`

| Item | fp32 | fp16/int16 | int8 |
|---|---|---|---|
| Score map, all heads `H*S^2` | 16.0 MiB | 8.0 MiB | 4.0 MiB |
| QKV all heads `3*S*D` | 1.5 MiB | 768.0 KiB | 384.0 KiB |
| QKV one head `3*S*d` | 384.0 KiB | 192.0 KiB | 96.0 KiB |
| Input `X` `S*D` | 512.0 KiB | 256.0 KiB | 128.0 KiB |
| Output `O` `S*D` | 512.0 KiB | 256.0 KiB | 128.0 KiB |
| KV cache @ S | 1.0 MiB | 512.0 KiB | 256.0 KiB |
| Weights/layer | 768.0 KiB | 384.0 KiB | 192.0 KiB |

**Verdict (case 13, S = 1024).** Dense materialized attention is impossible: the `S x S` map alone is 4-16 MiB. The only path is **exact tiled attention (DFT) + int8 + head-serial + token overwriting**:

- int8 per-head K,V = `2*S*d` = 64 KiB, Q = 32 KiB, score tile (e.g. 16 x 1024) ~ 16 KiB, and X/O can share one 128 KiB buffer -> ~**240-300 KiB** including runtime. That is at the absolute edge of 384 KiB for **one attention operator** (no FFN, no layer norm, no code margin).
- fp16/fp32 variants blow the budget immediately (QKV all-heads 768 KiB-1.5 MiB).
- Compute is also heavy: QK^T + P V = `2*S^2*d` = 2 x 1024^2 x 32 = **67.1M MAC/head**, 268M MAC/layer for H=4. On a 160 MHz scalar RV32IMC core with realistic scalar-MAC throughput (tens of millions MAC/s), one layer is roughly **seconds**; a multi-layer model is far from real-time, though it "runs" as a benchmark.

So: **S=1024, D=128 "could ever run" = technically yes as a single int8, weight-streamed, DFT-tiled attention block; no as a complete transformer**, and in any case only via tiling, never by materializing scores. For real streaming workloads use the StreamingLLM-style bounded window (Section 3.2) instead of full-S attention.

### (c) `S = 128, D = 1024`  (case 8)  ->  assume `H = 8, d = 128`

| Item | fp32 | fp16/int16 | int8 |
|---|---|---|---|
| Score map, all heads `H*S^2` | 512.0 KiB | 256.0 KiB | 128.0 KiB |
| QKV all heads `3*S*D` | 1.5 MiB | 768.0 KiB | 384.0 KiB |
| QKV one head `3*S*d` | 192.0 KiB | 96.0 KiB | 48.0 KiB |
| Input `X` `S*D` | 512.0 KiB | 256.0 KiB | 128.0 KiB |
| Output `O` `S*D` | 512.0 KiB | 256.0 KiB | 128.0 KiB |
| KV cache @ S | 1.0 MiB | 512.0 KiB | 256.0 KiB |
| **Weights/layer** `12*D^2` | 48.0 MiB | 24.0 MiB | 12.0 MiB |

**Verdict (case 8, D = 1024).** Fails on two independent counts:

1. **Activations:** even int8, all-head QKV = 384 KiB by itself; adding X (128 KiB) and any score tile exceeds 384 KiB. Head-serial helps SRAM (one head's QKV = 48 KiB) but X/O still need 128 KiB each, and you still cannot hold intra-layer working buffers comfortably.
2. **Weights do not fit flash:** 12 MiB/layer int8 (48 MiB fp32) against a 4-8 MB flash. One layer is already larger than the whole chip.

**D=1024 "could ever run" on an ESP32-C3: No.** Not as a dense attention transformer. The only way a "1024-dim" model touches this chip is with heavily factorized / low-rank / Mixture-of-Experts-style weight sparsity AND head-dim chunking (never holding a full D-width activation), which is a different model, not a D=1024 transformer. (Contrast: real MCU transformers in the literature keep D host + token-side overheads manageable, e.g. MCUFormer fits a ViT for ImageNet in 320 KB SRAM with int8, arXiv:2310.16898.)

### (d) `S = 100000, D = 1024`  (extreme)

Assume H=8, d=128 for like-with-like.

| Item | fp32 | fp16/int16 | int8 |
|---|---|---|---|
| Score map, all heads `H*S^2` | 305,176 MiB (320 GB) | 152,588 MiB | 76,294 MiB (80 GB) |
| QKV all heads `3*S*D` | 1,171.9 MiB | 585.9 MiB | 292.9 MiB |
| Input `X` `S*D` | 390.6 MiB | 195.3 MiB | 97.7 MiB |
| KV cache @ S | 781.3 MiB | 390.6 MiB | 195.3 MiB |
| Weights/layer | 48.0 MiB | 24.0 MiB | 12.0 MiB |

**Verdict.** Impossible as dense attention even on a server GPU's *SRAM* for the `H*S^2` block (320 GB fp32). On a 400 KB MCU it is off by six orders of magnitude. Three options if S=100000 must be processed on-device:

1. **Bounded-window attention** (StreamingLLM): only a fixed window + sinks resident; total context length is unbounded but memory is O(W), not O(S). Standard for streaming.
2. **Block-sparse big-context models** (BigBird/Longformer patterns): O(S * (w+g+r)) memory - still only viable here with very small w+g+r and tiny D.
3. **Linear-time / SSM sequence models (recommended):** Mamba-style selective SSMs carry a constant-size state per step (O(D*N_state), independent of S). MambaLite-Micro (arXiv:2509.05488) runs a fused Mamba for KWS in ~**230 KB peak RAM on an ESP32-S3**, vs 1,352 KB unfused (83% reduction). This is the realistic architecture for very long sequences on the C3 class - not a softmax transformer. It is still close to the C3's 384 KiB ceiling, and at 100k steps the *latency* (sequential recurrence, no FPU) is the real bottleneck.

---

## 5. Bottom line

| Case | Verdict on ESP32-C3 (384 KB usable SRAM, 4-8 MB flash, no FPU, no PSRAM) |
|---|---|
| **(a) S=128, D=128, H=4** | **Feasible.** int8 weights + int8 activations, head-serial DFT-tiled attention, weights streamed from flash. Peak ~130-160 KiB; ~20 layers at 192 KiB/layer in 4 MB flash. This is the sweet spot. |
| **(b) S=1024, D=128 (case 13)** | **Edge / demo-only.** Only as a single int8 DFT-tiled attention block at ~240-300 KiB; a full transformer does not fit, and fp32/fp16 do not fit at all. Use StreamingLLM windowing for long streams. |
| **(c) S=128, D=1024 (case 8)** | **No.** int8 all-head QKV = 384 KiB alone (no X, no score tile), and weights = 12 MiB/layer int8 > whole flash. |
| **(d) S=100000, D=1024** | **No (attention impossible).** `H*S^2` = 80 GB even int8; KV = 195 MB int8. Use an SSM/linear architecture (MambaLite-style, ~230 KB) or a bounded window, not attention. |

### What S=128 / D=128 needs, concretely (a working recipe for the C3)

1. **Quantization:** int8 (or int16) weights + int8 activations, int16/int32 accumulators, power-of-two scales (shifts, no division). No FPU -> no fp32 softmax anywhere.
2. **Integer softmax:** LUT-based exp (IntAttention-style "IndexSoftmax", 32-entry) with clipping + integer normalization.
3. **Tiling:** DFT / output-stationary tiling so the `S x S` matrix is never materialized (only a tile), per §3.1.
4. **Weight-stationary execution:** stream the 192 KiB/layer int8 weights from mapped flash; keep only activations in SRAM.
5. **Budget heads:** Transformer heads: compute one at a time and accumulate (head-serial), keeping only one head's Q/K/V + a score tile live.
6. **Memory accounting (int8):** X 16 KiB (shared with O), QKV 48 KiB, score tile ~8-16 KiB, O 16 KiB + ~32-64 KiB runtime/stack -> well under 384 KiB.
7. **For longer runs:** StreamingLLM (4 sinks + window W) or block-sparse window+global patterns; for very long S use an SSM (Mamba) instead of attention.

---

## 6. Sources (live-scraped)

1. FlashAttention - arXiv:2205.14135 (arxiv.org/html/2205.14135v2).
2. FlashAttention-2 blog - HazyResearch, Dao 2023 (hazyresearch.stanford.edu/blog/2023-07-17-flash2).
3. StreamingLLM - arXiv:2309.17453 (arxiv.org/html/2309.17453v3).
4. BigBird explainer - Hugging Face (huggingface.co/blog/big-bird).
5. Grouped-Query Attention / KV cache - ZeroEntropy (zeroentropy.dev/concepts/grouped-query-attention/).
6. Tiny Transformers on MCUs - arXiv:2404.02945 (arxiv.org/html/2404.02945v1).
7. MCUFormer - arXiv:2310.16898 (arxiv.org/html/2310.16898v3).
8. IntAttention (fully-integer attention) - arXiv:2511.21513 (arxiv.org/pdf/2511.21513).
9. MambaLite-Micro - arXiv:2509.05488 (arxiv.org/html/2509.05488v1).
10. ESP32-C3 Series Datasheet v2.4 - Espressif (espressif.com/documentation/esp32-c3_datasheet_en.pdf).
