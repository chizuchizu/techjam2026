# Existing Model Architecture — H200 Transformer Kernel Optimization

**Working directory:** `/Users/karthikgangula/Downloads/techjam2026`
**Primary source:** `torch_transformer_benchmark.py` (2,047 lines; static analysis, no torch import used in this research)
**Secondary sources:** `triton_kernels.py` (613 lines), `README.md`, `OPTIMIZATION_REPORT.md`, `TODO.md`, `STATEMENT.md`

All line references below are to `torch_transformer_benchmark.py` unless another file is named.
This is a code-architecture research document: it describes the *existing* baseline and the opt-in optimized path; it does not run the benchmark.

---

## 1. Model hyperparameters

### 1.1 TransformerConfig (dataclass) — L30-57

| Field | Type | Meaning |
|---|---|---|
| `batch_size` | int | batch dimension B |
| `seq_len` | int | sequence length S |
| `d_model` | int | hidden width D |
| `num_heads` | int | number of attention heads H |
| `ffn_dim` | int | FFN hidden width F |
| `num_layers` | int | number of transformer blocks L |
| `causal` | bool | enable causal self-attention |

`validate()` (L40-57) requires B, S, D, F, L > 0, num_heads > 0, and `d_model % num_heads == 0`.
The dataclass stores **no seed, no dtype, no head-dimension field**; `head_dim` is derived as `d_model // num_heads`.

### 1.2 Defaults (argparse, `parse_args()` L1521-1668)

| Flag | Default | Source |
|---|---|---|
| `--batch-size` | **8** | L1525 |
| `--seq-len` | **128** | L1526 |
| `--d-model` | **512** | L1527 |
| `--heads` | **8** | L1528 |
| `--ffn-dim` | **2048** | L1529 |
| `--layers` | **6** | L1530 |
| `--causal` | **off** (`store_true`) | L1531 |
| `--device` | `auto` (cuda if available else cpu) | L1533-1534; `resolve_device` L1056-1062 |
| `--dtype` | `float32` (`float32`/`float16`/`bfloat16`) | L1536-1540; `resolve_dtype` L1065-1071 |
| `--padding-ratio` | `0.0` (range [0,1); 0 => no padding mask) | L1541, L1672-1673 |
| `--input-scale` | `1.0` | L1542 |
| `--accuracy-trials` | **5** | L1544 |
| `--rtol` | **0.02** | L1545 |
| `--atol` | **0.002** | L1546 |
| `--seed` | **1234** | L1547 |
| `--warmup` | 20 | L1549 |
| `--repeats` | 100 | L1550 |
| `--benchmark-rounds` | 3 | L1551 |
| `--user-implementation` | `baseline` | L1560-1565 (see section 5) |

**Supported dtypes:** `float32`, `float16`, `bfloat16` only (L1536-1540, mapping L1065-1071). No FP8 path in the PyTorch script.

**Effective default model:** `B=8, S=128, D=512, H=8, head_dim=64, F=2048, L=6`, non-causal, no padding, FP32.
`OPTIMIZATION_REPORT.md` calls this the default benchmark shape and notes it is currently the *only* authoritative local target: the official test-shape appendix in `STATEMENT.md` failed to export ("This content is only supported in a Feishu Docs", STATEMENT section 3.7).

---

## 2. Exact layer structure and forward math of the baseline

### 2.1 Block-level structure: Pre-LN

`BaselineTransformerBlock` (L126-146):

```python
self.norm1     = nn.LayerNorm(d_model)            # L129
self.attention = BaselineSelfAttention(...)       # L130
self.norm2     = nn.LayerNorm(d_model)            # L131
self.ffn_in    = nn.Linear(d_model, ffn_dim)      # L132   (bias=True by default)
self.ffn_out   = nn.Linear(ffn_dim, d_model)      # L133   (bias=True by default)
```

`forward` (L135-146):
```python
x = x + self.attention(self.norm1(x), valid_token_mask, causal)     # L141
x = x + self.ffn_out(F.gelu(self.ffn_in(self.norm2(x)), approximate="none"))  # L142
```

Per-block order:
1. norm1 (LayerNorm, eps=1e-5, affine) -> attention
2. residual add of attention output
3. norm2 -> FFN
4. residual add of FFN output

This is **Pre-LN**: norms before each sublayer, residual branch around each sublayer.
No dropout, no per-block post-norm; the stack-level `final_norm` is the last operation.

### 2.2 Attention (`BaselineSelfAttention`, L60-123)

Modules (`__init__`, L63-76):
- `head_dim = d_model // num_heads` (L70) — **= 64** for the default shape.
- `scale = head_dim ** -0.5` (L71) — **= 1/8 = 0.125**.
- **Separate projections** q_proj, k_proj, v_proj, out_proj, each `nn.Linear(d_model, d_model, bias=True)` (L73-76). **All four carry bias.** Full MHA; no weight tying, no GQA/MQA.

Forward (L86-123), given `x: [B,S,D]`:
1. `q/k/v = _split_heads(q_proj(x) / k_proj(x) / v_proj(x))` (L94-96). `_split_heads` (L78-84) reshapes `[B,S,D] -> [B,S,H,head_dim]`, transposes to `[B,H,S,head_dim]`, `.contiguous()`.
2. Scores: `scores = torch.matmul(q, k.transpose(-2,-1)) * self.scale` (L98) — shape `[B,H,S,S]`.
3. **Causal mask** (only if causal=True): fresh bool matrix `torch.ones((S,S)).triu(diagonal=1)` built per call (L100-104); upper triangle (key j > query i) filled with `float("-inf")`.
4. **Padding/valid-key mask** (only if valid_token_mask is not None): `invalid_keys = ~valid_token_mask[:, None, None, :]` (`[B,1,1,S]`, L106-109) fills invalid key positions with -inf.
5. **Softmax with FP32 casting:** `probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)` (L112). Raw scores are cast to **FP32** for softmax, then cast back to the model dtype **before** the value matmul.
6. `context = torch.matmul(probs, v)` (L113) — `[B,H,S,head_dim]`.
7. Merge heads: transpose -> `[B,S,H,head_dim]` -> contiguous -> view `[B,S,D]` (L114-118).
8. `output = self.out_proj(context)` (L119) — bias present.
9. If padding mask: invalid query rows zeroed `output.masked_fill(~valid_token_mask[..., None], 0)` (L121-122).

No RoPE, no positional bias, no sliding window, no dropout, no scaling beyond `* scale`.

### 2.3 FFN

Inside the block (L132-133, L142):
- `ffn_in`: `Linear(512 -> 2048)`, **bias present** (`nn.Linear` default bias=True).
- Activation: `F.gelu(..., approximate="none")` = **exact erf form** `GELU(x) = 0.5*x*(1+erf(x/sqrt(2)))`. Not tanh (tanh only opt-in via `--gelu-approx-layer-indices`).
- `ffn_out`: `Linear(2048 -> 512)`, **bias present**.

### 2.4 Stack level (`BaselineTransformer`, L149-173)

```python
self.layers     = nn.ModuleList(BaselineTransformerBlock(...) for _ in range(num_layers))  # L153-160
self.final_norm = nn.LayerNorm(config.d_model)    # L161
def forward(x, valid_token_mask=None):            # L163-173
    for layer in self.layers:
        x = layer(x, valid_token_mask, self.config.causal)   # L169
    x = self.final_norm(x)                        # L170
    if valid_token_mask is not None:
        x = x.masked_fill(~valid_token_mask[..., None], 0)   # L171-172
```

`config.causal` is frozen at construction from the CLI flag and threaded into every layer attention.

### 2.5 Embeddings / positions / RoPE — NONE

The model adds **no token embeddings, no positional embeddings, no learnable positions, no RoPE**. Forward operates directly on the `[B,S,D]` tensor; the first op on the input is layer-0 norm1 (L141 via L129). `generate_random_case` (L1131-1168) confirms the input contract: `x = torch.randn(B,S,D,...) * input_scale` (L1142-1150) — a raw random tensor already in the hidden dimension. The only ordering signal in the model is the opt-in causal mask. **Input shape is exactly [B,S,d_model]; there is no vocab lookup and no [B,S] token-ID tensor anywhere.**

### 2.6 Parameter count (verified against report)

Per block: attention 4*(512*512+512) = 1,050,624; norm1+norm2 = 2*1024 = 2,048; ffn_in = 512*2048+2048 = 1,050,624; ffn_out = 2048*512+512 = 1,049,088 -> **3,152,384/block**; x6 = 18,914,304; + final_norm 1,024 = **18,915,328 parameters (37.83 MB at FP16)** — matches OPTIMIZATION_REPORT.md.

---

## 3. Data flow and masks

### 3.1 Tensor shapes per stage (default B=8, S=128, D=512, H=8, head_dim=64, F=2048)

```
x (input, randn*scale)                   [8, 128, 512]      L1142-1150
per block:
  norm1(x)                               [8, 128, 512]
  q = k = v (separate Linear)            [8, 128, 512]      each L94-96
  split heads (view+transpose+contig)    [8, 8, 128, 64]    L78-84
  scores = q*k^T * scale                 [8, 8, 128, 128]   L98
  softmax(scores.float())->dtype         [8, 8, 128, 128]   L112
  context = probs*v                      [8, 8, 128, 64]    L113
  merge heads -> view                    [8, 128, 512]      L114-118
  out_proj                               [8, 128, 512]
  residual add                           [8, 128, 512]
  norm2(x)                               [8, 128, 512]
  ffn_in(x)                              [8, 128, 2048]     L142 (L132)
  GELU exact (erf)                       [8, 128, 2048]
  ffn_out                                [8, 128, 512]
  residual add                           [8, 128, 512]
end
  final_norm                             [8, 128, 512]      L170
output                                   [8, 128, 512]      (same contract L196)
```

There is **no initial input projection** — raw [B,S,D] goes straight into layer-0 norm1.

### 3.2 Where masks come from

Two independent, opt-in mechanisms:

1. **Causal mask** — from `--causal` (default off), stored in config.causal, threaded through stack -> blocks -> attention (L169, L141, L90). Built **per call** inside attention as a bool (S,S) triu(diagonal=1), masked_fill(-inf) (L100-104).
2. **Padding / valid-token mask** — from `--padding-ratio > 0` (default 0.0 => mask is None). `generate_random_case` (L1156-1168) samples a random **valid prefix length** per batch row `lengths ~ randint(min_valid, S+1)`, `min_valid = max(1, round(S*(1-padding_ratio)))`, then `valid_token_mask = arange(S)[None,:] < lengths[:,None]` -> `[B,S]` bool; invalid input rows zeroed (L1167). This is a **left-aligned (prefix) variable-length mask**, not center/right padding.

**Default mask regime: non-causal, no padding.** OPTIMIZATION_REPORT.md: the only authoritative target is "B=8, S=128, D=512, H=8, F=2048, L=6, noncausal, no padding".

### 3.3 Mask semantics recap

- invalid **key** positions -> -inf in the score row (L106-109; optimized fallback L413-415);
- invalid **query** rows -> output zeroed after projection (L121-122 attention, L144-145 block, L171-172 stack).

---

## 4. Numerics and accuracy gate

### 4.1 FP32 / TF32 / FP16 / BF16 handling

- **Common setup in main (L1845-1850):** `torch.manual_seed(args.seed)`; `torch.set_float32_matmul_precision(args.matmul_precision)` (default "high", L1657-1661); on CUDA `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32 = args.allow_tf32` (default **True**, `--allow-tf32` L1662-1667).
- **Consequence:** the FP32 "reference" itself runs with **TF32 enabled by default** on CUDA (`matmul_precision="high"` => FP32-storage matmuls may use TF32 tensor cores). Both baseline and candidate run under identical flags, so A/B is apples-to-apples; the gate is candidate-vs-baseline in one process, not vs an FP64 oracle.
- **Softmax always FP32** then recast to model dtype (L112) — the baseline's only in-op promotion, commented as a stable reference for fp16/bf16.
- **FP16:** weights/activations FP16 with the FP32-softmax boundary above.
- **BF16:** BF16 storage; on the optimized path BF16 is deliberately kept **bit-exact / on reference math** — forward returns `super().forward(...)` for BF16 (L510-511) and the fused-SDPA index set is emptied (L518-521), because "BF16 changes in attention algorithms exceed this benchmark's fixed 0.002 absolute tolerance" (L506-507).

### 4.2 Accuracy gate (`compare_outputs` L1185-1252)

Per element, both cast to FP32 (`.detach().float()`, L1202-1203):

```
finite(ref) and finite(opt)
and ( abs(opt-ref) <= atol  OR  abs(opt-ref) <= rtol*abs(ref) )
```

- L1205-1212 implements the exact **OR** semantics (deliberately *not* torch.isclose, which is more permissive; L1208-1209).
- **Defaults:** atol=0.002, rtol=0.02 (L1545-1546). These match STATEMENT.md ("relative error < 0.02, abs error < 0.002").
  - WARNING/discrepancy: the module docstring (L11) still says "default thresholds are atol=0.001 and rtol=0.01 (1%)" — **stale**; treat argparse (rtol=0.02, atol=0.002) as authoritative.
- Gate is **elementwise over the entire output tensor**; any failed element fails the trial (passed = failed_elements==0, L1242); any failed trial fails the accuracy phase (L1291).

### 4.3 Accuracy trials (`run_accuracy_tests` L1255-1320)

- `--accuracy-trials` (default 5; README best commands use 25) trials.
- Each trial: **fresh random input** with `seed = args.seed + trial` (L1279-1286); same tensor to baseline and candidate (same weights via `copy_model_weights` L1043-1053).
- Runs under torch.inference_mode (L1277). Reports PASS/FAIL, max abs, max rel, failed/total per trial (L1297-1319).
- If accuracy fails and `--benchmark-on-failure` is off (default), benchmark skipped, exit 2 (L2024-2027).

### 4.4 A/B measurement (warmup_model L1359-1370, benchmark_once L1373-1404, benchmark_models L1407-1479)

- **Fixed input:** one random case generated once with seed+100000 (L1425-1432); timing excludes data generation (L1421).
- **Warmup:** `--warmup` (default 20) passes through both models before timing (L1434-1436); CUDA synchronized (L1369-1370).
- **Rounds x repeats:** `--benchmark-rounds` (default 3) x `--repeats` (default 100). **Alternating order** to reduce thermal/clock bias: even rounds A-then-B, odd rounds B-then-A (L1441-1456).
- **CUDA timing:** per-iteration torch.cuda.Event on the current stream, sync before/after (L1383-1396); CPU fallback time.perf_counter_ns.
- **Metric:** `median` latency per model; speedup = baseline.median / optimized.median (L1460); tokens/s = B*S*1000/median_ms (L1461-1463); also mean/p90/min (L1465-1479).
- **Fresh processes:** the script is one Python process; the "fresh-process / order-balanced isolated-process pairs" A/B used for graph/Triton deltas in OPTIMIZATION_REPORT.md / README.md is report-level methodology (one script launch per branch), not an in-script feature.
- Optimized weights packed once after weight copy and device/dtype transfer, outside accuracy/timing regions (prepare_optimized_weights L298-335, called L1895).

---

## 5. Existing optimized path — flags and what they do

All optimizations are **opt-in**; `--user-implementation` defaults to `baseline` (L1560-1565).
Host code: `UserOptimizedTransformer` (L190-713) + `CudaGraphedTransformer` (L738-783) / `IntegratedInputCudaGraphedTransformer` (L823-1040).

### 5.1 Implementation selection (`--user-implementation`, L1560-1565)

| Choice | Behavior |
|---|---|
| `baseline` | exact `BaselineTransformer` path |
| `packed-qkv` | pack Q/K/V projections into one GEMM (`_packed_qkv` built L298-329, used L399/L438); attention math stays reference-style (`_baseline_attention_packed_qkv` L389-424) — **no fused SDPA** (index set forced empty, L518-521) |
| `sdpa` | baseline blocks, but selected trailing layers run fused `F.scaled_dot_product_attention` (`_sdpa_attention` L338-387: dropout_p=0.0, is_causal=causal, scale=attention.scale, attn_mask from valid-key mask L372-376) |
| `sdpa-packed-qkv` | fused SDPA **and** packed QKV on selected layers; non-selected layers use packed-QKV reference attention (L550-572) |

### 5.2 Dispatch and sensitivity flags

| Flag | Default | Effect | Constraints (validate_args L1671-1825) |
|---|---|---|---|
| `--sdpa-layers auto|N|all` | `auto` | number of **trailing** SDPA layers. Auto (resolve_sdpa_layers L1095-1128): **all layers for FP32**; 4 known default FP16 noncausal; 3 padded noncausal FP16; 2 causal FP16; 1 undisclosed FP16; 0 BF16 | — |
| `--sdpa-layer-indices` | None | explicit zero-based SDPA layer set, overrides `--sdpa-layers` (L1572-1577, L217-229) | indices in [0,L) |
| `--gelu-approx-layer-indices` | () | switch layers to **tanh GELU** (`F.gelu(..., approximate="tanh")` L524-528) — sensitivity only; **every individual layer failed** the FP16 25-trial gate (README, report), so best commands leave it empty | CUDA only (L1710-1711) |

### 5.3 Triton kernel flags (kernels in `triton_kernels.py`)

| Flag | Kernel / behavior |
|---|---|
| `--triton-rounded-attention` (all layers) or `--triton-rounded-attention-layer-indices` | shape-specialized **FP16 attention** for S=128, head_dim=64 (`rounded_attention` L549-612; kernel `_rounded_attention_kernel` L442-546). **Bit-exact** vs reference: keeps the two FP16 score-rounding boundaries (QK^T->FP16, scale->FP16, L509-510), FP32 softmax reproducing PyTorch PersistentSoftmax.cuh (lane-local sequential sums + XOR warp tree, `_pytorch_softmax_128` L32-44), libdevice exp, correctly-rounded tl.div_rn, probs->FP16 before value dot (L527-528). Tuned block_m=16 dense/causal, 64 padded; 4 warps; 3/1 stages (L587-588; README) |
| `--triton-exact-add-norm` (all 12 sites) or `--triton-fused-add-norm-sites` | exact **residual-add + following-LayerNorm** fusion (`add_layer_norm` L178-229; kernel L133-175). Reproduces PyTorch width-512 four-warp Welford tree (`_pytorch_layer_norm_stats_512` L104-130); residual sum stored in model dtype then normalized FP32 (L161-166). Bit-exact; 2 sites per layer |
| `--triton-exact-initial-norm` | exact **initial width-512 LayerNorm** (`layer_norm_512` L255-289) for the one norm not preceded by a residual add. Bit-exact |
| `--triton-linear-gelu` (all layers) or `--triton-linear-gelu-layer-indices` | fused **FFN input projection + exact erf GELU** (`linear_exact_gelu` L424-430; kernel `_linear_exact_gelu_kernel` L292-373). FP16 512->2048, block 128x128x64, 8 warps, 4 stages (L403-420); keeps the FP16 biased-GEMM materialization boundary, then exact libdevice.erf (L366-367). 0 failed elements per mask regime over 100 trials; unmasked bit-exact |
| `--triton-poly11-gelu-layer-indices` (retained: layer 5 only) | same fused kernel with use_poly11=True (L341-364): CUTLASS **degree-11 standard-normal-CDF polynomial**, clip to +-3.5, gelu = x * clamp(Phi(x), 0, 1). Accuracy-gated: only the final FFN passes (report; passes scale 0.1/10, padding 0.1/0.75) |
| `--pretranspose-ffn-output-weights` | prepack the six FFN-output weights once as contiguous [K=2048, N=512], dispatch via torch.addmm (`_linear_with_pretransposed_weight` L176-187; packing L330-336) — selects NVJet `NNT` tactic instead of `TNT`; QKV/attn-out weights deliberately unchanged. Bit-identical results |

### 5.4 Compile / CUDA-graph flags

| Flag | Effect | Notes / constraints |
|---|---|---|
| `--compile-baseline` / `--compile-user` | `torch.compile(model, mode=...)` (maybe_compile L1513-1518; applied L1898/L1917) | **FP32 recommended** (README); FP16 compilation **fails** the strict gate (report). Triton flags conflict with `--compile-user` (L1740-1741, L1762-1763, L1788-1789, L1814-1815) |
| `--compile-mode default|reduce-overhead|max-autotune` (default `default`) | compile mode (L1651-1655) | README best FP32 uses `reduce-overhead` |
| `--cuda-graph-user` | capture unchanged eager model via `torch.cuda.make_graphed_callables` into one CUDA graph (`CudaGraphedTransformer` L738-783); one cudaGraphLaunch/call; output cloned by default | requires CUDA (L1695-1696); mutually exclusive with `--compile-user` (L1697-1698); static shape + mask presence per capture |
| `--cuda-graph-static-output` | return graph-owned output directly (no clone); next replay overwrites it (L780-782, L1040) | requires `--cuda-graph-user` (L1699-1700) |
| `--cuda-graph-integrated-input-copy` | capture input (and optional mask) D2D copies as retargetable graph nodes via raw CUDA API (`IntegratedInputCudaGraphedTransformer` L823-1040); nodes matched by captured destination pointer, cudaGraphExecMemcpyNodeSetParams1D per call | requires `--cuda-graph-user` (L1701-1704) and contiguous CUDA storage, static shape/dtype/stride (L988-998, L843-850) |

### 5.5 Numerics-control flags

| Flag | Default | Effect |
|---|---|---|
| `--matmul-precision highest|high|medium` | `high` | `torch.set_float32_matmul_precision` (L1657-1661, L1846) |
| `--allow-tf32` / `--no-allow-tf32` | enabled | toggles torch.backends.cuda.matmul.allow_tf32 and torch.backends.cudnn.allow_tf32 for both implementations (L1662-1667, L1849-1850) |

### 5.6 Known best verified commands (README.md)

- **FP32 (default dtype):** `--user-implementation sdpa-packed-qkv --compile-user --compile-mode reduce-overhead` (+ 25 trials, 20 warmup, 100 repeats, 5 rounds).
- **FP16 (current best path):** `packed-qkv` + `--triton-rounded-attention --triton-exact-add-norm --triton-exact-initial-norm --triton-linear-gelu --triton-poly11-gelu-layer-indices 5 --pretranspose-ffn-output-weights` + `--cuda-graph-user --cuda-graph-static-output --cuda-graph-integrated-input-copy`.
- **BF16:** `packed-qkv` + raw graph, bit-exact (never enters SDPA/Triton).

### 5.7 Current headline numbers (OPTIMIZATION_REPORT.md)

- FP16 eager baseline ~2.4-3.3 ms; current Triton+graph FP16 path **0.27859 ms** -> **8.62x / +762% throughput vs eager**, 144.5 TFLOP/s = **17.3% of the 835.5 TFLOP/s dense roof**, 42.7% of the HBM logical-traffic roof.
- Best eager FP16 (packed QKV + trailing-4 SDPA) 1.325x; compiled FP32 ~4.2x; TensorRT FP32 engine + graph ~4.45x (0.5127 ms).
- Rejected for FP16: whole-model compile (45/13.1M fails), tanh GELU (every layer), TensorRT FP16 engine (61,494/13.1M fails), FP8, TMA/CUTLASS GELU variants (slower or non-bit-exact).

---

## 6. Comparison with a modern LLM decoder (LLaMA / Qwen)

The baseline is a **minimal, generic self-attention transformer stack** — notably *not* a modern autoregressive LLM decoder.

| Modern LLM decoder feature (LLaMA/Qwen) | Present in baseline? | Evidence |
|---|---|---|
| Token embedding lookup (vocab -> d_model) | **No** — input is a raw [B,S,D] random tensor | L1142-1150; L163-173 |
| Positional info: RoPE (or any positional embedding) | **No** — only ordering signal is the opt-in causal mask | L86-123; no rope/embedding modules |
| GQA / MQA (shared K/V heads) | **No** — full MHA: independent Q/K/V/O projections, width D, bias on | L73-76 |
| SwiGLU / gated FFN (LLaMA SiLU*gate) | **No** — plain exact-erf GELU FFN, 1->4->1 | L132-133, L142 |
| Causal-by-default attention | **No** — default **non-causal**; `--causal` opt-in (store_true, default off) | L1531; report default = "noncausal, no padding" |
| Output LM head (d_model -> vocab logits) | **No** — output is [B,S,D] after final_norm | L170-173 |
| KV cache / incremental decode structure | **No** — one dense prefill-style pass; causal is a static -inf triangle, not cache masking | forward L163-173 |
| Dropout / training machinery | **No** dropout; harness runs inference_mode | L1277, L1359 |
| Encoder-decoder / cross-attention | **No** — self-attention only | L86-123 |

Consequence for optimization work: the fixed B x S x D dense workload (T = B*S = 1024 tokens) has **no decode phase, no vocab/head GEMM, and (by default) no causal structure to exploit**. The 4*B*S^2*D attention term is small relative to projections+FFN at this shape — attention becomes dominant only when S > 2D + F (3072 for default D/F).

---

## 7. Constraints and notes relevant to further optimization

### 7.1 Fixed shapes / static contracts

- **Default authoritative shape:** B=8, S=128, D=512, H=8, head_dim=64, F=2048, L=6, noncausal, unmasked (report, Scope). Official test-shape table still missing (Feishu export failure; TODO P0).
- **Shape-specialized kernels are hard-coded:** rounded attention requires S=128, head_dim=64 + FP16 + CUDA (triton_kernels.py L563-564); linear+GELU requires width-512 input, 2048x512 FP16 weight (L385-388); add+LayerNorm width 512 (L199-200); exact initial LayerNorm width 512 FP16 (L266-268). validate_args rejects combination flags outside the audited shape/dtype/mask (e.g. L1725-1734, L1810-1825).
- **CUDA graphs require static shape AND fixed mask presence** per capture; the integrated-copy variant additionally requires contiguous CUDA inputs with unchanged shape/stride/dtype/device (L988-998).
- `--cuda-graph-user` and `--compile-user` mutually exclusive (L1697-1698); Triton flags incompatible with `--compile-user` (L1740-1815).

### 7.2 Accuracy gates that shape every optimization

- Elementwise `atol=0.002` OR `rtol=0.02*|ref|`, zero tolerated failures, per trial; any failed element fails the run (L1205-1242). Verified command sets use 25 trials and often 100-trial audits across four mask regimes (unmasked/padded/causal/causal+padded) plus input-scale 0.1/10 stress.
- BF16 must stay **bit-exact** on reference math — never touches SDPA/Triton (L510-511, L518-521).
- FP16 compile is disqualified numerically; only FP32 compile passes (report). Triton paths are bit-exact or near-bit-exact by reproducing the materialization boundaries (score->FP16, scale->FP16, softmax FP32 lane/XOR order, probs->FP16).
- Approximations (tanh GELU, poly11, FP8, TF) must pass the same elementwise gate; validated approximation retained so far is only the final-layer degree-11 polynomial GELU (README/report).

### 7.3 Roofline / bottleneck picture and next steps (OPTIMIZATION_REPORT.md)

- Model per-call arithmetic: **40.265 GFLOP/call** (6.711 GFLOP/block). Dominant block terms: FFN 64% (4.295 GFLOP), QKV projections 24% (1.611 GFLOP), attention output proj 8%, QK^T+PV 4%.
- Ideal compute-only dense roof on H200: 48.2 us at 835.5 TFLOP/s (dense, 2:4 sparsity unusable on random weights). Logical-traffic floor ~118.9 us at 4.8 TB/s. Eager baseline is ~50-69x the compute floor, dominated by 127 serial kernel launches and medium GEMMs.
- **Current bottleneck:** the sequence of medium, dependency-bound projection GEMMs, not HBM bandwidth or attention. In the final 44-node graph: all GEMM groups = 181.3 us (69.2% of GPU kernel time); the two FFN projections alone = 112.6 us (43.0%). FFN input+exact-GELU 65.4 us (25.0%), FFN output 47.2 us (18.0%). Each layer must finish normalization before its next projection, limiting occupancy.
- **Prioritized backlog (condensed):** P0 recover the official Feishu shape/dtype/mask matrix and enable ncu counters; P1 packed QKV, pretransposed FFN-output, compile/CUDA graph/TensorRT sweep, static-mask hoisting (all implemented); P2 fused residual+LN+linear (linear+FFN epilogue open), whole-block valid-token packing (1.2-2x with padding), per-shape autotuned dispatcher (5-30%); P3 FP8/Transformer Engine (1.1-1.7x, accuracy-gated), Hopper TMA/WGMMA where libraries underfill; P4 whole-block persistent megakernel.
- Compounding groups: packed QKV + BSHD + Flash SDPA; token packing + variable-length attention + FFN skipping; residual/LayerNorm + linear epilogues. FlashAttention / cuDNN SDPA / Transformer Engine attention are competing backends and must A/B, not stack. Approximate GELU and FP8 conflict most with the strict numerical gate.
- Measurement note: Nsight Compute blocked by ERR_NVGPUCTRPERM (needs admin/non-admin profiling enablement); roofs above are estimates, nsys gives launch/kernel-timing evidence.

---

## 8. Key findings (summary)

1. **Baseline = tiny Pre-LN dense MHA stack, NOT an LLM decoder.** B=8, S=128, D=512, H=8 (head_dim 64), F=2048, L=6; separate biased Q/K/V/O projections; scale = 1/sqrt(64) = 0.125; softmax in FP32 then cast back to model dtype; exact erf GELU; final LayerNorm; no embeddings, no RoPE/GQA/SwiGLU, no LM head, no KV cache. Default run is **non-causal and unpadded** (mask None); causal and a left-aligned prefix padding mask are both opt-in.
2. **Default dtype is FP32 with TF32 enabled** (`--dtype float32`, `matmul_precision=high`, allow_tf32=True), so the FP32 "reference" already uses TF32 tensor cores; FP16 and BF16 are available, and BF16 is held bit-exact on reference math (never SDPA/Triton).
3. **The accuracy gate is strict and elementwise** — abs <= 0.002 OR rel <= 0.02, zero failures allowed, fresh random input per trial (5 default / 25 verified), same weights both sides; failures abort the benchmark (exit 2). This gate, not speed, is what forced the bit-exact Triton kernels.
4. **Optimized path = opt-in flags composable under one wrapper:** packed QKV, fused SDPA dispatcher (dtype/mask-aware layer counts), bit-exact Triton attention / add+norm / initial norm / linear+erf-GELU / polynomial-GELU (final layer), pretransposed FFN-output weights, torch.compile (FP32 only), and CUDA-graph replay with retargetable integrated input copies (the biggest single win: 4-8x by removing launch overhead).
5. **Two documented inconsistencies to flag in later work:** module docstring says atol=0.001/rtol=0.01 while argparse+STATEMENT say 0.002/0.02 (argparse wins); and the official fixed test-shape table is missing from STATEMENT.md (Feishu export failure), so all shape specialization is currently anchored to the default shape only.


---

## A. VRAM / memory footprint (verified 2026-08-28, follow-up answer)

Weight memory (verified: `torch_transformer_benchmark.py` config D=512, F=2048, L=6, + final norm):
* Per block (×6) = 3,152,384 params:
  - attention (4 × Linear 512→512, bias): 1,050,624
  - norm1 + norm2 (LayerNorm 512): 2,048 each pair
  - ffn_in (Linear 512→2048, bias): 1,050,624
  - ffn_out (Linear 2048→512, bias): 1,049,088
* Final LayerNorm 512: 1,024
* **Total: 18,915,328 params (~18.9M / 0.019B)** — no vocab embeddings or LM head, so no large embedding block.
* Weight VRAM: FP16/BF16 = 37.83 MB; FP32 = 75.66 MB.
* Total inference footprint ≈ 100–250 MB (GPU: H200 NVL ~141 GB). Largest intermediates: FFN hidden ~4 MiB; attention scores/probs ~2–4 MiB.
* Note: the ~570 MB/call figure seen in profiling is cumulative data movement bandwidth, NOT resident VRAM.
