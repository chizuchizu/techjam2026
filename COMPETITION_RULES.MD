# TechJam 2026 — Transformer Optimization Competition: Rules & Reference

> Goal: use AI-assisted methods to optimize the runtime efficiency of a Transformer
> on a given GPU, improve performance, and keep output numerically correct versus the
> reference implementation.

---

## 1. Background

Transformer is a widely used neural network architecture in modern AI, core to NLP,
vision, speech, recommendation, and LLM systems. Its main idea is **self-attention**:
each token interacts directly with all other tokens, which (unlike recurrent models)
can be processed in parallel — hence its GPU suitability.

Given an input sequence `X ∈ R^(N×d)` (N = sequence length, d = hidden size), the
Transformer first projects the input into Query, Key, Value matrices:

```
Q = X·W_Q,   K = X·W_K,   V = X·W_V
```

Scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q·K^T / sqrt(d_k)) · V
```

where `d_k` is the per-head dimension; `1/sqrt(d_k)` keeps dot-product magnitudes from
degrading softmax stability. Costly parts: matmul, attention scores, softmax,
normalization, feed-forward. These may be limited by GPU **compute throughput, memory
bandwidth, cache efficiency, kernel launch overhead, and tensor-core utilization**.

---

## 2. Problem Statement

- Fixed Transformer layer formula. Submit **one or several GPU kernels** implementing
  the layer so that the provided test cases pass.
- Tests are written in **PyTorch** or **TensorFlow** (the organizers provided only
  the **PyTorch** script to us). You may modify the layer implementation, including
  deciding which parts to fuse into one kernel.
- The test compares your implementation against the original PyTorch/TensorFlow one;
  the difference must be small: **relative error < 0.02, abs error < 0.002**.
- Cases cover many input shapes (small/large batch, sequence length, dimensions, etc.).
  You may add shape checks in the layer and choose different implementations per shape.
  **All input-shape combinations are announced in advance.**
- AI tooling is encouraged so different kernels for different shapes can be produced
  quickly.
- Optimize and test on your own machine; methods depend on your GPU.
- Provide a clear tech report including the AI skills/tools used (for bonus points).

### What to do
1. Download the benchmark script (Torch or TensorFlow — one is enough;
   **we were provided only the Torch one**).
2. Implement the customized-implementation part and optimize it (by AI or by hand).
3. Run the script on your own machine.
4. Write a clear tech report: environment (CPU/GPU/DISK/etc.), optimizations performed,
   final test results.

---

## 3. Test Cases (announced shapes)

All 14 combinations. Columns: batch size, QKV dim (= `d_model`), heads, seq len,
layers, causal, FFN dim.

| # | Batch | QKV Dim | Heads | Seq Len | Layers | Causal | FFN Dim |
|---|------:|--------:|------:|--------:|-------:|:------:|--------:|
| 1 | 64    | 128     | 4     | 128     | 4      | TRUE   | 128     |
| 2 | 1     | 128     | 4     | 128     | 4      | TRUE   | 128     |
| 3 | 4     | 128     | 4     | 128     | 4      | TRUE   | 128     |
| 4 | 16    | 128     | 4     | 128     | 4      | TRUE   | 128     |
| 5 | 128   | 128     | 4     | 128     | 4      | TRUE   | 128     |
| 6 | 10000 | 128     | 4     | 128     | 4      | TRUE   | 128     |
| 7 | 64    | 32      | 4     | 128     | 4      | TRUE   | 32      |
| 8 | 64    | 1024    | 4     | 128     | 4      | TRUE   | 1024    |
| 9 | 64    | 128     | 1     | 128     | 4      | TRUE   | 128     |
| 10| 64    | 128     | 2     | 128     | 4      | TRUE   | 128     |
| 11| 64    | 128     | 16    | 128     | 4      | TRUE   | 128     |
| 12| 64    | 128     | 4     | 32      | 4      | TRUE   | 128     |
| 13| 64    | 128     | 4     | 1024    | 4      | TRUE   | 128     |
| 14| 32    | 1024    | 16    | 100000  | 2      | TRUE   | 1024    |

Observations worth exploiting (shape-aware kernels):
- Batch sweeps: 1 → 4 → 16 → 64 → 128 → **10000** (huge-batch case).
- Seq sweeps: 32 → 128 → 1024 → **100000** (very long sequence, 2 layers).
- Head-count sweeps at fixed dim: 1, 2, 4, 16.
- Dim sweeps at fixed batch/seq: QKV 32/128/1024 with FFN scaled to match (32/128/1024).
- **Every case is causal.**
- Case 14 (B=32, d=1024, H=16, S=100000, L=2) is the memory/compute stress case.

---

## 4. Available Resources / Data

**Only the PyTorch script was provided to us — nothing else:**

| Lane | Script |
|---|---|
| Torch | `torch_transformer_benchmark.py` (the only provided file) |

The competition text mentions a TensorFlow variant ("download 1 of these"), but no
TensorFlow script, no weights, no checkpoints, and no datasets were distributed; the
Torch script above is the complete resource for this task.

### No weights are needed — the script is self-contained
- The model is built from scratch with **randomly-initialized weights**, seeded
  reproducibly (`torch.manual_seed(args.seed)`, default `1234`).
- Input data is generated in-code with the same seed (`generate_random_case`), with
  optional padding via `--padding-ratio`.
- `copy_model_weights()` deep-copies the baseline state dict into the optimized
  model, so **baseline and optimized run on the identical random weights**.
- "Correct" therefore means *matching the baseline implementation on those same random
  weights* within `rtol=0.02 / atol=0.002`; there is no ground-truth or pretrained
  weight set to match against.

### What the uploaded Torch script actually does
- **Model**: pre-LN Transformer block — `x + Attn(LayerNorm(x))`, `x + FFN(GELU(FFN_in(LayerNorm(x))))`; final `LayerNorm`. Attention scales by `head_dim**-0.5`, applies a causal upper-triangular mask, computes softmax in fp32 (stable reference), then `P·V`, output projection. Optional `valid_token_mask` (padding) support.
- **Configuration fields** (`TransformerConfig`): `batch_size, seq_len, d_model, num_heads, ffn_dim, num_layers, causal` (validated, e.g. `d_model % num_heads == 0`).
- **Optimization point**: replace `UserOptimizedTransformer.forward()` — keep the signature, return `[B, S, d_model]`, and keep compatible parameter names or customize `copy_model_weights()`.
- **Correctness rule** (exact OR semantics, not `torch.isclose`):
  ```
  pass  ⇔  abs(user - ref) <= atol   OR   abs(user - ref) <= rtol * abs(ref)
  ```
  Defaults: `-rtol 0.02`, `-atol 0.002` (matches the stated 2% / 0.002). Elements must also be finite.
- **Accuracy**: 5 random trials per run (`--accuracy-trials 5`); a FAIL blocks the performance benchmark unless `--benchmark-on-failure`.
- **Latency**: fixed input, warmup then alternating rounds of timing; CUDA timing via `torch.cuda.Event` on the current stream; reports median/mean/p90/min, token/s, and **speedup = baseline median / optimized median**.
- **Key CLI flags**:
  ```
  --batch-size --seq-len --d-model --heads --ffn-dim --layers --causal
  --device auto|cpu|cuda --dtype float32|float16|bfloat16
  --padding-ratio --input-scale --accuracy-trials --rtol --atol --seed
  --warmup --repeats --benchmark-rounds --benchmark-on-failure
  --compile-baseline --compile-user --compile-mode default|reduce-overhead|max-autotune
  --non-strict-weight-copy --matmul-precision highest|high|medium --allow-tf32/--no-allow-tf32
  ```
- Suggested optimization directions (script comments): `scaled_dot_product_attention`,
  `torch.compile`, fused Triton/CUDA kernels, fused LayerNorm/residual/FFN, TF32/fp16.

---

## 5. Constraints & Scope

| In scope | Out of scope |
|---|---|
| AI-based code generation, GPU kernel fusion, profile tools usage, etc. | Production-ready deployment |

---

## 6. Deliverables

1. **Written Project Description (Devpost)**
   - How the solution addresses the problem statement
   - Development tools used (e.g. VSCode, Colab, Jupyter)
   - APIs used (e.g. OpenAI GPT-4o, Google Maps API)
   - Libraries/frameworks used (Hugging Face Transformers, PyTorch, scikit-learn, pandas, …)
   - Datasets/assets used

2. **Public Code/GitHub Repository**
   - Well-structured, commented code covering all solution components
   - README with: project overview; setup & install; reproduce-steps; reflection on
     limitations and future improvements; team contributions

3. **Demo Video** (YouTube, public, linked in Devpost)
   - End-to-end demonstration (inference/results/dashboard). For backend/NLP tracks a
     walkthrough of API usage, inference examples, or result analysis is accepted.
   - No third-party trademarks/copyrighted content without permission.

---

## 7. Judging Criteria

| Criterion | Definition | Weight |
|---|---|---|
| Technical Execution | Strong engineering: structured code, thoughtful architecture, effective API/model use; demo runs reliably; deliberate complexity | 35% |
| Innovation & Problem Insight | Originality of idea and approach; sharp problem framing; solution directly addresses it | 20% |
| Impact & Relevance | Clear real-user/stakeholder value; reach beyond the hackathon prompt | 20% |
| Feasibility & Practicality | Realistic and buildable beyond prototype; proportionate resource use; grounded, sustainable architecture | 15% |
| Presentation & Communication | (final event) Clear coherent story, problem→solution→potential; deep, genuine understanding when questioned | 10% |

---

## 8. Local notes / checklist

- [ ] Lane = **Torch** (the only script provided; no TensorFlow file was distributed).
- [ ] Enumerate the 14 announced shape combos and pick a per-shape kernel strategy
      (e.g. `scaled_dot_product_attention` for large seq/batch, fused Triton for small).
- [ ] Verify every case passes `rtol=0.02 / atol=0.002` (OR rule) before benchmarking.
- [ ] Report environment (CPU/GPU/DISK, CUDA, torch version) and per-case speedup.
- [ ] Document AI tools used (model names/versions, prompt style, iteration loop) for bonus points.
- [ ] Prepare Devpost write-up, GitHub repo + README, and public YouTube video.
