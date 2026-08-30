# Running a Transformer with Runtime-Variable Geometry on ESP32-C3 (no per-shape recompile)

**Question.** How do you write *one* C function/class that runs a transformer whose
model geometry — batch `B`, sequence length `S`, model dim `D`, heads `H`, FFN dim `F`,
layers `L` — is decided at **runtime**, without re-flashing firmware for each geometry
(no `#define D 64 / #define L 4` rebuild loop)?

**Short answer.** Yes, this is a solved pattern. Don't bake tensor shapes into macros.
Do what the data-driven runtimes do:

1. Store the geometry as **runtime data** (a small binary model header: a config struct
   followed by weights, or a GGUF/TFLite/ONNX-style file with an embedded metadata block).
2. Write every kernel as **loops over runtime variables** (`for l in 0..L`, `for h in 0..H`,
   `for i in 0..S`, `for j in 0..D`). Shape never appears as a compile-time constant.
3. Allocate one **workspace / arena** from SRAM. Either (a) size it for the *maximum*
   supported geometry at compile time, or (b) compute the required size from the loaded
   config and carve it out of a fixed heap/arena buffer at startup. Never sprinkle
   `malloc`/`free` inside per-layer kernels.
4. Keep weights resident in **flash** (ESP32-C3 flash is memory-mapped / XIP) and compute
   offsets from the runtime config (`wq = layer_base + l*4*D*D + ...`).

The same firmware image then accepts any model whose geometry fits the worst-case
workspace and whose operator set is compiled in.

---

## 1. Why geometry likes to be "compile-time" on MCUs

A naive hand-rolled transformer uses `#define`-sized arrays:

```c
#define D 64
#define L 4
#define S 64
static float state[L][S][D];   /* fixed in the image */
```

This makes stack/static sizing trivial and lets the compiler fully unroll loops, but it
forces a **recompile and re-flash for every `D/H/L/S` change** — exactly the problem to
avoid. The cost of escaping it is:

* all loops must carry runtime bounds (loop counters are cheap; unrolling is lost, but
  on a 160 MHz in-order RISC-V core unrolling matters less than memory traffic anyway),
* all buffers must either live behind pointers (`float *buf`) or be capped to a
  `MAX_*` worst case, and
* tensor offsets must be computed from the runtime config instead of being constants.

That is precisely the trade every portable inference runtime makes.

---

## 2. Existing projects that already do this

| Project | How geometry is decided at runtime | Allocator model | Recompile needed per model? |
|---|---|---|---|
| **TensorFlow Lite Micro** | `.tflite` FlatBuffer carries all tensor shapes; shapes are read at `AllocateTensors()`/first `Invoke()` | single shared `tensor_arena`; `MicroAllocator` plans head/temp/tail sections; greedy planner reuses non-overlapping tensors | No (same op set), only the arena must be sized for the worst-case model |
| **microTVM graph executor** | model graph + params shipped as JSON; parsed by C runtime on device | dynamic `TVMBackendDeviceAlloc()` calls; JSON parsing adds memory overhead | No, but memory-heavy |
| **microTVM AOT executor** | compiler emits a fixed C function with baked-in calls | static workspace array computed by compiler | **Yes** (regenerate C per model) |
| **llama.cpp / GGUF** | GGUF file holds key/value metadata (`block_count`, `embedding_length`, `head_count`, `feed_forward_length`, …); buffers sized from those values at load | host `malloc`/`mmap`; on MCU you'd swap in an arena | No |
| **karpathy llama2.c** | 7-int config header (`dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len`) read from checkpoint | host `malloc` for weights + run state | No |
| **gemma.cpp** | model config selected at runtime; weights file carries compressed tensors + metadata | host allocation | No |
| **cONNXr** | pure C99 ONNX interpreter; reads the `.onnx` graph at runtime | libc `malloc` | No |
| **tinyml-rs** | `no_std` runtime; model from a binary blob (`Model::from_bytes`), arena passed per call | fixed-size, stack-allocated **bump allocator** (`Arena<4096>`) | No |
| **CMSIS-NN** | kernels are plain functions (`arm_mat_mul_s8`, `arm_softmax_s8`, …) taking buffer pointers + runtime dims; **no graph runtime of its own** | you decide (static arrays or arena) | No (you hand-roll the graph loop) |

Key nuance for TFLM: the *binary* is geometry-agnostic, but the *arena* is not. If you
flash one firmware and then feed it several different models, you must give the arena a
size that covers the largest model, or re-derive it at boot. `MicroMutableOpResolver<N>`
lets you pre-register the union of all op codes the models use; `AllOpsResolver`
registers everything at the price of extra code size.

> **TFLM evidence.** `memory_management.md` describes the "online" strategy: "loading a
> model into a shared tensor arena" — the planner runs at load time, not compile time.
> A recording example shows a small model's arena: **head 7744 B + tail 1824 B =
> 9568 B total**. The greedy planner "looks at the entire graph of a model and tries to
> reuse as many buffers as possible." The `LinearMemoryPlanner` gives each tensor its
> own space (easier debug, more RAM).

> **microTVM evidence.** The design doc is explicit that the graph executor "needs to
> reconstruct the graph in memory … then invoke the operator implementations in the
> correct order", while the AOT executor "avoid[s] … any Graph JSON parsing and
> improve[s] inference speed by generating C code to call the generated operator
> implementations directly." So: *graph executor = runtime geometry, more memory*;
> *AOT executor = fast/small, per-model rebuild*.

> **GGUF evidence.** The format aims to contain "all information needed to load a model
> is contained in the model file". Hyperparameters are named keys:
> `[llm].block_count`, `[llm].embedding_length`, `[llm].feed_forward_length`,
> `[llm].context_length`, `[llm].attention.head_count`,
> `[llm].attention.head_count_kv`, `[llm].attention.key_length`, …
> `llama.cpp` reads these and sizes its graphs/buffers accordingly at load.

---

## 3. The hand-rolled pattern (single C function)

### 3.1 Runtime config + model header

```c
typedef struct {
    uint32_t magic;      /* 'T', 'R', 'F', 'M' */
    uint32_t version;
    uint32_t B;          /* batch (usually 1 on MCU)   */
    uint32_t S;          /* max sequence length        */
    uint32_t D;          /* model dim                  */
    uint32_t H;          /* heads                      */
    uint32_t F;          /* FFN hidden dim             */
    uint32_t L;          /* layers                     */
    uint32_t V;          /* vocab size                 */
    uint32_t n_q;        /* 0 = no quant, 4 = f32, 1 = int8, 2 = int16 */
} ModelCfg;              /* packed, fixed-size, read from flash */

/* on-disk layout: ModelCfg, then weight bytes in a canonical order */
```

The exact same code runs every geometry because `B..V` are variables, not macros.

### 3.2 Loop shapes are runtime values

```c
/* C = A (MxK) * B (KxN), all dims runtime. SIMD later; this is the semantic core. */
void matmul(const float *a, const float *b, float *c, int m, int k, int n) {
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float acc = 0;
            for (int p = 0; p < k; p++) acc += a[i*k + p] * b[p*n + j];
            c[i*n + j] = acc;
        }
}
```

Attention with runtime `B,S,D,H` and `hs = D/H`:

```c
/* Tensors are [B, S, H, hs]: element idx = ((b*S + t)*H + h)*hs + d */
static inline size_t qkv_idx(int b, int t, int h, int d, int S, int H, int hs) {
    return ((size_t)(b*S + t)*H + h)*hs + d;
}

void attention(const float *q, const float *k, const float *v,
               float *out, float *scores, int B, int S, int H, int hs) {
    float scale = 1.0f / sqrtf((float)hs);      /* ESP32-C3: software sqrt */
    for (int b = 0; b < B; b++)
        for (int h = 0; h < H; h++)
            for (int i = 0; i < S; i++) {       /* query position */
                float m = -1e30f, sum = 0;
                for (int j = 0; j <= i; j++) {  /* causal: attend past only */
                    float s = 0;
                    for (int d = 0; d < hs; d++)
                        s += q[qkv_idx(b,i,h,d,S,H,hs)] * k[qkv_idx(b,j,h,d,S,H,hs)];
                    s *= scale; scores[i*S + j] = s;
                    if (s > m) m = s;
                }
                for (int j = 0; j <= i; j++) sum += expf(scores[i*S+j] - m);
                for (int d = 0; d < hs; d++) {
                    float a = 0;
                    for (int j = 0; j <= i; j++)
                        a += expf(scores[i*S+j] - m) * v[qkv_idx(b,j,h,d,S,H,hs)];
                    out[qkv_idx(b,i,h,d,S,H,hs)] = a / sum;
                }
            }
}
```

### 3.3 Layers loop over `L`

```c
typedef struct { float *x; float *qkv, *scores, *att, *ffn, *wk1, *kvcache; } Workspace;

void transformer_run(const ModelCfg *cfg, const void *weights,
                     const int *tokens, float *logits, Workspace *ws) {
    const float *w = (const float*)weights;   /* offset base by cfg->L/D/F */
    int D=cfg->D, S=cfg->S, H=cfg->H, F=cfg->F, L=cfg->L, B=cfg->B, hs=D/H;
    embed(tokens, ws->x, w, B, S, D, cfg->V); w += (size_t)cfg->V * D;
    for (int l = 0; l < L; l++) {
        /* q,k,v projections: W{q,k,v} @ x  ->  3 * B*S*D */
        for (int i = 0; i < 3; i++) { matmul(ws->x, w, ws->qkv + i*B*S*D, B*S, D, D); w += (size_t)D*D; }
        attention(ws->qkv, ws->qkv + B*S*D, ws->qkv + 2*B*S*D, ws->att, ws->scores, B, S, H, hs);
        matmul(ws->att, w, ws->x, B*S, D, D);     w += (size_t)D*D;   /* Wo + residual */
        /* FFN: W1 -> GELU -> W2 */
        matmul(ws->x, w, ws->ffn, B*S, D, F);     w += (size_t)D*F;   /* up   */
        gelu(ws->ffn, B*S*F);
        matmul(ws->ffn, w, ws->x, B*S, F, D);     w += (size_t)F*D;   /* down + residual */
    }
    layernorm_final(ws->x, logits, w, B, S, D);
}
```

This function has **zero compile-time geometry constants**. Only `ws` must be large
enough — which leads to the allocator question.

---

## 4. Allocation: static vs dynamic, arenas, workspaces

### 4.1 Static allocation (compile-time sized)

All buffers are fixed, known at link time; zero `malloc`, WCET-friendly, no
fragmentation, but **per-geometry rebuild** unless you size everything for a worst
case. This is what CMSIS-NN examples and small hand-written firmware do.

* **Pro**: deterministic, provable peak RAM, no heap metadata.
* **Con**: "Model switching requires full recompile/reboot" (the one row in the
  static-vs-dynamic comparison that matters here).

### 4.2 Dynamic allocation with the platform heap

`malloc`/`free` from ESP-IDF's heap: flexible geometry, but

* per-call latency is unbounded (free-list search),
* **fragmentation** accumulates in a long-running device — you can end up with enough
  *total* free RAM but no *contiguous* block for a new `L×S×D` tensor,
* failure modes are `NULL` + `abort()`, and peak RAM is hard to bound.

An inference-first runtime-allocator paper, SynapticOS (arXiv 2607.12606), makes exactly
this point: NN kernels have a memory-access pattern "fundamentally different from the
control-loop and sensor-polling workloads that traditional RTOS allocators are tuned
for."

### 4.3 Arena / bump allocator (the right answer for this task)

Reserve one contiguous region (static array or one early `malloc`), then hand out
aligned slices with a moving pointer; never free individual pieces.

```c
typedef struct { uint8_t *base; size_t cap, used; } Arena;
static void *arena_alloc(Arena *a, size_t n, size_t align) {   /* bump, aligned */
    size_t off = (a->used + (align - 1)) & ~(align - 1);
    if (off + n > a->cap) return NULL;
    void *p = a->base + off;
    a->used = off + n;
    return p;
}
#define arena_reset(a) ((a)->used = 0)
```

Two usage modes:

* **Per-inference scratch.** Reset the arena before each `transformer_run`; allocate
  activations monotonically. This is what tinyml-rs does (`runtime.predict(&input,
  &arena)`).
* **Worst-case static pool.** Compute `max_workspace(configs...)` at startup and carve
  it once; use a liveness-aware planner (below) to overlap tensors and shrink the pool.

### 4.4 Planned reuse (what TFLM does "online")

Tensors have a lifetime `[first_created, last_used]`. A **greedy first-fit** planner
packs tensors whose lifetimes don't overlap into the same bytes:

```
|-- head (non-persistent, reused) --|<-- temp -->|-- tail (persistent) --|
```

`MicroAllocator` puts shared tensor buffers in the *head* (planned), scoped scratch in
the *temp* section, and persistent structs (nodes, registrations, planner, quant
metadata) in the *tail*. `RecordingMicroInterpreter::PrintAllocations()` prints each
section so you can right-size the arena ("Arena allocation total 9568 bytes… head 7744
bytes … tail 1824 bytes").

**The practical rule:** keep one compile-time arena big enough for the worst geometry
you will ever accept, then *verify* at load with an arena-used-bytes check —
`if (needed > ARENA_SIZE) return ERR_MODEL_TOO_BIG;`. If you can afford a one-time
startup cost, run your own greedy planner over the chosen `(B,S,D,H,F,L)` to compute the
exact offsets — then different geometries share one SRAM pool without leaving dead
slack.

---

## 5. Heap fragmentation specifics on ESP32-C3

* ESP32-C3 runs one app image (no MMU). All heap + static + stack live in the same
  400 KB SRAM; there is no virtual memory to make scattered pages contiguous.
* Fragmentation therefore **directly** reduces the maximum runnable model: you need one
  physically contiguous `B*S*F` or `B*H*S*S` buffer.
* Recommendations from the embedded-ML literature and TFLM/microTVM practice:
  * one arena **per model/geometry**, freed wholesale between models (never free
    individual tensors);
  * if you must use the heap: allocate the workspace **once** at model load, keep it
    for the model's lifetime, free it as one block on model swap;
  * keep long-lived buffers (Wi-Fi stack, ring buffers) in static/`.noinit` sections,
    not interleaved with per-model memory.

---

## 6. Concrete memory budgets

### 6.1 The target: ESP32-C3

* RISC-V single core, **RV32IMC**, up to **160 MHz**, **no FPU** (floating point is
  software-emulated → prefer int8/int16 or fixed-point kernels).
* **400 KB internal SRAM**, **4 MB flash** (typical module; memory-mapped/XIP).
* Budget is shared with the BLE/Wi-Fi stack, so realistic *free* SRAM for inference is
  often ~150–250 KB depending on stack usage.

### 6.2 Published transformer/MCU numbers

| Source | Result | Memory | Flash |
|---|---|---|---|
| TinyFormer (arXiv 2311.01759) | sparse transformer on MCUs | **320 KB** memory | **1 MB** storage constraint |
| MCUFormer (arXiv 2310.16898) ViT, ImageNet 73.6 % | optimized | **218–319 KB** | **0.89–0.90 MB** |
| MCUFormer naive baseline (AutoFormer) | unoptimized | **681 KB** (exceeds C3) | 1.24 MB |
| TFLM recording example (tiny model) | allocator audit | **9.6 KB** arena total (7.7 head + 1.8 tail) | — |
| DistilBERT-style TFLM on Cortex-M55 blog | `kTensorArenaSize` | **150 KB** arena | — |

Conclusion from the table: a *small* transformer (dim ≈ 64–128, ≤ 4 layers) **fits in
ESP32-C3 SRAM**, but a naive deployment of even a small ViT (681 KB) does **not** — you
need the same buffer-reuse/token-overwrite scheduling MCUFormer describes. MCUFormer's
peak-memory rule for its FC layers is:

```
M = sup_l ( h_f^l * w_f^l ) + max_l ( h_i^l * w_i^l ,  h_o^l * w_o^l )
      \________ weight buffer ___/     \____ max(input||output activation) ___/
```

i.e. **hold one layer's weight tile + the larger of input/output activations**, and
overwrite the input activation with the output in place ("token overwriting").

### 6.3 Analytical budget for a C3-sized decoder transformer

Choose `B=1, S=64, D=64, H=4 (hs=16), F=256, L=4, V=256`:

* Weights `4·D² + 2·F·D` per layer = 48 KB, ×4 layers + 16 KB embedding ≈ **208 KB int8**
  (or 416 KB int16 / 832 KB fp32). fp32 weights (832 KB) exceed the 400 KB SRAM (they would have to stream from flash), and fp32 math is very slow on a no-FPU core, so int8/int16 is the practical choice; int8 weights fit in flash with room to spare.
* Activation workspace (with reuse, fp32 scores worst case):
  embeddings 4 KB, QKV 12 KB, attention scores `B·H·S²` = 16 KB, FFN `B·S·F` = 16 KB,
  KV cache `2·L·B·S·D` = 32 KB → **≈ 48–64 KB peak** for prefill.
* Long-content risk: scores are **O(S²)** and KV cache is **O(L·S)** — on a 400 KB chip
  `S` must stay small (tens of tokens) or you must compute attention in `O(S)` per-head
  chunks and/or cap `S` at load time.

So a single ESP32-C3 binary can realistically carry a **configurable decoder with
D ≤ 128, H ≤ 8, F ≤ 512, L ≤ 6, S ≤ 64–128, int8 weights** — geometry chosen at boot
from a file header, one `static uint8_t arena[...]` sized for the max, zero per-shape
recompile.

---

## 7. Bottom-line recipe

1. Define a packed `ModelCfg {magic, B, S, D, H, F, L, V, quant}` header; store weights
   after it in flash at offsets computed from `cfg`.
2. Write all kernels as runtime-bound loops (matmul / qkv / attention / FFN / norms).
3. Provide `needed = workspace_bytes(cfg)` (sum the formulas above, apply MCUFormer-style
   reuse for FC layers).
4. At boot: read `cfg`, check `needed <= ARENA_MAX` and `weights <= FLASH_MAX`, else
   fail cleanly.
5. Carve `workspace` from one static arena (bump allocator; reset per inference).
6. Use int8/int16 arithmetic (no FPU on C3); optionally drop in CMSIS-NN per-kernel
   intrinsics for the matmuls.
7. One firmware image now runs any `(B,S,D,H,F,L)` that fits your caps — exactly the
   "no per-shape #define rebuild" target.

---

## 8. Sources

- TFLM memory_management.md (GitHub): https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/docs/memory_management.md
- TFLM deepwiki memory management: https://deepwiki.com/tensorflow/tflite-micro/3.2-memory-management
- TFLM deepwiki operator resolution: https://deepwiki.com/tensorflow/tflite-micro/3.3-operator-resolution
- microTVM design doc: https://octoml.github.io/relax-site/arch/microtvm_design.html
- Inferensys static memory allocation glossary: https://inferensys.com/glossary/tiny-machine-learning-deployment/microcontroller-inference-optimization/static-memory-allocation
- TinyFormer arXiv: https://arxiv.org/abs/2311.01759
- MCUFormer arXiv HTML: https://arxiv.org/html/2310.16898v3
- llama2.c run.c: https://github.com/karpathy/llama2.c/blob/master/run.c
- GGUF spec (ggml): https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp: https://github.com/ggml-org/llama.cpp
- gemma.cpp: https://github.com/google/gemma.cpp
- cONNXr: https://github.com/alrevuelta/cONNXr
- tinyml-rs: https://github.com/CanReader/tinyml-rs
- ESP32-C3 product page: https://www.espressif.com/en/products/socs/esp32-c3
- Memory-efficient transformer benchmarking blog: https://martinuke0.github.io/posts/2026-03-26-benchmarking-memoryefficient-transformer-architectures-for-realtime-inference-on-embedded-systems/
- SynapticOS (inference-first allocator paper): https://arxiv.org/html/2607.12606v1
