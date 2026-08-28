[![](https://www.antsand.com/uploads/b2124321cd9c9709061d9d1c112bc604/172222719d1cd65bb80573f36630fb75.svg)](https://www.shivasnotes.com/)

*   [About](https://www.shivasnotes.com/about)
    
*   [Projects](https://www.shivasnotes.com/projects)
    
*   [Support](https://www.shivasnotes.com/support)
    
*   [Search](https://www.shivasnotes.com/search)
    
*   [Work With Us](https://www.shivasnotes.com/work-with-us)
    

![Flash Attention on CPU: Online Softmax, Cache Discipline, and C-Kernel-Engine](https://www.antsand.com/uploads/b2124321cd9c9709061d9d1c112bc604/ed14ba6db95b2988b5972989aef254f9image.jpeg)

[attention](https://www.shivasnotes.com/tag/attention)

Flash Attention on CPU: Online Softmax, Cache Discipline, and C-Kernel-Engine
=============================================================================

21st Jun 2026 • By Anthony Shivakumar

[attention](https://www.shivasnotes.com/tag/attention)
[avx512](https://www.shivasnotes.com/tag/avx512)
[c-kernel-engine](https://www.shivasnotes.com/tag/c-kernel-engine)
[cpu-kernels](https://www.shivasnotes.com/tag/cpu-kernels)
[flash-attention](https://www.shivasnotes.com/tag/flash-attention)
[online-softmax](https://www.shivasnotes.com/tag/online-softmax)
[simd](https://www.shivasnotes.com/tag/simd)

ML fundamentals

This ShivasNotes deep dive is written for CPU silicon teams asking a very specific question: does [C-Kernel-Engine](https://github.com/antshiv/C-Kernel-Engine)
 really own attention on CPU down at the kernel layer? The answer is yes. The implementation spans [flash attention](https://c-kernel-engine.github.io/C-Kernel-Engine/flash_attention.html)
, [the public docs](https://c-kernel-engine.github.io/C-Kernel-Engine/)
, and a full attention surface of 9,187 lines across 10 files, with the video walkthrough companion on [youtube.com/@antshivrobotics](https://www.youtube.com/@antshivrobotics)
.

The thesis of this post is simple: **flash attention is a CPU story too**. On GPU the selling point is avoiding HBM↔SRAM traffic. On CPU the same algorithmic win shows up as cache discipline: do not materialize the full score matrix, keep a running max and running sum, tile the K dimension, and normalize only at the end. For a silicon team, this is the tell: [C-Kernel-Engine](https://github.com/antshiv/C-Kernel-Engine)
 does not treat attention as a call into a mystery library. It exposes the scalar reference, the AVX path, the AVX-512 path, the fast-exp toggle, the tile size policy, and the head-broadcast rule in plain C.

### What this post covers

The opening part frames why online softmax is the core idea behind flash-style attention on CPU and walks the clean scalar reference in `attention_flash_true.c`.

The middle part moves into the SIMD mechanics: custom exp approximations, AVX-512 and AVX decode kernels, and the standalone three-pass softmax that CKE keeps for score-matrix attention paths.

The final part zooms back out to GQA, BF16 and llama.cpp-parity hooks, the 10-file attention surface, and what this kernel stack signals to CPU vendors looking at cache hierarchy, FMA usage, and future AMX or SVE2 opportunities.

Flash Attention Is a CPU Story Too
----------------------------------

Flash attention entered the public conversation as a GPU optimization: do the work in tiles so the score matrix never sloshes back and forth between high-bandwidth memory and on-chip SRAM. But the deeper insight is algorithmic, not vendor-specific. You never actually need to hold all attention scores at once to produce the final weighted sum.

Flash attention is not only a decode trick. The same idea applies to prefill and decode. In prefill, many query tokens attend over many key/value tokens, so the avoided intermediate is the full `T_q × T_k` score matrix. In decode, one new query token scans the existing KV cache, so the avoided intermediate is the `1 × T_k` score row. This post focuses on CKE's decode-oriented flash path because it is the cleanest implementation to inspect line by line, but the algorithmic principle is broader.

On CPU that matters for a different physical reason. The enemy is not only raw DRAM bandwidth but avoidable footprint. A full `T×T` score matrix inflates working-set size, pollutes cache, and forces a clean separation between score generation and softmax normalization. [C-Kernel-Engine's flash path](https://c-kernel-engine.github.io/C-Kernel-Engine/flash_attention.html)
 collapses that into one streaming decode kernel: compute a score, update the online softmax statistics, accumulate the weighted value vector, and move on.

The public kernel surface is large enough that this is clearly not a wrapper. Attention plus softmax in [the C-Kernel-Engine repository](https://github.com/antshiv/C-Kernel-Engine)
 is 9,187 lines across 10 files, with `attention_flash_true.c` acting as the cleanest teaching specimen: one scalar reference, one AVX-512 path, one AVX path, one dispatcher, one custom fast-exp family, and one online-softmax contract.

[The SIMD deep dive](https://www.shivasnotes.com/blog/5911)
 established the x86 SIMD ladder. [The ARM NEON post](https://www.shivasnotes.com/blog/5912)
 showed that NEON belongs in the same conversation. [The quantization deep dive](https://www.shivasnotes.com/blog/5913)
 showed that quantization is a real kernel surface, not a marketing checkbox. This post adds the attention layer: the most numerically delicate, bandwidth-sensitive, and latency-critical part of the LLM runtime. 9,187 linesThe flash-style core file alone is 741 lines, and the total attention plus softmax surface reaches 9,187 lines across scalar, AVX, AVX-512, fused decode, sliding-window, oracle, and quantized variants.

![Flash attention reframed for CPU: avoid the full score matrix, keep state in cache, and stream K and V through online softmax.](https://www.antsand.com/uploads/b2124321cd9c9709061d9d1c112bc604/ed14ba6db95b2988b5972989aef254f9image.jpeg)attention\_flash\_true.c header comment — CKE states the CPU flash-attention contract upfrontcCopy

    /**
     * @file attention_flash_true.c
     * @brief Flash-style attention (online softmax, causal, streaming)
     *
     * CK-ENGINE KERNEL RULES:
     * =======================
     * 1. NO malloc/free - memory via bump allocator, pointers passed in
     * 2. NO OpenMP - parallelization at orchestrator/codegen layer
     * 3. API must define: inputs, outputs, workspace, and memory layouts
     * 4. Pure computation - deterministic, no side effects
     *
     * After changes: make test && make llamacpp-parity-full
     *
     * Layout:
     *   Q/K/V/Out: [T, H, D_h] contiguous
     *
     * Causal alignment:
     *   Queries are assumed to correspond to the last T_q positions in the KV cache.
     *   This makes T_q == T_k behave like standard causal prefill, and T_q == 1
     *   behave like decode over a full KV cache.
     *
     * Notes:
     * - This is O(T_k) per query head; it avoids materializing the score matrix.
     * - SIMD paths are provided for AVX-512 and AVX.
     */

Why Standard Attention Is O(T²) Memory
--------------------------------------

The textbook implementation of causal attention is easy to explain because it materializes the intermediate scores explicitly. First compute `S = Q·Kᵀ / √d`. Then run row-wise softmax over `S`. Then multiply the normalized weights by `V`. That decomposition is mathematically clean, but it means the score matrix exists as a real object in memory.

For `T = 4096`, one head’s score matrix has `4096 × 4096 = 16,777,216` FP32 values. That is about `67 MB` per head, just for the scores. Multiply that by many heads and the memory traffic story becomes obvious very quickly.

CKE keeps that materialized path because prefill sometimes wants it. But the flash path exists precisely because decode does not need it. When `T_q = 1`, the kernel can walk the entire KV cache once, update online softmax statistics, and finish with only O(`T_k`) streaming work per query head and O(`D_h`) state.

Fullscreen table

| Sequence length | Score elements per head | FP32 bytes | Why it hurts on CPU |
| --- | --- | --- | --- |
| 1024 | 1,048,576 | ≈ 4 MB | Already bigger than the “small temporary” most hot loops want. |
| 2048 | 4,194,304 | ≈ 16 MB | Now the score matrix competes with everything else for LLC space. |
| 4096 | 16,777,216 | ≈ 67 MB | The matrix becomes a bandwidth and footprint problem, not a harmless intermediate. |
| 8192 | 67,108,864 | ≈ 268 MB | Materialization starts looking absurd for single-token decode. |

The flash insight is not “compute less.” It is “store less.” The kernel still sees every relevant key token. What disappears is the requirement that all scores coexist in memory before softmax can proceed. ≈67 MBAt 4096 tokens, the naive per-head score matrix is 16.8 million floats. The more useful number for systems work is the footprint: about 67 MB of FP32 scores that the flash path never allocates.

![Materialized score matrix versus online softmax stream: one path stores T by T scores, the other only keeps running statistics and the output vector.](https://www.antsand.com/uploads/b2124321cd9c9709061d9d1c112bc604/e5d966c549dd86849b298429e9ff7452image.jpeg)attention\_kernels.c — the score-matrix path explicitly documents O(N²) storagecCopy

    /**
     * Causal attention forward (score-matrix version)
     * @test test_attention.py::TestAttentionForward::test_causal_forward
     * @test test_attention.py::TestAttentionForward::test_gqa_broadcast
     * @test test_attention.py::TestAttentionForward::test_exact_vs_fast
     * @test test_parity.py::test_attention_parity
     *
     * Computes softmax(Q @ K^T / sqrt(d)) @ V with causal masking.
     * Uses O(N^2) memory for scores matrix.
     *
     * After changes: make test && make llamacpp-parity-full
     */
    void attention_forward_causal_head_major(const float *q,
                                             const float *k,
                                             const float *v,
                                             float *scores,
                                             float *output,
                                             int num_heads,
                                             int num_tokens,
                                             int head_dim,
                                             int aligned_head_dim,
                                             int aligned_context_window)

attention\_kernels.c — materialized attention calls row-wise softmax over the stored scorescCopy

        // Phase 2: apply causal row-wise softmax in-place over j <= i.
        causal_softmax_head_major(scores,
                                  num_heads,
                                  num_tokens,
                                  aligned_context_window);
    
        // Phase 3: attention weights · V.
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < num_tokens; ++i) {
                size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
    
                // Zero the full aligned head slice so padded dims stay clean.
                for (int d = 0; d < aligned_head_dim; ++d) {
                    output[out_base + d] = 0.0f;
                }
    
                // Weighted sum over causal positions.
                for (int j = 0; j <= i; ++j) {
                    float w = scores[score_index(h, i, j, aligned_context_window)];
                    size_t v_base = qkv_index(h, j, 0, num_tokens, aligned_head_dim);
    
                    for (int d = 0; d < head_dim; ++d) {
                        output[out_base + d] += w * v[v_base + d];
                    }
                }
            }
        }
    }

Online Softmax — The Mathematical Foundation
--------------------------------------------

Standard softmax is typically taught as a two-pass procedure. First find the row maximum `m`. Then compute `exp(xᵢ - m)`, sum those exponentials, and divide each numerator by the shared denominator. The maximum is there for numerical stability.

The Milakov & Gimelshein trick is that `m` does not have to be known in advance. It can be updated online. If a new block maximum `m′` exceeds the running maximum `m`, the old partial numerator and denominator can simply be rescaled by `exp(m - m′)`. Because `m′ ≥ m`, that multiplier is always in `(0, 1]`. No overflow disaster. No need to restart.

That is exactly what CKE does. The kernel processes K tokens in tiles, computes all scores in that tile, finds the tile maximum, rescales the running state if needed, accumulates weighted `V`, and normalizes once at the end. The final answer is identical in form to ordinary softmax. The difference is that the kernel never needed the whole row at once.

For hardware people, online softmax is the important bridge between math and locality. It converts a global reduction problem into a streaming reduction with a tiny state vector: one running max, one running sum, and one running output accumulator per query head. The rescaling step is the heart of flash attention. When the maximum changes, CKE does not throw away prior work. It shrinks the old work by exp(old\_max − new\_max), then keeps going.

![Online softmax rescaling: when a new block maximum appears, the prior partial sum and output are multiplied by exp(old max minus new max).](https://www.antsand.com/uploads/b2124321cd9c9709061d9d1c112bc604/640d7002fa1c562dbdbf22ca7b8931c1image.jpeg)attention\_flash\_true.c — causal alignment helper for the last T\_q positions in the KV cachecCopy

    static inline int max_k_for_query(int t_q, int T_q, int T_k) {
        int q_pos_offset = (T_k > T_q) ? (T_k - T_q) : 0;
        int max_k = q_pos_offset + t_q;
        if (max_k >= T_k) {
            max_k = T_k - 1;
        }
        return max_k;

attention\_flash\_true.c — the scalar kernel rescales the running sum and running output when m\_block exceeds mcCopy

                if (m_block > m) {
                    float scale_old = (m == -INFINITY) ? 0.0f : ck_expf(m - m_block);
                    s *= scale_old;
                    for (int d = 0; d < D_h; ++d) {
                        out_head[d] *= scale_old;
                    }
                    m = m_block;
                }

attention\_flash\_true.c — after rescaling, each tile contributes new weights and weighted V vectorscCopy

                for (int bi = 0; bi < blk_len; ++bi) {
                    const int t_k = t_k0 + bi;
                    const float *v_head = v_base + (size_t)t_k * stride;
                    float w = ck_expf(scores[bi] - m);
                    s += w;
                    for (int d = 0; d < D_h; ++d) {
                        out_head[d] += w * v_head[d];
                    }
                }
            }

attention\_flash\_true.c — final normalization happens once, after all tiles are consumedcCopy

            if (s > 0.0f) {
                float inv_s = 1.0f / s;
                for (int d = 0; d < D_h; ++d) {
                    out_head[d] *= inv_s;
                }
            } else {
                for (int d = 0; d < D_h; ++d) {
                    out_head[d] = 0.0f;
                }
            }
        }

CKE's Flash Attention — The Scalar Reference
--------------------------------------------

The best part of `attention_flash_true.c` is that the scalar reference is clean enough to teach from directly. No macro maze. No allocator noise. No threading policy. Just the core decode algorithm over one contiguous `[T, H, D_h]` layout.

The function loops over `idx = t_q × H + h`, derives the query position and head index, computes the causal limit with `max_k_for_query()`, zeros the output head, and initializes the two online-softmax scalars: `m = -INFINITY` and `s = 0`.

Then the tiled loop begins. For each block of keys, the kernel computes dot products `Q·K`, scales them, caches those scores in a small stack array, records the tile maximum, performs the online rescale if necessary, and finally streams through the matching `V` vectors. At the end, one reciprocal finishes the job.

![Scalar flash-attention decode on CPU: query head selection, tiled K scan, online softmax rescaling, V accumulation, and final normalization.](https://www.antsand.com/uploads/b2124321cd9c9709061d9d1c112bc604/7f2cfee5767c6439ad0b8e502eb783faimage.jpeg)attention\_flash\_true.c — scalar flash decode setup, layout, and per-head initializationcCopy

    static void attention_flash_decode_scalar(
        float *out,
        const float *q,
        const float *k,
        const float *v,
        int T_q,
        int T_k,
        int H,
        int D_h,
        float scale)
    {
        const int total = T_q * H;
        const size_t stride = (size_t)H * (size_t)D_h;
        const int tile_k = ck_flash_attn_tile_k(D_h);
    
        for (int idx = 0; idx < total; ++idx) {
            const int t_q = idx / H;
            const int h = idx - t_q * H;
            const int max_k = max_k_for_query(t_q, T_q, T_k);
    
            const float *q_head = q + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
            float *out_head = out + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
            const float *k_base = k + (size_t)h * (size_t)D_h;
            const float *v_base = v + (size_t)h * (size_t)D_h;
    
            for (int d = 0; d < D_h; ++d) {
                out_head[d] = 0.0f;
            }
    
            float m = -INFINITY;
            float s = 0.0f;
    
            float scores[CK_FLASH_ATTN_TILE_K];

attention\_flash\_true.c — scalar tile loop computes dot products and tracks the tile maximumcCopy

            for (int t_k0 = 0; t_k0 <= max_k; t_k0 += tile_k) {
                int blk_len = max_k - t_k0 + 1;
                if (blk_len > tile_k) {
                    blk_len = tile_k;
                }
    
                float m_block = -INFINITY;
                for (int bi = 0; bi < blk_len; ++bi) {
                    const int t_k = t_k0 + bi;
                    const float *k_head = k_base + (size_t)t_k * stride;
    
                    float dot = 0.0f;
                    for (int d = 0; d < D_h; ++d) {
                        dot += q_head[d] * k_head[d];
                    }
    
                    float score = dot * scale;
                    scores[bi] = score;
                    if (score > m_block) {
                        m_block = score;
                    }
                }

attention\_flash\_true.c — scalar rescaling block keeps previous partial work numerically validcCopy

                if (m_block > m) {
                    float scale_old = (m == -INFINITY) ? 0.0f : ck_expf(m - m_block);
                    s *= scale_old;
                    for (int d = 0; d < D_h; ++d) {
                        out_head[d] *= scale_old;
                    }
                    m = m_block;
                }

attention\_flash\_true.c — scalar accumulation and final normalizationcCopy

                for (int bi = 0; bi < blk_len; ++bi) {
                    const int t_k = t_k0 + bi;
                    const float *v_head = v_base + (size_t)t_k * stride;
                    float w = ck_expf(scores[bi] - m);
                    s += w;
                    for (int d = 0; d < D_h; ++d) {
                        out_head[d] += w * v_head[d];
                    }
                }
            }
    
            if (s > 0.0f) {
                float inv_s = 1.0f / s;
                for (int d = 0; d < D_h; ++d) {
                    out_head[d] *= inv_s;
                }
            } else {
                for (int d = 0; d < D_h; ++d) {
                    out_head[d] = 0.0f;
                }
            }
        }

This is the reference every SIMD path has to preserve. If an AVX or AVX-512 optimization cannot be explained as “the scalar loop, widened and reduced more efficiently,” it is probably too clever for its own good. The scalar reference is not a fallback to be ignored. It is the semantic contract that lets the wider ISA tiers stay auditable.

Tile-K — Why Blocking the Key Dimension Matters
-----------------------------------------------

First, a naming clarification. In normal GEMM notation, `[M × N] · [N × K] = [M × K]`, the letter `K` usually names the output-column dimension. In attention, `K` also means the Key tensor. Those two meanings collide. For attention scores, `scores = Q · Kᵀ`, so the Key tensor becomes the second matrix after transpose: `Q [T_q × D_h] · Kᵀ [D_h × T_k] = scores [T_q × T_k]`. In this post, `tile_k` means blocking that `T_k` axis: the output-column dimension of the score matrix, which corresponds to key-token positions in the KV cache.

So from the computer's point of view, this is ordinary matrix blocking. The semantic label is “Key,” but the kernel is really keeping the query vector, a block of key vectors, the matching value vectors, and the online-softmax state hot while it walks the second-matrix/output-column side of `Q · Kᵀ`.

CKE does not scan keys one token at a time unless the head dimension forces it to. Instead it chooses a tile size for the K dimension. The compile-time default is `CK_FLASH_ATTN_TILE_K = 32`, but the helper shrinks that when `D_h` gets large so that the tile’s working set stays cache-friendly.

The logic is simple. Head dimensions above 128 drop the tile to 8. Above 64 drop to 16. Otherwise stay at 32. The code also clamps the result so it never exceeds the compile-time ceiling and never drops below a safe minimum when that ceiling is at least 8.

That small detail matters because the score buffer is stack-allocated as `float scores[CK_FLASH_ATTN_TILE_K]`. No heap allocation. No scratch planner. Just a short-lived tile buffer that keeps the scores close while the kernel decides whether the running max needs to change.

The trade-off is classic blocking theory. Bigger tiles mean fewer rescaling events and better amortization of control overhead. Smaller tiles mean less cache pressure and less transient state. CKE chooses a simple, explicit heuristic rather than pretending one tile size is perfect for all head widths. 32 → 16 → 8Default tile size is 32, but large heads push the effective tile down to 16 or 8 so the per-tile footprint stays friendly to L1 and L2 behavior.

![Adaptive K-dimension blocking in flash attention: smaller tiles for larger head dimensions to keep the working set hot in CPU caches.](https://www.antsand.com/uploads/b2124321cd9c9709061d9d1c112bc604/d6dbb06b5d470509fd08c02eb7ad39e2image.jpeg)attention\_flash\_true.c — adaptive tile\_k policycCopy

    static inline int ck_flash_attn_tile_k(int D_h) {
        int tile = CK_FLASH_ATTN_TILE_K;
        if (D_h > 128) {
            tile = CK_FLASH_ATTN_TILE_K / 4;
        } else if (D_h > 64) {
            tile = CK_FLASH_ATTN_TILE_K / 2;
        }
    
        if (CK_FLASH_ATTN_TILE_K >= 8 && tile < 8) {
            tile = 8;
        }
        if (tile > CK_FLASH_ATTN_TILE_K) {
            tile = CK_FLASH_ATTN_TILE_K;
        }
        if (tile < 1) {
            tile = 1;
        }
        return tile;
    }
    
    int ck_flash_attn_choose_tile_k(int D_h) {
        return ck_flash_attn_tile_k(D_h);
    }

attention\_flash\_true.c — the tile-local score array lives on the stackcCopy

            float m = -INFINITY;
            float s = 0.0f;
    
            float scores[CK_FLASH_ATTN_TILE_K];
    
            for (int t_k0 = 0; t_k0 <= max_k; t_k0 += tile_k) {
                int blk_len = max_k - t_k0 + 1;
                if (blk_len > tile_k) {
                    blk_len = tile_k;
                }

attention\_flash\_true.c — the header comment states the O(T\_k) streaming goal directlycCopy

     * Layout:
     *   Q/K/V/Out: [T, H, D_h] contiguous
     *
     * Causal alignment:
     *   Queries are assumed to correspond to the last T_q positions in the KV cache.
     *   This makes T_q == T_k behave like standard causal prefill, and T_q == 1
     *   behave like decode over a full KV cache.
     *
     * Notes:
     * - This is O(T_k) per query head; it avoids materializing the score matrix.
     * - SIMD paths are provided for AVX-512 and AVX.

The Fast Exp Approximation
--------------------------

Softmax lives or dies on exponentials. CKE therefore keeps a custom fast approximation for the flash path, toggled by the compile-time flag `CK_FLASH_ATTN_FAST_EXP`. When the flag is off, the kernel uses `expf()` for stricter parity. When it is on, the kernel uses a Schraudolph-style formulation built around `x · log₂(e)`, integer-and-fraction splitting, a 4th-degree polynomial, and direct IEEE-754 exponent construction.

The scalar helper is small and self-contained: clamp the argument to roughly `[-88, 88]`, convert to base-2 space, approximate `2^f` with the polynomial, then synthesize `2^n` by shifting the exponent bits into place. The vectorized AVX-512 and AVX versions do the same thing lane-wise.

This is one of those places where CPU implementation quality becomes visible immediately. A runtime that really owns softmax on CPU does not just say “fast exp.” It shows the approximation, the coefficients, the fallback rule, and the lane-width-specific mechanics.

attention\_flash\_true.c — scalar ck\_fast\_expf()cCopy

    static inline float ck_fast_expf(float x) {
        const float max_val = 88.0f;
        const float min_val = -88.0f;
        if (x > max_val) {
            x = max_val;
        } else if (x < min_val) {
            x = min_val;
        }
    
        const float log2e = 1.4426950408889634f;
        float z = x * log2e;
        float zf = nearbyintf(z);
        float f = z - zf;
    
        const float c0 = 1.0f;
        const float c1 = 0.6931471805599453f;
        const float c2 = 0.2402265069591007f;
        const float c3 = 0.05550410866482158f;
        const float c4 = 0.009618129107628478f;
    
        float poly = ((c4 * f + c3) * f + c2) * f + c1;
        poly = poly * f + c0;
    
        int32_t zi = (int32_t)zf + 127;
        uint32_t bits = (uint32_t)zi << 23;
        union {
            uint32_t i;
            float f;
        } u;
        u.i = bits;
        return poly * u.f;
    }

attention\_flash\_true.c — the compile-time switch between fast exp and libm expf()cCopy

    static inline float ck_expf(float x) {
    #if CK_FLASH_ATTN_FAST_EXP
        return ck_fast_expf(x);
    #else
        return expf(x);
    #endif
    }

attention\_flash\_true.c — AVX-512 vector fast exp helpercCopy

    #if CK_FLASH_ATTN_FAST_EXP
    static inline __m512 ck_fast_exp512_ps(__m512 x) {
        const __m512 max_val = _mm512_set1_ps(88.0f);
        const __m512 min_val = _mm512_set1_ps(-88.0f);
        x = _mm512_min_ps(x, max_val);
        x = _mm512_max_ps(x, min_val);
    
        const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
        __m512 z = _mm512_mul_ps(x, log2e);
        __m512 zf = _mm512_roundscale_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m512 f = _mm512_sub_ps(z, zf);
    
        const __m512 c0 = _mm512_set1_ps(1.0f);
        const __m512 c1 = _mm512_set1_ps(0.6931471805599453f);
        const __m512 c2 = _mm512_set1_ps(0.2402265069591007f);
        const __m512 c3 = _mm512_set1_ps(0.05550410866482158f);
        const __m512 c4 = _mm512_set1_ps(0.009618129107628478f);
    
    #if defined(__FMA__)
        __m512 poly = _mm512_fmadd_ps(c4, f, c3);
        poly = _mm512_fmadd_ps(poly, f, c2);
        poly = _mm512_fmadd_ps(poly, f, c1);
        poly = _mm512_fmadd_ps(poly, f, c0);
    #else
        __m512 poly = _mm512_add_ps(_mm512_mul_ps(c4, f), c3);
        poly = _mm512_add_ps(_mm512_mul_ps(poly, f), c2);
        poly = _mm512_add_ps(_mm512_mul_ps(poly, f), c1);
        poly = _mm512_add_ps(_mm512_mul_ps(poly, f), c0);
    #endif
    
        __m512i zi = _mm512_cvtps_epi32(zf);
        zi = _mm512_add_epi32(zi, _mm512_set1_epi32(127));
        zi = _mm512_slli_epi32(zi, 23);
        __m512 pow2 = _mm512_castsi512_ps(zi);
        return _mm512_mul_ps(poly, pow2);
    }

attention\_flash\_true.c — AVX helper builds 2^n by splitting the 256-bit vector into 128-bit halvescCopy

    static inline __m256 ck_pow2_256_ps(__m256 zf) {
        __m128 z0 = _mm256_castps256_ps128(zf);
        __m128 z1 = _mm256_extractf128_ps(zf, 1);
    
        __m128i i0 = _mm_cvtps_epi32(z0);
        __m128i i1 = _mm_cvtps_epi32(z1);
        i0 = _mm_add_epi32(i0, _mm_set1_epi32(127));
        i1 = _mm_add_epi32(i1, _mm_set1_epi32(127));
        i0 = _mm_slli_epi32(i0, 23);
        i1 = _mm_slli_epi32(i1, 23);
    
        __m128 f0 = _mm_castsi128_ps(i0);
        __m128 f1 = _mm_castsi128_ps(i1);
        __m256 out = _mm256_castps128_ps256(f0);
        return _mm256_insertf128_ps(out, f1, 1);
    }

attention\_flash\_true.c — AVX vector fast exp helper with optional FMAcCopy

    static inline __m256 ck_fast_exp256_ps(__m256 x) {
        const __m256 max_val = _mm256_set1_ps(88.0f);
        const __m256 min_val = _mm256_set1_ps(-88.0f);
        x = _mm256_min_ps(x, max_val);
        x = _mm256_max_ps(x, min_val);
    
        const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
        __m256 z = _mm256_mul_ps(x, log2e);
        __m256 zf = _mm256_round_ps(z, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        __m256 f = _mm256_sub_ps(z, zf);
    
        const __m256 c0 = _mm256_set1_ps(1.0f);
        const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
        const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
        const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
        const __m256 c4 = _mm256_set1_ps(0.009618129107628478f);
    
    #if defined(__FMA__)
        __m256 poly = _mm256_fmadd_ps(c4, f, c3);
        poly = _mm256_fmadd_ps(poly, f, c2);
        poly = _mm256_fmadd_ps(poly, f, c1);
        poly = _mm256_fmadd_ps(poly, f, c0);
    #else
        __m256 poly = _mm256_add_ps(_mm256_mul_ps(c4, f), c3);
        poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c2);
        poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c1);
        poly = _mm256_add_ps(_mm256_mul_ps(poly, f), c0);
    #endif
    
        __m256 pow2 = ck_pow2_256_ps(zf);
        return _mm256_mul_ps(poly, pow2);
    }

The accuracy target here is practical rather than ceremonial. Softmax cares about ratios. Relative error around `1e-4` is typically fine, especially when the fallback exists for strict parity testing. ~1e-4softmax\_kernels.c describes the approximation as “good for softmax, ~1e-4 relative error.” That is the right engineering language: quantify the error and state the intended use.

AVX-512 Flash Attention — 16 Lanes Wide
---------------------------------------

The AVX-512 decode path keeps the scalar algorithm intact and widens the obvious pieces. Output zeroing becomes 16-float vector stores. Dot products use a dual-accumulator helper that consumes 32 floats per loop with two FMAs. Rescaling the running output becomes a vector multiply. Weighted `V` accumulation becomes `_mm512_fmadd_ps`. Final normalization is another vector multiply by `1/s`.

CKE also vectorizes the fast-exp phase when the compile-time flag is enabled. Scores for a tile are first written into the temporary stack buffer, then a 16-lane exp helper transforms those centered scores in place. The result is still the same online-softmax algorithm; only the width changes.

The helper `ck_dot_f32_avx512()` is worth staring at because it captures the spirit of the whole file. It is not an exotic trick. It is just the scalar inner product rewritten so the core executes 32 FP32 multiplies-and-adds per loop body with a clean horizontal reduction at the end.

![AVX-512 flash-attention pipeline: 16-lane zeroing, 32-float dual-accumulator dot products, vector rescale, FMA accumulation, and vector normalization.](https://www.antsand.com/uploads/b2124321cd9c9709061d9d1c112bc604/898e4ba0b985c0a8d3ccbca16f59ecf7image.jpeg)attention\_flash\_true.c — AVX-512 dual-accumulator dot-product helpercCopy

    static inline float ck_dot_f32_avx512(const float *q, const float *k, int D_h) {
        __m512 sum0 = _mm512_setzero_ps();
        __m512 sum1 = _mm512_setzero_ps();
    
        int d = 0;
        for (; d + 32 <= D_h; d += 32) {
            __m512 q0 = _mm512_loadu_ps(q + d);
            __m512 k0 = _mm512_loadu_ps(k + d);
            __m512 q1 = _mm512_loadu_ps(q + d + 16);
            __m512 k1 = _mm512_loadu_ps(k + d + 16);
            sum0 = _mm512_fmadd_ps(q0, k0, sum0);
            sum1 = _mm512_fmadd_ps(q1, k1, sum1);
        }
        for (; d + 16 <= D_h; d += 16) {
            __m512 q0 = _mm512_loadu_ps(q + d);
            __m512 k0 = _mm512_loadu_ps(k + d);
            sum0 = _mm512_fmadd_ps(q0, k0, sum0);
        }
    
        sum0 = _mm512_add_ps(sum0, sum1);
        float dot = _mm512_reduce_add_ps(sum0);
        for (; d < D_h; ++d) {
            dot += q[d] * k[d];
        }
        return dot;

attention\_flash\_true.c — AVX-512 decode setup and tile scoringcCopy

    static void attention_flash_decode_avx512(
        float *out,
        const float *q,
        const float *k,
        const float *v,
        int T_q,
        int T_k,
        int H,
        int D_h,
        float scale)
    {
        const int total = T_q * H;
        const size_t stride = (size_t)H * (size_t)D_h;
        const int tile_k = ck_flash_attn_tile_k(D_h);
    
        for (int idx = 0; idx < total; ++idx) {
            const int t_q = idx / H;
            const int h = idx - t_q * H;
            const int max_k = max_k_for_query(t_q, T_q, T_k);
    
            const float *q_head = q + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
            float *out_head = out + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
            const float *k_base = k + (size_t)h * (size_t)D_h;
            const float *v_base = v + (size_t)h * (size_t)D_h;
    
            int d = 0;
            for (; d + 16 <= D_h; d += 16) {
                _mm512_storeu_ps(out_head + d, _mm512_setzero_ps());
            }
            for (; d < D_h; ++d) {
                out_head[d] = 0.0f;
            }
    
            float m = -INFINITY;
            float s = 0.0f;
    
            float scores[CK_FLASH_ATTN_TILE_K];
    
            for (int t_k0 = 0; t_k0 <= max_k; t_k0 += tile_k) {
                int blk_len = max_k - t_k0 + 1;
                if (blk_len > tile_k) {
                    blk_len = tile_k;
                }
    
                float m_block = -INFINITY;
                for (int bi = 0; bi < blk_len; ++bi) {
                    const int t_k = t_k0 + bi;
                    const float *k_head = k_base + (size_t)t_k * stride;
    
                    float dot = ck_dot_f32_avx512(q_head, k_head, D_h);
    
                    float score = dot * scale;
                    scores[bi] = score;
                    if (score > m_block) {
                        m_block = score;
                    }
                }

attention\_flash\_true.c — AVX-512 output rescaling and vector fast-exp pathcCopy

                if (m_block > m) {
                    float scale_old = (m == -INFINITY) ? 0.0f : ck_expf(m - m_block);
                    s *= scale_old;
                    __m512 scale_old_vec = _mm512_set1_ps(scale_old);
                    d = 0;
                    for (; d + 16 <= D_h; d += 16) {
                        __m512 out_v = _mm512_loadu_ps(out_head + d);
                        _mm512_storeu_ps(out_head + d, _mm512_mul_ps(out_v, scale_old_vec));
                    }
                    for (; d < D_h; ++d) {
                        out_head[d] *= scale_old;
                    }
                    m = m_block;
                }
    
    #if CK_FLASH_ATTN_FAST_EXP
                int bi_vec = 0;
                __m512 m_vec = _mm512_set1_ps(m);
                for (; bi_vec + 16 <= blk_len; bi_vec += 16) {
                    __m512 s_vec = _mm512_loadu_ps(scores + bi_vec);
                    s_vec = _mm512_sub_ps(s_vec, m_vec);
                    __m512 w_vec = ck_fast_exp512_ps(s_vec);
                    _mm512_storeu_ps(scores + bi_vec, w_vec);
                }
                for (; bi_vec < blk_len; ++bi_vec) {
                    scores[bi_vec] = ck_fast_expf(scores[bi_vec] - m);
                }

attention\_flash\_true.c — AVX-512 weighted V accumulation and final normalizationcCopy

                for (int bi = 0; bi < blk_len; ++bi) {
                    const int t_k = t_k0 + bi;
                    const float *v_head = v_base + (size_t)t_k * stride;
    #if CK_FLASH_ATTN_FAST_EXP
                    float w = scores[bi];
    #else
                    float w = ck_expf(scores[bi] - m);
    #endif
                    s += w;
    
                    __m512 w_vec = _mm512_set1_ps(w);
                    d = 0;
                    for (; d + 16 <= D_h; d += 16) {
                        __m512 out_v = _mm512_loadu_ps(out_head + d);
                        __m512 v_v = _mm512_loadu_ps(v_head + d);
                        out_v = _mm512_fmadd_ps(w_vec, v_v, out_v);
                        _mm512_storeu_ps(out_head + d, out_v);
                    }
                    for (; d < D_h; ++d) {
                        out_head[d] += w * v_head[d];
                    }
                }
            }
    
            if (s > 0.0f) {
                float inv_s = 1.0f / s;
                __m512 inv_s_vec = _mm512_set1_ps(inv_s);
                d = 0;
                for (; d + 16 <= D_h; d += 16) {
                    __m512 out_v = _mm512_loadu_ps(out_head + d);
                    _mm512_storeu_ps(out_head + d, _mm512_mul_ps(out_v, inv_s_vec));
                }
                for (; d < D_h; ++d) {
                    out_head[d] *= inv_s;
                }
            } else {
                for (int d0 = 0; d0 < D_h; ++d0) {
                    out_head[d0] = 0.0f;
                }
            }
        }
    }

For x86 vendors this is the exact kind of kernel that turns AVX-512 from a bullet point into a systems argument. Wider registers help, but the real story is the combination of width, built-in reductions, and FMA density inside a numerically stable streaming algorithm. The AVX-512 path is not “different attention.” It is the scalar algorithm with the memory traffic pattern preserved and the arithmetic widened to 16 FP32 lanes.

AVX Flash Attention — 8 Lanes, Same Algorithm
---------------------------------------------

The AVX implementation is structurally identical to the AVX-512 path, just at half the width. It still zeros the output head with vectors, still walks K in tiles, still rescales the running output when the max changes, still broadcasts `w` into a vector for the `V` accumulation, and still finishes with one reciprocal plus a vector multiply.

The interesting details are x86-generation specific. AVX1 does not have the same integer convenience as AVX2, so the power-of-two helper has to split the 256-bit register into two 128-bit halves. And the accumulation block explicitly guards on `__FMA__`, using fused multiply-add when the machine supports it and the old multiply-then-add sequence otherwise.

That is exactly the right kind of portability for CPU kernels. The algorithm does not fork. The available instructions do.

attention\_flash\_true.c — AVX horizontal sum and dot-product helpercCopy

    static inline float hsum256_ps(__m256 v) {
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 sum128 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(sum128);
        __m128 sums = _mm_add_ps(sum128, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ps(shuf, sums);
        return _mm_cvtss_f32(sums);
    }
    
    static inline float ck_dot_f32_avx(const float *q, const float *k, int D_h) {
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
    
        int d = 0;
        for (; d + 16 <= D_h; d += 16) {
            __m256 q0 = _mm256_loadu_ps(q + d);
            __m256 k0 = _mm256_loadu_ps(k + d);
            __m256 q1 = _mm256_loadu_ps(q + d + 8);
            __m256 k1 = _mm256_loadu_ps(k + d + 8);
        #if defined(__FMA__)
            sum0 = _mm256_fmadd_ps(q0, k0, sum0);
            sum1 = _mm256_fmadd_ps(q1, k1, sum1);
        #else
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(q0, k0));
            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(q1, k1));
        #endif
        }
        for (; d + 8 <= D_h; d += 8) {
            __m256 q0 = _mm256_loadu_ps(q + d);
            __m256 k0 = _mm256_loadu_ps(k + d);
        #if defined(__FMA__)
            sum0 = _mm256_fmadd_ps(q0, k0, sum0);
        #else
            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(q0, k0));
        #endif
        }
    
        __m256 sum = _mm256_add_ps(sum0, sum1);
        float dot = hsum256_ps(sum);
        for (; d < D_h; ++d) {
            dot += q[d] * k[d];
        }
        return dot;

attention\_flash\_true.c — AVX decode setup and tile scoringcCopy

    static void attention_flash_decode_avx(
        float *out,
        const float *q,
        const float *k,
        const float *v,
        int T_q,
        int T_k,
        int H,
        int D_h,
        float scale)
    {
        const int total = T_q * H;
        const size_t stride = (size_t)H * (size_t)D_h;
        const int tile_k = ck_flash_attn_tile_k(D_h);
    
        for (int idx = 0; idx < total; ++idx) {
            const int t_q = idx / H;
            const int h = idx - t_q * H;
            const int max_k = max_k_for_query(t_q, T_q, T_k);
    
            const float *q_head = q + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
            float *out_head = out + (size_t)t_q * stride + (size_t)h * (size_t)D_h;
            const float *k_base = k + (size_t)h * (size_t)D_h;
            const float *v_base = v + (size_t)h * (size_t)D_h;
    
            int d = 0;
            for (; d + 8 <= D_h; d += 8) {
                _mm256_storeu_ps(out_head + d, _mm256_setzero_ps());
            }
            for (; d < D_h; ++d) {
                out_head[d] = 0.0f;
            }
    
            float m = -INFINITY;
            float s = 0.0f;
    
            float scores[CK_FLASH_ATTN_TILE_K];
    
            for (int t_k0 = 0; t_k0 <= max_k; t_k0 += tile_k) {
                int blk_len = max_k - t_k0 + 1;
                if (blk_len > tile_k) {
                    blk_len = tile_k;
                }
    
                float m_block = -INFINITY;
                for (int bi = 0; bi < blk_len; ++bi) {
                    const int t_k = t_k0 + bi;
                    const float *k_head = k_base + (size_t)t_k * stride;
    
                    float dot = ck_dot_f32_avx(q_head, k_head, D_h);
    
                    float score = dot * scale;
                    scores[bi] = score;
                    if (score > m_block) {
                        m_block = score;
                    }
                }

attention\_flash\_true.c — AVX online-softmax rescaling and optional vector fast-expcCopy

                if (m_block > m) {
                    float scale_old = (m == -INFINITY) ? 0.0f : ck_expf(m - m_block);
                    s *= scale_old;
                    __m256 scale_old_vec = _mm256_set1_ps(scale_old);
                    d = 0;
                    for (; d + 8 <= D_h; d += 8) {
                        __m256 out_v = _mm256_loadu_ps(out_head + d);
                        _mm256_storeu_ps(out_head + d, _mm256_mul_ps(out_v, scale_old_vec));
                    }
                    for (; d < D_h; ++d) {
                        out_head[d] *= scale_old;
                    }
                    m = m_block;
                }
    
    #if CK_FLASH_ATTN_FAST_EXP
                int bi_vec = 0;
                __m256 m_vec = _mm256_set1_ps(m);
                for (; bi_vec + 8 <= blk_len; bi_vec += 8) {
                    __m256 s_vec = _mm256_loadu_ps(scores + bi_vec);
                    s_vec = _mm256_sub_ps(s_vec, m_vec);
                    __m256 w_vec = ck_fast_exp256_ps(s_vec);
                    _mm256_storeu_ps(scores + bi_vec, w_vec);
                }
                for (; bi_vec < blk_len; ++bi_vec) {
                    scores[bi_vec] = ck_fast_expf(scores[bi_vec] - m);
                }
    #endif

attention\_flash\_true.c — AVX weighted accumulation uses FMA when availablecCopy

                for (int bi = 0; bi < blk_len; ++bi) {
                    const int t_k = t_k0 + bi;
                    const float *v_head = v_base + (size_t)t_k * stride;
    #if CK_FLASH_ATTN_FAST_EXP
                    float w = scores[bi];
    #else
                    float w = ck_expf(scores[bi] - m);
    #endif
                    s += w;
    
                    __m256 w_vec = _mm256_set1_ps(w);
                    d = 0;
                    for (; d + 8 <= D_h; d += 8) {
                        __m256 out_v = _mm256_loadu_ps(out_head + d);
                        __m256 v_v = _mm256_loadu_ps(v_head + d);
                    #if defined(__FMA__)
                        out_v = _mm256_fmadd_ps(w_vec, v_v, out_v);
                    #else
                        out_v = _mm256_add_ps(out_v, _mm256_mul_ps(w_vec, v_v));
                    #endif
                        _mm256_storeu_ps(out_head + d, out_v);

attention\_flash\_true.c — AVX final normalization and zero fallbackcCopy

            if (s > 0.0f) {
                float inv_s = 1.0f / s;
                __m256 inv_s_vec = _mm256_set1_ps(inv_s);
                d = 0;
                for (; d + 8 <= D_h; d += 8) {
                    __m256 out_v = _mm256_loadu_ps(out_head + d);
                    _mm256_storeu_ps(out_head + d, _mm256_mul_ps(out_v, inv_s_vec));
                }
                for (; d < D_h; ++d) {
                    out_head[d] *= inv_s;
                }
            } else {
                for (int d0 = 0; d0 < D_h; ++d0) {
                    out_head[d0] = 0.0f;
                }
            }
        }
    }

Haswell-class cores and later get the nicer `_mm256_fmadd_ps` form. Earlier AVX machines still get the same kernel topology, just with separate multiply and add instructions. FMA is not a new algorithm. It is the same accumulation pattern with fewer instructions and often better latency-throughput behavior in the hot loop.

Standalone Softmax — The Three-Pass Kernel
------------------------------------------

Flash attention is not the only attention path in CKE. The runtime also carries a standalone softmax kernel for the cases where the score matrix does exist, especially materialized prefill-style paths. That kernel lives in `softmax_kernels.c` and follows the classic three-pass recipe: find the maximum, exponentiate-and-sum, then normalize.

The function `causal_softmax_head_major()` works over a head-major score tensor laid out as `[head][query_token][key_token]`. For each row it only treats positions `0..i` as valid, and then explicitly zeroes the future tokens. AVX-512 gets vector max, vector exp, vector sum reduction, vector normalization, and vector tail zeroing. AVX2 gets the same structure at 8 lanes. AVX1 keeps vector max and vector normalize but uses scalar `expf()`. Scalar remains the exact fallback.

That split is important. Sometimes materializing scores is still the right trade, especially when prefill can lean on larger matrix multiplications upstream. CKE therefore keeps both strategies instead of forcing one attention style onto every regime.

![Standalone causal softmax kernel: max pass, exp-plus-sum pass, normalize pass, and explicit zeroing of future tokens.](https://www.antsand.com/uploads/b2124321cd9c9709061d9d1c112bc604/515fb1705ca1b65c7a140813828e42feimage.jpeg)softmax\_kernels.c — AVX-512 exp approximation used inside standalone softmaxcCopy

    /* Fast vectorized exp approximation (good for softmax, ~1e-4 relative error) */
    // Based on Schraudolph's algorithm with improved coefficients
    #if defined(__AVX512F__)
    static inline __m512 exp512_approx(__m512 x) {
        // Clamp to avoid overflow/underflow
        x = _mm512_max_ps(x, _mm512_set1_ps(-88.0f));
        x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));
    
        // exp(x) = 2^(x * log2(e)) = 2^(x * 1.4426950408889634)
        const __m512 log2e = _mm512_set1_ps(1.4426950408889634f);
        const __m512 c1 = _mm512_set1_ps(0.693359375f);
        const __m512 c2 = _mm512_set1_ps(-2.12194440e-4f);
    
        __m512 t = _mm512_mul_ps(x, log2e);
        __m512 ti = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    
        // Reconstruct remainder: rx = x - ti * ln(2)
        __m512 rx = _mm512_sub_ps(x, _mm512_mul_ps(ti, c1));
        rx = _mm512_sub_ps(rx, _mm512_mul_ps(ti, c2));
    
        // Polynomial approximation for 2^tf on [-0.5, 0.5]
        const __m512 p0 = _mm512_set1_ps(1.0f);
        const __m512 p1 = _mm512_set1_ps(0.6931471805599453f);
        const __m512 p2 = _mm512_set1_ps(0.24022650695910071f);
        const __m512 p3 = _mm512_set1_ps(0.05550410866482157f);
        const __m512 p4 = _mm512_set1_ps(0.009618129107628477f);
    
        __m512 poly = _mm512_fmadd_ps(p4, rx, p3);
        poly = _mm512_fmadd_ps(poly, rx, p2);
        poly = _mm512_fmadd_ps(poly, rx, p1);
        poly = _mm512_fmadd_ps(poly, rx, p0);
    
        // Scale by 2^ti using integer manipulation
        __m512i ti_int = _mm512_cvtps_epi32(ti);
        ti_int = _mm512_add_epi32(ti_int, _mm512_set1_epi32(127));
        ti_int = _mm512_slli_epi32(ti_int, 23);
        __m512 scale = _mm512_castsi512_ps(ti_int);
    
        return _mm512_mul_ps(poly, scale);
    }

softmax\_kernels.c — AVX2 exp approximation with integer exponent synthesiscCopy

    #if defined(__AVX2__)
    // AVX2 version with integer operations
    static inline __m256 exp256_approx(__m256 x) {
        // Clamp to avoid overflow/underflow
        x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
        x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));
    
        const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
        const __m256 c1 = _mm256_set1_ps(0.693359375f);
        const __m256 c2 = _mm256_set1_ps(-2.12194440e-4f);
    
        __m256 t = _mm256_mul_ps(x, log2e);
        __m256 ti = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    
        __m256 rx = _mm256_sub_ps(x, _mm256_mul_ps(ti, c1));
        rx = _mm256_sub_ps(rx, _mm256_mul_ps(ti, c2));
    
        // Polynomial (use FMA if available)
        const __m256 p0 = _mm256_set1_ps(1.0f);
        const __m256 p1 = _mm256_set1_ps(0.6931471805599453f);
        const __m256 p2 = _mm256_set1_ps(0.24022650695910071f);
        const __m256 p3 = _mm256_set1_ps(0.05550410866482157f);
        const __m256 p4 = _mm256_set1_ps(0.009618129107628477f);
    
        __m256 poly = _mm256_fmadd_ps(p4, rx, p3);
        poly = _mm256_fmadd_ps(poly, rx, p2);
        poly = _mm256_fmadd_ps(poly, rx, p1);
        poly = _mm256_fmadd_ps(poly, rx, p0);
    
        // Scale by 2^ti using AVX2 integer ops
        __m256i ti_int = _mm256_cvtps_epi32(ti);
        ti_int = _mm256_add_epi32(ti_int, _mm256_set1_epi32(127));
        ti_int = _mm256_slli_epi32(ti_int, 23);
        __m256 scale = _mm256_castsi256_ps(ti_int);
    
        return _mm256_mul_ps(poly, scale);
    }

softmax\_kernels.c — AVX/AVX2 horizontal max and sum helperscCopy

    // AVX/AVX2 horizontal max helper (works for both, uses 256-bit ops only)
    #if defined(__AVX__) || defined(__AVX2__)
    static inline float hmax256_ps(__m256 v) {
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 max128 = _mm_max_ps(lo, hi);
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(2, 3, 0, 1)));
        max128 = _mm_max_ps(max128, _mm_shuffle_ps(max128, max128, _MM_SHUFFLE(1, 0, 3, 2)));
        return _mm_cvtss_f32(max128);
    }
    
    // AVX/AVX2 horizontal sum helper
    static inline float hsum256_ps_softmax(__m256 v) {
        __m128 hi = _mm256_extractf128_ps(v, 1);
        __m128 lo = _mm256_castps256_ps128(v);
        __m128 sum128 = _mm_add_ps(lo, hi);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        return _mm_cvtss_f32(sum128);
    }

softmax\_kernels.c — causal\_softmax\_head\_major(): AVX-512 max and exp-plus-sum passescCopy

    void causal_softmax_head_major(float *scores,
                                   int num_heads,
                                   int num_tokens,
                                   int aligned_context_window)
    {
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < num_tokens; ++i) {
                int base = h * aligned_context_window * aligned_context_window
                         + i * aligned_context_window;
                float *row = &scores[base];
                int len = i + 1;  // Number of valid elements (0..i inclusive)
    
    #if defined(__AVX512F__)
                // Find max (vectorized)
                __m512 max_vec = _mm512_set1_ps(-INFINITY);
                int j = 0;
                for (; j + 16 <= len; j += 16) {
                    __m512 v = _mm512_loadu_ps(&row[j]);
                    max_vec = _mm512_max_ps(max_vec, v);
                }
                float max_val = _mm512_reduce_max_ps(max_vec);
                for (; j < len; ++j) {
                    if (row[j] > max_val) max_val = row[j];
                }
    
                // Compute exp and sum (vectorized)
                __m512 max_broadcast = _mm512_set1_ps(max_val);
                __m512 sum_vec = _mm512_setzero_ps();
                j = 0;
                for (; j + 16 <= len; j += 16) {
                    __m512 v = _mm512_loadu_ps(&row[j]);
                    __m512 e = exp512_approx(_mm512_sub_ps(v, max_broadcast));
                    _mm512_storeu_ps(&row[j], e);
                    sum_vec = _mm512_add_ps(sum_vec, e);
                }
                float sum = _mm512_reduce_add_ps(sum_vec);
                for (; j < len; ++j) {
                    float e = expf(row[j] - max_val);
                    row[j] = e;
                    sum += e;
                }

softmax\_kernels.c — causal\_softmax\_head\_major(): AVX-512 normalize and causal mask zeroingcCopy

                // Normalize (vectorized)
                float inv_sum = 1.0f / sum;
                __m512 inv_sum_vec = _mm512_set1_ps(inv_sum);
                j = 0;
                for (; j + 16 <= len; j += 16) {
                    __m512 v = _mm512_loadu_ps(&row[j]);
                    _mm512_storeu_ps(&row[j], _mm512_mul_ps(v, inv_sum_vec));
                }
                for (; j < len; ++j) {
                    row[j] *= inv_sum;
                }
    
                // Zero out future tokens (vectorized)
                __m512 zero = _mm512_setzero_ps();
                for (; j + 16 <= num_tokens; j += 16) {
                    _mm512_storeu_ps(&row[j], zero);
                }
                for (; j < num_tokens; ++j) {
                    row[j] = 0.0f;
                }

softmax\_kernels.c — causal\_softmax\_head\_major(): AVX2 branchcCopy

    #elif defined(__AVX2__)
                // AVX2: Find max (vectorized)
                __m256 max_vec = _mm256_set1_ps(-INFINITY);
                int j = 0;
                for (; j + 8 <= len; j += 8) {
                    __m256 v = _mm256_loadu_ps(&row[j]);
                    max_vec = _mm256_max_ps(max_vec, v);
                }
                float max_val = hmax256_ps(max_vec);
                for (; j < len; ++j) {
                    if (row[j] > max_val) max_val = row[j];
                }
    
                // Compute exp and sum (vectorized with fast exp)
                __m256 max_broadcast = _mm256_set1_ps(max_val);
                __m256 sum_vec = _mm256_setzero_ps();
                j = 0;
                for (; j + 8 <= len; j += 8) {
                    __m256 v = _mm256_loadu_ps(&row[j]);
                    __m256 e = exp256_approx(_mm256_sub_ps(v, max_broadcast));
                    _mm256_storeu_ps(&row[j], e);
                    sum_vec = _mm256_add_ps(sum_vec, e);
                }
                float sum = hsum256_ps_softmax(sum_vec);
                for (; j < len; ++j) {
                    float e = expf(row[j] - max_val);
                    row[j] = e;
                    sum += e;
                }
    
                // Normalize (vectorized)
                float inv_sum = 1.0f / sum;
                __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);
                j = 0;
                for (; j + 8 <= len; j += 8) {
                    __m256 v = _mm256_loadu_ps(&row[j]);
                    _mm256_storeu_ps(&row[j], _mm256_mul_ps(v, inv_sum_vec));
                }
                for (; j < len; ++j) {
                    row[j] *= inv_sum;
                }
    
                // Zero out future tokens (vectorized)
                __m256 zero = _mm256_setzero_ps();
                for (; j + 8 <= num_tokens; j += 8) {
                    _mm256_storeu_ps(&row[j], zero);
                }
                for (; j < num_tokens; ++j) {
                    row[j] = 0.0f;
                }

softmax\_kernels.c — scalar fallback for exactness and portabilitycCopy

    #else
                // Scalar fallback
                float max_val = row[0];
                for (int j = 1; j < len; ++j) {
                    if (row[j] > max_val) max_val = row[j];
                }
    
                float sum = 0.0f;
                for (int j = 0; j < len; ++j) {
                    float e = expf(row[j] - max_val);
                    row[j] = e;
                    sum += e;
                }
    
                float inv_sum = 1.0f / sum;
                for (int j = 0; j < len; ++j) {
                    row[j] *= inv_sum;
                }
    
                for (int j = len; j < num_tokens; ++j) {
                    row[j] = 0.0f;
                }

Notice the division of labor. Flash attention removes the matrix when decode makes that attractive. The standalone softmax kernel optimizes the matrix form when upstream compute wants to keep it materialized. CPU runtimes need both, because prefill and decode are not the same problem. 493 linesThe standalone softmax file is 493 lines by itself. Even the “simple” normalization step is treated as first-class kernel code, not an afterthought.

The Exp Polynomial — Same Math, Three Widths
--------------------------------------------

There is a pleasing consistency across the scalar flash helper and the vector softmax helpers. All of them implement the same basic idea: map `eˣ` into base-2 form, separate the integer and fractional parts, approximate the fractional component with a short polynomial, and recover the integer component by constructing a float whose exponent bits represent `2ⁿ`.

In the flash scalar helper that appears as `nearbyintf()`, a fraction `f`, coefficients `c0..c4`, and a 32-bit shift into the exponent field. In the softmax AVX-512 and AVX2 helpers the same coefficients appear as vector constants `p0..p4`, while the remainder reconstruction uses a split `ln(2)` constant so the vector math stays stable.

The polynomial is therefore not an isolated trick inside one file. It is a shared numerical theme across the attention stack. That matters because silicon teams care about whether the math policy is coherent. Here it is.

Fullscreen table

| Variant | Width | Key idea | Implementation detail |
| --- | --- | --- | --- |
| Scalar flash exp | 1 lane | `x·log₂(e)` then polynomial on fractional part | Construct `2ⁿ` by shifting `(n + 127) << 23` into a float. |
| AVX flash exp | 8 lanes | Same polynomial, lane-wise | AVX1 requires 128-bit lane splitting for the exponent-build step. |
| AVX-512 flash exp | 16 lanes | Same polynomial, lane-wise | AVX-512 integer ops keep the whole exponent synthesis inside 512-bit vectors. |
| AVX2 softmax exp | 8 lanes | Same polynomial with remainder reconstruction | Uses `_mm256_fmadd_ps` for Horner evaluation. |
| AVX-512 softmax exp | 16 lanes | Same polynomial with remainder reconstruction | Uses `_mm512_fmadd_ps` and 512-bit integer shifts. |

GQA — Grouped-Query Attention in CKE
------------------------------------

`attention_kernels.c` is the broader attention file, and it is where CKE exposes grouped-query attention, BF16 conversion helpers, and llama.cpp-parity behavior. The top-of-file comments say the important part plainly: the implementation supports GQA with head broadcasting.

The layout in this file is head-major rather than the `[T, H, D_h]` flash teaching file. Here the helper `qkv_index()` assumes `[head][token][head_dim]` with stride `aligned_head_dim`. That tells you CKE is comfortable owning multiple layout contracts where it helps the kernel surface.

The decode-time flash GQA path then maps each query head to a KV head by integer ratio, slices the appropriate head out of the cache, and calls `attention_flash_decode()` with `T_q = 1` and `H = 1`. That is a clean composition: broad attention orchestration in one file, the flash decode microkernel in another.

attention\_kernels.c — file header, GQA note, BF16 conversion helper, and head-major layout commentcCopy

    /**
     * @file attention_kernels.c
     * @brief Attention score/softmax/output kernels with SIMD (SSE/AVX/AVX512)
     *
     * CK-ENGINE KERNEL RULES:
     * =======================
     * 1. NO malloc/free - memory via bump allocator, pointers passed in
     * 2. NO OpenMP - parallelization at orchestrator/codegen layer
     * 3. API must define: inputs, outputs, workspace, and memory layouts
     * 4. Pure computation - deterministic, no side effects
     *
     * After changes: make test && make llamacpp-parity-full
     *
     * Attention: softmax(Q @ K^T / sqrt(d)) @ V
     * Supports GQA (grouped-query attention) with head broadcasting.
     */
    
    #ifndef CK_ENABLE_LLAMA_CPP_PARITY
    #define CK_ENABLE_LLAMA_CPP_PARITY 0
    #endif
    
    #include "bf16_utils.h"
    #include "attention_oracle_ggml.h"
    #include "ckernel_engine.h"
    #if CK_ENABLE_LLAMA_CPP_PARITY
    #include "../../llama.cpp/ggml/include/ggml.h"
    #endif
    #include <dlfcn.h>
    #ifndef RTLD_DEFAULT
    #define RTLD_DEFAULT ((void *)0)
    #endif
    #include <math.h>
    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    
    #if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSE2__)
    #include <immintrin.h>
    #endif
    
    /* Convert BF16 tensor to FP32 using caller-provided buffer (no malloc!) */
    static void convert_bf16_tensor_to_buf(const uint16_t *src, float *dst, size_t count)
    {
        if (!dst || !src) return;
        bf16_tensor_to_float(src, dst, count);
    }
    
    // Helpers for head-major layouts used in attention.
    // Q/K/V layout: [head][token][head_dim] with stride aligned_head_dim.
    static inline size_t qkv_index(int h,
                                   int t,
                                   int d,
                                   int num_tokens,
                                   int aligned_head_dim)
    {
        return ((size_t)h * (size_t)num_tokens + (size_t)t) * (size_t)aligned_head_dim
             + (size_t)d;
    }

attention\_kernels.c — llama.cpp parity rounding through FP16cCopy

    // Match llama.cpp flash-attention input handling where F32 K/V are rounded through F16.
    static inline float ck_round_fp16_scalar(float x) {
        return CK_FP16_TO_FP32(CK_FP32_TO_FP16(x));
    }
    
    static inline void ck_local_fp16_to_fp32_row(const uint16_t *src, float *dst, int n)
    {
        if (!src || !dst || n <= 0) {
            return;
        }
        for (int i = 0; i < n; ++i) {
            dst[i] = CK_FP16_TO_FP32(src[i]);
        }

attention\_kernels.c — decode-time GQA maps query heads onto KV heads and calls flash decodecCopy

                                                       const float *k_cache,
                                                       const float *v_cache,
                                                       float *out_token,
                                                       int num_heads,
                                                       int num_kv_heads,
                                                       int kv_tokens,
                                                       int cache_capacity,
                                                       int head_dim,
                                                       int aligned_head_dim)
    {
        if (!q_token || !k_cache || !v_cache || !out_token) {
            return;
        }
        if (num_heads <= 0 || num_kv_heads <= 0 || kv_tokens <= 0 || cache_capacity <= 0) {
            return;
        }
        if (kv_tokens > cache_capacity || head_dim <= 0 || aligned_head_dim <= 0) {
            return;
        }
    
        const float scale = 1.0f / sqrtf((float)head_dim);
        const size_t head_stride = (size_t)cache_capacity * (size_t)aligned_head_dim;
    
        for (int h = 0; h < num_heads; ++h) {
            int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
            const float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            const float *k_head = k_cache + (size_t)kv_head * head_stride;
            const float *v_head = v_cache + (size_t)kv_head * head_stride;
            float *out_head = out_token + (size_t)h * (size_t)aligned_head_dim;
    
            attention_flash_decode(out_head,
                                   q_head,
                                   k_head,
                                   v_head,
                                   1,
                                   kv_tokens,
                                   1,
                                   aligned_head_dim,
                                   scale);
        }

This is the systems view: flash attention is not sitting alone in a toy file. It is integrated into a larger attention stack that already understands head broadcasting, alignment, BF16 conversion, and parity workflows. Grouped-query attention is where kernel ownership really shows. The runtime has to know how heads map, how caches are laid out, and where a specialized decode kernel can be dropped into the broader execution path.

The Full Attention File Landscape
---------------------------------

If you are evaluating software maturity for CPU inference, file surface matters. Tiny wrappers around external libraries do not look like this. [C-Kernel-Engine's attention surface](https://github.com/antshiv/C-Kernel-Engine)
 includes a clean flash core, a much larger general attention file, sliding-window attention, fused decode, an oracle path for parity testing, and several mega-fused or quantized variants.

The table below uses the current public line counts from the repository. Read it as a map of where engineering effort has actually gone.

Fullscreen table

| File | Lines | What it signals |
| --- | --- | --- |
| `attention_flash_true.c` | 741 | The clean flash-style kernel: online softmax, scalar + AVX + AVX-512, custom exp. |
| `attention_kernels.c` | 3992 | Full attention stack with GQA, llama.cpp parity hooks, BF16 support, and multiple wrappers. |
| `attention_kernels_sliding.c` | 750 | Sliding-window variants for context trimming strategies. |
| `attention_decode_fused.c` | 361 | Fused decode path work beyond the clean reference file. |
| `attention_oracle_ggml.c` | 960 | Parity and reference machinery for ggml-style validation. |
| `softmax_kernels.c` | 493 | Standalone SIMD softmax with causal masking. |
| `fused/mega_fused_attention_avx.c` | 709 | AVX fused path work tuned around broader inference structure. |
| `fused/mega_fused_attention_prefill.c` | 385 | Prefill-side fused attention work. |
| `fused/mega_fused_attention_decode_q5_0.c` | 504 | Quantized decode attention. |
| `fused/mega_fused_attention_prefill_q8_0.c` | 292 | Quantized prefill attention. |

![Map of the CKE attention surface: flash core, general attention, sliding window, fused decode, oracle parity, standalone softmax, and quantized fused variants.](https://www.antsand.com/uploads/b2124321cd9c9709061d9d1c112bc604/07949dd31938beae3e5b69c2a29d5c97image.jpeg)

Nearly four thousand lines in `attention_kernels.c` tells you where real complexity lives. The clean flash file is the teaching kernel. The big file is the production integration surface. 3,992 linesThe largest single file in this surface is `attention_kernels.c` at 3,992 lines. That is the kind of number that only appears when the runtime truly owns edge cases, parity hooks, and multiple execution modes.

Flash vs Materialized — When to Use Which
-----------------------------------------

The right answer is not “flash everywhere,” and it is also not “flash only for decode.” Flash-style attention can be implemented for prefill by tiling query blocks against key/value blocks and carrying online-softmax state for each query row. Decode is the simpler shape: one query row scans the KV cache. CKE's clean flash file emphasizes that decode case, while the broader runtime also keeps materialized score-matrix paths for regimes where they are still useful.

CKE therefore keeps both strategies in the tree. The materialized path computes the lower triangle, calls `causal_softmax_head_major()`, and then multiplies the normalized rows by `V`. The flash path goes straight through `attention_flash_decode()` and never stores the score matrix at all.

That duality is exactly what serious CPU software should look like. Cache hierarchy is not the same as GPU SRAM, but it still rewards algorithms that minimize footprint, stage working sets carefully, and match the regime at hand.

Fullscreen table

| Mode | Best-fit scenario | Memory behavior | Why CKE keeps it |
| --- | --- | --- | --- |
| Flash / online softmax for prefill | Many query tokens, many key/value tokens | No full `T_q × T_k` score matrix | Tiles Q/K/V blocks and carries online-softmax state per query row. |
| Flash / online softmax for decode | `T_q = 1`, long KV cache | No stored `1 × T_k` score row | Minimizes footprint and streams directly through K/V. |
| Materialized scores + standalone softmax | Prefill, `T_q = T_k`, score-matrix workflows | Explicit `[head][query][key]` matrix | Fits broader GEMM-style pipelines and keeps softmax reusable. |
| Fused / quantized variants | Production hot paths | Depends on kernel | Lets the runtime specialize around layout, dtype, and hardware tier. |

attention\_flash\_true.c — simple ISA dispatch for the flash decode kernelcCopy

    void attention_flash_decode(
        float *out,
        const float *q,
        const float *k,
        const float *v,
        int T_q,
        int T_k,
        int H,
        int D_h,
        float scale)
    {
        if (!out || !q || !k || !v) {
            return;
        }
        if (T_q <= 0 || T_k <= 0 || H <= 0 || D_h <= 0) {
            return;
        }
    
        // Dispatch based on CPU features
    #if defined(__AVX512F__)
        attention_flash_decode_avx512(out, q, k, v, T_q, T_k, H, D_h, scale);
    #elif defined(__AVX__) && !defined(__AVX512F__)
        attention_flash_decode_avx(out, q, k, v, T_q, T_k, H, D_h, scale);
    #else
        attention_flash_decode_scalar(out, q, k, v, T_q, T_k, H, D_h, scale);
    #endif

attention\_kernels.c — materialized attention explicitly hands scores to causal\_softmax\_head\_major()cCopy

        // Phase 2: apply causal row-wise softmax in-place over j <= i.
        causal_softmax_head_major(scores,
                                  num_heads,
                                  num_tokens,
                                  aligned_context_window);
    
        // Phase 3: attention weights · V.
        for (int h = 0; h < num_heads; ++h) {
            for (int i = 0; i < num_tokens; ++i) {
                size_t out_base = qkv_index(h, i, 0, num_tokens, aligned_head_dim);
    
                // Zero the full aligned head slice so padded dims stay clean.
                for (int d = 0; d < aligned_head_dim; ++d) {
                    output[out_base + d] = 0.0f;
                }
    
                // Weighted sum over causal positions.
                for (int j = 0; j <= i; ++j) {
                    float w = scores[score_index(h, i, j, aligned_context_window)];
                    size_t v_base = qkv_index(h, j, 0, num_tokens, aligned_head_dim);
    
                    for (int d = 0; d < head_dim; ++d) {
                        output[out_base + d] += w * v[v_base + d];

What Flash Attention on CPU Means for Silicon Vendors
-----------------------------------------------------

The attention kernel is where CPU inference becomes believable or collapses. It combines dot products, reductions, exponentials, normalization, cache-sensitive streaming, layout assumptions, and numerical stability in one loop nest. If a runtime has real answers here, it usually has real answers elsewhere too.

[C-Kernel-Engine's flash implementation](https://c-kernel-engine.github.io/C-Kernel-Engine/flash_attention.html)
 demonstrates several things that silicon vendors should care about. First, online softmax absolutely works on CPU. Second, the tiling strategy maps naturally onto cache hierarchy. Third, the exp approximation is ISA-shaped code: FMA matters, reduction quality matters, and integer-float conversion support matters. Fourth, the runtime already thinks in tiers: scalar reference, AVX, AVX-512 today; AMX or SVE2 tomorrow.

That last point matters most. A kernel surface of this size is an invitation to hardware co-design. AMX could accelerate the dot-product-heavy phases around broader attention workflows. ARM SVE2 could offer a scalable-vector version of the same widening story. The software stack is already exposing the right seams.

attention\_kernels.c — head-major score layout contract used by the materialized pathcCopy

    // Scores layout matches causal_softmax_head_major:
    // [head][query_token][key_token] with stride aligned_context_window.
    static inline size_t score_index(int h,
                                     int i,
                                     int j,
                                     int aligned_context_window)
    {
        return ((size_t)h * (size_t)aligned_context_window * (size_t)aligned_context_window)
             + (size_t)i * (size_t)aligned_context_window
             + (size_t)j;
    }

attention\_kernels.c — GQA flash decode calls the microkernel one head at a timecCopy

        for (int h = 0; h < num_heads; ++h) {
            int kv_head = (int)((long long)h * (long long)num_kv_heads / (long long)num_heads);
            const float *q_head = q_token + (size_t)h * (size_t)aligned_head_dim;
            const float *k_head = k_cache + (size_t)kv_head * head_stride;
            const float *v_head = v_cache + (size_t)kv_head * head_stride;
            float *out_head = out_token + (size_t)h * (size_t)aligned_head_dim;
    
            attention_flash_decode(out_head,
                                   q_head,
                                   k_head,
                                   v_head,
                                   1,
                                   kv_tokens,
                                   1,
                                   aligned_head_dim,
                                   scale);
        }

attention\_flash\_true.c — compile-time selection of AVX-512, AVX, or scalar implementationcCopy

        // Dispatch based on CPU features
    #if defined(__AVX512F__)
        attention_flash_decode_avx512(out, q, k, v, T_q, T_k, H, D_h, scale);
    #elif defined(__AVX__) && !defined(__AVX512F__)
        attention_flash_decode_avx(out, q, k, v, T_q, T_k, H, D_h, scale);
    #else
        attention_flash_decode_scalar(out, q, k, v, T_q, T_k, H, D_h, scale);
    #endif

This is the CPU-vendor takeaway. The implementation is already organized around architectural leverage points: vector width, FMA presence, fast-exp policy, cache-sized tiling, head mapping, and explicit parity escape hatches. 9,187 lines is not a wrapper. It is a kernel surface. And kernel surfaces are where hardware differentiation becomes legible.

Conclusion
----------

Flash attention is not GPU-only. CKE implements it on CPU with three ISA tiers: scalar reference, AVX, and AVX-512. The key idea is online softmax: keep a running maximum and running sum, rescale accumulated state when a larger score appears, and never materialize the full `T_q × T_k` score matrix.

The fast-exp story is equally important. CKE shares a coherent vectorized-exp philosophy across flash attention and standalone softmax, using polynomial approximations that are accurate enough for softmax while preserving strict-fallback paths for parity work. The result is a stack that feels engineered rather than improvised.

For silicon teams, that is the signal. The runtime does not just claim CPU support. It exposes the math, the layout, the tiling, the ISA branches, the parity hooks, and the production surface area in code you can actually audit.

### Follow the implementation

[CKE on GitHub](https://github.com/antshiv/C-Kernel-Engine)
 · [documentation hub](https://c-kernel-engine.github.io/C-Kernel-Engine/)
 · [flash-attention page](https://c-kernel-engine.github.io/C-Kernel-Engine/flash_attention.html)
 · [scaling page](https://c-kernel-engine.github.io/C-Kernel-Engine/scaling.html)
 · [YouTube companion](https://www.youtube.com/@antshivrobotics)

### Subscribe

Get my rants delivered to your inbox

I will send new posts as and when I write. No fixed cadence, just engineering notes, rants, and things I am thinking through.

Subscribe to emails from Anthony
================================

First Name

Last Name

Email

Submit

Submit

Need an intelligent system to work on real hardware?
====================================================

Embedded systems · Robotics · Constrained AI · CPU and HPC · Accelerators · Distributed systems
-----------------------------------------------------------------------------------------------

Work With Us / Antshiv Robotics

[See engineering services](https://www.shivasnotes.com/work-with-us)

#### ShivasNotes

Independent notes on software, systems, AI workflows, robotics, and building Antsand in public.

*   [](https://www.antshiv.com/)
    
*   [](https://www.youtube.com/@antshivrobotics)
    
*   [](https://www.antsand.ca/)
    

### Read

*   [Latest articles](https://www.shivasnotes.com/)
    
*   [About](https://www.shivasnotes.com/about)
    
*   [Projects](https://www.shivasnotes.com/projects)
    
*   [Resume](https://www.shivasnotes.com/resume)
    

### Support

*   [Support this work](https://www.shivasnotes.com/support)
    
*   [GitHub Sponsors](https://github.com/sponsors/antshiv)
    
*   [C-Kernel-Engine](https://www.shivasnotes.com/blog/5889/What-Is-the-C-Kernel-Engine)
    
*   [Work With Us](https://www.shivasnotes.com/work-with-us)
    

× ‹

![](https://www.shivasnotes.com/blog/5914)

›