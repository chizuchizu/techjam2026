Index your code with Devin

[DeepWiki](https://deepwiki.com/)

[DeepWiki](https://deepwiki.com/)

[karpathy/llm.c](https://github.com/karpathy/llm.c "Open repository")

Index your code with

Devin

Edit WikiShare

Last indexed: 30 March 2026 ([f1e2ac](https://github.com/karpathy/llm.c/commits/f1e2ace6)
)

*   [Overview](https://deepwiki.com/karpathy/llm.c/1-overview)
    
*   [Architecture](https://deepwiki.com/karpathy/llm.c/2-architecture)
    
*   [Training Pipeline](https://deepwiki.com/karpathy/llm.c/2.1-training-pipeline)
    
*   [Multi-GPU Training](https://deepwiki.com/karpathy/llm.c/2.2-multi-gpu-training)
    
*   [Testing Framework](https://deepwiki.com/karpathy/llm.c/2.3-testing-framework)
    
*   [Core Components](https://deepwiki.com/karpathy/llm.c/3-core-components)
    
*   [CUDA GPT-2 Implementation](https://deepwiki.com/karpathy/llm.c/3.1-cuda-gpt-2-implementation)
    
*   [PyTorch Reference Implementation](https://deepwiki.com/karpathy/llm.c/3.2-pytorch-reference-implementation)
    
*   [CPU Implementation](https://deepwiki.com/karpathy/llm.c/3.3-cpu-implementation)
    
*   [LLaMA 3.1 Implementation](https://deepwiki.com/karpathy/llm.c/3.4-llama-3.1-implementation)
    
*   [CUDA Kernel Library](https://deepwiki.com/karpathy/llm.c/4-cuda-kernel-library)
    
*   [Attention Mechanisms](https://deepwiki.com/karpathy/llm.c/4.1-attention-mechanisms)
    
*   [Layer Normalization](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization)
    
*   [Matrix Multiplication](https://deepwiki.com/karpathy/llm.c/4.3-matrix-multiplication)
    
*   [Activation Functions](https://deepwiki.com/karpathy/llm.c/4.4-activation-functions)
    
*   [Data Processing](https://deepwiki.com/karpathy/llm.c/5-data-processing)
    
*   [Performance Optimization](https://deepwiki.com/karpathy/llm.c/6-performance-optimization)
    
*   [Profiling and Benchmarking](https://deepwiki.com/karpathy/llm.c/6.1-profiling-and-benchmarking)
    
*   [Build System and Configuration](https://deepwiki.com/karpathy/llm.c/6.2-build-system-and-configuration)
    
*   [Glossary](https://deepwiki.com/karpathy/llm.c/7-glossary)
    

Menu

Layer Normalization
===================

Relevant source files

*   [dev/cuda/common.h](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/common.h)
    
*   [dev/cuda/fused\_residual\_forward.cu](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/fused_residual_forward.cu)
    
*   [dev/cuda/layernorm\_backward.cu](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_backward.cu)
    
*   [dev/cuda/layernorm\_forward.cu](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu)
    
*   [doc/layernorm/layernorm.c](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.c)
    
*   [doc/layernorm/layernorm.md](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.md?plain=1)
    
*   [doc/layernorm/layernorm.py](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.py)
    
*   [llmc/gelu.cuh](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/gelu.cuh)
    
*   [llmc/layernorm.cuh](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/layernorm.cuh)
    

This document explains the Layer Normalization implementation in the `llm.c` codebase, focusing on both the mathematical principles and the CUDA kernel optimizations. Layer Normalization is a critical component in transformer architectures, stabilizing hidden state dynamics by normalizing activations across the feature dimension.

1\. Overview of Layer Normalization
-----------------------------------

Layer Normalization normalizes the activations within each layer independently for each training example. Unlike Batch Normalization which normalizes across the batch dimension, Layer Normalization operates across the feature dimension, making it well-suited for sequence models like transformers where batch sizes may be small and sequence lengths variable.

### Mathematical Formulation

For an input tensor of shape `(B, T, C)` where:

*   `B` is the batch size
*   `T` is the sequence length
*   `C` is the hidden dimension (feature size)

Layer Normalization computes:

    y = ((x - μ) / √(σ² + ε)) * γ + β
    

Where:

*   `μ` (mean) is the average of the features.
*   `σ²` (variance) is the average squared deviation from the mean.
*   `ε` (epsilon) is a small constant (typically `1e-5f`) for numerical stability [dev/cuda/layernorm\_forward.cu38](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L38-L38)
    
*   `γ` (weight) and `β` (bias) are learnable parameters of shape `(C)`.

Sources: [dev/cuda/layernorm\_forward.cu35-70](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L35-L70)
 [doc/layernorm/layernorm.md63-67](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.md?plain=1#L63-L67)

2\. Implementation in llm.c
---------------------------

The repository implements both forward and backward passes of Layer Normalization, with a progression from simple CPU references to highly optimized CUDA kernels.

### Forward Pass Implementation

The forward pass calculates the mean and variance for each "fiber" (the $C$ dimension) of the input tensor.

1.  **Mean Calculation**: Sum elements along $C$ and divide by $C$ [dev/cuda/layernorm\_forward.cu42-48](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L42-L48)
    
2.  **Variance Calculation**: Sum $(x - \\mu)^2$ along $C$ and divide by $C$ [dev/cuda/layernorm\_forward.cu49-55](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L49-L55)
    
3.  **Normalization**: Compute reciprocal standard deviation `rstd = 1.0f / sqrtf(v + eps)` [dev/cuda/layernorm\_forward.cu57](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L57-L57)
    
4.  **Scale and Shift**: Multiply by `weight[i]` and add `bias[i]` [dev/cuda/layernorm\_forward.cu62](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L62-L62)
    
5.  **Caching**: Store `mean` and `rstd` (reciprocal standard deviation) for the backward pass to avoid recomputation [dev/cuda/layernorm\_forward.cu66-67](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L66-L67)
    

Sources: [dev/cuda/layernorm\_forward.cu35-70](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L35-L70)
 [llmc/layernorm.cuh20-65](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/layernorm.cuh#L20-L65)

### Backward Pass Overview

The backward pass computes gradients for the input (`dinp`), weights (`dweight`), and biases (`dbias`). Mathematically, the gradient for the input involves three terms derived from the chain rule applied to the normalization formula [doc/layernorm/layernorm.md80](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.md?plain=1#L80-L80)

Sources: [dev/cuda/layernorm\_backward.cu69-110](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_backward.cu#L69-L110)
 [doc/layernorm/layernorm.md70-83](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.md?plain=1#L70-L83)

3\. CUDA Kernel Evolution
-------------------------

The codebase provides multiple CUDA kernel implementations with progressive optimizations, documented in the `dev/cuda` directory.

### Forward Kernel Progression

*   **Kernel 1**: Naive port parallelizing over $B \\times T$, looping over $C$ [dev/cuda/layernorm\_forward.cu76-111](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L76-L111)
    
*   **Kernel 2**: Parallelizes over $B, T, C$ using multiple kernels (mean, rstd, normalization) [dev/cuda/layernorm\_forward.cu113-179](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L113-L179)
    
*   **Kernel 3**: Uses `cooperative_groups` and warp-level reductions (`cg::reduce`) to process one row per warp [dev/cuda/layernorm\_forward.cu181-224](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L181-L224)
    
*   **Kernel 4**: Implements the online variance estimation formula: $var(x) = mean(x^2) - mean(x)^2$, allowing a single pass over the data [dev/cuda/layernorm\_forward.cu16-18](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L16-L18)
    
*   **Kernel 6**: Production-grade kernel using `load128`/`store128` for vectorized access and shared memory for weights/biases [llmc/layernorm.cuh67-140](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/layernorm.cuh#L67-L140)
    

### Backward Kernel Progression

*   **Kernel 1**: Naive port using `atomicAdd` for weight and bias gradients [dev/cuda/layernorm\_backward.cu143-185](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_backward.cu#L143-L185)
    
*   **Kernel 2**: Uses shared memory for reductions within a block to minimize global memory atomics [dev/cuda/layernorm\_backward.cu188-267](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_backward.cu#L188-L267)
    

Sources: [dev/cuda/layernorm\_forward.cu1-22](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L1-L22)
 [dev/cuda/layernorm\_backward.cu7-12](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_backward.cu#L7-L12)
 [llmc/layernorm.cuh1-140](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/layernorm.cuh#L1-L140)

4\. Key Optimization Techniques
-------------------------------

### Vectorized Memory Access

The code uses a custom `Packed128` data structure (aliased as `f128` for floats or `x128` for `floatX`) to force the compiler to use 128-bit instructions (`LDG.128` and `STS.128`). This is achieved via `alignas(16)` and `int4` reinterpretation.

Sources: [dev/cuda/common.h103-179](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/common.h#L103-L179)
 [llmc/layernorm.cuh97-103](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/layernorm.cuh#L97-L103)

### Shared Memory and Streaming Hints

*   **Shared Memory**: In `layernorm_forward_kernel6`, weights and biases are loaded into `__shared__` memory once per block to reduce global memory pressure [llmc/layernorm.cuh72-86](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/layernorm.cuh#L72-L86)
    
*   **Streaming Hints**: The use of `__ldcs` (load cached streaming) and `__stcs` (store cached streaming) signals to the GPU that the activation data should not pollute the L1/L2 caches, as it is typically read once and then discarded [llmc/layernorm.cuh59-64](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/layernorm.cuh#L59-L64)
    

### Atomic Reductions for BF16

Since standard `atomicAdd` is not available for `__nv_bfloat16` on all architectures, `llm.c` implements `atomicAddX`. This function uses a 32-bit `atomicAdd` on a `__nv_bfloat162` by masking the address and preparing a half-zeroed 32-bit word.

Sources: [dev/cuda/layernorm\_backward.cu115-126](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_backward.cu#L115-L126)
 [llmc/cuda\_utils.cuh1-50](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/cuda_utils.cuh#L1-L50)

5\. Fused Residual Connections
------------------------------

In the transformer architecture, LayerNorm often follows a residual addition ($x = x + shortcut$). To improve memory bandwidth, `llm.c` provides `fused_residual_forward_kernel5`.

This fusion saves one round-trip to global memory by performing the addition and normalization in a single kernel pass [llmc/layernorm.cuh142-205](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/layernorm.cuh#L142-L205)

Sources: [llmc/layernorm.cuh142-205](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/layernorm.cuh#L142-L205)
 [dev/cuda/fused\_residual\_forward.cu116-157](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/fused_residual_forward.cu#L116-L157)

6\. Educational Tutorial: doc/layernorm
---------------------------------------

The repository includes a standalone tutorial in `doc/layernorm/` to bridge the gap between PyTorch and C/CUDA implementations.

| File | Purpose |
| --- | --- |
| `layernorm.py` | Reference implementation using PyTorch and manual math derivation [doc/layernorm/layernorm.py5-32](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.py#L5-L32) |
| `layernorm.c` | Pure C implementation matching the Python math [doc/layernorm/layernorm.c9-87](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.c#L9-L87) |
| `layernorm.md` | Detailed walkthrough of the LayerNorm gradient derivation [doc/layernorm/layernorm.md1-83](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.md?plain=1#L1-L83) |

### Gradient Derivation

The tutorial emphasizes that while PyTorch Autograd is convenient, manual derivation allows for optimized C/CUDA code. The backward pass for input $x$ is simplified to: `dx = rstd * (dnorm - mean(dnorm) - norm * mean(dnorm * norm))` where `dnorm = dout * w` [doc/layernorm/layernorm.md79-81](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.md?plain=1#L79-L81)

Sources: [doc/layernorm/layernorm.md1-83](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.md?plain=1#L1-L83)
 [doc/layernorm/layernorm.py1-71](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.py#L1-L71)
 [doc/layernorm/layernorm.c1-176](https://github.com/karpathy/llm.c/blob/f1e2ace6/doc/layernorm/layernorm.c#L1-L176)

7\. Implementation Mapping
--------------------------

Sources: [llmc/layernorm.cuh20-205](https://github.com/karpathy/llm.c/blob/f1e2ace6/llmc/layernorm.cuh#L20-L205)
 [dev/cuda/common.h16-179](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/common.h#L16-L179)
 [dev/cuda/layernorm\_forward.cu35-70](https://github.com/karpathy/llm.c/blob/f1e2ace6/dev/cuda/layernorm_forward.cu#L35-L70)

Dismiss

Refresh this wiki

Enter email to refresh

### On this page

*   [Layer Normalization](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#layer-normalization)
    
*   [1\. Overview of Layer Normalization](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#1-overview-of-layer-normalization)
    
*   [Mathematical Formulation](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#mathematical-formulation)
    
*   [2\. Implementation in llm.c](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#2-implementation-in-llmc)
    
*   [Forward Pass Implementation](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#forward-pass-implementation)
    
*   [Backward Pass Overview](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#backward-pass-overview)
    
*   [3\. CUDA Kernel Evolution](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#3-cuda-kernel-evolution)
    
*   [Forward Kernel Progression](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#forward-kernel-progression)
    
*   [Backward Kernel Progression](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#backward-kernel-progression)
    
*   [4\. Key Optimization Techniques](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#4-key-optimization-techniques)
    
*   [Vectorized Memory Access](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#vectorized-memory-access)
    
*   [Shared Memory and Streaming Hints](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#shared-memory-and-streaming-hints)
    
*   [Atomic Reductions for BF16](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#atomic-reductions-for-bf16)
    
*   [5\. Fused Residual Connections](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#5-fused-residual-connections)
    
*   [6\. Educational Tutorial: doc/layernorm](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#6-educational-tutorial-doclayernorm)
    
*   [Gradient Derivation](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#gradient-derivation)
    
*   [7\. Implementation Mapping](https://deepwiki.com/karpathy/llm.c/4.2-layer-normalization#7-implementation-mapping)