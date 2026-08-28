"""Optional Triton inference kernels used by explicit benchmark experiments."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers


@triton.jit
def _add_layer_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    sum_ptr,
    output_ptr,
    width: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    """Add two low-precision rows, then normalize the rounded sum in FP32."""
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < width
    row_offsets = row * width + offsets

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + row_offsets, mask=mask, other=0.0
    ).to(tl.float32)

    # Eager stores the residual add in the model dtype before LayerNorm reads it.
    summed = (x + residual).to(sum_ptr.dtype.element_ty)
    tl.store(sum_ptr + row_offsets, summed, mask=mask)
    summed_fp32 = summed.to(tl.float32)

    initial_mean = tl.where(mask, summed_fp32, 0.0)
    initial_m2 = tl.zeros_like(initial_mean)
    initial_weight = tl.where(mask, 1.0, 0.0)
    mean, m2, _ = triton_helpers.welford(
        initial_mean,
        initial_m2,
        initial_weight,
        0,
    )
    centered = tl.where(mask, summed_fp32 - mean, 0.0)
    variance = m2 / width
    normalized = centered * tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row_offsets,
        normalized * weight + bias,
        mask=mask,
    )


def add_layer_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``x + residual`` and its LayerNorm using one Triton launch."""
    if not x.is_cuda:
        raise ValueError("Triton add+LayerNorm requires CUDA tensors")
    if x.shape != residual.shape:
        raise ValueError("residual shape must match the input")
    if not x.is_contiguous() or not residual.is_contiguous():
        raise ValueError("Triton add+LayerNorm requires contiguous inputs")
    width = x.shape[-1]
    if weight.shape != (width,) or bias.shape != (width,):
        raise ValueError("LayerNorm weight and bias must match the final dimension")
    block_size = triton.next_power_of_2(width)
    if block_size > 65536:
        raise ValueError("feature dimension is too large for the fused kernel")

    summed = torch.empty_like(x)
    output = torch.empty_like(x)
    rows = x.numel() // width
    num_warps = 4 if block_size < 2048 else 8
    _add_layer_norm_kernel[(rows,)](
        x,
        residual,
        weight,
        bias,
        summed,
        output,
        width,
        eps,
        block_size=block_size,
        num_warps=num_warps,
    )
    return summed, output


@triton.jit
def _rounded_attention_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    valid_mask_ptr,
    output_ptr,
    stride_qb: tl.constexpr,
    stride_qh: tl.constexpr,
    stride_qs: tl.constexpr,
    stride_qd: tl.constexpr,
    stride_kb: tl.constexpr,
    stride_kh: tl.constexpr,
    stride_ks: tl.constexpr,
    stride_kd: tl.constexpr,
    stride_vb: tl.constexpr,
    stride_vh: tl.constexpr,
    stride_vs: tl.constexpr,
    stride_vd: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_oh: tl.constexpr,
    stride_os: tl.constexpr,
    stride_od: tl.constexpr,
    stride_mb: tl.constexpr,
    stride_ms: tl.constexpr,
    heads: tl.constexpr,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    causal: tl.constexpr,
    has_valid_mask: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
):
    """Attention with the reference path's two explicit FP16 score rounds."""
    query_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads

    offsets_m = query_block * block_m + tl.arange(0, block_m)
    offsets_n = tl.arange(0, block_n)
    offsets_d = tl.arange(0, head_dim)
    mask_m = offsets_m < seq_len
    mask_n = offsets_n < seq_len

    q_base = q_ptr + batch * stride_qb + head * stride_qh
    k_base = k_ptr + batch * stride_kb + head * stride_kh
    v_base = v_ptr + batch * stride_vb + head * stride_vh

    q = tl.load(
        q_base
        + offsets_m[:, None] * stride_qs
        + offsets_d[None, :] * stride_qd,
        mask=mask_m[:, None],
        other=0.0,
    )
    k_transposed = tl.load(
        k_base
        + offsets_n[None, :] * stride_ks
        + offsets_d[:, None] * stride_kd,
        mask=mask_n[None, :],
        other=0.0,
    )

    # Baseline matmul returns FP16, and the following scale multiply returns
    # FP16 again. Keep both rounding boundaries before the FP32 softmax.
    unscaled_scores = tl.dot(q, k_transposed).to(tl.float16)
    scores = (unscaled_scores.to(tl.float32) * scale).to(tl.float16)
    scores = scores.to(tl.float32)

    score_mask = mask_m[:, None] & mask_n[None, :]
    if causal:
        score_mask &= offsets_m[:, None] >= offsets_n[None, :]
    if has_valid_mask:
        valid_keys = tl.load(
            valid_mask_ptr
            + batch * stride_mb
            + offsets_n * stride_ms,
            mask=mask_n,
            other=0,
        ).to(tl.int1)
        score_mask &= valid_keys[None, :]
    scores = tl.where(score_mask, scores, float("-inf"))

    scores -= tl.max(scores, axis=1)[:, None]
    probabilities = tl.exp(scores)
    probabilities /= tl.sum(probabilities, axis=1)[:, None]
    probabilities = probabilities.to(tl.float16)

    values = tl.load(
        v_base
        + offsets_n[:, None] * stride_vs
        + offsets_d[None, :] * stride_vd,
        mask=mask_n[:, None],
        other=0.0,
    )
    context = tl.dot(probabilities, values)

    output_base = output_ptr + batch * stride_ob + head * stride_oh
    tl.store(
        output_base
        + offsets_m[:, None] * stride_os
        + offsets_d[None, :] * stride_od,
        context,
        mask=mask_m[:, None],
    )


def rounded_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    valid_token_mask: torch.Tensor | None,
    causal: bool,
    scale: float,
) -> torch.Tensor:
    """Run fixed-shape FP16 attention with baseline-compatible score rounds."""
    if q.dtype != torch.float16 or not q.is_cuda:
        raise ValueError("rounded Triton attention currently requires CUDA FP16")
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 4:
        raise ValueError("q, k, and v must share [B, H, S, D] shape")
    batch, heads, seq_len, head_dim = q.shape
    if seq_len != 128 or head_dim != 64:
        raise ValueError("rounded Triton attention is specialized for S=128, D=64")
    if valid_token_mask is not None and valid_token_mask.shape != (
        batch,
        seq_len,
    ):
        raise ValueError("valid-token mask must have [B, S] shape")

    # Store into BSHD backing storage while exposing the BHSD attention shape.
    # The caller's BHSD -> BSHD transpose then becomes a view before out_proj.
    output = torch.empty(
        (batch, seq_len, heads, head_dim),
        device=q.device,
        dtype=q.dtype,
    ).transpose(1, 2)
    mask = valid_token_mask if valid_token_mask is not None else q
    mask_strides = (
        valid_token_mask.stride()
        if valid_token_mask is not None
        else (0, 0)
    )
    # H200 tuning for B=8, H=8, S=128, D=64. A 64-row tile with four warps
    # outperformed the 16/32-row variants while preserving rowwise arithmetic.
    block_m = 64
    block_n = 128
    _rounded_attention_kernel[(triton.cdiv(seq_len, block_m), batch * heads)](
        q,
        k,
        v,
        mask,
        output,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *output.stride(),
        *mask_strides,
        heads,
        seq_len,
        head_dim,
        scale,
        causal,
        valid_token_mask is not None,
        block_m=block_m,
        block_n=block_n,
        num_warps=4,
        num_stages=1,
    )
    return output
