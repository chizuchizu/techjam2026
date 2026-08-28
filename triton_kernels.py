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
