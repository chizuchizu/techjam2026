"""Optional Triton inference kernels used by explicit benchmark experiments."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _xor_sum_32(values, block_m: tl.constexpr):
    """Return lane-zero's CUDA XOR-shuffle reduction tree."""
    values = values.reshape(block_m, 2, 16).permute(0, 2, 1)
    left, right = values.split()
    values = left + right
    values = values.reshape(block_m, 2, 8).permute(0, 2, 1)
    left, right = values.split()
    values = left + right
    values = values.reshape(block_m, 2, 4).permute(0, 2, 1)
    left, right = values.split()
    values = left + right
    values = values.reshape(block_m, 2, 2).permute(0, 2, 1)
    left, right = values.split()
    values = left + right
    values = values.reshape(block_m, 2)
    left, right = values.split()
    return left + right


@triton.jit
def _pytorch_softmax_128(scores, block_m: tl.constexpr):
    """Match PersistentSoftmax.cuh's lane-local sum order for 128 columns."""
    scores -= tl.max(scores, axis=1)[:, None]
    probabilities = libdevice.exp(scores)
    lane_values = probabilities.reshape(block_m, 4, 32).permute(0, 2, 1)
    even_values, odd_values = lane_values.reshape(block_m, 32, 2, 2).split()
    value_0, value_2 = even_values.split()
    value_1, value_3 = odd_values.split()
    lane_sum = value_0 + value_1
    lane_sum += value_2
    lane_sum += value_3
    denominator = _xor_sum_32(lane_sum, block_m)
    return tl.div_rn(probabilities, denominator[:, None])


@triton.jit
def _welford_online(value, mean, m2, count: tl.constexpr):
    """Match cuWelfordOnlineSum's separately rounded reciprocal multiply."""
    delta = value - mean
    new_mean = mean + delta * (1.0 / count)
    return new_mean, m2 + delta * (value - new_mean)


@triton.jit
def _welford_combine(
    left_mean,
    left_m2,
    left_count,
    right_mean,
    right_m2,
    right_count,
):
    """Match the CUDA LayerNorm combine order for two Welford accumulators."""
    count = left_count + right_count
    reciprocal = 1.0 / count
    right_fraction = right_count * reciprocal
    left_fraction = left_count * reciprocal
    delta = left_mean - right_mean
    mean = right_fraction * right_mean + left_fraction * left_mean
    m2 = (
        left_m2
        + right_m2
        + delta * delta * right_count * left_fraction
    )
    return mean, m2, count


@triton.jit
def _welford_pair_halves(
    mean,
    m2,
    count,
    groups: tl.constexpr,
    lanes: tl.constexpr,
):
    """Apply one CUDA shuffle-down reduction stage to adjacent lane halves."""
    mean = mean.reshape(groups, 2, lanes).permute(0, 2, 1)
    m2 = m2.reshape(groups, 2, lanes).permute(0, 2, 1)
    count = count.reshape(groups, 2, lanes).permute(0, 2, 1)
    left_mean, right_mean = mean.split()
    left_m2, right_m2 = m2.split()
    left_count, right_count = count.split()
    return _welford_combine(
        left_mean,
        left_m2,
        left_count,
        right_mean,
        right_m2,
        right_count,
    )


@triton.jit
def _pytorch_layer_norm_stats_512(values):
    """Reproduce vectorized_layer_norm_kernel's four-warp Welford tree."""
    vector_values = values.reshape(128, 4)
    even_values, odd_values = vector_values.reshape(128, 2, 2).split()
    value_0, value_2 = even_values.split()
    value_1, value_3 = odd_values.split()

    mean = value_0
    m2 = tl.zeros_like(mean)
    mean, m2 = _welford_online(value_1, mean, m2, 2.0)
    mean, m2 = _welford_online(value_2, mean, m2, 3.0)
    mean, m2 = _welford_online(value_3, mean, m2, 4.0)
    count = tl.full(mean.shape, 4.0, tl.float32)

    mean = mean.reshape(4, 32)
    m2 = m2.reshape(4, 32)
    count = count.reshape(4, 32)
    mean, m2, count = _welford_pair_halves(mean, m2, count, 4, 16)
    mean, m2, count = _welford_pair_halves(mean, m2, count, 4, 8)
    mean, m2, count = _welford_pair_halves(mean, m2, count, 4, 4)
    mean, m2, count = _welford_pair_halves(mean, m2, count, 4, 2)
    mean, m2, count = _welford_pair_halves(mean, m2, count, 4, 1)

    mean, m2, count = _welford_pair_halves(mean, m2, count, 1, 2)
    mean, m2, _ = _welford_pair_halves(mean, m2, count, 1, 1)
    return mean.reshape(1), m2.reshape(1) * (1.0 / 512.0)


@triton.jit
def _add_layer_norm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    bias_ptr,
    valid_mask_ptr,
    sum_ptr,
    output_ptr,
    width: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
    apply_valid_mask: tl.constexpr,
):
    """Add two low-precision rows, then normalize the rounded sum in FP32."""
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    feature_mask = offsets < width
    row_valid = tl.load(valid_mask_ptr + row) if apply_valid_mask else True
    mask = feature_mask & row_valid
    row_offsets = row * width + offsets

    x = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
    residual = tl.load(
        residual_ptr + row_offsets, mask=mask, other=0.0
    ).to(tl.float32)

    # Eager stores the residual add in the model dtype before LayerNorm reads it.
    summed = (x + residual).to(sum_ptr.dtype.element_ty)
    tl.store(sum_ptr + row_offsets, summed, mask=feature_mask)
    summed_fp32 = summed.to(tl.float32)

    mean, variance = _pytorch_layer_norm_stats_512(summed_fp32)
    centered = tl.where(mask, summed_fp32 - mean, 0.0)
    inverse_std = tl.rsqrt(variance + eps)

    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        output_ptr + row_offsets,
        tl.where(row_valid, weight * (inverse_std * centered) + bias, 0.0),
        mask=feature_mask,
    )


def add_layer_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    valid_token_mask: torch.Tensor | None = None,
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
    if width != 512:
        raise ValueError("exact Triton add+LayerNorm is specialized for width 512")
    if valid_token_mask is not None:
        if valid_token_mask.shape != x.shape[:-1]:
            raise ValueError("valid-token mask must match all non-feature dimensions")
        if (
            valid_token_mask.device != x.device
            or valid_token_mask.dtype != torch.bool
            or not valid_token_mask.is_contiguous()
        ):
            raise ValueError("valid-token mask must be contiguous CUDA bool")

    summed = torch.empty_like(x)
    output = torch.empty_like(x)
    rows = x.numel() // width
    num_warps = 4 if block_size < 2048 else 8
    _add_layer_norm_kernel[(rows,)](
        x,
        residual,
        weight,
        bias,
        valid_token_mask if valid_token_mask is not None else x,
        summed,
        output,
        width,
        eps,
        block_size=block_size,
        apply_valid_mask=valid_token_mask is not None,
        num_warps=num_warps,
    )
    return summed, output


@triton.jit
def _layer_norm_512_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    eps: tl.constexpr,
):
    """Normalize one FP16 width-512 row using PyTorch's exact Welford tree."""
    row = tl.program_id(0)
    offsets = tl.arange(0, 512)
    row_offsets = row * 512 + offsets
    values = tl.load(input_ptr + row_offsets).to(tl.float32)
    mean, variance = _pytorch_layer_norm_stats_512(values)
    inverse_std = tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + offsets).to(tl.float32)
    bias = tl.load(bias_ptr + offsets).to(tl.float32)
    tl.store(
        output_ptr + row_offsets,
        weight * (inverse_std * (values - mean)) + bias,
    )


def layer_norm_512(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Run the fixed-width FP16 LayerNorm with PyTorch-equivalent reduction."""
    if (
        not inputs.is_cuda
        or inputs.dtype != torch.float16
        or not inputs.is_contiguous()
        or inputs.shape[-1] != 512
    ):
        raise ValueError("exact Triton LayerNorm requires contiguous CUDA FP16 width 512")
    if weight.shape != (512,) or bias.shape != (512,):
        raise ValueError("LayerNorm parameters must have width 512")
    if any(
        parameter.device != inputs.device
        or parameter.dtype != torch.float16
        or not parameter.is_contiguous()
        for parameter in (weight, bias)
    ):
        raise ValueError("LayerNorm parameters must be contiguous CUDA FP16")

    output = torch.empty_like(inputs)
    rows = inputs.numel() // 512
    _layer_norm_512_kernel[(rows,)](
        inputs,
        weight,
        bias,
        output,
        eps=eps,
        num_warps=4,
    )
    return output


@triton.jit
def _linear_exact_gelu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    rows,
    input_features: tl.constexpr,
    output_features: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
    block_k: tl.constexpr,
    group_m: tl.constexpr,
    use_poly11: tl.constexpr,
):
    """FP16 linear materialization followed by exact or polynomial GELU."""
    program_id = tl.program_id(0)
    programs_m = tl.cdiv(rows, block_m)
    programs_n = tl.cdiv(output_features, block_n)
    programs_per_group = group_m * programs_n
    group_id = program_id // programs_per_group
    first_program_m = group_id * group_m
    group_size_m = tl.minimum(programs_m - first_program_m, group_m)
    program_m = first_program_m + (program_id % group_size_m)
    program_n = (program_id % programs_per_group) // group_size_m

    offsets_m = program_m * block_m + tl.arange(0, block_m)
    offsets_n = program_n * block_n + tl.arange(0, block_n)
    offsets_k = tl.arange(0, block_k)
    accumulator = tl.zeros((block_m, block_n), tl.float32)

    for k_start in range(0, input_features, block_k):
        input_offsets = offsets_m[:, None] * input_features + (
            k_start + offsets_k[None, :]
        )
        weight_offsets = offsets_n[None, :] * input_features + (
            k_start + offsets_k[:, None]
        )
        input_values = tl.load(
            input_ptr + input_offsets,
            mask=offsets_m[:, None] < rows,
            other=0.0,
        )
        weight_values = tl.load(weight_ptr + weight_offsets)
        accumulator += tl.dot(input_values, weight_values)

    bias = tl.load(bias_ptr + offsets_n)
    linear_fp16 = (accumulator + bias[None, :]).to(tl.float16)
    linear_fp32 = linear_fp16.to(tl.float32)
    if use_poly11:
        clipped = tl.maximum(-3.5, tl.minimum(3.5, linear_fp32))
        clipped_sq = clipped * clipped
        cdf = 0.5 + clipped * (
            0.39639100184010506
            + clipped_sq
            * (
                -0.06247543670746915
                + clipped_sq
                * (
                    0.0077960139440769235
                    + clipped_sq
                    * (
                        -0.000606554913963472
                        + clipped_sq
                        * (
                            2.5871031157700182e-05
                            + clipped_sq * -4.5557832301351553e-07
                        )
                    )
                )
            )
        )
        gelu = linear_fp32 * tl.maximum(0.0, tl.minimum(1.0, cdf))
    else:
        erf_values = libdevice.erf(linear_fp32 * 0.7071067811865476)
        gelu = (linear_fp32 * 0.5) * (1.0 + erf_values)
    output_offsets = offsets_m[:, None] * output_features + offsets_n[None, :]
    tl.store(
        output_ptr + output_offsets,
        gelu,
        mask=(offsets_m[:, None] < rows),
    )


def _linear_gelu(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    use_poly11: bool,
) -> torch.Tensor:
    if not inputs.is_cuda or inputs.dtype != torch.float16:
        raise ValueError("Triton linear+GELU requires a CUDA FP16 input")
    if not inputs.is_contiguous() or inputs.shape[-1] != 512:
        raise ValueError("Triton linear+GELU requires contiguous width-512 input")
    if weight.shape != (2048, 512) or bias.shape != (2048,):
        raise ValueError("Triton linear+GELU requires 2048x512 weight and bias")
    if (
        weight.device != inputs.device
        or bias.device != inputs.device
        or weight.dtype != torch.float16
        or bias.dtype != torch.float16
        or not weight.is_contiguous()
        or not bias.is_contiguous()
    ):
        raise ValueError("Triton linear+GELU parameters must be contiguous CUDA FP16")

    rows = inputs.numel() // inputs.shape[-1]
    output = torch.empty(
        (*inputs.shape[:-1], 2048), device=inputs.device, dtype=inputs.dtype
    )
    block_m, block_n, block_k = 128, 128, 64
    grid = (triton.cdiv(rows, block_m) * triton.cdiv(2048, block_n),)
    _linear_exact_gelu_kernel[grid](
        inputs,
        weight,
        bias,
        output,
        rows,
        input_features=512,
        output_features=2048,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        group_m=8,
        use_poly11=use_poly11,
        num_warps=8,
        num_stages=4,
    )
    return output


def linear_exact_gelu(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Run the fixed FP16 FFN input projection and exact GELU in one kernel."""
    return _linear_gelu(inputs, weight, bias, use_poly11=False)


def linear_poly11_gelu(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Run the fixed FFN projection with CUTLASS's degree-11 CDF polynomial."""
    return _linear_gelu(inputs, weight, bias, use_poly11=True)


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

    probabilities = _pytorch_softmax_128(scores, block_m)
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
    # H200 tuning for the exact-division kernel selected 16 rows and three
    # stages for dense/causal inputs. Padded softmax follows a different eager
    # arithmetic path and needs the audited 64-row, one-stage geometry.
    block_m = 64 if valid_token_mask is not None else 16
    num_stages = 1 if valid_token_mask is not None else 3
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
        num_stages=num_stages,
    )
    return output
