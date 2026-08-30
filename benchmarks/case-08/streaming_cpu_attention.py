#!/usr/bin/env python3
"""Benchmark exact dense versus block-online causal attention for case 8."""

from __future__ import annotations

import argparse
import math
import statistics
import time

import numpy as np


def dense_attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Reference attention. Q/K/V are [B,H,S,HD], output has the same shape."""
    sequence = q.shape[2]
    scale = np.float32(q.shape[-1] ** -0.5)
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    future = np.triu(np.ones((sequence, sequence), dtype=np.bool_), 1)
    scores[..., future] = -np.inf
    maximum = np.max(scores, axis=-1, keepdims=True)
    probabilities = np.exp(scores - maximum)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    return np.matmul(probabilities, v)


def streaming_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    query_tile: int,
    kv_tile: int,
) -> np.ndarray:
    """Exact causal attention using fp32 online-softmax block merging."""
    batch, heads, sequence, head_dim = q.shape
    scale = np.float32(head_dim ** -0.5)
    output = np.empty_like(q)

    for query_begin in range(0, sequence, query_tile):
        query_end = min(sequence, query_begin + query_tile)
        query = q[:, :, query_begin:query_end, :]
        query_positions = np.arange(query_begin, query_end)[:, None]
        rows = query_end - query_begin
        running_max = np.full((batch, heads, rows), -np.inf, dtype=np.float32)
        running_sum = np.zeros((batch, heads, rows), dtype=np.float32)
        running_num = np.zeros(
            (batch, heads, rows, head_dim), dtype=np.float32
        )

        # Future-only blocks are skipped entirely.
        for kv_begin in range(0, query_end, kv_tile):
            kv_end = min(sequence, kv_begin + kv_tile)
            key = k[:, :, kv_begin:kv_end, :]
            value = v[:, :, kv_begin:kv_end, :]
            scores = np.matmul(query, np.swapaxes(key, -1, -2)) * scale
            key_positions = np.arange(kv_begin, kv_end)[None, :]
            scores = np.where(
                key_positions[None, None, :, :] <=
                query_positions[None, None, :, :],
                scores,
                -np.inf,
            )

            block_max = np.max(scores, axis=-1)
            valid = np.isfinite(block_max)
            safe_max = np.where(valid, block_max, np.float32(0.0))
            weights = np.exp(scores - safe_max[..., None])
            weights = np.where(valid[..., None], weights, np.float32(0.0))
            block_sum = np.sum(weights, axis=-1)
            block_num = np.matmul(weights, value)

            merged_max = np.maximum(running_max, block_max)
            old_scale = np.where(
                np.isfinite(running_max),
                np.exp(running_max - merged_max),
                np.float32(0.0),
            )
            block_scale = np.where(
                valid, np.exp(block_max - merged_max), np.float32(0.0)
            )
            running_num = (
                running_num * old_scale[..., None]
                + block_num * block_scale[..., None]
            )
            running_sum = running_sum * old_scale + block_sum * block_scale
            running_max = merged_max

        output[:, :, query_begin:query_end, :] = (
            running_num / running_sum[..., None]
        )
    return output


def measure(function, repeats: int) -> tuple[np.ndarray, list[float]]:
    samples: list[float] = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = function()
        samples.append((time.perf_counter_ns() - started) / 1e6)
    assert result is not None
    return result, samples


def compare(reference: np.ndarray, candidate: np.ndarray) -> tuple[int, float, float]:
    error = np.abs(candidate - reference)
    passed = np.isfinite(candidate) & (
        (error <= 0.001) | (error <= 0.01 * np.abs(reference))
    )
    return (
        int(passed.size - np.count_nonzero(passed)),
        float(np.max(error)),
        float(np.mean(error)),
    )


def mib(elements: int) -> float:
    return elements * np.dtype(np.float32).itemsize / (1024 * 1024)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--tiles",
        default="16,32,64,32x64,64x32",
        help="comma-separated square sizes or QxKV pairs",
    )
    args = parser.parse_args()

    batch, model_dim, heads, sequence = 64, 1024, 4, 128
    head_dim = model_dim // heads
    rng = np.random.default_rng(args.seed)
    # Scaling is not required for correctness, but keeps scores representative.
    q = rng.normal(0, 1, (batch, heads, sequence, head_dim)).astype(np.float32)
    k = rng.normal(0, 1, (batch, heads, sequence, head_dim)).astype(np.float32)
    v = rng.normal(0, 1, (batch, heads, sequence, head_dim)).astype(np.float32)

    dense, dense_samples = measure(lambda: dense_attention(q, k, v), args.repeats)
    dense_ms = statistics.median(dense_samples)
    score_elements = batch * heads * sequence * sequence
    print(
        f"case=8 B={batch} D={model_dim} H={heads} S={sequence} "
        f"HD={head_dim} causal=true"
    )
    print(
        f"dense median_ms={dense_ms:.3f} "
        f"score_tile_mib={mib(score_elements):.3f}"
    )

    failed_any = False
    tile_pairs = []
    for item in args.tiles.split(","):
        if not item:
            continue
        if "x" in item.lower():
            query_text, kv_text = item.lower().split("x", 1)
            tile_pairs.append((int(query_text), int(kv_text)))
        else:
            tile_pairs.append((int(item), int(item)))
    for query_tile, kv_tile in tile_pairs:
        candidate, samples = measure(
            lambda query_tile=query_tile, kv_tile=kv_tile: streaming_attention(
                q, k, v, query_tile, kv_tile
            ),
            args.repeats,
        )
        median_ms = statistics.median(samples)
        failed, max_abs, mean_abs = compare(dense, candidate)
        failed_any |= failed != 0
        tile_elements = batch * heads * query_tile * kv_tile
        print(
            f"stream qtile={query_tile} kvtile={kv_tile} "
            f"median_ms={median_ms:.3f} "
            f"speedup={dense_ms / median_ms:.3f}x "
            f"score_tile_mib={mib(tile_elements):.3f} "
            f"failed={failed}/{candidate.size} max_abs={max_abs:.9g} "
            f"mean_abs={mean_abs:.9g}"
        )
    return 2 if failed_any else 0


if __name__ == "__main__":
    raise SystemExit(main())
