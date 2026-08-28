#!/usr/bin/env python3
"""Run official case-2 layer-0 LayerNorm, QKV, and attention on ESP32 workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import socket
import statistics
import struct
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


MAGIC = 0x45535033
VERSION = 1
TYPE_NORM_HEAD_TASK = 11
TYPE_NORM_HEAD_RESULT = 12
TCP_PORT = 4211
SEQUENCE = 128
MODEL_DIMENSION = 128
HEADS = 4
HEAD_DIMENSION = 32
LAYER_NORM_EPSILON = np.float32(1.0e-5)
HEADER = struct.Struct("!IBBHI")
TASK_FIXED = struct.Struct("!HHBBHff")
RESULT_FIXED = struct.Struct("!HHBBBBHHIIII")
ABSOLUTE_TOLERANCE = 0.002
RELATIVE_TOLERANCE = 0.02


@dataclass
class References:
    input_quantized: np.ndarray
    input_scale: np.float32
    norm_kernel: np.ndarray
    norm_official: np.ndarray
    context_kernel: list[np.ndarray]
    context_official: list[np.ndarray]


def quantize(values: np.ndarray, maximum: int, dtype: np.dtype) -> tuple[np.ndarray, np.float32]:
    maximum_absolute = np.max(np.abs(values)).astype(np.float32)
    if maximum_absolute == 0:
        maximum_absolute = np.float32(1.0)
    scale = np.float32(maximum_absolute / np.float32(maximum))
    quantized = np.rint(values / scale)
    quantized = np.clip(quantized, -maximum, maximum).astype(dtype)
    return quantized, scale


def layer_norm(values: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    mean = np.sum(values, axis=1, dtype=np.float32) / np.float32(MODEL_DIMENSION)
    centered = values - mean[:, None]
    variance = (
        np.sum(centered * centered, axis=1, dtype=np.float32)
        / np.float32(MODEL_DIMENSION)
    )
    reciprocal = np.float32(1.0) / np.sqrt(
        variance + LAYER_NORM_EPSILON, dtype=np.float32
    )
    return ((centered * reciprocal[:, None]) * gamma + beta).astype(np.float32)


def parse_q12(blob: bytes) -> list[tuple[np.ndarray, np.float32]]:
    matrices = []
    offset = 0
    for _ in range(3):
        count, scale = struct.unpack_from("<If", blob, offset)
        offset += 8
        if count != MODEL_DIMENSION * MODEL_DIMENSION:
            raise ValueError(f"unexpected Q12 matrix size: {count}")
        matrix = np.frombuffer(
            blob, dtype="<i2", count=count, offset=offset
        ).copy().reshape(MODEL_DIMENSION, MODEL_DIMENSION)
        offset += count * 2
        matrices.append((matrix, np.float32(scale)))
    return matrices


def load_layer0_weights(baseline: Path) -> dict:
    flat = np.fromfile(baseline / "weights.bin", dtype="<f4")
    if flat.size != 398_592:
        raise ValueError(f"unexpected fp32 weight count: {flat.size}")
    cursor = 0
    result = {}
    result["norm_weight"] = flat[cursor:cursor + MODEL_DIMENSION]
    cursor += MODEL_DIMENSION
    result["norm_bias"] = flat[cursor:cursor + MODEL_DIMENSION]
    cursor += MODEL_DIMENSION
    for name in ("query", "key", "value"):
        result[f"{name}_weight"] = flat[
            cursor:cursor + MODEL_DIMENSION * MODEL_DIMENSION
        ].reshape(MODEL_DIMENSION, MODEL_DIMENSION)
        cursor += MODEL_DIMENSION * MODEL_DIMENSION
        result[f"{name}_bias"] = flat[cursor:cursor + MODEL_DIMENSION]
        cursor += MODEL_DIMENSION
    q12 = parse_q12((baseline / "weights_q12.bin").read_bytes())
    for name, (matrix, scale) in zip(("query", "key", "value"), q12):
        result[f"{name}_q12"] = matrix
        result[f"{name}_q12_scale"] = scale
    return result


def project_q12(
    normalized_quantized: np.ndarray,
    normalized_scale: np.float32,
    weight: np.ndarray,
    weight_scale: np.float32,
    bias: np.ndarray,
) -> np.ndarray:
    accumulator = (
        normalized_quantized.astype(np.int32)
        @ weight.astype(np.int32).T
    )
    return (
        accumulator.astype(np.float32)
        * np.float32(normalized_scale * weight_scale)
        + bias
    ).astype(np.float32)


def attention(
    query: np.ndarray,
    query_scale: np.float32,
    key: np.ndarray,
    key_scale: np.float32,
    value: np.ndarray,
    value_scale: np.float32,
) -> np.ndarray:
    reconstructed_query = query.astype(np.float32) * query_scale
    reconstructed_key = key.astype(np.float32) * key_scale
    scores = (reconstructed_query @ reconstructed_key.T) * np.float32(
        1.0 / np.sqrt(np.float32(HEAD_DIMENSION))
    )
    scores[np.triu_indices(SEQUENCE, k=1)] = -np.inf
    maximum = np.max(scores, axis=1, keepdims=True)
    probability = np.exp(scores - maximum).astype(np.float32)
    probability /= np.sum(probability, axis=1, keepdims=True, dtype=np.float32)
    return (
        probability @ (value.astype(np.float32) * value_scale)
    ).astype(np.float32)


def official_attention(
    normalized: np.ndarray, weights: dict, head: int
) -> np.ndarray:
    projected = []
    for name in ("query", "key", "value"):
        full = (
            normalized @ weights[f"{name}_weight"].T
            + weights[f"{name}_bias"]
        ).astype(np.float32)
        projected.append(
            full[:, head * HEAD_DIMENSION:(head + 1) * HEAD_DIMENSION]
        )
    query, key, value = projected
    scores = (query @ key.T) * np.float32(
        1.0 / np.sqrt(np.float32(HEAD_DIMENSION))
    )
    scores[np.triu_indices(SEQUENCE, k=1)] = -np.inf
    maximum = np.max(scores, axis=1, keepdims=True)
    probability = np.exp(scores - maximum).astype(np.float32)
    probability /= np.sum(probability, axis=1, keepdims=True, dtype=np.float32)
    return (probability @ value).astype(np.float32)


def build_references(baseline: Path, seed: int) -> References:
    weights = load_layer0_weights(baseline)
    original = np.fromfile(
        baseline / "testdata" / f"input_{seed}.bin", dtype="<f4"
    ).reshape(SEQUENCE, MODEL_DIMENSION)
    input_quantized, input_scale = quantize(original, 32767, np.int16)
    reconstructed = input_quantized.astype(np.float32) * input_scale
    norm_kernel = layer_norm(
        reconstructed, weights["norm_weight"], weights["norm_bias"]
    )
    norm_official = layer_norm(
        original, weights["norm_weight"], weights["norm_bias"]
    )
    normalized_quantized, normalized_scale = quantize(
        norm_kernel, 32767, np.int16
    )

    projections = {}
    for name in ("query", "key", "value"):
        projections[name] = project_q12(
            normalized_quantized,
            normalized_scale,
            weights[f"{name}_q12"],
            weights[f"{name}_q12_scale"],
            weights[f"{name}_bias"],
        )

    context_kernel = []
    context_official = []
    for head in range(HEADS):
        begin = head * HEAD_DIMENSION
        end = begin + HEAD_DIMENSION
        query, query_scale = quantize(
            projections["query"][:, begin:end], 32767, np.int16
        )
        key, key_scale = quantize(
            projections["key"][:, begin:end], 32767, np.int16
        )
        value, value_scale = quantize(
            projections["value"][:, begin:end], 32767, np.int16
        )
        context_kernel.append(
            attention(query, query_scale, key, key_scale, value, value_scale)
        )
        context_official.append(official_attention(norm_official, weights, head))
    return References(
        input_quantized=input_quantized,
        input_scale=input_scale,
        norm_kernel=norm_kernel,
        norm_official=norm_official,
        context_kernel=context_kernel,
        context_official=context_official,
    )


def build_task(head: int, references: References) -> bytes:
    fixed = TASK_FIXED.pack(
        SEQUENCE,
        MODEL_DIMENSION,
        head,
        1,
        0,
        float(references.input_scale),
        float(LAYER_NORM_EPSILON),
    )
    return fixed + references.input_quantized.astype(">i2").tobytes()


def receive_exact(connection: socket.socket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = connection.recv(length - len(data))
        if not chunk:
            raise ConnectionError("worker closed the connection")
        data.extend(chunk)
    return bytes(data)


def send(
    connection: socket.socket, head: int, payload: bytes, iteration: int
) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int, int]:
    request_id = 0xC3A00000 | ((iteration & 0xFFFF) << 4) | head
    message = HEADER.pack(
        MAGIC, VERSION, TYPE_NORM_HEAD_TASK, len(payload), request_id
    ) + payload
    start = time.perf_counter_ns()
    connection.sendall(message)
    reply_header = receive_exact(connection, HEADER.size)
    magic, version, kind, payload_bytes, reply_id = HEADER.unpack(reply_header)
    reply = receive_exact(connection, payload_bytes)
    elapsed_us = (time.perf_counter_ns() - start) // 1000
    if (magic, version, kind, reply_id) != (
        MAGIC, VERSION, TYPE_NORM_HEAD_RESULT, request_id
    ):
        raise RuntimeError("invalid Norm-to-head response header")
    fields = RESULT_FIXED.unpack_from(reply)
    (
        sequence,
        dimension,
        reply_head,
        flags,
        status,
        reserved,
        elements,
        reserved_word,
        decode_us,
        layer_norm_us,
        projection_us,
        attention_us,
    ) = fields
    expected_bytes = RESULT_FIXED.size + elements * 8
    if (
        sequence != SEQUENCE
        or dimension != HEAD_DIMENSION
        or reply_head != head
        or flags != 1
        or status != 0
        or reserved != 0
        or reserved_word != 0
        or elements != SEQUENCE * HEAD_DIMENSION
        or payload_bytes != expected_bytes
    ):
        raise RuntimeError("invalid Norm-to-head response shape")
    context_begin = RESULT_FIXED.size
    norm_begin = context_begin + elements * 4
    context = np.frombuffer(
        reply, dtype=">f4", count=elements, offset=context_begin
    ).astype(np.float32).reshape(SEQUENCE, HEAD_DIMENSION)
    normalized = np.frombuffer(
        reply, dtype=">f4", count=elements, offset=norm_begin
    ).astype(np.float32).reshape(SEQUENCE, HEAD_DIMENSION)
    return (
        context,
        normalized,
        elapsed_us,
        decode_us,
        layer_norm_us,
        projection_us,
        attention_us,
        payload_bytes,
    )


def compare(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float, int]:
    difference = np.abs(
        actual.astype(np.float64) - expected.astype(np.float64)
    )
    relative = difference / np.maximum(np.abs(expected.astype(np.float64)), 1e-12)
    failed = (~np.isfinite(actual)) | (
        (difference > ABSOLUTE_TOLERANCE)
        & (difference > RELATIVE_TOLERANCE * np.abs(expected))
    )
    return float(np.max(difference)), float(np.max(relative)), int(np.sum(failed))


def run(
    workers: list[str], references: References, seed: int, timeout: float,
    warmups: int, repetitions: int,
) -> list[dict]:
    tasks = [build_task(head, references) for head in range(HEADS)]
    assignment = [head % len(workers) for head in range(HEADS)]
    assignment_text = ";".join(str(worker) for worker in assignment)
    connections = [
        socket.create_connection((worker, TCP_PORT), timeout) for worker in workers
    ]
    for connection in connections:
        connection.settimeout(timeout)

    tasks_by_worker: list[list[tuple[int, bytes]]] = [[] for _ in workers]
    for head, payload in enumerate(tasks):
        tasks_by_worker[assignment[head]].append((head, payload))

    def dispatch(iteration: int):
        begin = time.perf_counter_ns()

        def worker_batch(worker: int):
            return [
                (head, worker, *send(
                    connections[worker], head, payload, iteration
                ))
                for head, payload in tasks_by_worker[worker]
            ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(workers)
        ) as pool:
            results = [
                item
                for batch in pool.map(worker_batch, range(len(workers)))
                for item in batch
            ]
        return results, (time.perf_counter_ns() - begin) // 1000

    samples = [[] for _ in range(HEADS)]
    wall_samples = []
    try:
        for iteration in range(warmups):
            dispatch(iteration)
        for repetition in range(repetitions):
            results, wall_us = dispatch(warmups + repetition)
            wall_samples.append(wall_us)
            for result in results:
                samples[result[0]].append(result)
    finally:
        for connection in connections:
            connection.close()

    rows = []
    for head, head_samples in enumerate(samples):
        norm_slice = references.norm_kernel[
            :, head * HEAD_DIMENSION:(head + 1) * HEAD_DIMENSION
        ]
        norm_official_slice = references.norm_official[
            :, head * HEAD_DIMENSION:(head + 1) * HEAD_DIMENSION
        ]
        norm_kernel_comparisons = [
            compare(norm_slice, sample[3]) for sample in head_samples
        ]
        norm_official_comparisons = [
            compare(norm_official_slice, sample[3]) for sample in head_samples
        ]
        context_kernel_comparisons = [
            compare(references.context_kernel[head], sample[2])
            for sample in head_samples
        ]
        context_official_comparisons = [
            compare(references.context_official[head], sample[2])
            for sample in head_samples
        ]

        def worst(comparisons: list[tuple[float, float, int]], index: int):
            return max(result[index] for result in comparisons)

        norm_kernel_failed = worst(norm_kernel_comparisons, 2)
        norm_official_failed = worst(norm_official_comparisons, 2)
        context_kernel_failed = worst(context_kernel_comparisons, 2)
        context_official_failed = worst(context_official_comparisons, 2)
        status = "PASS" if not any((
            norm_kernel_failed,
            norm_official_failed,
            context_kernel_failed,
            context_official_failed,
        )) else "FAIL"
        representative = head_samples[-1]
        rows.append({
            "seed": seed,
            "kind": "head",
            "head": head,
            "worker": representative[1],
            "assignment": assignment_text,
            "request_payload_bytes": len(tasks[head]),
            "response_payload_bytes": representative[9],
            "elapsed_us": int(statistics.median(sample[4] for sample in head_samples)),
            "decode_us": int(statistics.median(sample[5] for sample in head_samples)),
            "layer_norm_us": int(statistics.median(sample[6] for sample in head_samples)),
            "projection_us": int(statistics.median(sample[7] for sample in head_samples)),
            "attention_us": int(statistics.median(sample[8] for sample in head_samples)),
            "norm_kernel_max_abs": worst(norm_kernel_comparisons, 0),
            "norm_official_max_abs": worst(norm_official_comparisons, 0),
            "norm_official_failed": norm_official_failed,
            "context_kernel_max_abs": worst(context_kernel_comparisons, 0),
            "context_official_max_abs": worst(context_official_comparisons, 0),
            "context_official_failed": context_official_failed,
            "status": status,
        })

    def critical_path(field: str) -> int:
        return max(
            sum(row[field] for row in rows if row["worker"] == worker)
            for worker in range(len(workers))
        )

    rows.append({
        "seed": seed,
        "kind": "full",
        "head": -1,
        "worker": -1,
        "assignment": assignment_text,
        "request_payload_bytes": sum(row["request_payload_bytes"] for row in rows),
        "response_payload_bytes": sum(row["response_payload_bytes"] for row in rows),
        "elapsed_us": int(statistics.median(wall_samples)),
        "decode_us": critical_path("decode_us"),
        "layer_norm_us": critical_path("layer_norm_us"),
        "projection_us": critical_path("projection_us"),
        "attention_us": critical_path("attention_us"),
        "norm_kernel_max_abs": max(row["norm_kernel_max_abs"] for row in rows),
        "norm_official_max_abs": max(row["norm_official_max_abs"] for row in rows),
        "norm_official_failed": sum(row["norm_official_failed"] for row in rows),
        "context_kernel_max_abs": max(row["context_kernel_max_abs"] for row in rows),
        "context_official_max_abs": max(row["context_official_max_abs"] for row in rows),
        "context_official_failed": sum(row["context_official_failed"] for row in rows),
        "status": "PASS" if all(row["status"] == "PASS" for row in rows) else "FAIL",
    })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", required=True)
    parser.add_argument("--baseline", type=Path, default=Path("esp32-baseline"))
    parser.add_argument(
        "--seeds", default="0", help="comma-separated exported input seeds"
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workers = [item.strip() for item in args.workers.split(",") if item.strip()]
    if not workers:
        parser.error("at least one worker is required")
    if len(workers) > HEADS:
        parser.error("at most four workers can receive the four heads")

    try:
        seeds = [int(item.strip()) for item in args.seeds.split(",")]
    except ValueError as error:
        parser.error(f"invalid --seeds value: {error}")
    if not seeds:
        parser.error("at least one seed is required")
    rows = []
    for seed in seeds:
        references = build_references(args.baseline, seed)
        rows.extend(run(
            workers, references, seed, args.timeout,
            args.warmups, args.repetitions,
        ))
    writer = csv.DictWriter(
        sys.stdout, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output:
            file_writer = csv.DictWriter(
                output, fieldnames=list(rows[0]), lineterminator="\n"
            )
            file_writer.writeheader()
            file_writer.writerows(rows)
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
