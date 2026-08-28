#!/usr/bin/env python3
"""Dispatch deterministic attention heads to ESP32 UDP workers."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import socket
import statistics
import struct
import time
from dataclasses import dataclass
from pathlib import Path

from validate_e2e import (
    HEADS,
    HEAD_DIMENSION,
    MODEL_DIMENSION,
    SEQUENCE,
    compare,
    fixture_input,
    project,
    reference,
    token_is_valid,
)

MAGIC = 0x45535033
VERSION = 1
TYPE_HEAD_TASK = 5
TYPE_HEAD_RESULT = 6
UDP_PORT = 4210
HEADER = struct.Struct("!IBBHI")
TASK_FIXED = struct.Struct("!HHBBHfff")
RESULT_FIXED = struct.Struct("!HHBBHII")
TILE_SIZE = 8


@dataclass
class HeadResult:
    head: int
    worker: int
    output: list[float]
    rtt_us: int
    decode_us: int
    compute_us: int
    request_payload_bytes: int
    response_payload_bytes: int


def round_away_from_zero(value: float) -> int:
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def quantize(values: list[list[float]], limit: int) -> tuple[list[int], float]:
    flat = [value for row in values for value in row]
    maximum = max(abs(value) for value in flat)
    scale = maximum / limit if maximum > 0.0 else 1.0
    quantized = [
        max(-limit, min(limit, round_away_from_zero(value / scale)))
        for value in flat
    ]
    return quantized, scale


def visible_mask() -> bytes:
    mask = bytearray((SEQUENCE + 7) // 8)
    for token in range(SEQUENCE):
        if token_is_valid(token):
            mask[token // 8] |= 1 << (token % 8)
    return bytes(mask)


def extract_head(values: list[int], head: int) -> list[int]:
    begin = head * HEAD_DIMENSION
    return [
        values[token * MODEL_DIMENSION + begin + feature]
        for token in range(SEQUENCE)
        for feature in range(HEAD_DIMENSION)
    ]


def build_task(
    head: int,
    causal: bool,
    query: list[int],
    key: list[int],
    value: list[int],
    scales: tuple[float, float, float],
) -> bytes:
    mask = visible_mask()
    query_head = extract_head(query, head)
    key_head = extract_head(key, head)
    value_head = extract_head(value, head)
    fixed = TASK_FIXED.pack(
        SEQUENCE,
        HEAD_DIMENSION,
        1 if causal else 0,
        TILE_SIZE,
        len(mask),
        *scales,
    )
    return (
        fixed
        + mask
        + bytes(value & 0xFF for value in query_head)
        + bytes(value & 0xFF for value in key_head)
        + struct.pack(f"!{len(value_head)}h", *value_head)
    )


def send_task(
    host: str,
    worker: int,
    head: int,
    causal: bool,
    payload: bytes,
    timeout: float,
    retries: int,
    iteration: int,
) -> HeadResult:
    request_id = (
        0xC3000000
        | ((1 if causal else 0) << 20)
        | ((iteration & 0xFFFF) << 4)
        | head
    )
    message = HEADER.pack(
        MAGIC, VERSION, TYPE_HEAD_TASK, len(payload), request_id
    ) + payload
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.settimeout(timeout)
        for attempt in range(retries + 1):
            start = time.perf_counter_ns()
            connection.sendto(message, (host, UDP_PORT))
            try:
                reply, _ = connection.recvfrom(2048)
            except TimeoutError:
                if attempt == retries:
                    raise
                continue
            elapsed_us = (time.perf_counter_ns() - start) // 1000
            if len(reply) < HEADER.size + RESULT_FIXED.size:
                continue
            magic, version, kind, payload_bytes, reply_id = HEADER.unpack_from(reply)
            if (
                magic != MAGIC
                or version != VERSION
                or kind != TYPE_HEAD_RESULT
                or reply_id != request_id
                or len(reply) != HEADER.size + payload_bytes
            ):
                continue
            fixed = RESULT_FIXED.unpack_from(reply, HEADER.size)
            sequence, dimension, flags, status, elements, decode_us, compute_us = fixed
            if (
                sequence != SEQUENCE
                or dimension != HEAD_DIMENSION
                or flags != (1 if causal else 0)
                or status != 0
                or elements != SEQUENCE * HEAD_DIMENSION
                or payload_bytes != RESULT_FIXED.size + elements * 4
            ):
                continue
            output = list(
                struct.unpack_from(
                    f"!{elements}f", reply, HEADER.size + RESULT_FIXED.size
                )
            )
            return HeadResult(
                head,
                worker,
                output,
                elapsed_us,
                decode_us,
                compute_us,
                len(payload),
                payload_bytes,
            )
    raise RuntimeError(f"invalid replies for head {head} from worker {worker}")


def reference_context(
    query: list[list[float]],
    key: list[list[float]],
    value: list[list[float]],
    head: int,
    causal: bool,
) -> list[float]:
    begin = head * HEAD_DIMENSION
    end = begin + HEAD_DIMENSION
    output = [[0.0] * HEAD_DIMENSION for _ in range(SEQUENCE)]
    scale = 1.0 / math.sqrt(HEAD_DIMENSION)
    for query_index in range(SEQUENCE):
        if not token_is_valid(query_index):
            continue
        keys = [
            key_index
            for key_index in range(SEQUENCE)
            if token_is_valid(key_index)
            and (not causal or key_index <= query_index)
        ]
        scores = [
            sum(
                query[query_index][feature] * key[key_index][feature]
                for feature in range(begin, end)
            )
            * scale
            for key_index in keys
        ]
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        denominator = sum(weights)
        for key_index, weight in zip(keys, weights):
            for local_feature, feature in enumerate(range(begin, end)):
                output[query_index][local_feature] += (
                    weight / denominator * value[key_index][feature]
                )
    return [item for row in output for item in row]


def run_case(
    workers: list[str],
    causal: bool,
    timeout: float,
    retries: int,
    warmups: int,
    repetitions: int,
) -> tuple[list[dict], dict]:
    inputs = [
        [fixture_input(token, feature) for feature in range(MODEL_DIMENSION)]
        for token in range(SEQUENCE)
    ]
    query_float = project(inputs, 0)
    key_float = project(inputs, 1)
    value_float = project(inputs, 2)
    query, query_scale = quantize(query_float, 127)
    key, key_scale = quantize(key_float, 127)
    value, value_scale = quantize(value_float, 32767)
    scales = (query_scale, key_scale, value_scale)
    tasks = [
        (
            workers[head % len(workers)],
            head % len(workers),
            head,
            build_task(head, causal, query, key, value, scales),
        )
        for head in range(HEADS)
    ]

    def dispatch(iteration: int) -> tuple[list[HeadResult], int]:
        batch_start = time.perf_counter_ns()
        if len(workers) == 1:
            dispatched = [
                send_task(
                    host,
                    worker,
                    head,
                    causal,
                    payload,
                    timeout,
                    retries,
                    iteration,
                )
                for host, worker, head, payload in tasks
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(workers)
            ) as pool:
                futures = [
                    pool.submit(
                        send_task,
                        host,
                        worker,
                        head,
                        causal,
                        payload,
                        timeout,
                        retries,
                        iteration,
                    )
                    for host, worker, head, payload in tasks
                ]
                dispatched = [future.result() for future in futures]
        batch_elapsed = (time.perf_counter_ns() - batch_start) // 1000
        dispatched.sort(key=lambda result: result.head)
        return dispatched, batch_elapsed

    for warmup in range(warmups):
        dispatch(warmup)
    samples: list[list[HeadResult]] = [[] for _ in range(HEADS)]
    batch_samples = []
    for repetition in range(repetitions):
        dispatched, batch_elapsed = dispatch(warmups + repetition)
        batch_samples.append(batch_elapsed)
        for result in dispatched:
            samples[result.head].append(result)

    results = []
    for head_samples in samples:
        representative = head_samples[-1]
        results.append(
            HeadResult(
                representative.head,
                representative.worker,
                representative.output,
                int(statistics.median(item.rtt_us for item in head_samples)),
                int(statistics.median(item.decode_us for item in head_samples)),
                int(statistics.median(item.compute_us for item in head_samples)),
                representative.request_payload_bytes,
                representative.response_payload_bytes,
            )
        )
    batch_elapsed_us = int(statistics.median(batch_samples))
    rows = []
    context = [[0.0] * MODEL_DIMENSION for _ in range(SEQUENCE)]
    for result in results:
        expected = reference_context(
            query_float, key_float, value_float, result.head, causal
        )
        max_abs, max_relative, failed = compare(expected, result.output)
        for token in range(SEQUENCE):
            for feature in range(HEAD_DIMENSION):
                context[token][result.head * HEAD_DIMENSION + feature] = (
                    result.output[token * HEAD_DIMENSION + feature]
                )
        rows.append(
            {
                "kind": "head",
                "causal": int(causal),
                "head": result.head,
                "worker": result.worker,
                "request_payload_bytes": result.request_payload_bytes,
                "response_payload_bytes": result.response_payload_bytes,
                "elapsed_us": result.rtt_us,
                "decode_us": result.decode_us,
                "compute_us": result.compute_us,
                "max_abs_error": max_abs,
                "max_relative_error": max_relative,
                "failed_elements": failed,
                "status": "PASS" if failed == 0 else "FAIL",
            }
        )

    candidate = project(context, 3)
    for token in range(SEQUENCE):
        if not token_is_valid(token):
            candidate[token] = [0.0] * MODEL_DIMENSION
    candidate_flat = [item for row in candidate for item in row]
    full_stats = compare(reference(causal), candidate_flat)
    compute_by_worker = [0] * len(workers)
    for result in results:
        compute_by_worker[result.worker] += result.compute_us
    full_row = {
        "kind": "full",
        "causal": int(causal),
        "head": -1,
        "worker": -1,
        "request_payload_bytes": sum(r.request_payload_bytes for r in results),
        "response_payload_bytes": sum(r.response_payload_bytes for r in results),
        "elapsed_us": batch_elapsed_us,
        "decode_us": sum(r.decode_us for r in results),
        "compute_us": max(compute_by_worker),
        "max_abs_error": full_stats[0],
        "max_relative_error": full_stats[1],
        "failed_elements": full_stats[2],
        "status": "PASS" if full_stats[2] == 0 else "FAIL",
    }
    return rows, full_row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", required=True, help="comma-separated ESP32 worker IPs"
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workers = [worker.strip() for worker in args.workers.split(",") if worker.strip()]
    if not workers:
        parser.error("at least one worker IP is required")
    if args.warmups < 0 or args.repetitions <= 0 or args.retries < 0:
        parser.error("warmups/retries must be non-negative and repetitions positive")

    rows = []
    for causal in (False, True):
        head_rows, full_row = run_case(
            workers,
            causal,
            args.timeout,
            args.retries,
            args.warmups,
            args.repetitions,
        )
        rows.extend(head_rows)
        rows.append(full_row)

    fieldnames = list(rows[0])
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output_file:
            output_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            output_writer.writeheader()
            output_writer.writerows(rows)
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
