#!/usr/bin/env python3
"""Run exact distributed online-softmax KV shards on ESP32 workers."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "..", "..", "..", "experiments",
                                                 "attention-layer", "tools")))


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

from run_head_parallel import (
    quantize,
    reference_context,
    visible_mask,
)
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
TYPE_KV_SHARD_TASK = 7
TYPE_KV_SHARD_RESULT = 8
UDP_PORT = 4210
HEADER = struct.Struct("!IBBHI")
TASK_FIXED = struct.Struct("!HHBBHHHfff")
RESULT_FIXED = struct.Struct("!HHBBHHHII")
SHARDS = 4


@dataclass
class ShardResult:
    head: int
    shard: int
    worker: int
    maximums: list[float]
    sums: list[float]
    numerators: list[list[float]]
    rtt_us: int
    decode_us: int
    compute_us: int
    request_payload_bytes: int
    response_payload_bytes: int


def extract_query_head(values: list[int], head: int) -> list[int]:
    begin = head * HEAD_DIMENSION
    return [
        values[token * MODEL_DIMENSION + begin + feature]
        for token in range(SEQUENCE)
        for feature in range(HEAD_DIMENSION)
    ]


def extract_kv_shard(
    values: list[int], head: int, shard_begin: int, shard_end: int
) -> list[int]:
    begin = head * HEAD_DIMENSION
    return [
        values[token * MODEL_DIMENSION + begin + feature]
        for token in range(shard_begin, shard_end)
        for feature in range(HEAD_DIMENSION)
    ]


def build_task(
    head: int,
    shard_begin: int,
    shard_end: int,
    causal: bool,
    query: list[int],
    key: list[int],
    value: list[int],
    scales: tuple[float, float, float],
) -> bytes:
    mask = visible_mask()
    query_head = extract_query_head(query, head)
    key_shard = extract_kv_shard(key, head, shard_begin, shard_end)
    value_shard = extract_kv_shard(value, head, shard_begin, shard_end)
    fixed = TASK_FIXED.pack(
        SEQUENCE,
        HEAD_DIMENSION,
        1 if causal else 0,
        0,
        len(mask),
        shard_begin,
        shard_end,
        *scales,
    )
    return (
        fixed
        + mask
        + bytes(item & 0xFF for item in query_head)
        + bytes(item & 0xFF for item in key_shard)
        + struct.pack(f"!{len(value_shard)}h", *value_shard)
    )


def send_task(
    host: str,
    worker: int,
    head: int,
    shard: int,
    causal: bool,
    payload: bytes,
    timeout: float,
    retries: int,
    iteration: int,
) -> ShardResult:
    request_id = (
        0xD3000000
        | ((1 if causal else 0) << 20)
        | ((iteration & 0xFFF) << 8)
        | (head << 4)
        | shard
    )
    message = HEADER.pack(
        MAGIC, VERSION, TYPE_KV_SHARD_TASK, len(payload), request_id
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
                or kind != TYPE_KV_SHARD_RESULT
                or reply_id != request_id
                or len(reply) != HEADER.size + payload_bytes
            ):
                continue
            fixed = RESULT_FIXED.unpack_from(reply, HEADER.size)
            (
                sequence,
                dimension,
                flags,
                status,
                shard_begin,
                shard_end,
                query_count,
                decode_us,
                compute_us,
            ) = fixed
            statistics_count = sequence * (dimension + 2)
            if (
                sequence != SEQUENCE
                or dimension != HEAD_DIMENSION
                or flags != (1 if causal else 0)
                or status != 0
                or query_count != SEQUENCE
                or payload_bytes != RESULT_FIXED.size + statistics_count * 4
            ):
                continue
            values = struct.unpack_from(
                f"!{statistics_count}f",
                reply,
                HEADER.size + RESULT_FIXED.size,
            )
            maximums = []
            sums = []
            numerators = []
            cursor = 0
            for _ in range(SEQUENCE):
                maximums.append(values[cursor])
                sums.append(values[cursor + 1])
                numerators.append(list(values[cursor + 2:cursor + 2 + HEAD_DIMENSION]))
                cursor += HEAD_DIMENSION + 2
            return ShardResult(
                head,
                shard,
                worker,
                maximums,
                sums,
                numerators,
                elapsed_us,
                decode_us,
                compute_us,
                len(payload),
                payload_bytes,
            )
    raise RuntimeError(
        f"invalid replies for head {head}, shard {shard} from worker {worker}"
    )


def merge_shards(shards: list[ShardResult], value_scale: float) -> list[float]:
    output = [[0.0] * HEAD_DIMENSION for _ in range(SEQUENCE)]
    for query_index in range(SEQUENCE):
        active = [shard for shard in shards if shard.sums[query_index] > 0.0]
        if not active:
            continue
        maximum = max(shard.maximums[query_index] for shard in active)
        denominator = 0.0
        numerator = [0.0] * HEAD_DIMENSION
        for shard in active:
            scale = math.exp(shard.maximums[query_index] - maximum)
            denominator += scale * shard.sums[query_index]
            for feature in range(HEAD_DIMENSION):
                numerator[feature] += (
                    scale * shard.numerators[query_index][feature]
                )
        for feature in range(HEAD_DIMENSION):
            output[query_index][feature] = (
                numerator[feature] * value_scale / denominator
            )
    return [item for row in output for item in row]


def run_case(
    workers: list[str],
    causal: bool,
    timeout: float,
    retries: int,
    warmups: int,
    repetitions: int,
) -> list[dict]:
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
    shard_width = SEQUENCE // SHARDS
    tasks = []
    for head in range(HEADS):
        for shard in range(SHARDS):
            shard_begin = shard * shard_width
            shard_end = SEQUENCE if shard == SHARDS - 1 else shard_begin + shard_width
            worker = shard % len(workers)
            tasks.append(
                (
                    workers[worker],
                    worker,
                    head,
                    shard,
                    build_task(
                        head,
                        shard_begin,
                        shard_end,
                        causal,
                        query,
                        key,
                        value,
                        scales,
                    ),
                )
            )

    def dispatch(iteration: int) -> tuple[list[ShardResult], int]:
        start = time.perf_counter_ns()
        if len(workers) == 1:
            dispatched = [
                send_task(
                    host,
                    worker,
                    head,
                    shard,
                    causal,
                    payload,
                    timeout,
                    retries,
                    iteration,
                )
                for host, worker, head, shard, payload in tasks
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
                        shard,
                        causal,
                        payload,
                        timeout,
                        retries,
                        iteration,
                    )
                    for host, worker, head, shard, payload in tasks
                ]
                dispatched = [future.result() for future in futures]
        elapsed_us = (time.perf_counter_ns() - start) // 1000
        dispatched.sort(key=lambda result: (result.head, result.shard))
        return dispatched, elapsed_us

    for warmup in range(warmups):
        dispatch(warmup)
    samples: list[list[ShardResult]] = [[] for _ in range(HEADS * SHARDS)]
    batch_samples = []
    for repetition in range(repetitions):
        dispatched, elapsed_us = dispatch(warmups + repetition)
        batch_samples.append(elapsed_us)
        for result in dispatched:
            samples[result.head * SHARDS + result.shard].append(result)

    results = []
    for shard_samples in samples:
        representative = shard_samples[-1]
        results.append(
            ShardResult(
                representative.head,
                representative.shard,
                representative.worker,
                representative.maximums,
                representative.sums,
                representative.numerators,
                int(statistics.median(item.rtt_us for item in shard_samples)),
                int(statistics.median(item.decode_us for item in shard_samples)),
                int(statistics.median(item.compute_us for item in shard_samples)),
                representative.request_payload_bytes,
                representative.response_payload_bytes,
            )
        )

    rows = []
    context = [[0.0] * MODEL_DIMENSION for _ in range(SEQUENCE)]
    for result in results:
        rows.append(
            {
                "kind": "shard",
                "causal": int(causal),
                "head": result.head,
                "shard": result.shard,
                "worker": result.worker,
                "request_payload_bytes": result.request_payload_bytes,
                "response_payload_bytes": result.response_payload_bytes,
                "elapsed_us": result.rtt_us,
                "decode_us": result.decode_us,
                "compute_us": result.compute_us,
                "max_abs_error": "",
                "max_relative_error": "",
                "failed_elements": 0,
                "status": "PASS",
            }
        )

    for head in range(HEADS):
        head_shards = [result for result in results if result.head == head]
        merged = merge_shards(head_shards, value_scale)
        expected = reference_context(
            query_float, key_float, value_float, head, causal
        )
        stats = compare(expected, merged)
        for token in range(SEQUENCE):
            for feature in range(HEAD_DIMENSION):
                context[token][head * HEAD_DIMENSION + feature] = (
                    merged[token * HEAD_DIMENSION + feature]
                )
        rows.append(
            {
                "kind": "head_merge",
                "causal": int(causal),
                "head": head,
                "shard": -1,
                "worker": -1,
                "request_payload_bytes": sum(
                    item.request_payload_bytes for item in head_shards
                ),
                "response_payload_bytes": sum(
                    item.response_payload_bytes for item in head_shards
                ),
                "elapsed_us": max(item.rtt_us for item in head_shards),
                "decode_us": sum(item.decode_us for item in head_shards),
                "compute_us": sum(item.compute_us for item in head_shards),
                "max_abs_error": stats[0],
                "max_relative_error": stats[1],
                "failed_elements": stats[2],
                "status": "PASS" if stats[2] == 0 else "FAIL",
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
    rows.append(
        {
            "kind": "full",
            "causal": int(causal),
            "head": -1,
            "shard": -1,
            "worker": -1,
            "request_payload_bytes": sum(
                result.request_payload_bytes for result in results
            ),
            "response_payload_bytes": sum(
                result.response_payload_bytes for result in results
            ),
            "elapsed_us": int(statistics.median(batch_samples)),
            "decode_us": sum(result.decode_us for result in results),
            "compute_us": max(compute_by_worker),
            "max_abs_error": full_stats[0],
            "max_relative_error": full_stats[1],
            "failed_elements": full_stats[2],
            "status": "PASS" if full_stats[2] == 0 else "FAIL",
        }
    )
    return rows


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
        rows.extend(
            run_case(
                workers,
                causal,
                args.timeout,
                args.retries,
                args.warmups,
                args.repetitions,
            )
        )
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
