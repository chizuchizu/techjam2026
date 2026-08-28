#!/usr/bin/env python3
"""Dispatch four official-size synthetic attention heads over persistent TCP."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import math
import socket
import statistics
import struct
import sys
import time
from pathlib import Path

MAGIC = 0x45535033
VERSION = 1
TYPE_HEAD_TASK = 5
TYPE_HEAD_RESULT = 6
TCP_PORT = 4211
SEQUENCE = 128
DIMENSION = 32
HEADS = 4
TILE_SIZE = 32
HEADER = struct.Struct("!IBBHI")
TASK_FIXED = struct.Struct("!HHBBHfff")
RESULT_FIXED = struct.Struct("!HHBBHII")
ABSOLUTE_TOLERANCE = 0.002
RELATIVE_TOLERANCE = 0.02


def inputs(head: int) -> tuple[list[int], list[int], list[int]]:
    query = [((token * 17 + feature * 29 + head * 11) % 255) - 127
             for token in range(SEQUENCE) for feature in range(DIMENSION)]
    key = [((token * 31 + feature * 13 + head * 19) % 255) - 127
           for token in range(SEQUENCE) for feature in range(DIMENSION)]
    value = [
        ((token * 977 + feature * 619 + head * 1237) % 60001) - 30000
        for token in range(SEQUENCE) for feature in range(DIMENSION)
    ]
    return query, key, value


def build_task(head: int, causal: bool) -> tuple[bytes, list[int], list[int], list[int]]:
    query, key, value = inputs(head)
    mask = bytes([0xFF] * ((SEQUENCE + 7) // 8))
    scales = (0.0075, 0.00625, 0.0005)
    fixed = TASK_FIXED.pack(
        SEQUENCE, DIMENSION, 1 if causal else 0, TILE_SIZE, len(mask), *scales
    )
    payload = (
        fixed + mask + bytes(item & 0xFF for item in query)
        + bytes(item & 0xFF for item in key)
        + struct.pack(f"!{len(value)}h", *value)
    )
    return payload, query, key, value


def reference(query: list[int], key: list[int], value: list[int], causal: bool) -> list[float]:
    query_scale, key_scale, value_scale = 0.0075, 0.00625, 0.0005
    score_scale = query_scale * key_scale / math.sqrt(DIMENSION)
    output = [0.0] * (SEQUENCE * DIMENSION)
    for query_index in range(SEQUENCE):
        last_key = query_index + 1 if causal else SEQUENCE
        scores = []
        for key_index in range(last_key):
            dot = sum(
                query[query_index * DIMENSION + feature]
                * key[key_index * DIMENSION + feature]
                for feature in range(DIMENSION)
            )
            scores.append(dot * score_scale)
        maximum = max(scores)
        weights = [math.exp(score - maximum) for score in scores]
        denominator = sum(weights)
        for key_index, weight in enumerate(weights):
            normalized = weight * value_scale / denominator
            for feature in range(DIMENSION):
                output[query_index * DIMENSION + feature] += (
                    normalized * value[key_index * DIMENSION + feature]
                )
    return output


def receive_exact(connection: socket.socket, length: int) -> bytes:
    data = bytearray()
    while len(data) < length:
        chunk = connection.recv(length - len(data))
        if not chunk:
            raise ConnectionError("worker closed the connection")
        data.extend(chunk)
    return bytes(data)


def send(
    connection: socket.socket,
    head: int,
    causal: bool,
    payload: bytes,
    iteration: int,
) -> tuple[list[float], int, int, int, int]:
    request_id = (
        0xC3800000 | ((1 if causal else 0) << 20)
        | ((iteration & 0xFFFF) << 4) | head
    )
    message = HEADER.pack(MAGIC, VERSION, TYPE_HEAD_TASK,
                          len(payload), request_id) + payload
    start = time.perf_counter_ns()
    connection.sendall(message)
    reply_header = receive_exact(connection, HEADER.size)
    magic, version, kind, payload_bytes, reply_id = HEADER.unpack(reply_header)
    reply = receive_exact(connection, payload_bytes)
    elapsed_us = (time.perf_counter_ns() - start) // 1000
    if (magic, version, kind, reply_id) != (
        MAGIC, VERSION, TYPE_HEAD_RESULT, request_id
    ):
        raise RuntimeError("invalid large-head response header")
    fields = RESULT_FIXED.unpack_from(reply)
    sequence, dimension, flags, status, elements, decode_us, compute_us = fields
    if (
        sequence != SEQUENCE or dimension != DIMENSION
        or flags != (1 if causal else 0) or status != 0
        or elements != SEQUENCE * DIMENSION
        or payload_bytes != RESULT_FIXED.size + elements * 4
    ):
        raise RuntimeError("invalid large-head response shape")
    output = list(struct.unpack_from(f"!{elements}f", reply, RESULT_FIXED.size))
    return output, elapsed_us, decode_us, compute_us, payload_bytes


def compare(expected: list[float], actual: list[float]) -> tuple[float, float, int]:
    maximum_absolute = 0.0
    maximum_relative = 0.0
    failed = 0
    for reference_value, actual_value in zip(expected, actual):
        absolute = abs(actual_value - reference_value)
        relative = absolute / max(abs(reference_value), 1e-12)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        if (
            absolute > ABSOLUTE_TOLERANCE
            and absolute > RELATIVE_TOLERANCE * abs(reference_value)
        ):
            failed += 1
    return maximum_absolute, maximum_relative, failed


def assign_heads(
    worker_elapsed_us: list[int], require_all_workers: bool = False
) -> list[int]:
    """Greedily minimize the predicted makespan for equal-size head tasks."""
    if not worker_elapsed_us or any(elapsed <= 0 for elapsed in worker_elapsed_us):
        raise ValueError("worker calibration times must be positive")
    if require_all_workers and len(worker_elapsed_us) > HEADS:
        raise ValueError("cannot assign at least one head to every worker")

    loads = [0] * len(worker_elapsed_us)
    assignment = []
    if require_all_workers:
        for worker in range(len(worker_elapsed_us)):
            assignment.append(worker)
            loads[worker] += worker_elapsed_us[worker]
    while len(assignment) < HEADS:
        worker = min(
            range(len(worker_elapsed_us)),
            key=lambda candidate: (
                loads[candidate] + worker_elapsed_us[candidate], candidate
            ),
        )
        assignment.append(worker)
        loads[worker] += worker_elapsed_us[worker]
    return assignment


def run_case(
    workers: list[str], causal: bool, timeout: float,
    warmups: int, repetitions: int, scheduler: str
) -> list[dict]:
    tasks = [build_task(head, causal) for head in range(HEADS)]
    references = [reference(query, key, value, causal)
                  for _, query, key, value in tasks]
    connections = [socket.create_connection((host, TCP_PORT), timeout)
                   for host in workers]
    for connection in connections:
        connection.settimeout(timeout)

    calibration_us = []
    if scheduler == "round-robin":
        assignment = [head % len(workers) for head in range(HEADS)]
    else:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(workers)
        ) as pool:
            calibration = list(pool.map(
                lambda worker: send(
                    connections[worker], 0, causal, tasks[0][0], 0xF000 + worker
                ),
                range(len(workers)),
            ))
        calibration_us = [sample[1] for sample in calibration]
        assignment = assign_heads(
            calibration_us, require_all_workers=scheduler == "calibrated-all"
        )

    tasks_by_worker = [[] for _ in workers]
    for head, task in enumerate(tasks):
        tasks_by_worker[assignment[head]].append((head, task[0]))
    assignment_text = ";".join(str(worker) for worker in assignment)
    calibration_text = ";".join(str(elapsed) for elapsed in calibration_us)

    def dispatch(iteration: int):
        begin = time.perf_counter_ns()

        def worker_batch(worker: int):
            return [
                (head, worker, *send(connections[worker], head, causal,
                                     payload, iteration))
                for head, payload in tasks_by_worker[worker]
            ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(workers)
        ) as pool:
            results = [item for batch in pool.map(worker_batch, range(len(workers)))
                       for item in batch]
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
        representative = head_samples[-1]
        _, worker, output, _, _, _, response_bytes = representative
        maximum_absolute, maximum_relative, failed = compare(
            references[head], output
        )
        rows.append(
            {
                "kind": "head",
                "causal": int(causal),
                "head": head,
                "worker": worker,
                "scheduler": scheduler,
                "assignment": assignment_text,
                "calibration_us": calibration_text,
                "request_payload_bytes": len(tasks[head][0]),
                "response_payload_bytes": response_bytes,
                "elapsed_us": int(statistics.median(item[3]
                                                     for item in head_samples)),
                "decode_us": int(statistics.median(item[4]
                                                    for item in head_samples)),
                "compute_us": int(statistics.median(item[5]
                                                     for item in head_samples)),
                "max_abs_error": maximum_absolute,
                "max_relative_error": maximum_relative,
                "failed_elements": failed,
                "status": "PASS" if failed == 0 else "FAIL",
            }
        )
    rows.append(
        {
            "kind": "full",
            "causal": int(causal),
            "head": -1,
            "worker": -1,
            "scheduler": scheduler,
            "assignment": assignment_text,
            "calibration_us": calibration_text,
            "request_payload_bytes": sum(len(task[0]) for task in tasks),
            "response_payload_bytes": sum(row["response_payload_bytes"]
                                          for row in rows),
            "elapsed_us": int(statistics.median(wall_samples)),
            "decode_us": sum(row["decode_us"] for row in rows),
            "compute_us": max(
                sum(row["compute_us"] for row in rows
                    if row["worker"] == worker)
                for worker in range(len(workers))
            ),
            "max_abs_error": max(row["max_abs_error"] for row in rows),
            "max_relative_error": max(row["max_relative_error"] for row in rows),
            "failed_elements": sum(row["failed_elements"] for row in rows),
            "status": "PASS" if all(row["status"] == "PASS" for row in rows)
            else "FAIL",
        }
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--scheduler",
        choices=("round-robin", "calibrated", "calibrated-all"),
        default="round-robin",
        help=(
            "head assignment policy; calibrated minimizes predicted latency, "
            "while calibrated-all also assigns at least one head to every worker"
        ),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    workers = [item.strip() for item in args.workers.split(",") if item.strip()]
    if not workers:
        parser.error("at least one worker is required")
    if args.scheduler == "calibrated-all" and len(workers) > HEADS:
        parser.error("calibrated-all supports at most four workers")
    rows = []
    for causal in (False, True):
        rows.extend(run_case(workers, causal, args.timeout,
                             args.warmups, args.repetitions, args.scheduler))
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
