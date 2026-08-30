#!/usr/bin/env python3
"""Run exact tiled causal attention across two ESP32 TCP workers.

The workers compute independent KV-block online-softmax states.  This
coordinator keeps query tiles stationary, schedules KV tiles round-robin, and
merges (maximum, denominator, numerator) exactly before normalising.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import socket
import struct
import time
from dataclasses import dataclass

import numpy as np
import serial


MAGIC = 0x45535033
VERSION = 1
TYPE_RING_TILE_TASK = 13
TYPE_RING_TILE_RESULT = 14
TCP_PORT = 4211
HEADER = struct.Struct("!IBBHI")
TASK_FIXED = struct.Struct("!IIHHHBBHHfff")
RESULT_FIXED = struct.Struct("!IIHHBBHII")


def recv_exact(connection: socket.socket, count: int) -> bytes:
    chunks: list[bytes] = []
    received = 0
    while received < count:
        chunk = connection.recv(count - received)
        if not chunk:
            raise ConnectionError("ESP32 closed the TCP connection")
        chunks.append(chunk)
        received += len(chunk)
    return b"".join(chunks)


def packed_valid_mask(count: int) -> bytes:
    result = bytearray((count + 7) // 8)
    for index in range(count):
        result[index // 8] |= 1 << (index % 8)
    return bytes(result)


def quantize(values: np.ndarray, limit: int) -> tuple[np.ndarray, float]:
    maximum = float(np.max(np.abs(values)))
    scale = maximum / limit if maximum else 1.0
    quantized = np.clip(np.rint(values / scale), -limit, limit)
    dtype = np.int8 if limit == 127 else np.int16
    return quantized.astype(dtype), scale


@dataclass
class TileResult:
    query_begin: int
    kv_begin: int
    maximum: np.ndarray
    denominator: np.ndarray
    numerator: np.ndarray
    decode_us: int
    compute_us: int
    elapsed_us: int


class Worker:
    def __init__(
        self,
        name: str,
        timeout: float,
        connection: socket.socket | serial.Serial,
    ) -> None:
        self.host = name
        self.connection = connection
        self.timeout = timeout

    @classmethod
    def tcp(cls, host: str, timeout: float) -> "Worker":
        connection = socket.create_connection((host, TCP_PORT), timeout)
        connection.settimeout(timeout)
        return cls(host, timeout, connection)

    @classmethod
    def serial(cls, port: str, timeout: float, baud: int) -> "Worker":
        connection = serial.Serial(port, baud, timeout=timeout, write_timeout=timeout)
        connection.dtr = False
        connection.rts = False
        ready_deadline = time.monotonic() + min(timeout, 10.0)
        ready = False
        while time.monotonic() < ready_deadline:
            line = connection.readline()
            if b"SERIAL_RING_READY" in line:
                ready = True
                break
        if not ready:
            connection.close()
            raise TimeoutError(f"{port} did not announce SERIAL_RING_READY")
        connection.reset_input_buffer()
        return cls(port, timeout, connection)

    def sendall(self, data: bytes) -> None:
        if isinstance(self.connection, socket.socket):
            self.connection.sendall(data)
            return
        written = self.connection.write(data)
        self.connection.flush()
        if written != len(data):
            raise ConnectionError(f"short serial write to {self.host}")

    def receive_exact(self, count: int) -> bytes:
        if isinstance(self.connection, socket.socket):
            return recv_exact(self.connection, count)
        chunks: list[bytes] = []
        received = 0
        deadline = time.monotonic() + self.timeout
        while received < count and time.monotonic() < deadline:
            chunk = self.connection.read(count - received)
            if chunk:
                chunks.append(chunk)
                received += len(chunk)
        if received != count:
            raise TimeoutError(
                f"serial timeout from {self.host}: received {received}/{count} bytes"
            )
        return b"".join(chunks)

    def close(self) -> None:
        self.connection.close()

    def run_tile(
        self,
        request_id: int,
        query_begin: int,
        kv_begin: int,
        query: np.ndarray,
        key: np.ndarray,
        value: np.ndarray,
        scales: tuple[float, float, float],
        float_input: bool,
    ) -> TileResult:
        query_count, dimension = query.shape
        kv_count = key.shape[0]
        query_mask = packed_valid_mask(query_count)
        kv_mask = packed_valid_mask(kv_count)
        fixed = TASK_FIXED.pack(
            query_begin,
            kv_begin,
            query_count,
            kv_count,
            dimension,
            3 if float_input else 1,
            0,
            len(query_mask),
            len(kv_mask),
            *scales,
        )
        if float_input:
            encoded_query = np.ascontiguousarray(query, dtype=">f4").tobytes()
            encoded_key = np.ascontiguousarray(key, dtype=">f4").tobytes()
            encoded_value = np.ascontiguousarray(value, dtype=">f4").tobytes()
        else:
            encoded_query = np.ascontiguousarray(query, dtype=np.int8).tobytes()
            encoded_key = np.ascontiguousarray(key, dtype=np.int8).tobytes()
            encoded_value = np.ascontiguousarray(value, dtype=">i2").tobytes()
        payload = fixed + query_mask + kv_mask + encoded_query + encoded_key + encoded_value
        message = HEADER.pack(
            MAGIC, VERSION, TYPE_RING_TILE_TASK, len(payload), request_id
        ) + payload
        started = time.perf_counter_ns()
        self.sendall(message)
        reply_header = self.receive_exact(HEADER.size)
        magic, version, kind, payload_bytes, reply_id = HEADER.unpack(reply_header)
        if (
            magic != MAGIC
            or version != VERSION
            or kind != TYPE_RING_TILE_RESULT
            or reply_id != request_id
        ):
            remainder = b""
            if not isinstance(self.connection, socket.socket):
                remainder = self.connection.readline()
            diagnostic = (reply_header + remainder).decode("utf-8", "replace").strip()
            raise RuntimeError(f"invalid reply from {self.host}: {diagnostic!r}")
        if payload_bytes < RESULT_FIXED.size:
            raise RuntimeError(f"short reply from {self.host}: {payload_bytes} bytes")
        reply = self.receive_exact(payload_bytes)
        elapsed_us = (time.perf_counter_ns() - started) // 1000
        (
            reply_query_begin,
            reply_kv_begin,
            reply_query_count,
            reply_dimension,
            flags,
            status,
            _reserved,
            decode_us,
            compute_us,
        ) = RESULT_FIXED.unpack_from(reply)
        expected = RESULT_FIXED.size + reply_query_count * (reply_dimension + 2) * 4
        if (
            reply_query_begin != query_begin
            or reply_kv_begin != kv_begin
            or reply_query_count != query_count
            or reply_dimension != dimension
            or flags != (3 if float_input else 1)
            or status != 0
            or payload_bytes != expected
        ):
            raise RuntimeError(f"shape/status mismatch from {self.host}")
        statistics = np.frombuffer(reply, dtype=">f4", offset=RESULT_FIXED.size)
        statistics = statistics.astype(np.float32).reshape(query_count, dimension + 2)
        return TileResult(
            query_begin,
            kv_begin,
            statistics[:, 0].copy(),
            statistics[:, 1].copy(),
            statistics[:, 2:].copy(),
            decode_us,
            compute_us,
            int(elapsed_us),
        )


def merge_tile(
    running_max: np.ndarray,
    running_sum: np.ndarray,
    running_num: np.ndarray,
    tile: TileResult,
) -> None:
    valid = np.isfinite(tile.maximum) & (tile.denominator > 0)
    if not np.any(valid):
        return
    old_max = running_max[valid]
    block_max = tile.maximum[valid]
    merged_max = np.maximum(old_max, block_max)
    old_scale = np.where(np.isfinite(old_max), np.exp(old_max - merged_max), 0.0)
    block_scale = np.exp(block_max - merged_max)
    running_num[valid] = (
        running_num[valid] * old_scale[:, None]
        + tile.numerator[valid] * block_scale[:, None]
    )
    running_sum[valid] = (
        running_sum[valid] * old_scale + tile.denominator[valid] * block_scale
    )
    running_max[valid] = merged_max


def dense_reference(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    query_scale: float,
    key_scale: float,
    value_scale: float,
) -> np.ndarray:
    dimension = query.shape[1]
    scores = (
        query.astype(np.float32)
        @ key.astype(np.float32).T
        * np.float32(query_scale * key_scale / math.sqrt(dimension))
    )
    scores[np.triu_indices(scores.shape[0], 1)] = -np.inf
    maximum = np.max(scores, axis=1, keepdims=True)
    probabilities = np.exp(scores - maximum)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    return probabilities @ (value.astype(np.float32) * np.float32(value_scale))


def compare(reference: np.ndarray, candidate: np.ndarray, atol: float, rtol: float) -> dict:
    error = np.abs(candidate - reference)
    passed = np.isfinite(candidate) & np.isfinite(reference) & (
        (error <= atol) | (error <= rtol * np.abs(reference))
    )
    return {
        "passed": bool(np.all(passed)),
        "failed": int(np.size(passed) - np.count_nonzero(passed)),
        "elements": int(np.size(passed)),
        "max_abs": float(np.max(error)),
        "mean_abs": float(np.mean(error)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group(required=True)
    location.add_argument("--workers", help="two comma-separated worker IPs")
    location.add_argument("--serial-ports", help="two comma-separated COM/TTY ports")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--query-tile", type=int, default=32)
    parser.add_argument("--kv-tile", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--mode", choices=("float32", "quantized"), default="float32"
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    locations = args.workers if args.workers is not None else args.serial_ports
    hosts = [item.strip() for item in locations.split(",") if item.strip()]
    if len(hosts) not in (1, 2):
        raise ValueError("one or two worker IP addresses or serial ports are required")
    if args.workers is not None and args.mode == "float32":
        raise ValueError("float32 mode requires the esp32-ring-serial firmware")
    if not 1 <= args.head_dim <= 64:
        raise ValueError("--head-dim must be in [1, 64]")
    if not 1 <= args.query_tile <= 64 or not 1 <= args.kv_tile <= 128:
        raise ValueError("query tile must be <=64 and KV tile <=128")
    if args.mode == "float32" and (
        args.query_tile * args.head_dim * 4
        + args.kv_tile * args.head_dim * 8
        + 64 > 33024
    ):
        raise ValueError("float32 tiles exceed the 33,024-byte worker payload")

    rng = np.random.default_rng(args.seed)
    query_float = rng.normal(0.0, 1.0, (args.sequence, args.head_dim)).astype(np.float32)
    key_float = rng.normal(0.0, 1.0, (args.sequence, args.head_dim)).astype(np.float32)
    value_float = rng.normal(0.0, 1.0, (args.sequence, args.head_dim)).astype(np.float32)
    if args.mode == "float32":
        query, key, value = query_float, key_float, value_float
        query_scale = key_scale = value_scale = 1.0
    else:
        query, query_scale = quantize(query_float, 127)
        key, key_scale = quantize(key_float, 127)
        value, value_scale = quantize(value_float, 32767)
    scales = (query_scale, key_scale, value_scale)
    workers = (
        [Worker.tcp(host, args.timeout) for host in hosts]
        if args.workers is not None
        else [Worker.serial(port, args.timeout, args.baud) for port in hosts]
    )
    output = np.zeros((args.sequence, args.head_dim), dtype=np.float32)
    worker_compute = [0] * len(workers)
    worker_wire = [0] * len(workers)
    started = time.perf_counter_ns()
    request_id = 0xEA000000
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
            for query_begin in range(0, args.sequence, args.query_tile):
                query_end = min(args.sequence, query_begin + args.query_tile)
                q = query[query_begin:query_end]
                assignments: list[list[tuple[int, int, int]]] = [
                    [] for _ in workers
                ]
                query_tile_index = query_begin // args.query_tile
                for tile_index, kv_begin in enumerate(range(0, args.sequence, args.kv_tile)):
                    if kv_begin > query_end - 1:
                        break
                    kv_end = min(args.sequence, kv_begin + args.kv_tile)
                    # Rotate ownership for successive query tiles.  This is
                    # the two-rank zigzag schedule: early causal rows do not
                    # leave rank 1 idle while rank 0 always owns KV block 0.
                    owner = (tile_index + query_tile_index) % len(workers)
                    assignments[owner].append(
                        (kv_begin, kv_end, request_id)
                    )
                    request_id += 1

                def run_assigned(worker_index: int) -> list[TileResult]:
                    return [
                        workers[worker_index].run_tile(
                            rid,
                            query_begin,
                            kv_begin,
                            q,
                            key[kv_begin:kv_end],
                            value[kv_begin:kv_end],
                            scales,
                            args.mode == "float32",
                        )
                        for kv_begin, kv_end, rid in assignments[worker_index]
                    ]

                futures = [
                    pool.submit(run_assigned, index) for index in range(len(workers))
                ]
                results: list[TileResult] = []
                for worker_index, future in enumerate(futures):
                    worker_results = future.result()
                    results.extend(worker_results)
                    worker_compute[worker_index] += sum(x.compute_us for x in worker_results)
                    worker_wire[worker_index] += sum(x.elapsed_us for x in worker_results)
                results.sort(key=lambda item: item.kv_begin)
                count = query_end - query_begin
                running_max = np.full(count, -np.inf, dtype=np.float32)
                running_sum = np.zeros(count, dtype=np.float32)
                running_num = np.zeros((count, args.head_dim), dtype=np.float32)
                for result in results:
                    merge_tile(running_max, running_sum, running_num, result)
                output[query_begin:query_end] = (
                    running_num / running_sum[:, None] * np.float32(value_scale)
                )
    finally:
        for worker in workers:
            worker.close()
    wall_us = (time.perf_counter_ns() - started) // 1000

    if args.sequence <= 4096:
        reference = dense_reference(
            query, key, value, query_scale, key_scale, value_scale
        )
        stats = compare(reference, output, args.atol, args.rtol)
        status = "PASS" if stats["passed"] else "FAIL"
        print(
            f"accuracy={status} failed={stats['failed']}/{stats['elements']} "
            f"max_abs={stats['max_abs']:.9g} mean_abs={stats['mean_abs']:.9g}"
        )
        if args.mode == "quantized":
            fp32_reference = dense_reference(
                query_float, key_float, value_float, 1.0, 1.0, 1.0
            )
            fp32_stats = compare(fp32_reference, output, args.atol, args.rtol)
            fp32_status = "PASS" if fp32_stats["passed"] else "FAIL"
            print(
                f"quantized_vs_fp32={fp32_status} "
                f"failed={fp32_stats['failed']}/{fp32_stats['elements']} "
                f"max_abs={fp32_stats['max_abs']:.9g} "
                f"mean_abs={fp32_stats['mean_abs']:.9g}"
            )
    else:
        print("accuracy=SKIP reason=dense_reference_limited_to_4096_tokens")
        stats = {"passed": True}
    print(
        f"mode={args.mode} shape=S{args.sequence}xHD{args.head_dim} qtile={args.query_tile} "
        f"kvtile={args.kv_tile} workers={','.join(hosts)}"
    )
    print(
        f"wall_ms={wall_us / 1000:.3f} "
        f"worker_compute_ms={','.join(f'{item / 1000:.3f}' for item in worker_compute)} "
        f"worker_wire_ms={','.join(f'{item / 1000:.3f}' for item in worker_wire)} "
        f"checksum={float(np.sum(output, dtype=np.float64)):.9g}"
    )
    return 0 if stats["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
