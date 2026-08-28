#!/usr/bin/env python3
"""Benchmark the ESP32 binary UDP/TCP echo transport from the host."""

from __future__ import annotations

import argparse
import csv
import math
import random
import socket
import struct
import time
from pathlib import Path

MAGIC = 0x45535033
VERSION = 1
TYPE_ECHO_REQUEST = 1
TYPE_ECHO_REPLY = 2
TYPE_DISCOVER_REQUEST = 3
TYPE_DISCOVER_REPLY = 4
UDP_PORT = 4210
TCP_PORT = 4211
HEADER = struct.Struct("!IBBHI")
UDP_SIZES = (0, 64, 256, 1024, 1400)
TCP_SIZES = (0, 256, 1024, 4096, 16384, 32768)


def packet(kind: int, request_id: int, payload: bytes) -> bytes:
    return HEADER.pack(MAGIC, VERSION, kind, len(payload), request_id) + payload


def parse_reply(
    data: bytes, expected_type: int, request_id: int, expected_payload: bytes
) -> bool:
    if len(data) != HEADER.size + len(expected_payload):
        return False
    magic, version, kind, payload_bytes, reply_id = HEADER.unpack_from(data)
    return (
        magic == MAGIC
        and version == VERSION
        and kind == expected_type
        and payload_bytes == len(expected_payload)
        and reply_id == request_id
        and data[HEADER.size:] == expected_payload
    )


def discover(timeout: float) -> str:
    request_id = random.getrandbits(32)
    message = packet(TYPE_DISCOVER_REQUEST, request_id, b"")
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        udp.settimeout(0.2)
        udp.bind(("", 0))
        for destination in ("255.255.255.255", "192.168.0.255"):
            udp.sendto(message, (destination, UDP_PORT))

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                reply, address = udp.recvfrom(2048)
            except TimeoutError:
                continue
            if parse_reply(reply, TYPE_DISCOVER_REPLY, request_id, b""):
                return address[0]
    raise TimeoutError("no broadcast reply; pass the IP with --host")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def payload_for(size: int, request_id: int) -> bytes:
    return bytes(((index * 131 + request_id * 17) & 0xFF) for index in range(size))


def result_row(
    protocol: str,
    size: int,
    timings: list[float],
    sent: int,
    invalid: int,
) -> dict[str, str | int | float]:
    received = len(timings)
    median = percentile(timings, 0.5) if timings else math.nan
    p90 = percentile(timings, 0.9) if timings else math.nan
    round_trip_mbps = (
        2.0 * size * 8.0 / median / 1_000_000.0
        if timings and size > 0
        else 0.0
    )
    return {
        "protocol": protocol,
        "payload_bytes": size,
        "trials_sent": sent,
        "trials_received": received,
        "lost": sent - received,
        "invalid": invalid,
        "median_rtt_ms": median * 1000.0,
        "p90_rtt_ms": p90 * 1000.0,
        "round_trip_payload_mbps": round_trip_mbps,
    }


def benchmark_udp(host: str, trials: int, timeout: float) -> list[dict]:
    rows = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp:
        udp.settimeout(timeout)
        for size in UDP_SIZES:
            timings: list[float] = []
            invalid = 0
            for trial in range(trials):
                request_id = (size << 16) ^ trial ^ 0xA5A50000
                payload = payload_for(size, request_id)
                start = time.perf_counter()
                udp.sendto(packet(TYPE_ECHO_REQUEST, request_id, payload),
                           (host, UDP_PORT))
                try:
                    reply, _ = udp.recvfrom(HEADER.size + size + 32)
                except TimeoutError:
                    continue
                elapsed = time.perf_counter() - start
                if parse_reply(reply, TYPE_ECHO_REPLY, request_id, payload):
                    timings.append(elapsed)
                else:
                    invalid += 1
            rows.append(result_row("UDP", size, timings, trials, invalid))
    return rows


def receive_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("ESP32 closed the TCP connection")
        chunks.extend(chunk)
    return bytes(chunks)


def benchmark_tcp(host: str, trials: int, timeout: float) -> list[dict]:
    rows = []
    with socket.create_connection((host, TCP_PORT), timeout=timeout) as connection:
        connection.settimeout(timeout)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for size in TCP_SIZES:
            timings: list[float] = []
            invalid = 0
            for trial in range(trials):
                request_id = (size << 16) ^ trial ^ 0x5A5A0000
                payload = payload_for(size, request_id)
                start = time.perf_counter()
                connection.sendall(packet(TYPE_ECHO_REQUEST, request_id, payload))
                reply_header = receive_exact(connection, HEADER.size)
                fields = HEADER.unpack(reply_header)
                reply_payload = receive_exact(connection, fields[3])
                elapsed = time.perf_counter() - start
                if parse_reply(
                    reply_header + reply_payload,
                    TYPE_ECHO_REPLY,
                    request_id,
                    payload,
                ):
                    timings.append(elapsed)
                else:
                    invalid += 1
            rows.append(result_row("TCP", size, timings, trials, invalid))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="ESP32 IP; omit to use UDP discovery")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    host = args.host or discover(args.timeout)
    print(f"WORKER,{host},udp={UDP_PORT},tcp={TCP_PORT}")

    rows = benchmark_udp(host, args.trials, args.timeout)
    rows.extend(benchmark_tcp(host, args.trials, args.timeout))
    fieldnames = list(rows[0])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    all_valid = all(row["lost"] == 0 and row["invalid"] == 0 for row in rows)
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
