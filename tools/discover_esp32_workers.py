#!/usr/bin/env python3
"""Query ESP32 attention-worker capabilities over the LAN."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import socket
import struct
import sys
import time

MAGIC = 0x45535033
VERSION = 1
TYPE_CAPABILITIES_REQUEST = 9
TYPE_CAPABILITIES_REPLY = 10
UDP_PORT = 4210
HEADER = struct.Struct("!IBBHI")
CAPABILITIES_FIXED = struct.Struct("!IHHHHIHBB")

CAPABILITY_NAMES = {
    1 << 0: "head_udp",
    1 << 1: "head_tcp",
    1 << 2: "kv_shard_udp",
}


def query(host: str, timeout: float, index: int) -> dict:
    request_id = 0xC3CA0000 | (index & 0xFFFF)
    message = HEADER.pack(
        MAGIC, VERSION, TYPE_CAPABILITIES_REQUEST, 0, request_id
    )
    start = time.perf_counter_ns()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
        connection.settimeout(timeout)
        connection.sendto(message, (host, UDP_PORT))
        reply, source = connection.recvfrom(512)
    elapsed_us = (time.perf_counter_ns() - start) // 1000
    if len(reply) < HEADER.size + CAPABILITIES_FIXED.size:
        raise RuntimeError(f"short capability reply from {host}")
    magic, version, kind, payload_bytes, reply_id = HEADER.unpack_from(reply)
    if (
        magic != MAGIC
        or version != VERSION
        or kind != TYPE_CAPABILITIES_REPLY
        or reply_id != request_id
        or len(reply) != HEADER.size + payload_bytes
    ):
        raise RuntimeError(f"invalid capability reply from {host}")
    fields = CAPABILITIES_FIXED.unpack_from(reply, HEADER.size)
    (
        flags,
        max_sequence,
        max_head_dimension,
        max_udp_payload,
        max_tcp_payload,
        free_heap,
        cpu_mhz,
        cores,
        model_length,
    ) = fields
    if payload_bytes != CAPABILITIES_FIXED.size + model_length:
        raise RuntimeError(f"invalid model name length from {host}")
    model_begin = HEADER.size + CAPABILITIES_FIXED.size
    model = reply[model_begin:model_begin + model_length].decode(
        "ascii", errors="replace"
    )
    capabilities = "+".join(
        name for bit, name in CAPABILITY_NAMES.items() if flags & bit
    )
    return {
        "worker": index,
        "requested_host": host,
        "reply_host": source[0],
        "model": model,
        "cores": cores,
        "cpu_mhz": cpu_mhz,
        "free_heap": free_heap,
        "max_sequence": max_sequence,
        "max_head_dimension": max_head_dimension,
        "max_udp_payload": max_udp_payload,
        "max_tcp_payload": max_tcp_payload,
        "capabilities": capabilities,
        "rtt_us": elapsed_us,
        "status": "READY",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workers", required=True, help="comma-separated worker IP addresses"
    )
    parser.add_argument("--timeout", type=float, default=2.0)
    args = parser.parse_args()
    workers = [item.strip() for item in args.workers.split(",") if item.strip()]
    if not workers:
        parser.error("at least one worker IP is required")

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(workers)) as pool:
        futures = {
            pool.submit(query, host, args.timeout, index): (index, host)
            for index, host in enumerate(workers)
        }
        for future, (index, host) in futures.items():
            try:
                rows.append(future.result())
            except (TimeoutError, OSError, RuntimeError) as error:
                rows.append(
                    {
                        "worker": index,
                        "requested_host": host,
                        "reply_host": "",
                        "model": "",
                        "cores": "",
                        "cpu_mhz": "",
                        "free_heap": "",
                        "max_sequence": "",
                        "max_head_dimension": "",
                        "max_udp_payload": "",
                        "max_tcp_payload": "",
                        "capabilities": "",
                        "rtt_us": "",
                        "status": f"ERROR:{error}",
                    }
                )
    rows.sort(key=lambda row: int(row["worker"]))
    writer = csv.DictWriter(
        sys.stdout, fieldnames=list(rows[0]), lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return 0 if all(row["status"] == "READY" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
