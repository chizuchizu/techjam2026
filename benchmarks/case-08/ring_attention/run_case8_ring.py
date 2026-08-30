#!/usr/bin/env python3
"""Run the complete case-8 Transformer across two USB-connected ESP32-C3s."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import serial

from protocol import (
    HEADER,
    MAGIC,
    REPLY_BIT,
    Message,
    Status,
    decode_header,
    decode_status,
    encode_frame,
)


HERE = Path(__file__).resolve().parent
MODEL_DIM = 1024
SEQUENCE = 128
HEADS = 4
HEAD_DIM = MODEL_DIM // HEADS
LAYERS = 4
TOKEN_SPLIT = 64
LINEAR_TILE = 4
QUERY_TILE = 16
KV_TILE = 8
FLAG_GELU = 1 << 0
FLAG_RESIDUAL = 1 << 1
FLAG_CAUSAL = 1 << 0


def f32_bytes(values: np.ndarray) -> bytes:
    return np.ascontiguousarray(values, dtype="<f4").tobytes()


def compare(reference: np.ndarray, candidate: np.ndarray, atol: float, rtol: float) -> dict:
    reference = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(candidate, dtype=np.float32)
    error = np.abs(candidate - reference)
    finite = np.isfinite(reference) & np.isfinite(candidate)
    passed = finite & ((error <= atol) | (error <= rtol * np.abs(reference)))
    worst = np.unravel_index(int(np.argmax(error)), error.shape)
    return {
        "passed": bool(np.all(passed)),
        "failed": int(passed.size - np.count_nonzero(passed)),
        "elements": int(passed.size),
        "max_abs": float(np.max(error)),
        "mean_abs": float(np.mean(error)),
        "worst_index": [int(item) for item in worst],
        "reference_at_worst": float(reference[worst]),
        "candidate_at_worst": float(candidate[worst]),
    }


@dataclass
class WorkerMetrics:
    wire_us: int = 0
    device_us: int = 0
    requests: int = 0
    staged_bytes: int = 0


class Worker:
    def __init__(self, port: str, baud: int, timeout: float) -> None:
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.lock = threading.Lock()
        self.next_request = 0xC8000000
        self.metrics = WorkerMetrics()
        self.connection: serial.Serial
        self.identity = ""
        self._connect()

    def _connect(self) -> None:
        self.connection = serial.Serial(
            self.port, self.baud, timeout=self.timeout, write_timeout=self.timeout
        )
        self.connection.dtr = False
        self.connection.rts = False
        # Opening native USB can reset a C3. Consume the ROM/Arduino banners so
        # the first binary response cannot be mistaken for boot text.
        self.connection.timeout = 1.0
        ready_deadline = time.monotonic() + min(self.timeout, 10.0)
        while time.monotonic() < ready_deadline:
            line = self.connection.readline()
            if b"CASE8_RING_READY" in line:
                break
        self.connection.timeout = self.timeout
        self.connection.reset_input_buffer()
        identity = self.request(Message.HELLO)
        self.identity = identity.decode("utf-8", "replace")

    def reconnect(self) -> None:
        try:
            self.connection.close()
        finally:
            time.sleep(1.0)
        self._connect()

    def close(self) -> None:
        self.connection.close()

    def _read_exact(self, count: int) -> bytes:
        chunks: list[bytes] = []
        received = 0
        while received < count:
            chunk = self.connection.read(count - received)
            if not chunk:
                raise TimeoutError(f"{self.port}: received {received}/{count} bytes")
            chunks.append(chunk)
            received += len(chunk)
        return b"".join(chunks)

    def _read_header(self) -> bytes:
        target = MAGIC.to_bytes(4, "little")
        window = bytearray()
        while True:
            window += self._read_exact(1)
            if len(window) > 4:
                del window[0]
            if bytes(window) == target:
                return target + self._read_exact(HEADER.size - 4)

    def _write_all(self, data: bytes) -> None:
        # Native USB-CDC on the C3 has a shallow receive queue. Conservative
        # 256-byte writes keep the device draining while a frame is in flight.
        offset = 0
        pacing_seconds = 0.010
        while offset < len(data):
            chunk = data[offset : offset + 256]
            written = self.connection.write(chunk)
            if written != len(chunk):
                raise ConnectionError(
                    f"{self.port}: short serial write {written}/{len(chunk)}"
                )
            offset += written
            if offset < len(data):
                time.sleep(pacing_seconds)

    def request(self, kind: Message, payload: bytes = b"", flags: int = 0) -> bytes:
        with self.lock:
            request_id = self.next_request
            self.next_request += 1
            frame = encode_frame(kind, request_id, payload, flags)
            started = time.perf_counter_ns()
            self._write_all(frame)
            self.connection.flush()
            header = self._read_header()
            reply_kind, _reply_flags, reply_id, length, checksum = decode_header(header)
            reply = self._read_exact(length)
            elapsed_us = (time.perf_counter_ns() - started) // 1000
            if reply_kind != (int(kind) | REPLY_BIT) or reply_id != request_id:
                raise RuntimeError(
                    f"{self.port}: mismatched reply kind={reply_kind} id={reply_id}"
                )
            if zlib.crc32(reply) & 0xFFFFFFFF != checksum:
                raise RuntimeError(f"{self.port}: reply CRC mismatch")
            status, device_us, data = decode_status(reply)
            self.metrics.requests += 1
            self.metrics.wire_us += int(elapsed_us)
            self.metrics.device_us += int(device_us)
            if status != Status.OK:
                raise RuntimeError(f"{self.port}: {kind.name} failed with {status.name}")
            return data

    def _stage_matrix_once(self, path: Path, entry: dict, chunk_bytes: int) -> None:
        blob_bytes = path.stat().st_size
        if blob_bytes != entry["bytes"]:
            raise ValueError(f"artifact size mismatch for {path}")
        begin = struct.pack(
            "<IIII", entry["rows"], entry["cols"], blob_bytes, entry["crc32"]
        )
        self.request(Message.STAGE_BEGIN, begin)
        offset = 0
        running_crc = 0
        with path.open("rb") as source:
            while True:
                chunk = source.read(chunk_bytes)
                if not chunk:
                    break
                running_crc = zlib.crc32(chunk, running_crc)
                self.request(Message.STAGE_CHUNK, struct.pack("<I", offset) + chunk)
                offset += len(chunk)
        if offset != blob_bytes or running_crc & 0xFFFFFFFF != entry["crc32"]:
            raise ValueError(f"artifact CRC mismatch for {path}")
        self.request(Message.STAGE_COMMIT)
        self.metrics.staged_bytes += blob_bytes

    def stage_matrix(self, path: Path, entry: dict, chunk_bytes: int = 4096) -> None:
        for attempt in range(3):
            try:
                self._stage_matrix_once(path, entry, chunk_bytes)
                return
            except Exception as exc:
                if attempt == 2:
                    raise
                print(
                    f"stage_retry port={self.port} file={path.name} "
                    f"attempt={attempt + 2} error={exc}",
                    flush=True,
                )
                self.reconnect()

    def set_norm(self, blob: bytes, dimension: int = MODEL_DIM) -> None:
        expected = dimension * 2 * 4
        if len(blob) != expected:
            raise ValueError(f"norm blob is {len(blob)} bytes, expected {expected}")
        self.request(Message.SET_NORM, struct.pack("<HHf", dimension, 0, 1.0e-5) + blob)

    def run_norm(self, values: np.ndarray) -> np.ndarray:
        rows, dimension = values.shape
        data = self.request(
            Message.RUN_NORM, struct.pack("<HH", rows, dimension) + f32_bytes(values)
        )
        return np.frombuffer(data, dtype="<f4").copy().reshape(rows, dimension)

    def run_linear(
        self,
        values: np.ndarray,
        output_columns: int,
        *,
        gelu: bool = False,
        residual: np.ndarray | None = None,
    ) -> np.ndarray:
        rows, input_columns = values.shape
        flags = (FLAG_GELU if gelu else 0) | (FLAG_RESIDUAL if residual is not None else 0)
        body = struct.pack("<HHHH", rows, input_columns, output_columns, flags)
        body += f32_bytes(values)
        if residual is not None:
            if residual.shape != (rows, output_columns):
                raise ValueError("residual shape mismatch")
            body += f32_bytes(residual)
        data = self.request(Message.RUN_LINEAR, body)
        return np.frombuffer(data, dtype="<f4").copy().reshape(rows, output_columns)

    def attention_begin(self, query_begin: int, query: np.ndarray) -> None:
        count, dimension = query.shape
        self.request(
            Message.ATTN_BEGIN,
            struct.pack("<IHH", query_begin, count, dimension) + f32_bytes(query),
        )

    def attention_block(self, kv_begin: int, key: np.ndarray, value: np.ndarray) -> None:
        if key.shape != value.shape:
            raise ValueError("K/V shape mismatch")
        count, dimension = key.shape
        fixed = struct.pack("<IHHHH", kv_begin, count, dimension, FLAG_CAUSAL, 0)
        self.request(Message.ATTN_BLOCK, fixed + f32_bytes(key) + f32_bytes(value))

    def attention_end(self, count: int, dimension: int) -> np.ndarray:
        data = self.request(Message.ATTN_END)
        return np.frombuffer(data, dtype="<f4").copy().reshape(count, dimension)


@dataclass
class RunMetrics:
    phases: dict[str, float] = field(default_factory=dict)

    def record(self, name: str, started_ns: int) -> None:
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1.0e6
        self.phases[name] = elapsed_ms
        print(f"phase_complete name={name} ms={elapsed_ms:.3f}", flush=True)


class Coordinator:
    def __init__(self, workers: list[Worker], artifact_dir: Path, manifest: dict) -> None:
        if len(workers) != 2:
            raise ValueError("case 8 requires exactly two workers")
        self.workers = workers
        self.artifact_dir = artifact_dir
        self.manifest = manifest
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.metrics = RunMetrics()

    def close(self) -> None:
        self.pool.shutdown(wait=True)

    def _parallel(self, fn):
        futures = [self.pool.submit(fn, index, worker) for index, worker in enumerate(self.workers)]
        return [future.result() for future in futures]

    def stage_matrix(self, entry: dict, phase: str) -> None:
        started = time.perf_counter_ns()
        path = self.artifact_dir / entry["file"]
        self._parallel(lambda _index, worker: worker.stage_matrix(path, entry))
        self.metrics.record(phase, started)

    def set_norm(self, entry: dict, phase: str) -> None:
        started = time.perf_counter_ns()
        blob = (self.artifact_dir / entry["file"]).read_bytes()
        self._parallel(lambda _index, worker: worker.set_norm(blob, entry["dimension"]))
        self.metrics.record(phase, started)

    def distributed_norm(self, values: np.ndarray, phase: str) -> np.ndarray:
        started = time.perf_counter_ns()
        output = np.empty_like(values, dtype=np.float32)

        def run_half(worker_index: int, worker: Worker) -> None:
            begin, end = (0, TOKEN_SPLIT) if worker_index == 0 else (TOKEN_SPLIT, SEQUENCE)
            for batch in range(values.shape[0]):
                for token in range(begin, end, LINEAR_TILE):
                    stop = min(end, token + LINEAR_TILE)
                    output[batch, token:stop] = worker.run_norm(values[batch, token:stop])

        self._parallel(run_half)
        self.metrics.record(phase, started)
        return output

    def distributed_linear(
        self,
        values: np.ndarray,
        phase: str,
        *,
        gelu: bool = False,
        residual: np.ndarray | None = None,
    ) -> np.ndarray:
        started = time.perf_counter_ns()
        output = np.empty((values.shape[0], SEQUENCE, MODEL_DIM), dtype=np.float32)

        def run_half(worker_index: int, worker: Worker) -> None:
            begin, end = (0, TOKEN_SPLIT) if worker_index == 0 else (TOKEN_SPLIT, SEQUENCE)
            for batch in range(values.shape[0]):
                for token in range(begin, end, LINEAR_TILE):
                    stop = min(end, token + LINEAR_TILE)
                    residual_tile = None if residual is None else residual[batch, token:stop]
                    output[batch, token:stop] = worker.run_linear(
                        values[batch, token:stop], MODEL_DIM,
                        gelu=gelu, residual=residual_tile,
                    )

        self._parallel(run_half)
        self.metrics.record(phase, started)
        return output

    def ring_attention(self, query: np.ndarray, key: np.ndarray, value: np.ndarray, phase: str) -> np.ndarray:
        started = time.perf_counter_ns()
        context = np.empty_like(query, dtype=np.float32)

        def run_half(worker_index: int, worker: Worker) -> None:
            begin, end = (0, TOKEN_SPLIT) if worker_index == 0 else (TOKEN_SPLIT, SEQUENCE)
            for batch in range(query.shape[0]):
                for head in range(HEADS):
                    feature_begin = head * HEAD_DIM
                    feature_end = feature_begin + HEAD_DIM
                    for query_begin in range(begin, end, QUERY_TILE):
                        query_end = min(end, query_begin + QUERY_TILE)
                        for attempt in range(3):
                            try:
                                worker.attention_begin(
                                    query_begin,
                                    query[batch, query_begin:query_end, feature_begin:feature_end],
                                )
                                # Traverse every KV block that can contain a valid causal
                                # key. Blocks wholly to the right of this query tile are
                                # future-only and do not need to cross USB.
                                for kv_begin in range(0, query_end, KV_TILE):
                                    kv_end = min(query_end, kv_begin + KV_TILE)
                                    worker.attention_block(
                                        kv_begin,
                                        key[batch, kv_begin:kv_end, feature_begin:feature_end],
                                        value[batch, kv_begin:kv_end, feature_begin:feature_end],
                                    )
                                tile = worker.attention_end(query_end - query_begin, HEAD_DIM)
                                context[batch, query_begin:query_end, feature_begin:feature_end] = tile
                                break
                            except Exception as exc:
                                if attempt == 2:
                                    raise
                                print(
                                    f"attention_retry port={worker.port} batch={batch} head={head} "
                                    f"query={query_begin} attempt={attempt + 2} error={exc}",
                                    flush=True,
                                )
                                worker.reconnect()
                    print(
                        f"attention_progress port={worker.port} batch={batch} head={head + 1}/{HEADS}",
                        flush=True,
                    )

        self._parallel(run_half)
        self.metrics.record(phase, started)
        return context

    def forward(
        self,
        input_values: np.ndarray,
        checkpoint_dir: Path | None = None,
        start_layer: int = 0,
    ) -> np.ndarray:
        x = np.ascontiguousarray(input_values, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (SEQUENCE, MODEL_DIM):
            raise ValueError(f"input must have shape [B,{SEQUENCE},{MODEL_DIM}]")
        for layer_index, layer in enumerate(self.manifest["layers"]):
            if layer_index < start_layer:
                continue
            prefix = f"layer{layer_index}"
            self.set_norm(layer["norms"]["norm1"], f"{prefix}.norm1.stage")
            norm1 = self.distributed_norm(x, f"{prefix}.norm1")
            projections = {}
            for name in ("q", "k", "v"):
                self.stage_matrix(layer["matrices"][name], f"{prefix}.{name}.stage")
                projections[name] = self.distributed_linear(norm1, f"{prefix}.{name}")
            context = self.ring_attention(
                projections["q"], projections["k"], projections["v"],
                f"{prefix}.attention",
            )
            self.stage_matrix(layer["matrices"]["o"], f"{prefix}.o.stage")
            x_after_attention = self.distributed_linear(
                context, f"{prefix}.o_residual", residual=x
            )
            self.set_norm(layer["norms"]["norm2"], f"{prefix}.norm2.stage")
            norm2 = self.distributed_norm(x_after_attention, f"{prefix}.norm2")
            self.stage_matrix(layer["matrices"]["f1"], f"{prefix}.f1.stage")
            hidden = self.distributed_linear(norm2, f"{prefix}.f1_gelu", gelu=True)
            self.stage_matrix(layer["matrices"]["f2"], f"{prefix}.f2.stage")
            x = self.distributed_linear(
                hidden, f"{prefix}.f2_residual", residual=x_after_attention
            )
            if checkpoint_dir is not None:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                np.save(checkpoint_dir / f"after_layer{layer_index}.npy", x)
                (checkpoint_dir / "progress.json").write_text(
                    json.dumps({"completed_layer": layer_index}, indent=2) + "\n",
                    encoding="utf-8",
                )
        self.set_norm(self.manifest["final_norm"], "final_norm.stage")
        return self.distributed_norm(x, "final_norm")


def dense_attention(query: np.ndarray, key: np.ndarray, value: np.ndarray) -> np.ndarray:
    output = np.empty_like(query)
    causal = np.triu(np.ones((SEQUENCE, SEQUENCE), dtype=bool), 1)
    for head in range(HEADS):
        begin = head * HEAD_DIM
        end = begin + HEAD_DIM
        q = query[:, :, begin:end]
        k = key[:, :, begin:end]
        v = value[:, :, begin:end]
        scores = q @ np.swapaxes(k, 1, 2) / np.float32(math.sqrt(HEAD_DIM))
        scores[:, causal] = -np.inf
        maximum = np.max(scores, axis=-1, keepdims=True)
        weights = np.exp(scores - maximum)
        weights /= np.sum(weights, axis=-1, keepdims=True)
        output[:, :, begin:end] = weights @ v
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-ports", default="COM10,COM11")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--artifact-dir", type=Path, default=HERE / ".artifacts" / "seed1234")
    parser.add_argument("--batch", type=int, choices=(1, 64), default=1)
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--attention-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--atol", type=float, default=0.002)
    parser.add_argument("--rtol", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ports = [item.strip() for item in args.serial_ports.split(",") if item.strip()]
    if len(ports) != 2:
        raise ValueError("--serial-ports must contain exactly two ports")
    manifest_path = args.artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"layers": []}
    workers = [Worker(port, args.baud, args.timeout) for port in ports]
    coordinator = Coordinator(workers, args.artifact_dir, manifest)
    started = time.perf_counter_ns()
    try:
        print("workers=" + ",".join(f"{w.port}:{w.identity}" for w in workers), flush=True)
        if args.attention_only:
            rng = np.random.default_rng(args.seed)
            query = rng.normal(0, 0.2, (1, SEQUENCE, MODEL_DIM)).astype(np.float32)
            key = rng.normal(0, 0.2, (1, SEQUENCE, MODEL_DIM)).astype(np.float32)
            value = rng.normal(0, 0.2, (1, SEQUENCE, MODEL_DIM)).astype(np.float32)
            output = coordinator.ring_attention(query, key, value, "attention_only")
            reference = dense_attention(query, key, value)
        else:
            input_path = args.artifact_dir / f"input_b{args.batch}_trial{args.trial}.npy"
            reference_path = args.artifact_dir / f"reference_b{args.batch}_trial{args.trial}.npy"
            if not input_path.exists() or not reference_path.exists():
                raise FileNotFoundError(
                    f"missing {input_path.name}/{reference_path.name}; run export_case8.py first"
                )
            input_values = np.load(input_path)
            reference = np.load(reference_path)
            checkpoint_dir = args.output_dir / f"checkpoints_b{args.batch}_trial{args.trial}"
            start_layer = 0
            if args.resume and (checkpoint_dir / "progress.json").exists():
                progress = json.loads((checkpoint_dir / "progress.json").read_text(encoding="utf-8"))
                completed_layer = int(progress["completed_layer"])
                input_values = np.load(checkpoint_dir / f"after_layer{completed_layer}.npy")
                start_layer = completed_layer + 1
                print(f"resume completed_layer={completed_layer} next_layer={start_layer}", flush=True)
            output = coordinator.forward(
                input_values, checkpoint_dir, start_layer=start_layer
            )
        stats = compare(reference, output, args.atol, args.rtol)
        wall_ms = (time.perf_counter_ns() - started) / 1.0e6
        result = {
            "case": 8,
            "shape": {"B": int(output.shape[0]), "S": SEQUENCE, "D": MODEL_DIM, "H": HEADS, "L": LAYERS},
            "ports": ports,
            "attention_only": args.attention_only,
            "wall_ms": wall_ms,
            "accuracy": stats,
            "phases_ms": coordinator.metrics.phases,
            "workers": [worker.metrics.__dict__ for worker in workers],
            "checksum": float(np.sum(output, dtype=np.float64)),
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        stem = "attention_only" if args.attention_only else f"case8_b{args.batch}_trial{args.trial}"
        np.save(args.output_dir / f"{stem}_output.npy", output)
        (args.output_dir / f"{stem}.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2), flush=True)
        return 0 if stats["passed"] else 2
    finally:
        coordinator.close()
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
