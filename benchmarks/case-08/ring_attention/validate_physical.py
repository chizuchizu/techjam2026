#!/usr/bin/env python3
"""Fast physical checks for case-8 normalization and quantized GEMM kernels."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import zlib
from pathlib import Path

import numpy as np

from export_case8 import quantize_rows
from run_case8_ring import MODEL_DIM, Worker, compare


HERE = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial-ports", default="COM10,COM11")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ports = [item.strip() for item in args.serial_ports.split(",") if item.strip()]
    if len(ports) != 2:
        raise ValueError("exactly two serial ports are required")
    rng = np.random.default_rng(808)
    weight = rng.normal(0, 0.03, (16, MODEL_DIM)).astype(np.float32)
    bias = rng.normal(0, 0.01, 16).astype(np.float32)
    quantized, scales = quantize_rows(weight)
    blob = scales.tobytes() + np.asarray(bias, dtype="<f4").tobytes() + quantized.tobytes()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "kernel_check_matrix.bin"
    matrix_path.write_bytes(blob)
    entry = {
        "rows": 16,
        "cols": MODEL_DIM,
        "bytes": len(blob),
        "crc32": zlib.crc32(blob) & 0xFFFFFFFF,
    }
    workers = [Worker(port, 921600, args.timeout) for port in ports]
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda worker: worker.stage_matrix(matrix_path, entry), workers))

        values = rng.normal(0, 0.4, (4, MODEL_DIM)).astype(np.float32)
        maximum = float(np.max(np.abs(values)))
        input_scale = maximum / 32767.0 if maximum else 1.0
        q_input = np.clip(np.rint(values / input_scale), -32767, 32767).astype(np.int16)
        expected_linear = (
            q_input.astype(np.int64) @ quantized.astype(np.int64).T
        ).astype(np.float32) * (input_scale * scales[None, :]) + bias[None, :]

        gamma = rng.normal(1.0, 0.02, MODEL_DIM).astype(np.float32)
        beta = rng.normal(0.0, 0.01, MODEL_DIM).astype(np.float32)
        norm_blob = gamma.astype("<f4").tobytes() + beta.astype("<f4").tobytes()
        norm_mean = np.mean(values, axis=1, keepdims=True, dtype=np.float32)
        norm_var = np.mean((values - norm_mean) ** 2, axis=1, keepdims=True, dtype=np.float32)
        expected_norm = (values - norm_mean) / np.sqrt(norm_var + np.float32(1e-5))
        expected_norm = expected_norm * gamma + beta

        reports = []
        for worker in workers:
            linear = worker.run_linear(values, 16)
            residual = rng.normal(0, 0.1, (4, 16)).astype(np.float32)
            fused = worker.run_linear(values, 16, gelu=True, residual=residual)
            expected_fused = np.vectorize(
                lambda x: 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0))),
                otypes=[np.float32],
            )(expected_linear) + residual
            worker.set_norm(norm_blob)
            normalized = worker.run_norm(values)
            reports.append(
                {
                    "port": worker.port,
                    "identity": worker.identity,
                    "linear": compare(expected_linear, linear, 1e-5, 1e-5),
                    "gelu_residual": compare(expected_fused, fused, 1e-5, 1e-5),
                    "layer_norm": compare(expected_norm, normalized, 2e-5, 2e-5),
                }
            )
        result = {"passed": all(all(item[name]["passed"] for name in ("linear", "gelu_residual", "layer_norm")) for item in reports), "workers": reports}
        (args.output_dir / "physical_kernels.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2))
        return 0 if result["passed"] else 2
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
