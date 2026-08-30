#!/usr/bin/env python3
"""Export official case-8 weights into the staged ESP32 matrix format."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zlib
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))


def quantize_rows(weight: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    weight = np.asarray(weight, dtype=np.float32)
    maximum = np.max(np.abs(weight), axis=1)
    scales = np.where(maximum > 0, maximum / 32767.0, 1.0).astype("<f4")
    quantized = np.clip(np.rint(weight / scales[:, None]), -32767, 32767)
    return np.ascontiguousarray(quantized, dtype="<i2"), scales


def write_matrix(path: Path, weight: np.ndarray, bias: np.ndarray) -> dict:
    quantized, scales = quantize_rows(weight)
    bias = np.ascontiguousarray(bias, dtype="<f4")
    blob = scales.tobytes() + bias.tobytes() + quantized.tobytes()
    path.write_bytes(blob)
    return {
        "file": path.name,
        "rows": int(weight.shape[0]),
        "cols": int(weight.shape[1]),
        "dtype": "int16-q15-per-output-row",
        "layout": "scales-f32,bias-f32,weights-i16-row-major",
        "bytes": len(blob),
        "crc32": zlib.crc32(blob) & 0xFFFFFFFF,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def write_norm(path: Path, weight: np.ndarray, bias: np.ndarray) -> dict:
    values = np.concatenate(
        [np.asarray(weight, dtype="<f4"), np.asarray(bias, dtype="<f4")]
    )
    blob = values.tobytes()
    path.write_bytes(blob)
    return {
        "file": path.name,
        "dimension": int(weight.size),
        "dtype": "float32",
        "layout": "gamma,beta",
        "bytes": len(blob),
        "crc32": zlib.crc32(blob) & 0xFFFFFFFF,
        "sha256": hashlib.sha256(blob).hexdigest(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=HERE / ".artifacts" / "seed1234")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--reference-batch", type=int, default=1)
    parser.add_argument("--reference-trials", type=int, default=1)
    parser.add_argument("--skip-reference", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        import torch
        from torch_transformer_benchmark import (
            BaselineTransformer,
            TransformerConfig,
            generate_random_case,
        )
    except ModuleNotFoundError as exc:
        raise SystemExit("PyTorch is required to export official case-8 artifacts") from exc

    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    config = TransformerConfig(
        batch_size=args.reference_batch,
        seq_len=128,
        d_model=1024,
        num_heads=4,
        ffn_dim=1024,
        num_layers=4,
        causal=True,
    )
    torch.manual_seed(args.seed)
    model = BaselineTransformer(config).eval()
    manifest: dict = {
        "format": "techjam-case8-ring-v1",
        "seed": args.seed,
        "case": {"B": 64, "S": 128, "D": 1024, "H": 4, "F": 1024, "L": 4, "causal": True},
        "quantization": "symmetric int16 weights with one float32 scale per output row",
        "layers": [],
    }
    for layer_index, layer in enumerate(model.layers):
        layer_entry = {"index": layer_index, "norms": {}, "matrices": {}}
        for norm_name in ("norm1", "norm2"):
            norm = getattr(layer, norm_name)
            filename = f"layer{layer_index}_{norm_name}.bin"
            layer_entry["norms"][norm_name] = write_norm(
                args.outdir / filename,
                norm.weight.detach().cpu().numpy(),
                norm.bias.detach().cpu().numpy(),
            )
        matrices = {
            "q": layer.attention.q_proj,
            "k": layer.attention.k_proj,
            "v": layer.attention.v_proj,
            "o": layer.attention.out_proj,
            "f1": layer.ffn_in,
            "f2": layer.ffn_out,
        }
        for name, linear in matrices.items():
            filename = f"layer{layer_index}_{name}.bin"
            layer_entry["matrices"][name] = write_matrix(
                args.outdir / filename,
                linear.weight.detach().cpu().numpy(),
                linear.bias.detach().cpu().numpy(),
            )
        manifest["layers"].append(layer_entry)

    manifest["final_norm"] = write_norm(
        args.outdir / "final_norm.bin",
        model.final_norm.weight.detach().cpu().numpy(),
        model.final_norm.bias.detach().cpu().numpy(),
    )

    references = []
    if not args.skip_reference:
        with torch.inference_mode():
            for trial in range(args.reference_trials):
                x, mask = generate_random_case(
                    config, torch.device("cpu"), torch.float32,
                    args.seed + trial, 0.0, 1.0,
                )
                output = model(x, mask)
                input_name = f"input_b{args.reference_batch}_trial{trial}.npy"
                ref_name = f"reference_b{args.reference_batch}_trial{trial}.npy"
                np.save(args.outdir / input_name, x.numpy())
                np.save(args.outdir / ref_name, output.numpy())
                references.append({"trial": trial, "input": input_name, "reference": ref_name})
    manifest["references"] = references
    (args.outdir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(args.outdir / "manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
