#!/usr/bin/env python3
"""
export_case2.py - emit validation artifacts for the ESP32 case-2 baseline.

Reads the exact torch reference implementation from
techjam2026/torch_transformer_benchmark.py (official competition benchmark;
the same weight init seed, the same
random-input generator, same fp32 reference forward), then writes:

  weights.bin         flat fp32 weights, 398,592 floats (see tm_config.h layout)
  weights_q12.bin     Q12 int16 matrices + per-matrix scale for the FAST GEMM
  testdata/input_<s>.bin, ref_<s>.bin   torch input & reference output per seed
  manifest.json       seeds, layout, hashes

Requires the system python3 (torch + numpy). Usage:
  python3 tools/export_case2.py --outdir . [--seeds 25]
"""
import argparse
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np
import torch

repository_root = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repository_root))  # official benchmark at repo root

from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    generate_random_case,
)

# ---------- config (case 2) ----------
B, S, D, H, FH, L = 1, 128, 128, 4, 128, 4
HD = D // H
SEED = 1234        # == benchmark --seed; also weights-init RNG

LAYER_FLOATS = 2 * D + 4 * (D * D + D) + 2 * D + (FH * D + FH) + (D * FH + D)  # == 99,584
assert LAYER_FLOATS == 99_584, LAYER_FLOATS
TOTAL_FLOATS = L * LAYER_FLOATS + 2 * D
assert TOTAL_FLOATS == 398_592, TOTAL_FLOATS

# block order inside one layer (must mirror woff() in src/model.c)
BLOCKS = [
    ("norm1.weight", 0, D), ("norm1.bias", 0, D),
    ("q_proj.weight", 0, D * D), ("q_proj.bias", 0, D),
    ("k_proj.weight", 0, D * D), ("k_proj.bias", 0, D),
    ("v_proj.weight", 0, D * D), ("v_proj.bias", 0, D),
    ("out_proj.weight", 0, D * D), ("out_proj.bias", 0, D),
    ("norm2.weight", 0, D), ("norm2.bias", 0, D),
    ("ffn_in.weight", 0, FH * D), ("ffn_in.bias", 0, FH),
    ("ffn_out.weight", 0, D * FH), ("ffn_out.bias", 0, D),
]
assert sum(n for _, _, n in BLOCKS) == LAYER_FLOATS


def q12_block(w: np.ndarray):
    """Return (int16 flat, w_scale) for a [out,in] matrix, row-major out-major."""
    w = np.ascontiguousarray(w.astype(np.float32))
    amax = float(np.max(np.abs(w)))
    if amax == 0.0:
        amax = 1.0
    sw = 2047.0 / amax
    q = np.rint(w * sw).astype(np.int16)
    return q, amax / 2047.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=".", help="project root (embed files live here)")
    ap.add_argument("--seeds", type=int, default=25, help="number of accuracy seeds to export")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    td = out / "testdata"
    td.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    config = TransformerConfig(
        batch_size=B, seq_len=S, d_model=D, num_heads=H,
        ffn_dim=FH, num_layers=L, causal=True,
    )
    config.validate()

    # ---- reference model with the SAME init RNG as the benchmark harness ----
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()

    # ---- export flat fp32 weights ----
    sd = model.state_dict()
    flat = np.empty(TOTAL_FLOATS, dtype=np.float32)
    n = 0
    for l in range(L):
        for key, _, cnt in BLOCKS:
            if "proj" in key:
                name = f"layers.{l}.attention.{key}"
            else:
                name = f"layers.{l}.{key}"
            t = sd[name].detach().cpu().numpy().astype(np.float32).ravel()
            assert t.size == cnt, (name, t.size, cnt)
            flat[n:n + cnt] = t
            n += cnt
    for key in ("weight", "bias"):
        t = sd[f"final_norm.{key}"].detach().cpu().numpy().astype(np.float32).ravel()
        assert t.size == D
        flat[n:n + D] = t
        n += D
    assert n == TOTAL_FLOATS

    (out / "weights.bin").write_bytes(flat.tobytes())

    # ---- Q12 blob: [layer][q,k,v,o,f1,f2] = {u32 count, f32 w_scale, i16 data} ----
    qmats = ["q_proj", "k_proj", "v_proj", "out_proj", "ffn_in", "ffn_out"]
    blob = bytearray()
    for l in range(L):
        for mat in qmats:
            key = f"layers.{l}.attention.{mat}.weight" if "proj" in mat else f"layers.{l}.{mat}.weight"
            w = sd[key].detach().cpu().numpy().astype(np.float32)
            q, wscale = q12_block(w)  # [out,in] row-major == [N,K] expected by C
            cnt = w.shape[0] * w.shape[1]
            blob += struct.pack("<I", cnt)
            blob += struct.pack("<f", wscale)
            blob += q.ravel().astype("<i2").tobytes()
    (out / "weights_q12.bin").write_bytes(bytes(blob))

    # ---- per-seed inputs + torch references vs the real accuracy gate ----
    manifest = {"seed": SEED, "trials": args.seeds,
                "indexed_seed": list(range(args.seeds)),
                "rtol": 0.02, "atol": 0.002,
                "dims": {"B": B, "S": S, "D": D, "H": H, "HD": HD, "F": FH, "L": L}}
    for t in range(args.seeds):
        x, valid = generate_random_case(
            config=config, device=device, dtype=torch.float32,
            seed=SEED + t, padding_ratio=0.0, input_scale=1.0,
        )
        assert valid is None
        with torch.inference_mode():
            ref = model(x, None)
        xb = x.detach().cpu().numpy().astype(np.float32).ravel().tobytes()
        rb = ref.detach().cpu().numpy().astype(np.float32).ravel().tobytes()
        (td / f"input_{t}.bin").write_bytes(xb)
        (td / f"ref_{t}.bin").write_bytes(rb)
        manifest.setdefault("files", {})[str(t)] = {
            "input_sha256": hashlib.sha256(xb).hexdigest()[:16],
            "ref_sha256": hashlib.sha256(rb).hexdigest()[:16],
        }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    wq_bytes = (out / "weights_q12.bin").stat().st_size
    print(f"weights.bin      {(out/'weights.bin').stat().st_size:>9} bytes ({n} floats)")
    print(f"weights_q12.bin  {wq_bytes:>9} bytes")
    print(f"testdata:         {args.seeds} seeds -> input_<t>.bin / ref_<t>.bin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
