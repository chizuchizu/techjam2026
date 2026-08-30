#!/usr/bin/env python3
"""
gen_case.py - emit weights.bin, weights_q12.bin and one torch reference frame
for any case geometry, so the parameterized transformer can be host-tested
without depending on a specific case directory.

Usage:
  python3 tools/gen_case.py --S 128 --D 128 --H 4 --F 128 --L 4 \
                            --outdir /tmp/c02 --seeds 1
"""
import argparse
import hashlib
import json
import pathlib
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from torch_ref import BaselineTransformer, TransformerConfig, generate_random_case

SEED = 1234


def build_cfg(D, H, F, L):
    HD = D // H
    LAYER_FLOATS = 2*D + 4*(D*D + D) + 2*D + (F*D + F) + (D*F + D)
    TOTAL = L*LAYER_FLOATS + 2*D
    BLOCKS = [
        ("norm1.weight", 0, D), ("norm1.bias", 0, D),
        ("q_proj.weight", 0, D*D), ("q_proj.bias", 0, D),
        ("k_proj.weight", 0, D*D), ("k_proj.bias", 0, D),
        ("v_proj.weight", 0, D*D), ("v_proj.bias", 0, D),
        ("out_proj.weight", 0, D*D), ("out_proj.bias", 0, D),
        ("norm2.weight", 0, D), ("norm2.bias", 0, D),
        ("ffn_in.weight", 0, F*D), ("ffn_in.bias", 0, F),
        ("ffn_out.weight", 0, D*F), ("ffn_out.bias", 0, D),
    ]
    assert sum(c for _, _, c in BLOCKS) == LAYER_FLOATS
    return HD, LAYER_FLOATS, TOTAL, BLOCKS


def q12_block(w: np.ndarray):
    w = np.ascontiguousarray(w.astype(np.float32))
    amax = float(np.max(np.abs(w)))
    if amax == 0.0:
        amax = 1.0
    sw = 2047.0 / amax
    q = np.rint(w * sw).astype(np.int16)
    return q, amax / 2047.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--F", type=int, default=128)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--outdir", default=".", help="output directory")
    args = ap.parse_args()

    D, H, F, L, B = args.D, args.H, args.F, args.L, args.B
    HD, LAYER_FLOATS, TOTAL, BLOCKS = build_cfg(D, H, F, L)
    out = pathlib.Path(args.outdir)
    td = out / "testdata"
    td.mkdir(parents=True, exist_ok=True)

    cfg = TransformerConfig(B, args.S, D, H, F, L, True)
    cfg.validate()
    torch.manual_seed(SEED)
    model = BaselineTransformer(cfg).eval()
    sd = model.state_dict()

    flat = np.empty(TOTAL, dtype=np.float32)
    n = 0
    for l in range(L):
        for key, _, cnt in BLOCKS:
            name = f"layers.{l}.attention.{key}" if "proj" in key else f"layers.{l}.{key}"
            t = sd[name].detach().cpu().numpy().astype(np.float32).ravel()
            flat[n:n+cnt] = t
            n += cnt
    for key in ("weight", "bias"):
        t = sd[f"final_norm.{key}"].detach().cpu().numpy().astype(np.float32).ravel()
        flat[n:n+D] = t
        n += D
    (out / "weights.bin").write_bytes(flat.tobytes())

    qmats = ["q_proj", "k_proj", "v_proj", "out_proj", "ffn_in", "ffn_out"]
    blob = bytearray()
    for l in range(L):
        for mat in qmats:
            key = f"layers.{l}.attention.{mat}.weight" if "proj" in mat else f"layers.{l}.{mat}.weight"
            w = sd[key].detach().cpu().numpy().astype(np.float32)
            q, wscale = q12_block(w)
            cnt = w.shape[0] * w.shape[1]
            blob += struct.pack("<I", cnt)
            blob += struct.pack("<f", wscale)
            blob += q.ravel().astype("<i2").tobytes()
    (out / "weights_q12.bin").write_bytes(bytes(blob))

    manifest = {"dims": {"B": B, "S": args.S, "D": D, "H": H, "HD": HD,
                         "F": F, "L": L}, "seed": SEED}
    for t in range(args.seeds):
        x, valid = generate_random_case(cfg, torch.device("cpu"), torch.float32,
                                        SEED + t, 0.0, 1.0)
        assert valid is None or bool(valid.all())
        with torch.inference_mode():
            ref = model(x, None)
        xb = x[0].detach().cpu().numpy().astype(np.float32).ravel().tobytes()
        rb = ref[0].detach().cpu().numpy().astype(np.float32).ravel().tobytes()
        (td / f"input_{t}.bin").write_bytes(xb)
        (td / f"ref_{t}.bin").write_bytes(rb)
        manifest.setdefault("files", {})[str(t)] = {
            "input_sha256": hashlib.sha256(xb).hexdigest()[:16],
            "ref_sha256": hashlib.sha256(rb).hexdigest()[:16]}

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote weights.bin ({n} floats), weights_q12.bin, testdata to {out}")


if __name__ == "__main__":
    sys.exit(main())
