#!/usr/bin/env python3
"""
export_batch.py - emit the B input frames and torch references for a batch case.

Cases 1, 3, 4 and 5 are the case-2 geometry at a larger batch
(B = 64, 4, 16, 128 with S=128 D=128 H=4 F=128 L=4). The reference model is
built from (D, H, F, L) under a fixed seed before any input is drawn, so the
weights are identical across all of them and case 2 - the boards keep using
../../case-02/optimisation/esp32-baseline/weights*.bin unchanged. Only the
inputs differ, and each of the B inputs is an independent forward.

Writes testdata/B<batch>/input_<i>.bin and ref_<i>.bin, one frame per batch
element, so a data-parallel run can hand element i to whichever board owns it.

Usage: python3 tools/export_batch.py --batch 4 16 64 128
"""
import argparse
import pathlib
import sys

import numpy as np
import torch

BASE = pathlib.Path(__file__).resolve().parents[2] / "case-02" / "optimisation" / "esp32-baseline"
sys.path.insert(0, str(BASE / "tools"))

from torch_ref import BaselineTransformer, TransformerConfig, generate_random_case  # noqa: E402

SEED = 1234


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", nargs="+", type=int, default=[4, 16, 64, 128])
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--F", type=int, default=128)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--outdir", default=str(pathlib.Path(__file__).resolve().parents[1]))
    args = ap.parse_args()

    out = pathlib.Path(args.outdir) / "testdata"
    device = torch.device("cpu")

    for B in args.batch:
        cfg = TransformerConfig(batch_size=B, seq_len=args.S, d_model=args.D,
                                num_heads=args.H, ffn_dim=args.F,
                                num_layers=args.L, causal=True)
        cfg.validate()
        # weights: same construction order and seed as the single-board export,
        # so every batch case reuses the case-2 weight blobs verbatim
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        model = BaselineTransformer(cfg).to(device=device, dtype=torch.float32).eval()

        x, valid = generate_random_case(config=cfg, device=device,
                                        dtype=torch.float32, seed=SEED,
                                        padding_ratio=0.0, input_scale=1.0)
        assert valid is None or bool(valid.all())
        with torch.inference_mode():
            ref = model(x, None)

        d = out / f"B{B}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(B):
            (d / f"input_{i}.bin").write_bytes(
                x[i].detach().cpu().numpy().astype(np.float32).ravel().tobytes())
            (d / f"ref_{i}.bin").write_bytes(
                ref[i].detach().cpu().numpy().astype(np.float32).ravel().tobytes())
        print(f"B={B}: wrote {B} input/ref frames to {d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
