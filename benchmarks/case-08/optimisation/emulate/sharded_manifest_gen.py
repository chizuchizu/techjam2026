#!/usr/bin/env python3
"""
sharded_manifest_gen.py - emits per-shard weight files + a shard assignment
manifest for the Case 08 deployment (see SHARD_PLAN.md).

Output layout under shards/:
    attn_<a>_layer_<l>.npz    Q/K/V/O weight + bias slice for head shard a
    ffn_<f>_layer_<l>.npz     ffn_in / ffn_out slices for hidden shard f
    ln_<s>_layer_<l>.npz      LayerNorm weight/bias slab
    manifest.json             shard->role, feature ranges, byte budgets

The fp32 slices are written here; the on-device toolchain quantizes each slice
to Q12 with the same code used by case-02's export tool.  A conversion skeleton
(`q12_from_npz`) is provided so the deployment can be reproduced end to end.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from sharded_forward import Cfg, RefModel


def q12_scale(w: np.ndarray) -> float:
    """Q12 quantization scale (max-abs) matching case-02 export semantics."""
    m = float(np.abs(w).max())
    return m / 2047.0 if m > 0 else 1.0


def q12_from_npz(w: np.ndarray):
    s = q12_scale(w)
    return np.clip(np.round(w / s), -2048, 2047).astype(np.int16), float(s)


def main() -> None:
    cfg = Cfg()
    ref = RefModel(cfg, seed=1234)
    a_shards, f_shards, ln_slabs = 4, 16, 16
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shards")
    os.makedirs(out, exist_ok=True)

    manifest = {
        "geometry": dict(B=cfg.B, S=cfg.S, D=cfg.D, H=cfg.H, F=cfg.F, L=cfg.L,
                         causal=cfg.causal),
        "attention_shards": a_shards,
        "ffn_shards": f_shards,
        "layernorm_slabs": ln_slabs,
        "reduction": "coordinator all-reduce (see SHARD_PLAN.md)",
        "shards": [],
    }

    HD, fs, ls = cfg.HD, cfg.F // f_shards, cfg.D // ln_slabs
    for a in range(a_shards):
        heads = [a * (cfg.H // a_shards) + j for j in range(cfg.H // a_shards)]
        for l in range(cfg.L):
            lay = ref.layers[l]
            r0, r1 = heads[0] * HD, (heads[-1] + 1) * HD
            w = {
                "q_w": lay["q_w"][r0:r1], "q_b": lay["q_b"][r0:r1],
                "k_w": lay["k_w"][r0:r1], "k_b": lay["k_b"][r0:r1],
                "v_w": lay["v_w"][r0:r1], "v_b": lay["v_b"][r0:r1],
                "o_w": lay["o_w"][:, r0:r1],
            }
            np.savez(os.path.join(out, f"attn_{a}_layer_{l}.npz"), **w)
            nbytes = sum(x.nbytes for x in w.values())
            manifest["shards"].append(dict(role="attn", index=a, layer=l,
                                           heads=heads, byte_fp32=nbytes,
                                           byte_q12=nbytes // 2))
    for f in range(f_shards):
        for l in range(cfg.L):
            lay = ref.layers[l]
            r0, r1 = f * fs, (f + 1) * fs
            w = {"f1_w": lay["f1_w"][r0:r1], "f1_b": lay["f1_b"][r0:r1],
                 "f2_w": lay["f2_w"][:, r0:r1]}
            np.savez(os.path.join(out, f"ffn_{f}_layer_{l}.npz"), **w)
            nbytes = sum(x.nbytes for x in w.values())
            manifest["shards"].append(dict(role="ffn", index=f, layer=l,
                                           hidden_slice=[r0, r1], byte_fp32=nbytes,
                                           byte_q12=nbytes // 2))
    for s in range(ln_slabs):
        for l in range(cfg.L):
            lay = ref.layers[l]
            r0, r1 = s * ls, (s + 1) * ls
            w = {"ln1_g": lay["ln1_g"][r0:r1], "ln1_b": lay["ln1_b"][r0:r1],
                 "ln2_g": lay["ln2_g"][r0:r1], "ln2_b": lay["ln2_b"][r0:r1]}
            np.savez(os.path.join(out, f"ln_{s}_layer_{l}.npz"), **w)

    with open(os.path.join(out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"wrote {os.path.join(out, 'manifest.json')}")


if __name__ == "__main__":
    main()
