#!/usr/bin/env python3
"""
torch_crosscheck.py - validate the Case 08 sharded numpy forward against the
official torch reference (torch_ref.py, copied from the case-02 tooling).

The official benchmark model is `BaselineTransformer` (bias=True Linears,
erf GELU, pre-norm residual, final LayerNorm).  We build it, draw the random
weights/input exactly the way the benchmark does, run it once as the authority,
then feed the SAME weights and input through `sharded_forward.ShardedModel` and
require the two to agree inside the competition gate
(|a-b| <= atol OR |a-b| <= rtol*|b|, atol=0.002, rtol=0.02).

Run (system python3 has torch):
    python3 toolkit/torch_crosscheck.py --seeds 3
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "tools"))

from sharded_forward import Cfg, gelu_erf, layernorm, softmax_rows, RefModel, ShardedModel  # noqa: E402
import torch  # noqa: E402
from torch_ref import TransformerConfig, BaselineTransformer  # noqa: E402


def torch_state_to_numpy(model: BaselineTransformer, cfg: Cfg):
    sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
    ref = RefModel(cfg, seed=0)  # shapes/keys scaffold; weights replaced below
    for i in range(cfg.L):
        p = f"layers.{i}."
        ref.layers[i]["ln1_g"] = sd[p + "norm1.weight"]
        ref.layers[i]["ln1_b"] = sd[p + "norm1.bias"]
        ref.layers[i]["q_w"] = sd[p + "attention.q_proj.weight"]
        ref.layers[i]["q_b"] = sd[p + "attention.q_proj.bias"]
        ref.layers[i]["k_w"] = sd[p + "attention.k_proj.weight"]
        ref.layers[i]["k_b"] = sd[p + "attention.k_proj.bias"]
        ref.layers[i]["v_w"] = sd[p + "attention.v_proj.weight"]
        ref.layers[i]["v_b"] = sd[p + "attention.v_proj.bias"]
        ref.layers[i]["o_w"] = sd[p + "attention.out_proj.weight"]
        ref.layers[i]["o_b"] = sd[p + "attention.out_proj.bias"]
        ref.layers[i]["ln2_g"] = sd[p + "norm2.weight"]
        ref.layers[i]["ln2_b"] = sd[p + "norm2.bias"]
        ref.layers[i]["f1_w"] = sd[p + "ffn_in.weight"]
        ref.layers[i]["f1_b"] = sd[p + "ffn_in.bias"]
        ref.layers[i]["f2_w"] = sd[p + "ffn_out.weight"]
        ref.layers[i]["f2_b"] = sd[p + "ffn_out.bias"]
    ref.final_g = sd["final_norm.weight"]
    ref.final_b = sd["final_norm.bias"]
    return ref


def gate(a: np.ndarray, b: np.ndarray, atol: float, rtol: float):
    ok = (np.abs(a - b) <= atol) | (np.abs(a - b) <= rtol * np.abs(b))
    return ok.all(), float(np.abs(a - b).max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--a-shards", type=int, default=4)
    ap.add_argument("--f-shards", type=int, default=16)
    ap.add_argument("--ln-slabs", type=int, default=16)
    args = ap.parse_args()

    cfg = Cfg()
    a, f, s = args.a_shards, args.f_shards, args.ln_slabs

    for sd in range(args.seeds):
        tcfg = TransformerConfig(1, cfg.S, cfg.D, cfg.H, cfg.F, cfg.L, cfg.causal)
        model = BaselineTransformer(tcfg).eval()
        g = torch.Generator().manual_seed(1234 + sd)
        # reproducible init, same shape/scale family as the benchmark random case
        for p in model.parameters():
            if p.dim() >= 2:
                torch.nn.init.normal_(p, mean=0.0, std=1.0 / math.sqrt(cfg.D), generator=g)
            else:
                torch.nn.init.zeros_(p)
        ref = torch_state_to_numpy(model, cfg)
        sh = ShardedModel(cfg, ref, a, f, s)

        x = torch.randn(1, cfg.S, cfg.D, generator=g).numpy().astype(np.float32)

        with torch.no_grad():
            y_t = model(torch.from_numpy(x)).numpy().astype(np.float32)
        y_n = sh.forward(x[0])
        ok, mx = gate(y_n, y_t[0], 0.002, 0.02)
        print(f"seed {sd}: sharded-numpy vs torch max_abs={mx:.3e} gate={'PASS' if ok else 'FAIL'}")
        if not ok:
            sys.exit(1)
    print("crosscheck OK")


if __name__ == "__main__":
    main()
