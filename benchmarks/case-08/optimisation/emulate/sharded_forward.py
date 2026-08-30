#!/usr/bin/env python3
"""
Case 08 sharded emulation (host side, numpy only).

Geometry: B=64, S=128, D=1024, H=4, F=1024, L=4, causal.

A single XIAO ESP32-C3 (400 KB SRAM, 4 MB flash) cannot hold the case-08 model:
  * fp32 params = 25,208,832 floats = 100,835,328 bytes
  * Q12 params  = ~50.4 MB
  * one [S,D] activation tile in fp32 = 512 KB (already over SRAM)

This module proves the intended deployment recipe *numerically*:

  * attention       sharded by head   (A shards; per-head Q/K/V/O + bias slice)
  * FFN             sharded by F      (F shards; ffn_in rows + ffn_out columns)
  * LayerNorm       sharded by D      (per-shard partial sums, all-reduce mean/var)
  * residuals       reduced on a coordinator (all-reduce / parameter-server style)

The unsharded fp32 forward (RefModel) and the sharded forward (ShardedModel)
receive the exact same weights and are asserted to agree to <=1e-4 absolute in
fp32.  That agreement demonstrates sharding + inter-shard reduction is exact up
to fp32 accumulation order, which is the mechanism proof for the deployment
plan in SHARD_PLAN.md.

No torch is required here; torch_crosscheck.py validates the same sharded code
against the official torch reference.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
# geometry / config
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    B: int = 1      # per-frame forward; batch cases stream B independent frames
    S: int = 128
    D: int = 1024
    H: int = 4
    F: int = 1024
    L: int = 4
    causal: bool = True
    eps: float = 1e-5

    @property
    def HD(self) -> int:
        return self.D // self.H


def gelu_erf(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1.0 + np.erf(x / math.sqrt(2.0)))


def softmax_rows(x: np.ndarray, scale: float, causal: bool) -> np.ndarray:
    y = x.astype(np.float64) * scale
    if causal:
        n = y.shape[-1]
        y = y + np.triu(np.full((n, n), -np.inf), 1)[None, :, :]
    m = y.max(axis=-1, keepdims=True)
    e = np.exp(y - m)
    return (e / e.sum(axis=-1, keepdims=True)).astype(np.float32)


# --------------------------------------------------------------------------- #
# LayerNorm with an explicit per-shard all-reduce path
# --------------------------------------------------------------------------- #
def layernorm(x: np.ndarray, g: np.ndarray, b: np.ndarray,
              eps: float, slabs: int = 1) -> np.ndarray:
    """Exact LayerNorm. slabs>1 emulates feature-sharding: each slab computes
    partial sum / sum-of-squares, the coordinator all-reduces them into the
    global mean/var, and every slab then normalizes locally."""
    S, D = x.shape
    s1 = np.zeros((S, 1), np.float64)
    s2 = np.zeros((S, 1), np.float64)
    w = D // slabs
    for t in range(slabs):
        sl = slice(t * w, (t + 1) * w)
        xs = x[:, sl].astype(np.float64)
        s1 += xs.sum(axis=-1, keepdims=True)
        s2 += (xs * xs).sum(axis=-1, keepdims=True)
    mu = s1 / D
    var = s2 / D - mu * mu
    xn = (x.astype(np.float64) - mu) / np.sqrt(var + eps)
    return (xn.astype(np.float32) * g + b).astype(np.float32)


# --------------------------------------------------------------------------- #
# unsharded reference
# --------------------------------------------------------------------------- #
_GAUSS = {"loc": 0.0, "scale": 1.0}  # placeholder kept minimal


class RefModel:
    """fp32 reference with the exact torch_ref.py arithmetic (bias=True Linears,
    erf GELU, pre-norm residual blocks, final LayerNorm)."""

    def __init__(self, cfg: Cfg, seed: int = 1234):
        self.cfg = cfg
        rng = np.random.default_rng(seed)
        lim = 1.0 / math.sqrt(cfg.D)
        self.layers = []
        for _ in range(cfg.L):
            lay = {
                "ln1_g": rng.normal(0, 1, cfg.D).astype(np.float32),
                "ln1_b": np.zeros(cfg.D, np.float32),
                "q_w": rng.normal(0, lim, (cfg.D, cfg.D)).astype(np.float32),
                "q_b": np.zeros(cfg.D, np.float32),
                "k_w": rng.normal(0, lim, (cfg.D, cfg.D)).astype(np.float32),
                "k_b": np.zeros(cfg.D, np.float32),
                "v_w": rng.normal(0, lim, (cfg.D, cfg.D)).astype(np.float32),
                "v_b": np.zeros(cfg.D, np.float32),
                "o_w": rng.normal(0, lim, (cfg.D, cfg.D)).astype(np.float32),
                "o_b": np.zeros(cfg.D, np.float32),
                "ln2_g": rng.normal(0, 1, cfg.D).astype(np.float32),
                "ln2_b": np.zeros(cfg.D, np.float32),
                "f1_w": rng.normal(0, lim, (cfg.F, cfg.D)).astype(np.float32),
                "f1_b": np.zeros(cfg.F, np.float32),
                "f2_w": rng.normal(0, lim, (cfg.D, cfg.F)).astype(np.float32),
                "f2_b": np.zeros(cfg.D, np.float32),
            }
            self.layers.append(lay)
        self.final_g = rng.normal(0, 1, cfg.D).astype(np.float32)
        self.final_b = np.zeros(cfg.D, np.float32)

    def layer_forward(self, x: np.ndarray, w: dict) -> np.ndarray:
        cfg = self.cfg
        eps, H, HD, D = cfg.eps, cfg.H, cfg.HD, cfg.D
        xn = layernorm(x, w["ln1_g"], w["ln1_b"], eps)
        q = xn @ w["q_w"].T + w["q_b"]
        k = xn @ w["k_w"].T + w["k_b"]
        v = xn @ w["v_w"].T + w["v_b"]
        S = x.shape[0]
        qh = q.reshape(S, H, HD).transpose(1, 0, 2)
        kh = k.reshape(S, H, HD).transpose(1, 0, 2)
        vh = v.reshape(S, H, HD).transpose(1, 0, 2)
        scale = 1.0 / math.sqrt(HD)
        ctx = np.empty((H, S, HD), np.float32)
        for h in range(H):
            p = softmax_rows(qh[h] @ kh[h].T, scale, cfg.causal)
            ctx[h] = (p @ vh[h]).astype(np.float32)
        attn = ctx.transpose(1, 0, 2).reshape(S, D)
        x = (x + attn @ w["o_w"].T + w["o_b"]).astype(np.float32)
        xn = layernorm(x, w["ln2_g"], w["ln2_b"], eps)
        z = gelu_erf(xn @ w["f1_w"].T + w["f1_b"])
        x = (x + z @ w["f2_w"].T + w["f2_b"]).astype(np.float32)
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        for w in self.layers:
            x = self.layer_forward(x, w)
        return layernorm(x, self.final_g, self.final_b, self.cfg.eps)


# --------------------------------------------------------------------------- #
# sharded forward
# --------------------------------------------------------------------------- #
class ShardedModel:
    def __init__(self, cfg: Cfg, ref: RefModel, a_shards: int, f_shards: int,
                 ln_slabs: int):
        assert cfg.H % a_shards == 0 and cfg.F % f_shards == 0 and cfg.D % ln_slabs == 0
        self.cfg = cfg
        self.ref = ref
        self.a_shards = a_shards
        self.f_shards = f_shards
        self.ln_slabs = ln_slabs

    def shard_attention(self, x: np.ndarray, w: dict, a: int) -> np.ndarray:
        """Head shard `a`: compute its heads' attention and output-proj partials.
        Returns (partial_out: [S,D], no o_bias). o_bias is added once by coordinator."""
        cfg = self.cfg
        S, D, H, HD = x.shape[0], cfg.D, cfg.H, cfg.HD
        h0 = a * (H // self.a_shards)
        h1 = (a + 1) * (H // self.a_shards)
        scale = 1.0 / math.sqrt(HD)
        out = np.zeros((S, D), np.float32)
        for h in range(h0, h1):
            r0, r1 = h * HD, (h + 1) * HD
            q = (x[:, :, None] if False else x) @ w["q_w"][r0:r1].T + w["q_b"][r0:r1]
            k = x @ w["k_w"][r0:r1].T + w["k_b"][r0:r1]
            v = x @ w["v_w"][r0:r1].T + w["v_b"][r0:r1]
            p = softmax_rows(q @ k.T, scale, cfg.causal)
            ctx = p @ v
            out += ctx @ w["o_w"][:, r0:r1].T
        return out

    def shard_ffn(self, x: np.ndarray, w: dict, f: int) -> np.ndarray:
        """FFN shard `f`: gated hidden slice + output-proj partial. f2_bias added once."""
        cfg = self.cfg
        fs = cfg.F // self.f_shards
        r0, r1 = f * fs, (f + 1) * fs
        h = gelu_erf(x @ w["f1_w"][r0:r1].T + w["f1_b"][r0:r1])
        return h @ w["f2_w"][:, r0:r1].T

    def layer_forward(self, x: np.ndarray, w: dict) -> np.ndarray:
        cfg = self.cfg
        # LayerNorm with per-shard partials + all-reduce (slabs == feature shards)
        xn = layernorm(x, w["ln1_g"], w["ln1_b"], cfg.eps, self.ln_slabs)
        attn = np.zeros((x.shape[0], cfg.D), np.float32)
        for a in range(self.a_shards):
            attn += self.shard_attention(xn, w, a)
        attn += w["o_b"]  # output-projection bias counted exactly once (coordinator)
        x = (x + attn).astype(np.float32)

        xn = layernorm(x, w["ln2_g"], w["ln2_b"], cfg.eps, self.ln_slabs)
        z = np.zeros((x.shape[0], cfg.D), np.float32)
        for f in range(self.f_shards):
            z += self.shard_ffn(xn, w, f)
        z += w["f2_b"]  # ffn_out bias counted exactly once (coordinator)
        return (x + z).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for w in self.ref.layers:
            x = self.layer_forward(x, w)
        return layernorm(x, self.ref.final_g, self.ref.final_b, self.cfg.eps,
                         self.ln_slabs)


def param_budget(cfg: Cfg, a_shards: int, f_shards: int, ln_slabs: int) -> str:
    D, H, F, HD = cfg.D, cfg.H, cfg.F, cfg.HD
    per_layer_f32 = 2 * D + 4 * (D * D + D) + 2 * D + (F * D + F) + (D * F + D)
    total_f32 = cfg.L * per_layer_f32 + 2 * D
    total_q12 = total_f32 * 2
    a_w = 4 * (D * HD)  # Q,K,V,O per attention shard (heads/share)
    f_w = (F // f_shards) * D + D * (F // f_shards)
    lines = [
        f"fp32 params total      : {total_f32:,} floats = {total_f32*4:,} bytes",
        f"Q12 params total       : {total_q12:,} bytes",
        f"activation tile [S,D]  : {cfg.S*D:,} fp32 = {cfg.S*D*4:,} bytes",
        f"attention shard weights: {a_w:,} fp32 = {a_w*4:,} bytes (Q12 {a_w*2:,} bytes)",
        f"FFN shard weights      : {f_w:,} fp32 = {f_w*4:,} bytes (Q12 {f_w*2:,} bytes)",
        f"ln feature slabs       : {ln_slabs} (each {D//ln_slabs} features)",
        "per-layer fp32 weights : %d floats = %d bytes" % (per_layer_f32, per_layer_f32 * 4),
    ]
    return "\n".join(lines)


def main() -> None:
    cfg = Cfg()
    a_shards, f_shards, ln_slabs = 4, 16, 16
    ref = RefModel(cfg, seed=1234)
    sh = ShardedModel(cfg, ref, a_shards, f_shards, ln_slabs)

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1.0 / math.sqrt(cfg.D), (1, cfg.D)).astype(np.float32)
    x = np.repeat(x, cfg.S, axis=0)

    y_ref = ref.forward(x)
    y_sh = sh.forward(x)
    err = float(np.abs(y_ref - y_sh).max())
    print("case-08 sharded emulation (numpy fp32)")
    print(param_budget(cfg, a_shards, f_shards, ln_slabs))
    print(f"max abs diff ref vs sharded: {err:.3e}")
    assert err <= 1e-4, f"sharded forward diverged: {err}"
    print("OK: sharded == unsharded reference")


if __name__ == "__main__":
    main()
