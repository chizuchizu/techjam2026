#!/usr/bin/env python3
"""Train and export the complete tiny Transformer deployed on ESP32-C3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

CONTEXT = 16
MODEL_DIMENSION = 32
HEADS = 4
HEAD_DIMENSION = MODEL_DIMENSION // HEADS
LAYERS = 2
FFN_DIMENSION = 64

CORPUS = (
    "tiny chips can run transformers.\n"
    "esp32 nodes share attention.\n"
    "measure memory speed and accuracy.\n"
    "four heads work in parallel.\n"
)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        inverse_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-5)
        return values * inverse_rms * self.weight


class TransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = RMSNorm(MODEL_DIMENSION)
        self.query = nn.Linear(MODEL_DIMENSION, MODEL_DIMENSION, bias=False)
        self.key = nn.Linear(MODEL_DIMENSION, MODEL_DIMENSION, bias=False)
        self.value = nn.Linear(MODEL_DIMENSION, MODEL_DIMENSION, bias=False)
        self.output = nn.Linear(MODEL_DIMENSION, MODEL_DIMENSION, bias=False)
        self.norm2 = RMSNorm(MODEL_DIMENSION)
        self.ffn1 = nn.Linear(MODEL_DIMENSION, FFN_DIMENSION, bias=False)
        self.ffn2 = nn.Linear(FFN_DIMENSION, MODEL_DIMENSION, bias=False)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(values)
        batch, sequence, _ = normalized.shape
        query = self.query(normalized).view(batch, sequence, HEADS, HEAD_DIMENSION)
        key = self.key(normalized).view(batch, sequence, HEADS, HEAD_DIMENSION)
        value = self.value(normalized).view(batch, sequence, HEADS, HEAD_DIMENSION)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        scores = query @ key.transpose(-2, -1) / HEAD_DIMENSION**0.5
        mask = torch.ones(sequence, sequence, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
        context = functional.softmax(scores, dim=-1) @ value
        context = context.transpose(1, 2).contiguous().view(
            batch, sequence, MODEL_DIMENSION
        )
        values = values + self.output(context)
        values = values + self.ffn2(functional.relu(self.ffn1(self.norm2(values))))
        return values


class TinyTransformer(nn.Module):
    def __init__(self, vocabulary: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary, MODEL_DIMENSION)
        self.position_embedding = nn.Parameter(
            torch.empty(CONTEXT, MODEL_DIMENSION)
        )
        self.blocks = nn.ModuleList(TransformerBlock() for _ in range(LAYERS))
        self.final_norm = RMSNorm(MODEL_DIMENSION)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        values = self.token_embedding(tokens) + self.position_embedding
        for block in self.blocks:
            values = block(values)
        values = self.final_norm(values)
        return values @ self.token_embedding.weight.T


def make_dataset(vocabulary: str) -> tuple[torch.Tensor, torch.Tensor]:
    token_for = {character: index for index, character in enumerate(vocabulary)}
    doubled = CORPUS + CORPUS[:CONTEXT]
    inputs = []
    targets = []
    for begin in range(len(CORPUS)):
        inputs.append([token_for[character] for character in doubled[begin:begin + CONTEXT]])
        targets.append(token_for[doubled[begin + CONTEXT]])
    return torch.tensor(inputs), torch.tensor(targets)


def quantize_weight(weight: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    # Torch Linear is [output, input]; firmware stores [input, output].
    values = weight.detach().cpu().to(torch.float32)
    maximum = values.abs().amax(dim=1)
    scales = torch.where(maximum > 0, maximum / 127.0, torch.ones_like(maximum))
    quantized = torch.round(values / scales[:, None]).clamp(-127, 127).to(torch.int8)
    return quantized.T.contiguous().numpy(), scales.numpy()


def float_array(name: str, values: np.ndarray) -> str:
    flat = values.astype(np.float32).reshape(-1)
    body = ",".join(f"{float(value):.9g}f" for value in flat)
    return f"inline const float {name}[{len(flat)}] = {{{body}}};\n"


def int8_array(name: str, values: np.ndarray) -> str:
    flat = values.astype(np.int8).reshape(-1)
    body = ",".join(str(int(value)) for value in flat)
    return f"inline const int8_t {name}[{len(flat)}] = {{{body}}};\n"


def uint8_array(name: str, values: list[int]) -> str:
    body = ",".join(str(value) for value in values)
    return f"inline const uint8_t {name}[{len(values)}] = {{{body}}};\n"


def export_model(
    model: TinyTransformer,
    vocabulary: str,
    output_header: Path,
    output_npz: Path,
    output_manifest: Path,
    metrics: dict,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "token_embedding": model.token_embedding.weight.detach().cpu().numpy(),
        "position_embedding": model.position_embedding.detach().cpu().numpy(),
        "final_norm": model.final_norm.weight.detach().cpu().numpy(),
        "vocab_bytes": np.array([ord(character) for character in vocabulary], dtype=np.uint8),
    }
    prompt = CORPUS[:CONTEXT]
    token_for = {character: index for index, character in enumerate(vocabulary)}
    prompt_tokens = [token_for[character] for character in prompt]
    arrays["prompt_tokens"] = np.array(prompt_tokens, dtype=np.uint8)

    for layer, block in enumerate(model.blocks):
        arrays[f"layer{layer}_norm1"] = block.norm1.weight.detach().cpu().numpy()
        arrays[f"layer{layer}_norm2"] = block.norm2.weight.detach().cpu().numpy()
        for name, linear in (
            ("query", block.query),
            ("key", block.key),
            ("value", block.value),
            ("output", block.output),
            ("ffn1", block.ffn1),
            ("ffn2", block.ffn2),
        ):
            quantized, scales = quantize_weight(linear.weight)
            arrays[f"layer{layer}_{name}_weight"] = quantized
            arrays[f"layer{layer}_{name}_scale"] = scales
    lm_weight, lm_scale = quantize_weight(model.token_embedding.weight)
    arrays["lm_head_weight"] = lm_weight
    arrays["lm_head_scale"] = lm_scale

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_npz, **arrays)

    lines = [
        "#pragma once\n",
        "#include <stdint.h>\n\n",
        "namespace tiny_transformer_weights {\n",
        f"constexpr uint16_t CONTEXT = {CONTEXT};\n",
        f"constexpr uint16_t MODEL_DIMENSION = {MODEL_DIMENSION};\n",
        f"constexpr uint16_t HEADS = {HEADS};\n",
        f"constexpr uint16_t HEAD_DIMENSION = {HEAD_DIMENSION};\n",
        f"constexpr uint16_t LAYERS = {LAYERS};\n",
        f"constexpr uint16_t FFN_DIMENSION = {FFN_DIMENSION};\n",
        f"constexpr uint16_t VOCABULARY = {len(vocabulary)};\n",
        uint8_array("VOCAB_BYTES", arrays["vocab_bytes"].tolist()),
        uint8_array("PROMPT_TOKENS", prompt_tokens),
        float_array("TOKEN_EMBEDDING", arrays["token_embedding"]),
        float_array("POSITION_EMBEDDING", arrays["position_embedding"]),
    ]
    for layer in range(LAYERS):
        lines.append(float_array(f"LAYER{layer}_NORM1", arrays[f"layer{layer}_norm1"]))
        lines.append(float_array(f"LAYER{layer}_NORM2", arrays[f"layer{layer}_norm2"]))
        for name in ("query", "key", "value", "output", "ffn1", "ffn2"):
            upper = name.upper()
            lines.append(int8_array(
                f"LAYER{layer}_{upper}_WEIGHT", arrays[f"layer{layer}_{name}_weight"]
            ))
            lines.append(float_array(
                f"LAYER{layer}_{upper}_SCALE", arrays[f"layer{layer}_{name}_scale"]
            ))
    lines.extend(
        [
            float_array("FINAL_NORM", arrays["final_norm"]),
            int8_array("LM_HEAD_WEIGHT", arrays["lm_head_weight"]),
            float_array("LM_HEAD_SCALE", arrays["lm_head_scale"]),
            "}  // namespace tiny_transformer_weights\n",
        ]
    )
    output_header.parent.mkdir(parents=True, exist_ok=True)
    output_header.write_text("".join(lines))

    manifest = {
        "architecture": {
            "context": CONTEXT,
            "model_dimension": MODEL_DIMENSION,
            "heads": HEADS,
            "head_dimension": HEAD_DIMENSION,
            "layers": LAYERS,
            "ffn_dimension": FFN_DIMENSION,
            "vocabulary_size": len(vocabulary),
            "trained_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "normalization": "RMSNorm",
            "activation": "ReLU",
            "linear_weight_format": "int8 per-output-channel",
            "linear_activation_format": "dynamic int16",
            "attention_format": "int8 Q/K, int16 V, stable float row softmax",
        },
        "vocabulary": vocabulary,
        "prompt": prompt,
        "corpus": CORPUS,
        "metrics": metrics,
    }
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    output_manifest.write_text(json.dumps(manifest, indent=2) + "\n")


def greedy_generate(model: TinyTransformer, tokens: list[int], steps: int) -> list[int]:
    generated = list(tokens)
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            window = torch.tensor(generated[-CONTEXT:]).unsqueeze(0)
            next_token = int(model(window)[0, -1].argmax())
            generated.append(next_token)
    return generated


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--header", type=Path,
        default=Path("esp32_tiny_transformer/tiny_transformer_weights.h")
    )
    parser.add_argument(
        "--npz", type=Path, default=Path("models/tiny_transformer_weights.npz")
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("models/tiny_transformer_manifest.json")
    )
    args = parser.parse_args()

    torch.manual_seed(20260828)
    torch.set_num_threads(4)
    vocabulary = "".join(sorted(set(CORPUS)))
    inputs, targets = make_dataset(vocabulary)
    model = TinyTransformer(len(vocabulary))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    for step in range(args.steps):
        logits = model(inputs)[:, -1]
        loss = functional.cross_entropy(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 250 == 0 or step + 1 == args.steps:
            accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
            print(f"TRAIN,{step + 1},{loss.item():.8f},{accuracy:.6f}")

    model.eval()
    with torch.no_grad():
        logits = model(inputs)[:, -1]
        loss = functional.cross_entropy(logits, targets).item()
        accuracy = (logits.argmax(dim=-1) == targets).float().mean().item()
    token_for = {character: index for index, character in enumerate(vocabulary)}
    prompt_tokens = [token_for[character] for character in CORPUS[:CONTEXT]]
    generated = greedy_generate(model, prompt_tokens, 80)
    generated_text = "".join(vocabulary[token] for token in generated)
    metrics = {
        "training_steps": args.steps,
        "full_corpus_loss": loss,
        "full_corpus_next_token_accuracy": accuracy,
        "float_generation": generated_text,
        "seed": 20260828,
    }
    print(f"FINAL,{loss:.8f},{accuracy:.6f}")
    print("GENERATED," + generated_text.replace("\n", "\\n"))
    export_model(model, vocabulary, args.header, args.npz, args.manifest, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
