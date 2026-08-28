#!/usr/bin/env python3
"""Independent NumPy reference and serial validator for the tiny Transformer."""

from __future__ import annotations

import argparse
import json
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

import numpy as np

ABSOLUTE_TOLERANCE = 0.002
RELATIVE_TOLERANCE = 0.02


def round_away(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, np.floor(values + 0.5), np.ceil(values - 0.5))


def quantize(values: np.ndarray, limit: int) -> tuple[np.ndarray, np.float32]:
    maximum = np.max(np.abs(values)).astype(np.float32)
    scale = np.float32(maximum / limit) if maximum > 0 else np.float32(1.0)
    quantized = np.clip(round_away(values / scale), -limit, limit)
    dtype = np.int8 if limit == 127 else np.int16
    return quantized.astype(dtype), scale


def quantized_linear(
    values: np.ndarray, weight: np.ndarray, weight_scale: np.ndarray
) -> np.ndarray:
    quantized, activation_scale = quantize(values, 32767)
    dot = quantized.astype(np.int32) @ weight.astype(np.int32)
    return (
        dot.astype(np.float32)
        * activation_scale
        * weight_scale.astype(np.float32)
    ).astype(np.float32)


def rms_norm(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    mean_square = np.mean(values * values, axis=-1, keepdims=True, dtype=np.float32)
    inverse_rms = np.float32(1.0) / np.sqrt(mean_square + np.float32(1e-5))
    return (values * inverse_rms * weight).astype(np.float32)


def mixed_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    heads: int,
) -> np.ndarray:
    sequence, dimension = query.shape
    head_dimension = dimension // heads
    query_int8, query_scale = quantize(query, 127)
    key_int8, key_scale = quantize(key, 127)
    value_int16, value_scale = quantize(value, 32767)
    context = np.zeros_like(query, dtype=np.float32)
    score_scale = np.float32(
        query_scale * key_scale / np.sqrt(np.float32(head_dimension))
    )
    for head in range(heads):
        begin = head * head_dimension
        end = begin + head_dimension
        for query_index in range(sequence):
            scores = []
            for key_index in range(query_index + 1):
                dot = np.dot(
                    query_int8[query_index, begin:end].astype(np.int32),
                    key_int8[key_index, begin:end].astype(np.int32),
                )
                scores.append(np.float32(dot) * score_scale)
            scores_array = np.asarray(scores, dtype=np.float32)
            weights = np.exp(scores_array - np.max(scores_array)).astype(np.float32)
            denominator = np.sum(weights, dtype=np.float32)
            numerator = np.zeros(head_dimension, dtype=np.float32)
            for key_index, attention_weight in enumerate(weights):
                numerator += attention_weight * value_int16[key_index, begin:end]
            context[query_index, begin:end] = (
                numerator * value_scale / denominator
            )
    return context


def infer(tokens: list[int], weights: dict[str, np.ndarray], manifest: dict) -> np.ndarray:
    architecture = manifest["architecture"]
    context = architecture["context"]
    heads = architecture["heads"]
    if len(tokens) != context:
        raise ValueError(f"expected exactly {context} tokens")
    values = (
        weights["token_embedding"][np.asarray(tokens)]
        + weights["position_embedding"]
    ).astype(np.float32)
    for layer in range(architecture["layers"]):
        normalized = rms_norm(values, weights[f"layer{layer}_norm1"])
        query = quantized_linear(
            normalized,
            weights[f"layer{layer}_query_weight"],
            weights[f"layer{layer}_query_scale"],
        )
        key = quantized_linear(
            normalized,
            weights[f"layer{layer}_key_weight"],
            weights[f"layer{layer}_key_scale"],
        )
        value = quantized_linear(
            normalized,
            weights[f"layer{layer}_value_weight"],
            weights[f"layer{layer}_value_scale"],
        )
        attended = mixed_attention(query, key, value, heads)
        projected = quantized_linear(
            attended,
            weights[f"layer{layer}_output_weight"],
            weights[f"layer{layer}_output_scale"],
        )
        values = (values + projected).astype(np.float32)

        normalized = rms_norm(values, weights[f"layer{layer}_norm2"])
        hidden = quantized_linear(
            normalized,
            weights[f"layer{layer}_ffn1_weight"],
            weights[f"layer{layer}_ffn1_scale"],
        )
        hidden = np.maximum(hidden, np.float32(0.0)).astype(np.float32)
        feed_forward = quantized_linear(
            hidden,
            weights[f"layer{layer}_ffn2_weight"],
            weights[f"layer{layer}_ffn2_scale"],
        )
        values = (values + feed_forward).astype(np.float32)

    normalized = rms_norm(values, weights["final_norm"])
    return quantized_linear(
        normalized[-1:], weights["lm_head_weight"], weights["lm_head_scale"]
    )[0]


def generate(
    prompt: list[int],
    steps: int,
    weights: dict[str, np.ndarray],
    manifest: dict,
) -> list[int]:
    generated = list(prompt)
    for _ in range(steps):
        logits = infer(generated[-manifest["architecture"]["context"]:], weights, manifest)
        generated.append(int(np.argmax(logits)))
    return generated


def load_model(npz_path: Path, manifest_path: Path) -> tuple[dict, dict]:
    archive = np.load(npz_path)
    weights = {name: archive[name] for name in archive.files}
    manifest = json.loads(manifest_path.read_text())
    return weights, manifest


def corpus_accuracy(weights: dict, manifest: dict) -> tuple[int, int]:
    corpus = manifest["corpus"]
    vocabulary = manifest["vocabulary"]
    token_for = {character: index for index, character in enumerate(vocabulary)}
    context = manifest["architecture"]["context"]
    doubled = corpus + corpus[:context]
    correct = 0
    for begin in range(len(corpus)):
        tokens = [token_for[character] for character in doubled[begin:begin + context]]
        expected = token_for[doubled[begin + context]]
        correct += int(np.argmax(infer(tokens, weights, manifest)) == expected)
    return correct, len(corpus)


def capture_serial(port: str, timeout: float) -> str:
    descriptor = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        tty.setraw(descriptor)
        attributes = termios.tcgetattr(descriptor)
        attributes[4] = termios.B115200
        attributes[5] = termios.B115200
        termios.tcsetattr(descriptor, termios.TCSANOW, attributes)
        termios.tcflush(descriptor, termios.TCIOFLUSH)
        time.sleep(0.25)
        os.write(descriptor, b"f\n")
        deadline = time.monotonic() + timeout
        captured = bytearray()
        while time.monotonic() < deadline:
            readable, _, _ = select.select([descriptor], [], [], 0.25)
            if not readable:
                continue
            captured.extend(os.read(descriptor, 4096))
            if b"TINY_DONE" in captured:
                break
        if b"TINY_DONE" not in captured:
            raise TimeoutError("no TINY_DONE marker from board")
        os.write(descriptor, b"g\n")
        generation_deadline = time.monotonic() + timeout
        while time.monotonic() < generation_deadline:
            readable, _, _ = select.select([descriptor], [], [], 0.25)
            if not readable:
                continue
            captured.extend(os.read(descriptor, 4096))
            if b"TINY_GENERATION_DONE" in captured:
                break
        if b"TINY_GENERATION_DONE" not in captured:
            raise TimeoutError("no TINY_GENERATION_DONE marker from board")
        return captured.decode("utf-8", errors="replace")
    finally:
        os.close(descriptor)


def validate_capture(
    captured: str, weights: dict, manifest: dict
) -> tuple[float, float, int, bool]:
    expected = infer(weights["prompt_tokens"].tolist(), weights, manifest)
    actual = np.full_like(expected, np.nan)
    generated: dict[int, int] = {}
    board_status = False
    for line in captured.splitlines():
        fields = line.strip().split(",")
        if fields[0] == "TINY_LOGIT" and len(fields) == 3:
            actual[int(fields[1])] = float(fields[2])
        elif fields[0] == "TINY_GENERATED" and len(fields) >= 3:
            generated[int(fields[1])] = int(fields[2])
        elif fields[0] == "TINY_RESULT" and fields[-1] == "PASS":
            board_status = True
    maximum_absolute = 0.0
    maximum_relative = 0.0
    failed = 0
    for expected_value, actual_value in zip(expected, actual):
        if np.isnan(actual_value):
            failed += 1
            continue
        absolute = abs(float(actual_value - expected_value))
        relative = absolute / max(abs(float(expected_value)), 1e-12)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        if absolute > ABSOLUTE_TOLERANCE and absolute > RELATIVE_TOLERANCE * abs(expected_value):
            failed += 1

    expected_generation = generate(
        weights["prompt_tokens"].tolist(), 48, weights, manifest
    )[manifest["architecture"]["context"]:]
    generation_ok = (
        len(generated) == len(expected_generation)
        and all(generated.get(index) == token for index, token in enumerate(expected_generation))
    )
    return maximum_absolute, maximum_relative, failed, board_status and generation_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz", type=Path, default=Path("models/tiny_transformer_weights.npz")
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("models/tiny_transformer_manifest.json")
    )
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--input-capture", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    weights, manifest = load_model(args.npz, args.manifest)

    correct, total = corpus_accuracy(weights, manifest)
    prompt = weights["prompt_tokens"].tolist()
    generated = generate(prompt, 80, weights, manifest)
    generated_text = "".join(manifest["vocabulary"][token] for token in generated)
    print(f"QUANTIZED_CORPUS_ACCURACY,{correct},{total},{correct / total:.9f}")
    print("QUANTIZED_GENERATION," + generated_text.replace("\n", "\\n"))
    if args.self_test and not args.input_capture and not args.capture:
        return 0 if correct == total else 1

    if args.input_capture:
        captured = args.input_capture.read_text()
    else:
        captured = capture_serial(args.port, args.timeout)
    if args.capture:
        args.capture.parent.mkdir(parents=True, exist_ok=True)
        args.capture.write_text(captured)
    maximum_absolute, maximum_relative, failed, full_status = validate_capture(
        captured, weights, manifest
    )
    status = "PASS" if failed == 0 and full_status else "FAIL"
    print(
        f"HOST_VALIDATION,{maximum_absolute:.9g},{maximum_relative:.9g},"
        f"{failed},{status}"
    )
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
