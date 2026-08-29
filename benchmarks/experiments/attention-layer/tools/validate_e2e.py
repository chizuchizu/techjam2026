#!/usr/bin/env python3
"""Run and independently validate the ESP32 end-to-end attention fixture."""

from __future__ import annotations

import argparse
import math
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path

SEQUENCE = 16
MODEL_DIMENSION = 32
HEADS = 4
HEAD_DIMENSION = MODEL_DIMENSION // HEADS
ABSOLUTE_TOLERANCE = 0.002
RELATIVE_TOLERANCE = 0.02


def fixture_input(token: int, feature: int) -> float:
    value = (token * 37 + feature * 17 + token * feature * 3 + 13) % 101 - 50
    return value / 100.0


def fixture_weight(matrix: int, input_feature: int, output_feature: int) -> float:
    mixed = (
        (matrix + 1) * 29
        + input_feature * 31
        + output_feature * 17
        + input_feature * output_feature * 3
    )
    return (mixed % 127 - 63) / 256.0


def fixture_bias(projection: int, feature: int) -> float:
    return (((projection + 1) * 11 + feature * 7) % 31 - 15) / 512.0


def token_is_valid(token: int) -> bool:
    return token % 7 != 5


def project(values: list[list[float]], projection: int) -> list[list[float]]:
    output: list[list[float]] = []
    for row in values:
        output_row = []
        for output_feature in range(MODEL_DIMENSION):
            total = fixture_bias(projection, output_feature)
            for input_feature, value in enumerate(row):
                total += value * fixture_weight(
                    projection, input_feature, output_feature
                )
            output_row.append(total)
        output.append(output_row)
    return output


def reference(causal: bool) -> list[float]:
    inputs = [
        [fixture_input(token, feature) for feature in range(MODEL_DIMENSION)]
        for token in range(SEQUENCE)
    ]
    query = project(inputs, 0)
    key = project(inputs, 1)
    value = project(inputs, 2)
    context = [[0.0] * MODEL_DIMENSION for _ in range(SEQUENCE)]
    score_scale = 1.0 / math.sqrt(HEAD_DIMENSION)

    for head in range(HEADS):
        begin = head * HEAD_DIMENSION
        end = begin + HEAD_DIMENSION
        for query_index in range(SEQUENCE):
            if not token_is_valid(query_index):
                continue
            visible_keys = [
                key_index
                for key_index in range(SEQUENCE)
                if token_is_valid(key_index)
                and (not causal or key_index <= query_index)
            ]
            scores = [
                sum(
                    query[query_index][feature] * key[key_index][feature]
                    for feature in range(begin, end)
                )
                * score_scale
                for key_index in visible_keys
            ]
            maximum = max(scores)
            weights = [math.exp(score - maximum) for score in scores]
            weight_sum = sum(weights)
            for key_index, weight in zip(visible_keys, weights):
                probability = weight / weight_sum
                for feature in range(begin, end):
                    context[query_index][feature] += (
                        probability * value[key_index][feature]
                    )

    output = project(context, 3)
    for token in range(SEQUENCE):
        if not token_is_valid(token):
            output[token] = [0.0] * MODEL_DIMENSION
    return [value for row in output for value in row]


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
        os.write(descriptor, b"e\n")

        deadline = time.monotonic() + timeout
        captured = bytearray()
        while time.monotonic() < deadline:
            readable, _, _ = select.select([descriptor], [], [], 0.25)
            if not readable:
                continue
            chunk = os.read(descriptor, 4096)
            if not chunk:
                continue
            captured.extend(chunk)
            if b"E2E_DONE" in captured:
                break
        text = captured.decode("utf-8", errors="replace")
        if "E2E_DONE" not in text:
            raise TimeoutError(f"no E2E_DONE marker from {port} after {timeout}s")
        return text
    finally:
        os.close(descriptor)


def parse_outputs(text: str) -> tuple[dict[tuple[str, int], list[float]], list[str]]:
    outputs: dict[tuple[str, int], list[float]] = {}
    result_lines: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("E2E_RESULT,"):
            result_lines.append(line)
        elif line.startswith("E2E_OUTPUT,"):
            _, candidate, causal_text, index_text, value_text = line.split(",", 4)
            causal = int(causal_text)
            index = int(index_text)
            key = (candidate, causal)
            if key not in outputs:
                outputs[key] = [math.nan] * (SEQUENCE * MODEL_DIMENSION)
            if 0 <= index < len(outputs[key]):
                outputs[key][index] = float(value_text)
    return outputs, result_lines


def compare(expected: list[float], actual: list[float]) -> tuple[float, float, int]:
    maximum_absolute = 0.0
    maximum_relative = 0.0
    failed = 0
    for expected_value, actual_value in zip(expected, actual):
        if math.isnan(actual_value):
            failed += 1
            continue
        absolute = abs(actual_value - expected_value)
        relative = absolute / max(abs(expected_value), 1.0e-12)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        if absolute > ABSOLUTE_TOLERANCE and (
            absolute > RELATIVE_TOLERANCE * abs(expected_value)
        ):
            failed += 1
    return maximum_absolute, maximum_relative, failed


def main() -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--port", default="/dev/ttyACM0")
    source.add_argument("--input-capture", type=Path)
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    if args.input_capture:
        captured = args.input_capture.read_text()
    else:
        captured = capture_serial(args.port, args.timeout)
    if args.capture:
        args.capture.parent.mkdir(parents=True, exist_ok=True)
        args.capture.write_text(captured)

    outputs, board_results = parse_outputs(captured)
    for line in board_results:
        print(f"BOARD,{line}")
    expected_candidates = {
        "float_proj_mixed_attention",
        "int16_act_int8_proj_mixed_attention",
    }
    all_passed = len(board_results) == 4
    print(
        "HOST,candidate,causal,max_abs_error,max_relative_error,"
        "failed_elements,status"
    )
    for candidate in sorted(expected_candidates):
        for causal in (0, 1):
            actual = outputs.get(
                (candidate, causal), [math.nan] * (SEQUENCE * MODEL_DIMENSION)
            )
            stats = compare(reference(bool(causal)), actual)
            status = "PASS" if stats[2] == 0 else "FAIL"
            print(
                f"HOST,{candidate},{causal},{stats[0]:.9g},{stats[1]:.9g},"
                f"{stats[2]},{status}"
            )
            all_passed = all_passed and stats[2] == 0
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
