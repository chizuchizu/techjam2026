#!/usr/bin/env python3
"""Build and accuracy-gate a fixed-shape TensorRT Transformer engine."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # official benchmark at repo root

from torch_transformer_benchmark import (
    BaselineTransformer,
    TransformerConfig,
    benchmark_models,
    run_accuracy_tests,
)


class _NoMaskExportAdapter(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x, None)


def export_onnx(model: nn.Module, path: Path, config: TransformerConfig) -> str:
    """Export fixed-shape FP32 ONNX and return a content digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter = _NoMaskExportAdapter(model).eval()
    sample = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        dtype=torch.float32,
    )
    torch.onnx.export(
        adapter,
        (sample,),
        path,
        input_names=("input",),
        output_names=("output",),
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_engine(onnx_path: Path, engine_path: Path) -> None:
    """Parse ONNX and build a strongly typed TensorRT 11 engine."""
    try:
        import tensorrt as trt
    except ImportError as error:
        raise RuntimeError(
            "TensorRT is optional; install requirements-tensorrt.txt first"
        ) from error

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        errors = "\n".join(
            str(parser.get_error(index)) for index in range(parser.num_errors)
        )
        raise RuntimeError(f"TensorRT failed to parse ONNX:\n{errors}")

    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    serialized = builder.build_serialized_network(network, build_config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_path.write_bytes(bytes(serialized))


class TensorRTGraphTransformer(nn.Module):
    """TensorRT execution captured in a CUDA graph with PyTorch tensor I/O."""

    def __init__(
        self,
        engine_path: Path,
        shape: tuple[int, int, int],
        clone_output: bool,
    ) -> None:
        super().__init__()
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError(
                "TensorRT is optional; install requirements-tensorrt.txt first"
            ) from error

        self.clone_output = clone_output
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("TensorRT execution-context creation failed")

        self.static_input = torch.empty(shape, device="cuda", dtype=torch.float32)
        self.static_output = torch.empty_like(self.static_input)
        self.context.set_tensor_address("input", self.static_input.data_ptr())
        self.context.set_tensor_address("output", self.static_output.data_ptr())

        # TensorRT must initialize tactics/resources before stream capture.
        self.capture_stream = torch.cuda.Stream()
        self.capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.capture_stream):
            for _ in range(20):
                if not self.context.execute_async_v3(
                    self.capture_stream.cuda_stream
                ):
                    raise RuntimeError("TensorRT warmup execution failed")
        self.capture_stream.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            if not self.context.execute_async_v3(self.capture_stream.cuda_stream):
                raise RuntimeError("TensorRT graph-capture execution failed")
        torch.cuda.current_stream().wait_stream(self.capture_stream)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if valid_token_mask is not None:
            raise ValueError("the current TensorRT experiment is unmasked")
        self.static_input.copy_(x)
        self.graph.replay()
        if self.clone_output:
            return self.static_output.clone()
        return self.static_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export, build, and benchmark the fixed FP32 TensorRT model"
    )
    parser.add_argument("--artifacts-dir", type=Path, default=Path(".tensorrt-cache"))
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--static-output", action="store_true")
    parser.add_argument("--accuracy-trials", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--benchmark-rounds", type=int, default=5)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.002)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT benchmark requires CUDA")

    config = TransformerConfig(8, 128, 512, 8, 2048, 6, False)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Export before device transfer so ONNX owns the exact reference weights.
    baseline = BaselineTransformer(config).eval()
    onnx_path = args.artifacts_dir / "transformer_fp32.onnx"
    digest = export_onnx(baseline, onnx_path, config)
    engine_path = args.artifacts_dir / f"transformer_fp32_{digest[:16]}.engine"
    if args.rebuild or not engine_path.exists():
        print(f"building TensorRT engine: {engine_path}")
        build_engine(onnx_path, engine_path)
    else:
        print(f"reusing TensorRT engine: {engine_path}")

    device = torch.device("cuda:0")
    baseline = baseline.to(device).eval()
    candidate = TensorRTGraphTransformer(
        engine_path,
        (config.batch_size, config.seq_len, config.d_model),
        clone_output=not args.static_output,
    )

    print("=== TensorRT configuration ===")
    print(config)
    print(f"dtype=torch.float32, engine={engine_path}")
    print(f"static_output={args.static_output}")
    accuracy_passed = run_accuracy_tests(
        baseline=baseline,
        optimized=candidate,
        config=config,
        device=device,
        dtype=torch.float32,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=0.0,
        input_scale=1.0,
        rtol=args.rtol,
        atol=args.atol,
    )
    if not accuracy_passed:
        print("TensorRT benchmark skipped because accuracy validation failed")
        return 1

    benchmark_models(
        baseline=baseline,
        optimized=candidate,
        config=config,
        device=device,
        dtype=torch.float32,
        seed=args.seed,
        padding_ratio=0.0,
        input_scale=1.0,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
