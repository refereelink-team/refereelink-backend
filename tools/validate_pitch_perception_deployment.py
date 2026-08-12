#!/usr/bin/env python3
"""Validate numerical parity and runtime for a deployed pitch model.

This command does not measure model accuracy. It compares framework outputs on
the same deterministic input and records latency for deployment smoke testing.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import torch

try:
    from tools._bootstrap import ensure_repository_root
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from _bootstrap import ensure_repository_root

ensure_repository_root(__file__)

from tools.export_pitch_perception import load_trained_model


def summarize_latencies(values_ms: Sequence[float]) -> dict[str, float]:
    if not values_ms:
        raise ValueError("at least one latency observation is required")
    values = np.asarray(values_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(values)),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "max_ms": float(np.max(values)),
    }


def compare_output_sets(
    reference_outputs: Sequence[np.ndarray],
    deployed_outputs: Sequence[np.ndarray],
    *,
    atol: float,
    rtol: float,
) -> list[dict[str, object]]:
    if len(reference_outputs) != len(deployed_outputs):
        raise ValueError("frameworks returned different output counts")
    comparisons: list[dict[str, object]] = []
    for reference, deployed in zip(
        reference_outputs, deployed_outputs, strict=True
    ):
        if reference.shape != deployed.shape:
            raise ValueError(
                f"frameworks returned different shapes: {reference.shape} "
                f"and {deployed.shape}"
            )
        deployed_float = deployed.astype(np.float32, copy=False)
        reference_float = reference.astype(np.float32, copy=False)
        delta = np.abs(reference_float - deployed_float)
        comparisons.append(
            {
                "shape": list(reference.shape),
                "max_abs_error": float(np.max(delta)),
                "mean_abs_error": float(np.mean(delta)),
                "allclose": bool(
                    np.allclose(
                        reference_float,
                        deployed_float,
                        atol=atol,
                        rtol=rtol,
                    )
                ),
            }
        )
    return comparisons


def require_active_provider(
    requested_provider: str,
    available_providers: Sequence[str],
    active_providers: Sequence[str],
) -> None:
    if requested_provider not in available_providers:
        raise RuntimeError(
            f"requested provider {requested_provider!r} is unavailable; "
            f"available providers: {list(available_providers)}"
        )
    if not active_providers or active_providers[0] != requested_provider:
        raise RuntimeError(
            f"requested provider {requested_provider!r} is not active; "
            f"session providers: {list(active_providers)}"
        )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_pytorch(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    *,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, float]:
    device = input_tensor.device
    with torch.inference_mode():
        for _ in range(warmup_iterations):
            model(input_tensor)
        synchronize(device)
        latencies: list[float] = []
        for _ in range(measured_iterations):
            started = time.perf_counter()
            model(input_tensor)
            synchronize(device)
            latencies.append((time.perf_counter() - started) * 1000.0)
    return summarize_latencies(latencies)


def benchmark_onnx_session(
    session: Any,
    input_array: np.ndarray,
    *,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, float]:
    for _ in range(warmup_iterations):
        session.run(None, {"images": input_array})
    latencies: list[float] = []
    for _ in range(measured_iterations):
        started = time.perf_counter()
        session.run(None, {"images": input_array})
        latencies.append((time.perf_counter() - started) * 1000.0)
    return summarize_latencies(latencies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--provider", default="CUDAExecutionProvider")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--atol", type=float, default=0.02)
    parser.add_argument("--rtol", type=float, default=0.02)
    arguments = parser.parse_args()
    if arguments.warmup < 0 or arguments.iterations < 1:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    if importlib.util.find_spec("onnxruntime") is None:
        raise RuntimeError(
            "ONNX Runtime is unavailable; install a package compatible with the "
            "target device before deployment validation"
        )
    import onnxruntime as ort

    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but PyTorch cannot access it")
    dtype = torch.float16 if arguments.precision == "fp16" else torch.float32
    if dtype == torch.float16 and device.type != "cuda":
        raise ValueError("FP16 validation requires a CUDA device")

    model, payload = load_trained_model(arguments.checkpoint, device)
    model = model.to(dtype=dtype).eval()
    width, height = (int(value) for value in payload["input_size"])
    rng = np.random.default_rng(arguments.seed)
    input_array = rng.random((1, 3, height, width), dtype=np.float32).astype(
        np.float16 if dtype == torch.float16 else np.float32
    )
    input_tensor = torch.from_numpy(input_array).to(device)

    session = ort.InferenceSession(
        str(arguments.model),
        providers=[arguments.provider, "CPUExecutionProvider"],
    )
    available_providers = ort.get_available_providers()
    active_providers = session.get_providers()
    require_active_provider(
        arguments.provider,
        available_providers,
        active_providers,
    )

    with torch.inference_mode():
        reference_outputs = [
            output.float().cpu().numpy() for output in model(input_tensor)
        ]
    deployed_outputs = session.run(None, {"images": input_array})
    comparisons = compare_output_sets(
        reference_outputs,
        deployed_outputs,
        atol=arguments.atol,
        rtol=arguments.rtol,
    )
    report = {
        "status": "ok",
        "purpose": "numerical_parity_and_runtime_only",
        "deployment_accuracy_valid": False,
        "checkpoint_validation": payload["validation"],
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "available_providers": available_providers,
        "active_providers": active_providers,
        "input_shape": list(input_array.shape),
        "precision": arguments.precision,
        "tolerance": {"atol": arguments.atol, "rtol": arguments.rtol},
        "outputs": comparisons,
        "numerically_usable": all(item["allclose"] for item in comparisons),
        "latency": {
            "pytorch_forward_cuda_sync": benchmark_pytorch(
                model,
                input_tensor,
                warmup_iterations=arguments.warmup,
                measured_iterations=arguments.iterations,
            ),
            "onnxruntime_session_run": benchmark_onnx_session(
                session,
                input_array,
                warmup_iterations=arguments.warmup,
                measured_iterations=arguments.iterations,
            ),
        },
        "notes": [
            "This run checks numerical parity and runtime, not football accuracy.",
            "PyTorch timing synchronizes CUDA after each forward pass.",
            "ONNX Runtime session.run includes transfers and CPU output materialization.",
        ],
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not report["numerically_usable"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
