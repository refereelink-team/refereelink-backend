#!/usr/bin/env python3
"""Measure dual-head pitch-perception latency without substituting other models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from app.field_registration.models import build_pitch_perception_model
from app.field_registration.perception import PitchPerceptionVocabulary
from app.field_registration.pitch_model import PitchModel
from experiments.field_registration.research_assets import sha256_file
from tools.export_pitch_perception import load_trained_model


@torch.inference_mode()
def benchmark_model(
    architecture: str,
    *,
    device_name: str,
    input_size: tuple[int, int],
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    use_fp16: bool,
    checkpoint: Path | None = None,
) -> dict[str, object]:
    device = torch.device(device_name)
    vocabulary = PitchPerceptionVocabulary.from_pitch_model(PitchModel())
    try:
        if checkpoint is None:
            model = build_pitch_perception_model(
                architecture,
                vocabulary.semantic_class_count,
                len(vocabulary.landmark_labels),
                pretrained=False,
            ).eval().to(device)
            weights = "random_latency_only"
            checkpoint_validation = None
        else:
            model, payload = load_trained_model(checkpoint, device)
            checkpoint_architecture = str(payload["architecture"])
            if checkpoint_architecture != architecture:
                raise ValueError(
                    f"checkpoint architecture {checkpoint_architecture!r} does not "
                    f"match requested architecture {architecture!r}"
                )
            weights = {
                "checkpoint": str(checkpoint),
                "sha256": sha256_file(checkpoint),
            }
            checkpoint_validation = payload.get("validation")
    except ValueError as error:
        return {
            "architecture": architecture,
            "status": "not_implemented",
            "error": str(error),
        }
    width, height = input_size
    inputs = torch.randn(batch_size, 3, height, width, device=device)
    fp16_enabled = bool(use_fp16 and device.type == "cuda")
    for _ in range(warmup_iterations):
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16_enabled,
        ):
            outputs = model(inputs)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    latencies: list[float] = []
    for _ in range(measured_iterations):
        started = time.perf_counter()
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=fp16_enabled,
        ):
            outputs = model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        latencies.append((time.perf_counter() - started) * 1000.0)
    values = np.asarray(latencies, dtype=np.float64)
    return {
        "architecture": architecture,
        "status": "ok",
        "device": str(device),
        "precision": "fp16" if fp16_enabled else "fp32",
        "batch_size": batch_size,
        "input_size": [width, height],
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "latency_ms": {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "maximum": float(np.max(values)),
        },
        "throughput_fps": float(batch_size * 1000.0 / np.mean(values)),
        "peak_gpu_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))
            if device.type == "cuda"
            else None
        ),
        "output_shapes": [list(output.shape) for output in outputs],
        "weights": weights,
        "checkpoint_validation": checkpoint_validation,
        "accuracy_valid": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        nargs="+",
        default=["mobilenet_v3_dual_head"],
        choices=(
            "mobilenet_v3_dual_head",
            "pidnet_s_dual_head",
            "segformer_b0_dual_head",
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if arguments.checkpoint is not None and len(arguments.architectures) != 1:
        parser.error("--checkpoint requires exactly one architecture")
    rows = [
        benchmark_model(
            architecture,
            device_name=arguments.device,
            input_size=(arguments.input_width, arguments.input_height),
            batch_size=arguments.batch_size,
            warmup_iterations=arguments.warmup,
            measured_iterations=arguments.iterations,
            use_fp16=not arguments.fp32,
            checkpoint=arguments.checkpoint,
        )
        for architecture in arguments.architectures
    ]
    rendered = json.dumps(rows, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
