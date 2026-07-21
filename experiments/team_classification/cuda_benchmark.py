"""Benchmark the real MobileNetV3 appearance extractor on a CUDA device.

This benchmark measures preprocessing, host-to-device transfer, inference, and
feature readback through the same ``AppearanceFeatureExtractor`` used by the
runtime.  Inputs are synthetic crops, so the result is a performance report,
not a football-team accuracy evaluation.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from app.classification.team_calibration.appearance_features import AppearanceFeatureExtractor


def run_benchmark(
    *,
    batch_sizes: list[int],
    iterations: int,
    warmup: int,
    seed: int,
    device: str,
    pretrained: bool,
) -> dict:
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise RuntimeError("CUDA is not available on this host")

    torch.manual_seed(seed)
    np.random.seed(seed)
    extractor_results: list[dict] = []

    for batch_size in batch_sizes:
        extractor = AppearanceFeatureExtractor(
            device=device,
            pretrained=pretrained,
            batch_size=batch_size,
        ).load()
        crops = [
            np.random.default_rng(seed + index).integers(
                0,
                256,
                size=(128, 64, 3),
                dtype=np.uint8,
            )
            for index in range(batch_size)
        ]

        for _ in range(warmup):
            extractor.extract_batch(crops)
        if extractor.device.type == "cuda":
            torch.cuda.synchronize(extractor.device)
            torch.cuda.reset_peak_memory_stats(extractor.device)

        latencies_ms: list[float] = []
        for _ in range(iterations):
            start = time.perf_counter()
            extractor.extract_batch(crops)
            if extractor.device.type == "cuda":
                torch.cuda.synchronize(extractor.device)
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

        latency_array = np.asarray(latencies_ms, dtype=np.float64)
        memory_mb = 0.0
        if extractor.device.type == "cuda":
            memory_mb = torch.cuda.max_memory_allocated(extractor.device) / (1024 * 1024)
        extractor_results.append(
            {
                "batch_size": batch_size,
                "feature_dim": extractor.feature_dim,
                "mean_batch_latency_ms": round(float(latency_array.mean()), 3),
                "p95_batch_latency_ms": round(float(np.percentile(latency_array, 95)), 3),
                "mean_crop_latency_ms": round(float(latency_array.mean() / batch_size), 3),
                "throughput_crops_per_second": round(
                    float(batch_size * 1000.0 / latency_array.mean()), 3
                ),
                "peak_allocated_memory_mb": round(float(memory_mb), 3),
            }
        )

    return {
        "benchmark": "mobilenet_v3_small_cuda_appearance",
        "warning": "Synthetic crops measure performance only, not real-match accuracy.",
        "device": device,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pretrained": pretrained,
        "iterations": iterations,
        "warmup": warmup,
        "results": extractor_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-sizes", default="1,4,8,16")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    batch_sizes = [int(value) for value in args.batch_sizes.split(",") if value.strip()]
    if not batch_sizes or any(value < 1 for value in batch_sizes):
        raise ValueError("--batch-sizes must contain positive integers")
    if args.iterations < 1 or args.warmup < 0:
        raise ValueError("--iterations must be positive and --warmup cannot be negative")

    result = run_benchmark(
        batch_sizes=batch_sizes,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        device=args.device,
        pretrained=not args.no_pretrained,
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
