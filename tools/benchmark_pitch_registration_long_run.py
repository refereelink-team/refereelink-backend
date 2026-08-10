#!/usr/bin/env python3
"""Long-run CUDA benchmark for the integrated field-registration V2 pipeline."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

from app.constants.paths import PITCH_DETECTION_MODEL_PATH, PLAYER_DETECTION_MODEL_PATH
from app.vision.core import VisionCore


def _distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p95": None, "maximum": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def _memory_slope_mb_per_minute(samples: list[tuple[float, float]]) -> float | None:
    # Short smoke runs are dominated by allocator warm-up and produce wildly
    # misleading slopes. Five minutes is the minimum useful observation span.
    if len(samples) < 2 or samples[-1][0] - samples[0][0] < 300.0:
        return None
    elapsed_minutes = np.asarray([row[0] / 60.0 for row in samples])
    rss_mb = np.asarray([row[1] for row in samples])
    return float(np.polyfit(elapsed_minutes, rss_mb, 1)[0])


@torch.inference_mode()
def run_benchmark(
    *,
    source: Path,
    device: str,
    player_model_path: str,
    pitch_model_path: str,
    camera_calibration_path: str | None,
    camera_rig_profile_path: str | None,
    duration_minutes: float,
    max_frames: int | None,
    warmup_frames: int,
    imgsz: int,
    pitch_detection_interval: int,
    sample_interval_frames: int,
) -> dict[str, object]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open source video: {source}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 0:
        capture.release()
        raise ValueError("source video reports an invalid frame rate")
    core = VisionCore(
        device=device,
        fps=source_fps,
        player_model_path=player_model_path,
        pitch_model_path=pitch_model_path,
        camera_calibration_path=camera_calibration_path,
        enable_undistortion=camera_calibration_path is not None,
        pitch_detection_interval=pitch_detection_interval,
        imgsz=imgsz,
        enable_field_registration_v2=True,
        camera_rig_profile_path=camera_rig_profile_path,
    )
    core.load_models()
    process = psutil.Process(os.getpid())
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    frame_latencies_ms: list[float] = []
    coordinate_sigmas_m: list[float] = []
    camera_statuses: Counter[str] = Counter()
    memory_samples: list[tuple[float, float]] = []
    coordinate_count = 0
    usable_coordinate_count = 0
    frame_count = 0
    absolute_frame_index = 0
    video_loops = 0
    for _ in range(max(warmup_frames, 0)):
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_loops += 1
            ok, frame = capture.read()
            if not ok:
                capture.release()
                raise RuntimeError("source video could not be rewound during warm-up")
        absolute_frame_index += 1
        core.process(frame, absolute_frame_index)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    semantic_start = core.pitch_detection_count
    reuse_start = core.pitch_reuse_count
    started = time.perf_counter()
    deadline = started + duration_minutes * 60.0
    try:
        while (max_frames is None or frame_count < max_frames) and time.perf_counter() < deadline:
            ok, frame = capture.read()
            if not ok:
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                video_loops += 1
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("source video could not be rewound")
            frame_count += 1
            absolute_frame_index += 1
            frame_started = time.perf_counter()
            result = core.process(frame, absolute_frame_index)
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_latencies_ms.append((time.perf_counter() - frame_started) * 1000.0)
            camera_statuses[result.homography_status] += 1
            coordinate_count += len(result.pitch_coordinates)
            for coordinate in result.pitch_coordinates:
                if coordinate.xy_m is not None:
                    usable_coordinate_count += 1
                if coordinate.sigma_m is not None and np.isfinite(coordinate.sigma_m):
                    coordinate_sigmas_m.append(float(coordinate.sigma_m))
            if frame_count == 1 or frame_count % max(sample_interval_frames, 1) == 0:
                elapsed = time.perf_counter() - started
                rss_mb = process.memory_info().rss / (1024 * 1024)
                memory_samples.append((elapsed, rss_mb))
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    if not memory_samples or memory_samples[-1][0] < elapsed:
        memory_samples.append((elapsed, process.memory_info().rss / (1024 * 1024)))
    return {
        "status": "ok",
        "accuracy_valid": False,
        "source": str(source),
        "device": device,
        "source_fps": source_fps,
        "warmup_frames": warmup_frames,
        "frames": frame_count,
        "video_loops": video_loops,
        "elapsed_sec": elapsed,
        "end_to_end_fps": frame_count / max(elapsed, 1e-9),
        "frame_latency_ms": _distribution(frame_latencies_ms),
        "camera_status_counts": dict(camera_statuses),
        "homography_available_ratio": core.homography_available_count / max(frame_count, 1),
        "semantic_inference_count": core.pitch_detection_count - semantic_start,
        "semantic_reuse_ratio": (core.pitch_reuse_count - reuse_start) / max(frame_count, 1),
        "coordinate_usable_ratio": usable_coordinate_count / max(coordinate_count, 1),
        "coordinate_sigma_m": _distribution(coordinate_sigmas_m),
        "rss_mb": {
            "first": memory_samples[0][1],
            "last": memory_samples[-1][1],
            "growth": memory_samples[-1][1] - memory_samples[0][1],
            "slope_per_minute": _memory_slope_mb_per_minute(memory_samples),
        },
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
        "model_weights": {
            "player": player_model_path,
            "pitch": pitch_model_path,
            "field_perception": "legacy_keypoint_adapter",
        },
        "notes": [
            "This is a stability/performance run, not an accuracy evaluation.",
            "The V2 tracker uses the legacy keypoint model until a trained dual-head checkpoint exists.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--player-model-path", default=PLAYER_DETECTION_MODEL_PATH)
    parser.add_argument("--pitch-model-path", default=PITCH_DETECTION_MODEL_PATH)
    parser.add_argument("--camera-calibration-path")
    parser.add_argument("--camera-rig-profile-path")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pitch-detection-interval", type=int, default=5)
    parser.add_argument("--sample-interval-frames", type=int, default=250)
    arguments = parser.parse_args()
    if arguments.duration_minutes <= 0:
        parser.error("--duration-minutes must be positive")
    if arguments.max_frames is not None and arguments.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if arguments.warmup_frames < 0:
        parser.error("--warmup-frames cannot be negative")
    report = run_benchmark(
        source=arguments.source,
        device=arguments.device,
        player_model_path=arguments.player_model_path,
        pitch_model_path=arguments.pitch_model_path,
        camera_calibration_path=arguments.camera_calibration_path,
        camera_rig_profile_path=arguments.camera_rig_profile_path,
        duration_minutes=arguments.duration_minutes,
        max_frames=arguments.max_frames,
        warmup_frames=arguments.warmup_frames,
        imgsz=arguments.imgsz,
        pitch_detection_interval=arguments.pitch_detection_interval,
        sample_interval_frames=arguments.sample_interval_frames,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
