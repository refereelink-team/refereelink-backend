#!/usr/bin/env python3
"""Long-run CUDA benchmark for the integrated field-registration V2 pipeline."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time

import cv2
import numpy as np
import psutil
import torch

try:
    from tools._bootstrap import ensure_repository_root
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from _bootstrap import ensure_repository_root

ensure_repository_root(__file__)

from app.constants.paths import PITCH_DETECTION_MODEL_PATH, PLAYER_DETECTION_MODEL_PATH
from app.vision.core import VisionCore


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model_benchmark_evidence(path: Path) -> dict[str, object]:
    """Load one successful model-only CUDA benchmark for report composition."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("model benchmark must contain exactly one result")
        payload = payload[0]
    if not isinstance(payload, dict) or payload.get("status") != "ok":
        raise ValueError("model benchmark must contain one successful result")
    latency = payload.get("latency_ms")
    p95 = latency.get("p95") if isinstance(latency, dict) else None
    memory = payload.get("peak_gpu_memory_mb")
    weights = payload.get("weights")
    checkpoint = weights.get("checkpoint") if isinstance(weights, dict) else None
    checksum = weights.get("sha256") if isinstance(weights, dict) else None
    if p95 is None or not np.isfinite(float(p95)) or float(p95) <= 0.0:
        raise ValueError("model benchmark is missing a positive P95 latency")
    if memory is None or not np.isfinite(float(memory)) or float(memory) < 0.0:
        raise ValueError("model benchmark is missing GPU memory evidence")
    if not checkpoint or not checksum:
        raise ValueError("model benchmark must identify its checkpoint and SHA-256")
    return {
        "path": str(path),
        "architecture": payload.get("architecture"),
        "device": payload.get("device"),
        "precision": payload.get("precision"),
        "weights": weights,
        "latency_ms": latency,
        "gpu_memory_mb": float(memory),
    }


def validate_model_benchmark_checkpoint(
    evidence: dict[str, object], checkpoint: Path
) -> None:
    """Prevent performance evidence from being attached to another model."""

    weights = evidence.get("weights")
    if not isinstance(weights, dict):
        raise ValueError("model benchmark weights are unavailable")
    evidence_path = Path(str(weights.get("checkpoint", ""))).expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    if evidence_path != checkpoint:
        raise ValueError(
            "model benchmark checkpoint does not match "
            "--pitch-perception-checkpoint"
        )
    expected = str(weights.get("sha256", "")).lower()
    if _sha256(checkpoint) != expected:
        raise ValueError("model benchmark checkpoint SHA-256 does not match the file")


class BoundedReservoir:
    """Fixed-memory deterministic sample with an exact observed maximum."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("reservoir capacity must be positive")
        self.capacity = int(capacity)
        self._rng = np.random.default_rng(seed)
        self._values: list[float] = []
        self.count = 0
        self.maximum: float | None = None

    def add(self, value: float) -> None:
        if not np.isfinite(value):
            return
        numeric = float(value)
        self.count += 1
        self.maximum = numeric if self.maximum is None else max(self.maximum, numeric)
        if len(self._values) < self.capacity:
            self._values.append(numeric)
            return
        replacement = int(self._rng.integers(0, self.count))
        if replacement < self.capacity:
            self._values[replacement] = numeric

    def summary(self) -> dict[str, float | int | None]:
        if not self._values:
            return {
                "count": 0,
                "sample_count": 0,
                "median": None,
                "p95": None,
                "maximum": None,
            }
        array = np.asarray(self._values, dtype=np.float64)
        return {
            "count": self.count,
            "sample_count": len(self._values),
            "median": float(np.median(array)),
            "p95": float(np.percentile(array, 95)),
            "maximum": self.maximum,
        }


def _memory_slope_mb_per_minute(samples: list[tuple[float, float]]) -> float | None:
    # Short smoke runs are dominated by allocator warm-up and produce wildly
    # misleading slopes. Five minutes is the minimum useful observation span.
    if len(samples) < 2 or samples[-1][0] - samples[0][0] < 300.0:
        return None
    elapsed_minutes = np.asarray([row[0] / 60.0 for row in samples])
    rss_mb = np.asarray([row[1] for row in samples])
    return float(np.polyfit(elapsed_minutes, rss_mb, 1)[0])


def _counter_ratio(current: int, baseline: int, measured_frames: int) -> float:
    return max(int(current) - int(baseline), 0) / max(int(measured_frames), 1)


@torch.inference_mode()
def run_benchmark(
    *,
    source: Path,
    device: str,
    player_model_path: str,
    pitch_model_path: str,
    pitch_perception_checkpoint_path: str | None,
    camera_calibration_path: str | None,
    camera_rig_profile_path: str | None,
    duration_minutes: float,
    max_frames: int | None,
    warmup_frames: int,
    imgsz: int,
    pitch_detection_interval: int,
    sample_interval_frames: int,
    baseline_fps: float | None = None,
    model_benchmark: dict[str, object] | None = None,
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
        pitch_perception_checkpoint_path=pitch_perception_checkpoint_path,
    )
    core.load_models()
    process = psutil.Process(os.getpid())
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    frame_latencies_ms = BoundedReservoir(capacity=10_000, seed=7)
    coordinate_sigmas_m = BoundedReservoir(capacity=50_000, seed=11)
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
    homography_available_start = core.homography_available_count
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
            frame_latencies_ms.add((time.perf_counter() - frame_started) * 1000.0)
            camera_statuses[result.homography_status] += 1
            coordinate_count += len(result.pitch_coordinates)
            for coordinate in result.pitch_coordinates:
                if coordinate.xy_m is not None:
                    usable_coordinate_count += 1
                if coordinate.sigma_m is not None and np.isfinite(coordinate.sigma_m):
                    coordinate_sigmas_m.add(float(coordinate.sigma_m))
            if frame_count == 1 or frame_count % max(sample_interval_frames, 1) == 0:
                elapsed = time.perf_counter() - started
                rss_mb = process.memory_info().rss / (1024 * 1024)
                memory_samples.append((elapsed, rss_mb))
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    if not memory_samples or memory_samples[-1][0] < elapsed:
        memory_samples.append((elapsed, process.memory_info().rss / (1024 * 1024)))
    report = {
        "status": "ok",
        "accuracy_valid": False,
        "source": str(source),
        "device": device,
        "camera_rig_profile": camera_rig_profile_path,
        "physical_pan_constraint_active": camera_rig_profile_path is not None,
        "source_fps": source_fps,
        "warmup_frames": warmup_frames,
        "frames": frame_count,
        "video_loops": video_loops,
        "elapsed_sec": elapsed,
        "end_to_end_fps": frame_count / max(elapsed, 1e-9),
        "baseline_fps": baseline_fps,
        "frame_latency_ms": frame_latencies_ms.summary(),
        "camera_status_counts": dict(camera_statuses),
        "homography_available_ratio": _counter_ratio(
            core.homography_available_count,
            homography_available_start,
            frame_count,
        ),
        "semantic_inference_count": core.pitch_detection_count - semantic_start,
        "semantic_reuse_ratio": _counter_ratio(
            core.pitch_reuse_count,
            reuse_start,
            frame_count,
        ),
        "coordinate_usable_ratio": usable_coordinate_count / max(coordinate_count, 1),
        "coordinate_sigma_m": coordinate_sigmas_m.summary(),
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
        "model_latency_ms": (
            model_benchmark.get("latency_ms")
            if model_benchmark is not None
            else None
        ),
        "model_gpu_memory_mb": (
            model_benchmark.get("gpu_memory_mb")
            if model_benchmark is not None
            else None
        ),
        "model_benchmark": model_benchmark,
        "model_weights": {
            "player": player_model_path,
            "pitch": pitch_model_path,
            "field_perception": (
                pitch_perception_checkpoint_path or "legacy_keypoint_adapter"
            ),
        },
        "notes": [
            "This is a stability/performance run, not an accuracy evaluation.",
            (
                "The V2 tracker uses the configured dual-head checkpoint."
                if pitch_perception_checkpoint_path is not None
                else "The V2 tracker uses the legacy keypoint compatibility model."
            ),
            (
                "A calibrated physical pan constraint is active."
                if camera_rig_profile_path is not None
                else "No rig profile was supplied; generic homography relocalization is not production geometry."
            ),
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--player-model-path", default=PLAYER_DETECTION_MODEL_PATH)
    parser.add_argument("--pitch-model-path", default=PITCH_DETECTION_MODEL_PATH)
    parser.add_argument("--pitch-perception-checkpoint", type=str)
    parser.add_argument("--camera-calibration-path")
    parser.add_argument("--camera-rig-profile-path")
    parser.add_argument("--duration-minutes", type=float, default=30.0)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--pitch-detection-interval", type=int, default=5)
    parser.add_argument("--sample-interval-frames", type=int, default=250)
    parser.add_argument(
        "--baseline-fps",
        type=float,
        help="Full-pipeline FPS measured with the current production baseline",
    )
    parser.add_argument(
        "--model-benchmark-report",
        type=Path,
        help="Single-result JSON from benchmark_pitch_perception.py",
    )
    arguments = parser.parse_args()
    if arguments.duration_minutes <= 0:
        parser.error("--duration-minutes must be positive")
    if arguments.max_frames is not None and arguments.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if arguments.warmup_frames < 0:
        parser.error("--warmup-frames cannot be negative")
    if arguments.baseline_fps is not None and arguments.baseline_fps <= 0.0:
        parser.error("--baseline-fps must be positive")
    try:
        model_benchmark = (
            load_model_benchmark_evidence(arguments.model_benchmark_report)
            if arguments.model_benchmark_report is not None
            else None
        )
        if model_benchmark is not None:
            if arguments.pitch_perception_checkpoint is None:
                raise ValueError(
                    "--model-benchmark-report requires "
                    "--pitch-perception-checkpoint"
                )
            validate_model_benchmark_checkpoint(
                model_benchmark,
                Path(arguments.pitch_perception_checkpoint),
            )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    report = run_benchmark(
        source=arguments.source,
        device=arguments.device,
        player_model_path=arguments.player_model_path,
        pitch_model_path=arguments.pitch_model_path,
        pitch_perception_checkpoint_path=arguments.pitch_perception_checkpoint,
        camera_calibration_path=arguments.camera_calibration_path,
        camera_rig_profile_path=arguments.camera_rig_profile_path,
        duration_minutes=arguments.duration_minutes,
        max_frames=arguments.max_frames,
        warmup_frames=arguments.warmup_frames,
        imgsz=arguments.imgsz,
        pitch_detection_interval=arguments.pitch_detection_interval,
        sample_interval_frames=arguments.sample_interval_frames,
        baseline_fps=arguments.baseline_fps,
        model_benchmark=model_benchmark,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
