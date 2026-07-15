#!/usr/bin/env python3
"""Reproducible first-stage performance benchmark.

The script deliberately reports missing checkpoints as rows instead of
silently substituting the legacy role-aware model.  This keeps comparisons
between YOLOv11n and YOLOv11s honest when weights are installed later.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import supervision as sv

from app.constants.paths import (
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
    YOLO11N_PLAYER_MODEL_PATH,
    YOLO11S_PLAYER_MODEL_PATH,
)
from app.vision.core import VisionCore


def _run_case(
    source_video_path: str,
    device: str,
    player_model_path: str,
    pitch_detection_interval: int,
    imgsz: int,
    camera_calibration_path: str | None,
    enable_undistortion: bool,
    max_frames: int,
) -> dict:
    case = {
        "player_model": Path(player_model_path).name,
        "imgsz": imgsz,
        "pitch_detection_interval": pitch_detection_interval,
        "undistortion": enable_undistortion,
        "status": "ok",
    }
    try:
        process = None
        try:
            import psutil
            process = psutil.Process(os.getpid())
        except Exception:
            pass
        video_info = sv.VideoInfo.from_video_path(source_video_path)
        core = VisionCore(
            device=device,
            fps=video_info.fps,
            player_model_path=player_model_path,
            pitch_model_path=PITCH_DETECTION_MODEL_PATH,
            camera_calibration_path=camera_calibration_path,
            enable_undistortion=enable_undistortion,
            pitch_detection_interval=pitch_detection_interval,
            imgsz=imgsz,
        )
        core.load_models()
        start = time.perf_counter()
        frame_count = 0
        for frame_count, frame in enumerate(
            sv.get_video_frames_generator(source_path=source_video_path),
            start=1,
        ):
            core.process(frame, frame_count)
            if frame_count >= max_frames:
                break
        elapsed = time.perf_counter() - start
        case.update(
            {
                "frames": frame_count,
                "elapsed_sec": round(elapsed, 3),
                "end_to_end_fps": round(frame_count / max(elapsed, 1e-6), 2),
                "player_inference_latency_ms": round(
                    core.player_inference_time_ms / max(core.player_inference_count, 1), 2
                ),
                "pitch_inference_latency_ms": round(
                    core.pitch_inference_time_ms / max(core.pitch_detection_count, 1), 2
                ),
                "pitch_detection_count": core.pitch_detection_count,
                "homography_reuse_ratio": round(
                    core.pitch_reuse_count / max(core.frames_processed, 1), 3
                ),
                "homography_available_ratio": round(
                    core.homography_available_count / max(core.frames_processed, 1), 3
                ),
                "track_id_interruptions": core.track_id_interruptions,
                "memory_mb": round(
                    process.memory_info().rss / (1024 * 1024), 1
                ) if process is not None else None,
            }
        )
        try:
            import torch
            if device.startswith("cuda") and torch.cuda.is_available():
                case["gpu_memory_mb"] = round(
                    torch.cuda.memory_allocated() / (1024 * 1024), 1
                )
            else:
                case["gpu_memory_mb"] = None
        except Exception:
            case["gpu_memory_mb"] = None
    except FileNotFoundError as error:
        case.update({"status": "missing_file", "error": str(error)})
    except Exception as error:  # pragma: no cover - hardware/model dependent
        case.update({"status": "error", "error": f"{type(error).__name__}: {error}"})
    return case


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark phase-one vision configurations")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/phase1_benchmark.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--camera-calibration-path", default=CAMERA_CALIBRATION_PATH)
    parser.add_argument("--max-frames", type=int, default=300)
    args = parser.parse_args()

    cases = []
    for model_path in (YOLO11N_PLAYER_MODEL_PATH, YOLO11S_PLAYER_MODEL_PATH):
        for imgsz in (640, 960):
            for interval in (1, 5, 10):
                cases.append(
                    _run_case(
                        source_video_path=args.source,
                        device=args.device,
                        player_model_path=model_path,
                        pitch_detection_interval=interval,
                        imgsz=imgsz,
                        camera_calibration_path=args.camera_calibration_path,
                        enable_undistortion=True,
                        max_frames=args.max_frames,
                    )
                )

    # One explicit ablation row isolates the rectification cost using the
    # default recommended model/configuration.
    cases.append(
        _run_case(
            source_video_path=args.source,
            device=args.device,
            player_model_path=YOLO11S_PLAYER_MODEL_PATH,
            pitch_detection_interval=5,
            imgsz=640,
            camera_calibration_path=args.camera_calibration_path,
            enable_undistortion=False,
            max_frames=args.max_frames,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(f"Wrote {len(cases)} benchmark rows to {args.output}")


if __name__ == "__main__":
    main()
