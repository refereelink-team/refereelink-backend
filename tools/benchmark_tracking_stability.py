#!/usr/bin/env python3
"""Compare the current ByteTrack path with low-risk stability profiles.

This benchmark intentionally does not claim IDF1/HOTA without ground-truth
tracks.  It reports the lifecycle counters that can be collected directly
from the production VisionCore and is suitable for running on test1/test2 on
the CUDA host.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import supervision as sv

from app.constants.paths import (
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_DETECTION_MODEL_PATH,
)
from app.vision.core import VisionCore
from app.vision.entities import TrackEntityManager


CASES = {
    "E0_current": {
        "player_confidence": 0.25,
        "player_iou": 0.7,
        "track_lost_buffer": 45,
        "max_prediction_gap_frames": 0,
        "reactivation_window_frames": 0,
    },
    "E1_bytetrack_tuned": {
        "player_confidence": 0.20,
        "player_iou": 0.8,
        "track_lost_buffer": 60,
        "max_prediction_gap_frames": 0,
        "reactivation_window_frames": 0,
    },
    "E2_short_prediction_entity": {
        "player_confidence": 0.20,
        "player_iou": 0.8,
        "track_lost_buffer": 60,
        "max_prediction_gap_frames": 6,
        "reactivation_window_frames": 12,
    },
}


def run_case(
    *,
    source: str,
    device: str,
    player_model_path: str,
    pitch_model_path: str,
    calibration_path: str | None,
    imgsz: int,
    max_frames: int,
    name: str,
    settings: dict[str, int | float],
) -> dict[str, object]:
    result: dict[str, object] = {
        "case": name,
        "source": str(source),
        "device": device,
        "imgsz": imgsz,
        "settings": settings,
        "ground_truth_metrics": False,
        "status": "ok",
    }
    try:
        video_info = sv.VideoInfo.from_video_path(source)
        entity_manager = TrackEntityManager(
            max_prediction_gap_frames=int(settings["max_prediction_gap_frames"]),
            reactivation_window_frames=int(settings["reactivation_window_frames"]),
        )
        core = VisionCore(
            device=device,
            fps=video_info.fps,
            player_model_path=player_model_path,
            pitch_model_path=pitch_model_path,
            camera_calibration_path=calibration_path,
            pitch_detection_interval=5,
            imgsz=imgsz,
            player_confidence=float(settings["player_confidence"]),
            player_iou=float(settings["player_iou"]),
            track_lost_buffer=int(settings["track_lost_buffer"]),
            max_prediction_gap_frames=int(settings["max_prediction_gap_frames"]),
            reactivation_window_frames=int(settings["reactivation_window_frames"]),
            entity_manager=entity_manager,
        )
        core.load_models()
        start = time.perf_counter()
        frames = 0
        for frames, frame in enumerate(
            sv.get_video_frames_generator(source_path=source),
            start=1,
        ):
            core.process(frame, frames)
            if max_frames > 0 and frames >= max_frames:
                break
        elapsed = time.perf_counter() - start
        result.update(
            {
                "frames": frames,
                "elapsed_sec": round(elapsed, 3),
                "end_to_end_fps": round(frames / max(elapsed, 1e-6), 2),
                "player_inference_latency_ms": round(
                    core.player_inference_time_ms / max(core.player_inference_count, 1),
                    2,
                ),
                "track_id_interruptions": core.track_id_interruptions,
                "track_occlusion_events": core.track_occlusion_events,
                "track_predicted_frames": core.track_predicted_frames,
                "track_recovered_count": core.track_recovered_count,
                "track_reactivated_count": core.track_reactivated_count,
                "track_id_switches": core.track_id_switches,
                "track_fragmentations": core.track_fragmentations,
                "track_max_missing_frames": core.track_max_missing_frames,
                "track_entity_rebinds": core.track_entity_rebinds,
                "track_entity_fragmentations": core.track_entity_fragmentations,
                "track_lifecycle_counts": dict(core.track_lifecycle_counts),
                "pitch_detection_count": core.pitch_detection_count,
                "homography_available_ratio": round(
                    core.homography_available_count / max(core.frames_processed, 1),
                    3,
                ),
            }
        )
    except Exception as exc:  # pragma: no cover - model/hardware dependent
        result.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/tracking_stability.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--player-model-path", default=PLAYER_DETECTION_MODEL_PATH)
    parser.add_argument("--pitch-model-path", default=PITCH_DETECTION_MODEL_PATH)
    parser.add_argument("--camera-calibration-path", default=CAMERA_CALIBRATION_PATH)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args()

    rows = [
        run_case(
            source=args.source,
            device=args.device,
            player_model_path=args.player_model_path,
            pitch_model_path=args.pitch_model_path,
            calibration_path=args.camera_calibration_path,
            imgsz=args.imgsz,
            max_frames=args.max_frames,
            name=name,
            settings=settings,
        )
        for name, settings in CASES.items()
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
