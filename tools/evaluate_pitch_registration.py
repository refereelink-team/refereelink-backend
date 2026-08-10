#!/usr/bin/env python3
"""Evaluate image-to-pitch predictions against an explicit JSON ground truth.

The format intentionally stores neither pickles nor executable objects.  A
minimal input is::

    {
      "version": 1,
      "image_size": [1280, 720],
      "frames": [{
        "frame_index": 0,
        "ground_truth_image_to_pitch": [[...], [...], [...]],
        "predicted_image_to_pitch": [[...], [...], [...]],
        "correspondences": [
          {"image_xy": [640, 360], "pitch_xy_m": [52.5, 34.0]}
        ],
        "status": "corrected",
        "latency_ms": 2.3
      }]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from app.field_registration.metrics import (
    ErrorSummary,
    correspondence_errors,
    grid_projection_errors,
    template_jitter_px,
)
from app.field_registration.pitch_model import PitchDimensions, PitchModel


def _matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix")
    return matrix


def evaluate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if int(payload.get("version", 0)) != 1:
        raise ValueError("unsupported evaluation format version")
    image_size = tuple(int(value) for value in payload["image_size"])
    if len(image_size) != 2 or min(image_size) <= 0:
        raise ValueError("image_size must contain positive width and height")
    pitch_payload = payload.get("pitch_dimensions_m", {})
    pitch_dimensions = PitchDimensions(
        length_m=float(pitch_payload.get("length", 105.0)),
        width_m=float(pitch_payload.get("width", 68.0)),
    )
    pitch_grid = PitchModel(pitch_dimensions).grid()

    image_errors: list[float] = []
    pitch_errors: list[float] = []
    grid_errors: list[float] = []
    latencies: list[float] = []
    predicted_pitch_to_image: list[np.ndarray] = []
    status_counts: dict[str, int] = {}
    total_frames = 0
    available_frames = 0
    annotated_frames = 0
    per_frame: list[dict[str, Any]] = []

    for frame in payload.get("frames", []):
        total_frames += 1
        frame_index = int(frame["frame_index"])
        status = str(frame.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if frame.get("latency_ms") is not None:
            latencies.append(float(frame["latency_ms"]))
        predicted_raw = frame.get("predicted_image_to_pitch")
        truth_raw = frame.get("ground_truth_image_to_pitch")
        frame_result: dict[str, Any] = {"frame_index": frame_index, "status": status}
        if predicted_raw is None:
            frame_result["available"] = False
            per_frame.append(frame_result)
            continue
        predicted = _matrix(predicted_raw, "predicted_image_to_pitch")
        available_frames += 1
        frame_result["available"] = True
        predicted_pitch_to_image.append(np.linalg.inv(predicted))
        if truth_raw is None:
            per_frame.append(frame_result)
            continue
        truth = _matrix(truth_raw, "ground_truth_image_to_pitch")
        annotated_frames += 1
        correspondence_payload = frame.get("correspondences", [])
        if correspondence_payload:
            image = np.asarray(
                [item["image_xy"] for item in correspondence_payload], dtype=np.float64
            )
            pitch = np.asarray(
                [item["pitch_xy_m"] for item in correspondence_payload], dtype=np.float64
            )
            frame_image_errors, frame_pitch_errors = correspondence_errors(
                image, pitch, predicted
            )
            image_errors.extend(frame_image_errors.tolist())
            pitch_errors.extend(frame_pitch_errors.tolist())
            frame_result["image_reprojection_px"] = ErrorSummary.from_values(
                frame_image_errors
            ).to_dict()
            frame_result["pitch_projection_m"] = ErrorSummary.from_values(
                frame_pitch_errors
            ).to_dict()
        frame_grid_errors, valid_ratio = grid_projection_errors(
            predicted, truth, image_size
        )
        grid_errors.extend(frame_grid_errors.tolist())
        frame_result["grid_projection_m"] = ErrorSummary.from_values(
            frame_grid_errors
        ).to_dict()
        frame_result["valid_grid_ratio"] = valid_ratio
        per_frame.append(frame_result)

    report = {
        "version": 1,
        "frame_count": total_frames,
        "annotated_frame_count": annotated_frames,
        "available_frame_count": available_frames,
        "availability_ratio": available_frames / total_frames if total_frames else 0.0,
        "status_counts": status_counts,
        "image_reprojection_px": ErrorSummary.from_values(image_errors).to_dict(),
        "pitch_projection_m": ErrorSummary.from_values(pitch_errors).to_dict(),
        "grid_projection_m": ErrorSummary.from_values(grid_errors).to_dict(),
        "latency_ms": ErrorSummary.from_values(latencies).to_dict(),
        "template_jitter_px": template_jitter_px(
            predicted_pitch_to_image, pitch_grid
        ).to_dict(),
        "frames": per_frame,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="ground-truth/prediction JSON")
    parser.add_argument("--output", type=Path, help="write report JSON")
    arguments = parser.parse_args()
    payload = json.loads(arguments.input.read_text(encoding="utf-8"))
    report = evaluate_payload(payload)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
