#!/usr/bin/env python3
"""Evaluate persisted offline registration against held-out frame annotations."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools._bootstrap import ensure_repository_root
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from _bootstrap import ensure_repository_root

ensure_repository_root(__file__)

from app.field_registration.annotations import (
    correspondence_arrays,
    fit_annotation_homography,
    pitch_dimensions_from_payload,
    validate_annotation_payload,
)
from app.field_registration.metrics import (
    ErrorSummary,
    correspondence_errors,
    grid_projection_errors,
)
from app.field_registration.offline import load_offline_smoothing_result
from app.field_registration.pitch_model import PitchModel


SAFE = "safe"
PREVIEW = "preview"


def _finite_homography(value: np.ndarray) -> bool:
    return value.shape == (3, 3) and bool(np.all(np.isfinite(value)))


def _summary(values: list[float]) -> dict[str, float | int | None]:
    return ErrorSummary.from_values(values).to_dict()


def evaluate_offline_registration(
    annotation_path: str | Path,
    registration_path: str | Path,
    *,
    maximum_safe_median_720p_px: float = 4.0,
    maximum_safe_p95_720p_px: float = 8.0,
    maximum_safe_grid_median_m: float = 0.75,
    maximum_safe_grid_p95_m: float = 1.5,
) -> dict[str, Any]:
    annotation_file = Path(annotation_path)
    annotations = json.loads(annotation_file.read_text(encoding="utf-8"))
    validation = validate_annotation_payload(
        annotations,
        root=annotation_file.parent,
        require_annotated_frames=True,
        require_contact_points=False,
    )
    if not validation.valid:
        raise ValueError(
            "annotation validation failed: " + ", ".join(validation.issues)
        )
    metadata, arrays = load_offline_smoothing_result(registration_path)
    frame_indices = arrays["frame_indices"].astype(np.int64)
    if len(frame_indices) != len(set(frame_indices.tolist())):
        raise ValueError("offline registration contains duplicate frame indices")
    positions = {int(value): index for index, value in enumerate(frame_indices)}
    tiers = list(metadata["measurement_tiers"])
    statuses = list(metadata["statuses"])
    if len(tiers) != len(frame_indices) or len(statuses) != len(frame_indices):
        raise ValueError("offline registration status arrays have the wrong length")

    image_size = tuple(int(value) for value in annotations["image_size"])
    scale_720p = 720.0 / image_size[1]
    pitch_model = PitchModel(pitch_dimensions_from_payload(annotations))
    projectable_count = 0
    safe_count = 0
    display_count = 0
    false_safe_count = 0
    landmark_errors_720p: list[float] = []
    safe_landmark_errors_720p: list[float] = []
    grid_errors_m: list[float] = []
    safe_grid_errors_m: list[float] = []
    per_frame: list[dict[str, Any]] = []
    boundary_frames: list[int] = []

    for frame in annotations["frames"]:
        frame_index = int(frame["frame_index"])
        image_points, pitch_points = correspondence_arrays(frame, pitch_model)
        explicitly_projectable = frame.get("projectable")
        projectable = (
            bool(explicitly_projectable)
            if explicitly_projectable is not None
            else image_points.shape[0] >= 4
        )
        if frame.get("shot_boundary") is True:
            boundary_frames.append(frame_index)
        row: dict[str, Any] = {
            "frame_index": frame_index,
            "projectable": projectable,
            "annotation_correspondence_count": int(image_points.shape[0]),
        }
        if projectable:
            projectable_count += 1
        position = positions.get(frame_index)
        if position is None:
            row.update(
                {
                    "available": False,
                    "measurement_tier": "unavailable",
                    "reason": "prediction_frame_missing",
                }
            )
            per_frame.append(row)
            continue
        tier = str(tiers[position])
        status = str(statuses[position])
        matrix = np.asarray(arrays["pitch_to_image"][position], dtype=np.float64)
        available = _finite_homography(matrix)
        row.update(
            {
                "available": available,
                "measurement_tier": tier,
                "status": status,
            }
        )
        if projectable and tier == SAFE and available:
            safe_count += 1
        if projectable and tier in {SAFE, PREVIEW} and available:
            display_count += 1
        truth_image_to_pitch, truth_residual = fit_annotation_homography(
            image_points, pitch_points
        )
        if not projectable or truth_image_to_pitch is None or not available:
            if projectable and truth_image_to_pitch is None:
                row["reason"] = "ground_truth_homography_unavailable"
            per_frame.append(row)
            continue
        predicted_image_to_pitch = np.linalg.inv(matrix)
        image_error, _ = correspondence_errors(
            image_points,
            pitch_points,
            predicted_image_to_pitch,
        )
        image_error_720p = image_error * scale_720p
        frame_grid, valid_grid_ratio = grid_projection_errors(
            predicted_image_to_pitch,
            truth_image_to_pitch,
            image_size,
        )
        landmark_errors_720p.extend(image_error_720p.tolist())
        grid_errors_m.extend(frame_grid.tolist())
        landmark_summary = ErrorSummary.from_values(image_error_720p)
        grid_summary = ErrorSummary.from_values(frame_grid)
        row.update(
            {
                "ground_truth_fit_median_px": truth_residual,
                "landmark_reprojection_720p_px": landmark_summary.to_dict(),
                "grid_projection_m": grid_summary.to_dict(),
                "valid_grid_ratio": valid_grid_ratio,
            }
        )
        if tier == SAFE:
            safe_landmark_errors_720p.extend(image_error_720p.tolist())
            safe_grid_errors_m.extend(frame_grid.tolist())
            unsafe_error = (
                landmark_summary.median is None
                or landmark_summary.p95 is None
                or grid_summary.median is None
                or grid_summary.p95 is None
                or landmark_summary.median > maximum_safe_median_720p_px
                or landmark_summary.p95 > maximum_safe_p95_720p_px
                or grid_summary.median > maximum_safe_grid_median_m
                or grid_summary.p95 > maximum_safe_grid_p95_m
            )
            row["false_safe"] = unsafe_error
            false_safe_count += int(unsafe_error)
        per_frame.append(row)

    recovery_frames: list[int] = []
    for boundary in boundary_frames:
        recovery = None
        for frame_index in frame_indices[frame_indices >= boundary]:
            position = positions[int(frame_index)]
            if tiers[position] == SAFE and _finite_homography(
                np.asarray(arrays["pitch_to_image"][position])
            ):
                recovery = int(frame_index) - boundary
                break
        if recovery is not None:
            recovery_frames.append(recovery)

    split = validation.split
    accuracy_valid = (
        split in {"validation", "test"}
        and projectable_count > 0
        and len(safe_landmark_errors_720p) > 0
    )
    return {
        "format_version": 1,
        "accuracy_valid": accuracy_valid,
        "split": split,
        "annotation_path": str(annotation_file),
        "registration_path": str(registration_path),
        "annotated_frame_count": validation.annotated_frame_count,
        "projectable_frame_count": projectable_count,
        "safe_frame_count": safe_count,
        "safe_plus_preview_frame_count": display_count,
        "safe_coverage": safe_count / max(projectable_count, 1),
        "safe_plus_preview_coverage": display_count / max(projectable_count, 1),
        "false_safe_count": false_safe_count,
        "landmark_reprojection_720p_px": _summary(landmark_errors_720p),
        "safe_landmark_reprojection_720p_px": _summary(
            safe_landmark_errors_720p
        ),
        "grid_projection_m": _summary(grid_errors_m),
        "safe_grid_projection_m": _summary(safe_grid_errors_m),
        "hard_cut_count": len(boundary_frames),
        "hard_cut_recovery_frames": _summary(recovery_frames),
        "thresholds": {
            "safe_landmark_median_720p_px": maximum_safe_median_720p_px,
            "safe_landmark_p95_720p_px": maximum_safe_p95_720p_px,
            "safe_grid_median_m": maximum_safe_grid_median_m,
            "safe_grid_p95_m": maximum_safe_grid_p95_m,
        },
        "frames": per_frame,
        "notes": (
            []
            if accuracy_valid
            else [
                "Final accuracy requires held-out validation/test annotations; "
                "training or calibration splits are diagnostic only."
            ]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    report = evaluate_offline_registration(
        arguments.annotations, arguments.registration
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
