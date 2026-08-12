#!/usr/bin/env python3
"""Build a fixed-position horizontal-pan rig profile from annotated anchor frames."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from tools._bootstrap import ensure_repository_root
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from _bootstrap import ensure_repository_root

ensure_repository_root(__file__)

from app.field_registration.annotations import (
    correspondence_arrays,
    pitch_dimensions_from_payload,
    validate_annotation_payload,
)
from app.field_registration.lens import LensCalibration, LensModel, LensUndistorter
from app.field_registration.pitch_model import PitchModel
from app.field_registration.rig_calibration import PanAnchor, calibrate_fixed_pan_rig
from app.field_registration.types import PointObservation


def _anchors_from_manifest(
    path: Path,
    lens: LensCalibration,
    balance: float,
) -> tuple[list[PanAnchor], LensCalibration, tuple[float, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_annotation_payload(
        payload,
        root=path.parent,
        require_annotated_frames=False,
        require_contact_points=False,
    )
    image_size = tuple(int(value) for value in payload["image_size"])
    pitch_model = PitchModel(pitch_dimensions_from_payload(payload))
    undistorter = LensUndistorter(lens, balance=balance)
    rectified_matrix = undistorter.rectified_camera_matrix(image_size)
    rectified_lens = LensCalibration(
        lens_model=LensModel.PINHOLE,
        camera_matrix=rectified_matrix,
        distortion_coefficients=np.zeros(5),
        image_size=image_size,
        reprojection_error_px=lens.reprojection_error_px,
        validation_median_error_px=lens.validation_median_error_px,
        validation_p95_error_px=lens.validation_p95_error_px,
        valid_image_count=lens.valid_image_count,
    )
    anchors: list[PanAnchor] = []
    for frame in payload["frames"]:
        image_points, pitch_points = correspondence_arrays(frame, pitch_model)
        if image_points.shape[0] < 4:
            continue
        rectified_points = undistorter.rectify_points(image_points, image_size)
        observations = tuple(
            PointObservation(
                label=str(item.get("label", index)),
                image_xy=(float(image_xy[0]), float(image_xy[1])),
                pitch_xy_m=(float(pitch_xy[0]), float(pitch_xy[1])),
                confidence=1.0,
                sigma_px=float(item.get("sigma_px", 1.0)),
                source="manual_rig_anchor",
            )
            for index, (item, image_xy, pitch_xy) in enumerate(
                zip(frame["correspondences"], rectified_points, pitch_points)
            )
        )
        encoder_pan = frame.get("encoder_pan_rad")
        anchors.append(
            PanAnchor(
                anchor_id=f"{path.stem}:frame-{int(frame['frame_index'])}",
                observations=observations,
                encoder_pan_rad=(float(encoder_pan) if encoder_pan is not None else None),
            )
        )
    dimensions = pitch_model.dimensions
    return anchors, rectified_lens, (dimensions.length_m, dimensions.width_m)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--lens", type=Path, required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--balance", type=float, default=0.0)
    parser.add_argument("--pan-min-deg", type=float, default=-90.0)
    parser.add_argument("--pan-max-deg", type=float, default=90.0)
    parser.add_argument("--max-centre-spread-m", type=float, default=0.5)
    parser.add_argument("--max-anchor-error-px", type=float, default=3.0)
    arguments = parser.parse_args()

    lens = LensCalibration.load(arguments.lens)
    anchors: list[PanAnchor] = []
    rectified_lens: LensCalibration | None = None
    pitch_size_m: tuple[float, float] | None = None
    for annotation_path in arguments.annotations:
        manifest_anchors, manifest_lens, manifest_pitch_size = _anchors_from_manifest(
            annotation_path,
            lens,
            arguments.balance,
        )
        if rectified_lens is not None and manifest_lens.image_size != rectified_lens.image_size:
            raise ValueError("all rig anchor manifests must use the same image size")
        if pitch_size_m is not None and not np.allclose(manifest_pitch_size, pitch_size_m):
            raise ValueError("all rig anchor manifests must use the same pitch dimensions")
        rectified_lens = manifest_lens
        pitch_size_m = manifest_pitch_size
        anchors.extend(manifest_anchors)
    if rectified_lens is None or pitch_size_m is None or len(anchors) < 2:
        raise ValueError("at least two annotated pan anchor frames are required")

    result = calibrate_fixed_pan_rig(
        arguments.camera_id,
        rectified_lens,
        anchors,
        pitch_size_m,
        pan_limits_rad=(
            float(np.deg2rad(arguments.pan_min_deg)),
            float(np.deg2rad(arguments.pan_max_deg)),
        ),
    )
    maximum_error = max(result.anchor_reprojection_error_px.values())
    issues: list[str] = []
    if result.camera_centre_spread_m > arguments.max_centre_spread_m:
        issues.append("camera_centre_spread_exceeds_limit")
    if maximum_error > arguments.max_anchor_error_px:
        issues.append("anchor_reprojection_error_exceeds_limit")
    report = {
        "version": 1,
        "camera_id": arguments.camera_id,
        "anchor_count": len(anchors),
        "coordinate_system": "rectified_pinhole_pixels_to_metric_pitch",
        "source_lens_model": lens.lens_model.value,
        "rectified_image_size": list(rectified_lens.image_size),
        "camera_centre_spread_m": result.camera_centre_spread_m,
        "anchor_pan_rad": result.anchor_pan_rad,
        "anchor_reprojection_error_px": result.anchor_reprojection_error_px,
        "maximum_anchor_reprojection_error_px": maximum_error,
        "quality_limits": {
            "max_centre_spread_m": arguments.max_centre_spread_m,
            "max_anchor_error_px": arguments.max_anchor_error_px,
        },
        "issues": issues,
        "accepted": not issues,
    }
    report_path = arguments.report or arguments.output.with_suffix(".report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if issues:
        raise RuntimeError(
            "rig calibration failed quality gates: " + ", ".join(issues)
        )
    result.profile.save(arguments.output)
    print(f"Saved {arguments.output}")
    print(f"Saved {report_path}")


if __name__ == "__main__":
    main()
