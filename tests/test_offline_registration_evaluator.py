from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from app.field_registration.broadcast_camera import BroadcastCameraParameters
from app.field_registration.offline import (
    OfflineSmoothingResult,
    save_offline_smoothing_result,
)
from app.field_registration.pitch_model import PitchModel
from app.field_registration.types import (
    CameraState,
    CameraTrackingStatus,
    MeasurementTier,
    RegistrationMode,
)
from tools.check_pitch_registration_accuracy import check_accuracy_report


def _load_evaluator():
    path = Path(__file__).parents[1] / "tools" / "evaluate_offline_pitch_registration.py"
    spec = importlib.util.spec_from_file_location("offline_registration_evaluator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _camera() -> BroadcastCameraParameters:
    return BroadcastCameraParameters(
        np.asarray([-18.0, 30.0, 24.0]),
        0.05,
        0.31,
        -0.02,
        float(np.log(1120.0)),
        (1280, 720),
    )


def _state(camera: BroadcastCameraParameters, tier: MeasurementTier) -> CameraState:
    covariance = np.diag([1e-4] * 7)
    return CameraState(
        status=CameraTrackingStatus.CORRECTED,
        pan_rad=camera.pan_rad,
        pan_velocity_rad_s=0.0,
        covariance=covariance[np.ix_([0, 4], [0, 4])],
        image_to_pitch=camera.image_to_pitch_homography(),
        pitch_to_image=camera.pitch_to_image_homography(),
        confidence=0.95,
        registration_mode=RegistrationMode.BROADCAST,
        measurement_tier=tier,
        camera_model="broadcast_tripod_pan_tilt_zoom_offline_rts",
        focal_px=camera.focal_px,
        tilt_rad=camera.tilt_rad,
        roll_rad=camera.roll_rad,
        camera_center_xyz_m=camera.camera_center_xyz_m,
        camera_parameter_covariance=covariance,
    )


def _annotation(tmp_path: Path, split: str) -> Path:
    camera = _camera()
    pitch = PitchModel()
    metric = pitch.grid(6, 5)
    labels = tuple(f"grid-{index}" for index in range(len(metric)))
    image = cv2.perspectiveTransform(
        metric.reshape(-1, 1, 2).astype(np.float64),
        camera.pitch_to_image_homography(),
    ).reshape(-1, 2)
    frame_path = tmp_path / "frame.jpg"
    cv2.imwrite(str(frame_path), np.zeros((720, 1280, 3), dtype=np.uint8))
    payload = {
        "version": 1,
        "split": split,
        "source": {"name": f"held-out-{split}"},
        "image_size": [1280, 720],
        "pitch_dimensions_m": {"length": 105.0, "width": 68.0},
        "frames": [
            {
                "frame_index": 0,
                "image_path": frame_path.name,
                "projectable": True,
                "shot_boundary": True,
                "correspondences": [
                    {
                        "label": label,
                        "image_xy": image_xy.tolist(),
                        "pitch_xy_m": pitch_xy.tolist(),
                    }
                    for label, image_xy, pitch_xy in zip(labels, image, metric)
                ],
                "contact_points": [],
            }
        ],
    }
    path = tmp_path / f"{split}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _registration(tmp_path: Path) -> Path:
    camera = _camera()
    result = OfflineSmoothingResult(
        states=(_state(camera, MeasurementTier.SAFE),),
        frame_indices=(0,),
        smoothed_frame_count=1,
        preview_only_frame_count=0,
        shot_count=1,
    )
    path, _ = save_offline_smoothing_result(result, tmp_path / "registration")
    return path


def test_offline_evaluator_reports_exact_held_out_prediction(tmp_path) -> None:
    evaluator = _load_evaluator()

    report = evaluator.evaluate_offline_registration(
        _annotation(tmp_path, "test"), _registration(tmp_path)
    )

    assert report["accuracy_valid"] is True
    assert report["safe_coverage"] == 1.0
    assert report["safe_plus_preview_coverage"] == 1.0
    assert report["false_safe_count"] == 0
    assert report["safe_landmark_reprojection_720p_px"]["p95"] < 1e-5
    assert report["safe_grid_projection_m"]["p95"] < 0.005
    assert report["hard_cut_recovery_frames"]["p95"] == 0.0


def test_training_split_never_produces_valid_final_accuracy(tmp_path) -> None:
    evaluator = _load_evaluator()

    report = evaluator.evaluate_offline_registration(
        _annotation(tmp_path, "train"), _registration(tmp_path)
    )

    assert report["accuracy_valid"] is False
    assert report["notes"]


def test_accuracy_gate_requires_independent_coverage_and_zero_false_safe() -> None:
    valid = {
        "accuracy_valid": True,
        "projectable_frame_count": 50,
        "safe_coverage": 0.8,
        "safe_plus_preview_coverage": 0.98,
        "false_safe_count": 0,
        "safe_landmark_reprojection_720p_px": {"median": 3.0, "p95": 7.0},
        "safe_grid_projection_m": {"median": 0.5, "p95": 1.2},
        "hard_cut_count": 2,
        "hard_cut_recovery_frames": {"p95": 12.0},
    }

    assert check_accuracy_report(valid) == []

    invalid = dict(valid)
    invalid.update(
        {
            "accuracy_valid": False,
            "safe_coverage": 0.5,
            "safe_plus_preview_coverage": 0.7,
            "false_safe_count": 1,
        }
    )
    issues = check_accuracy_report(invalid)
    assert any("accuracy_valid" in issue for issue in issues)
    assert any("SAFE coverage" in issue for issue in issues)
    assert any("SAFE+PREVIEW" in issue for issue in issues)
    assert any("erroneous matrices" in issue for issue in issues)
