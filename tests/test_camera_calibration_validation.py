from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

from app.field_registration.camera_model import CameraRigProfile
from app.field_registration.geometry import transform_points
from app.field_registration.lens import LensCalibration, LensModel
from app.field_registration.pitch_model import PitchModel
from app.field_registration.rig_calibration import calibrate_fixed_pan_rig


def _load_calibrator():
    path = Path(__file__).parents[1] / "tools" / "calibrate_camera.py"
    spec = importlib.util.spec_from_file_location("camera_calibrator", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_rig_calibrator():
    path = Path(__file__).parents[1] / "tools" / "calibrate_camera_rig.py"
    spec = importlib.util.spec_from_file_location("rig_calibrator_tool", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _look_at_rotation(camera_xyz: np.ndarray, target_xyz: np.ndarray) -> np.ndarray:
    forward = target_xyz - camera_xyz
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.array([0.0, 0.0, 1.0]))
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.vstack((right, down, forward))


def _synthetic_observations(module):
    columns, rows = 9, 6
    objects = np.zeros((columns * rows, 3), dtype=np.float32)
    objects[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * 0.04
    camera_matrix = np.array(
        [[900.0, 0.0, 640.0], [0.0, 910.0, 360.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    distortion = np.array([-0.08, 0.015, 0.0005, -0.0003, 0.0])
    image_points = []
    object_points = []
    for index in range(8):
        rotation = np.array(
            [0.05 + 0.015 * index, -0.12 + 0.03 * index, 0.01 * index],
            dtype=np.float64,
        )
        translation = np.array(
            [-0.15 + 0.04 * index, -0.08 + 0.015 * index, 1.6 + 0.05 * index],
            dtype=np.float64,
        )
        projected, _ = cv2.projectPoints(
            objects,
            rotation,
            translation,
            camera_matrix,
            distortion,
        )
        noise = np.random.default_rng(index).normal(0.0, 0.08, projected.shape)
        object_points.append(objects.copy())
        image_points.append((projected + noise).astype(np.float32))
    return module.ChessboardObservations(
        object_points=tuple(object_points),
        image_points=tuple(image_points),
        image_size=(1280, 720),
        image_paths=tuple(Path(f"view-{index}.png") for index in range(8)),
    )


def test_pinhole_holdout_validation_uses_unseen_views() -> None:
    calibrator = _load_calibrator()
    observations = _synthetic_observations(calibrator)

    fit = calibrator.fit_camera_model(observations, LensModel.PINHOLE)
    validation = calibrator.cross_validate_camera_model(
        observations,
        LensModel.PINHOLE,
        maximum_folds=4,
    )

    assert fit.rms_px < 0.2
    assert validation.fold_count == 4
    assert validation.median_error_px < 0.2
    assert validation.p95_error_px < 0.35


def test_auto_selection_prefers_pinhole_for_near_tie() -> None:
    calibrator = _load_calibrator()
    pinhole = calibrator.ModelValidation(
        LensModel.PINHOLE,
        median_error_px=0.20,
        p95_error_px=0.40,
        maximum_error_px=0.8,
        fold_count=5,
        corner_count=200,
    )
    fisheye = calibrator.ModelValidation(
        LensModel.FISHEYE,
        median_error_px=0.18,
        p95_error_px=0.38,
        maximum_error_px=0.7,
        fold_count=5,
        corner_count=200,
    )

    selected = calibrator._select_candidate([fisheye, pinhole], near_tie_px=0.05)

    assert selected.lens_model is LensModel.PINHOLE


def test_holdout_validation_requires_six_views() -> None:
    calibrator = _load_calibrator()
    observations = _synthetic_observations(calibrator)
    observations = calibrator.ChessboardObservations(
        observations.object_points[:5],
        observations.image_points[:5],
        observations.image_size,
        observations.image_paths[:5],
    )

    with pytest.raises(ValueError, match="at least six"):
        calibrator.cross_validate_camera_model(observations, LensModel.PINHOLE)


def test_lens_v2_round_trip_preserves_validation_metadata(tmp_path) -> None:
    calibration = LensCalibration(
        lens_model=LensModel.PINHOLE,
        camera_matrix=np.array(
            [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=np.zeros(5),
        image_size=(1280, 720),
        reprojection_error_px=0.2,
        validation_median_error_px=0.3,
        validation_p95_error_px=0.7,
        valid_image_count=18,
    )
    path = tmp_path / "lens.npz"

    calibration.save(path)
    loaded = LensCalibration.load(path)

    assert loaded.version == 2
    assert loaded.validation_median_error_px == 0.3
    assert loaded.validation_p95_error_px == 0.7
    assert loaded.valid_image_count == 18


def test_rig_tool_rectifies_manual_anchor_manifests(tmp_path) -> None:
    tool = _load_rig_calibrator()
    camera = np.array([-12.0, 34.0, 22.0])
    camera_matrix = np.array(
        [[920.0, 0.0, 640.0], [0.0, 920.0, 360.0], [0.0, 0.0, 1.0]]
    )
    rig = CameraRigProfile(
        camera_id="truth",
        image_size=(1280, 720),
        lens_model=LensModel.PINHOLE,
        camera_matrix=camera_matrix,
        distortion_coefficients=np.zeros(5),
        pitch_length_m=105.0,
        pitch_width_m=68.0,
        pan_axis_origin_xyz_m=camera,
        pan_axis_direction=np.array([0.0, 0.0, 1.0]),
        optical_center_offset_m=np.zeros(3),
        base_rotation_world_to_camera=_look_at_rotation(
            camera, np.array([52.5, 34.0, 0.0])
        ),
    )
    pitch_points = PitchModel().grid(8, 6)
    frame_image = tmp_path / "anchor.jpg"
    cv2.imwrite(str(frame_image), np.zeros((720, 1280, 3), dtype=np.uint8))
    frames = []
    for frame_index, pan in ((0, -0.1), (1, 0.1)):
        image_points = transform_points(
            pitch_points, rig.pitch_to_image_homography(pan)
        )
        visible = (
            (image_points[:, 0] > -100.0)
            & (image_points[:, 0] < 1380.0)
            & (image_points[:, 1] > -100.0)
            & (image_points[:, 1] < 820.0)
        )
        frames.append(
            {
                "frame_index": frame_index,
                "image_path": frame_image.name,
                "correspondences": [
                    {
                        "label": f"point-{index}",
                        "image_xy": image.tolist(),
                        "pitch_xy_m": pitch.tolist(),
                    }
                    for index, (image, pitch) in enumerate(
                        zip(image_points[visible], pitch_points[visible])
                    )
                ],
                "contact_points": [],
            }
        )
    manifest_path = tmp_path / "anchors.json"
    manifest_path.write_text(
        json.dumps(
            {
                "version": 1,
                "split": "calibration",
                "image_size": [1280, 720],
                "pitch_dimensions_m": {"length": 105.0, "width": 68.0},
                "frames": frames,
            }
        ),
        encoding="utf-8",
    )
    lens = LensCalibration(
        LensModel.PINHOLE,
        camera_matrix,
        np.zeros(5),
        (1280, 720),
    )

    anchors, rectified_lens, pitch_size = tool._anchors_from_manifest(
        manifest_path, lens, balance=0.0
    )
    result = calibrate_fixed_pan_rig(
        "recovered", rectified_lens, anchors, pitch_size
    )

    assert len(anchors) == 2
    assert rectified_lens.lens_model is LensModel.PINHOLE
    assert np.allclose(rectified_lens.distortion_coefficients, 0.0)
    assert result.camera_centre_spread_m < 1e-3
