from __future__ import annotations

import numpy as np
import pytest

from app.geometry.camera import CameraCalibrationError, CameraUndistorter


def _write_calibration(
    path,
    width: int = 640,
    height: int = 480,
    lens_model: str = "pinhole",
) -> None:
    np.savez(
        path,
        camera_matrix=np.array(
            [[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=np.zeros(
            (1, 4 if lens_model == "fisheye" else 5), dtype=np.float64
        ),
        image_width=np.array(width),
        image_height=np.array(height),
        reprojection_error=np.array(0.2),
        lens_model=np.array(lens_model),
    )


def test_missing_calibration_bypasses_undistortion(tmp_path):
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    undistorter = CameraUndistorter(str(tmp_path / "missing.npz"))

    assert not undistorter.available
    assert undistorter.apply(frame) is frame


def test_calibration_loads_and_scales_maps_for_resolution_change(tmp_path):
    path = tmp_path / "camera.npz"
    _write_calibration(path)
    undistorter = CameraUndistorter(str(path), alpha=0.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    rectified = undistorter.apply(frame)

    assert undistorter.available
    assert rectified.shape == frame.shape
    assert (320, 240) in undistorter._maps


def test_calibration_rejects_incompatible_aspect_ratio(tmp_path):
    path = tmp_path / "camera.npz"
    _write_calibration(path)
    undistorter = CameraUndistorter(str(path))

    with pytest.raises(CameraCalibrationError, match="aspect ratio"):
        undistorter.apply(np.zeros((300, 320, 3), dtype=np.uint8))


def test_fisheye_calibration_uses_fisheye_rectification_maps(tmp_path):
    path = tmp_path / "fisheye.npz"
    _write_calibration(path, lens_model="fisheye")
    undistorter = CameraUndistorter(str(path), alpha=0.0)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    rectified = undistorter.apply(frame)

    assert undistorter.calibration is not None
    assert undistorter.calibration.lens_model == "fisheye"
    assert rectified.shape == frame.shape
