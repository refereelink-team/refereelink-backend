from __future__ import annotations

import cv2
import numpy as np

from app.geometry.camera import CameraMotionEstimator


def _field_like_frame(shift_x: int = 0) -> np.ndarray:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    frame[:] = (0, 96, 0)
    cv2.line(frame, (20, 30), (300, 70), (255, 255, 255), 3)
    cv2.line(frame, (10, 210), (300, 150), (255, 255, 255), 3)
    cv2.rectangle(frame, (70, 60), (250, 190), (255, 255, 255), 2)
    if shift_x == 0:
        return frame
    return cv2.warpAffine(
        frame,
        np.float32([[1, 0, shift_x], [0, 1, 0]]),
        (frame.shape[1], frame.shape[0]),
    )


def test_camera_motion_estimator_detects_horizontal_view_shift() -> None:
    estimator = CameraMotionEstimator(threshold_px=6.0, analysis_width=320)
    reference = _field_like_frame()
    estimator.mark_reference(reference)

    estimate = estimator.measure(_field_like_frame(shift_x=10))

    assert estimate is not None
    assert estimate.shift_x_px > 8.0
    assert estimate.magnitude_px > 8.0
    assert estimate.response >= 0.15
    assert estimate.requires_refresh


def test_camera_motion_reference_resets_after_refresh() -> None:
    estimator = CameraMotionEstimator(threshold_px=6.0, analysis_width=320)
    reference = _field_like_frame()
    shifted = _field_like_frame(shift_x=10)
    estimator.mark_reference(reference)

    assert estimator.measure(shifted).requires_refresh
    estimator.mark_reference(shifted)
    stable = estimator.measure(shifted)

    assert stable is not None
    assert not stable.requires_refresh
