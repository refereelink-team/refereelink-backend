from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from app.config.pitch import SoccerPitchConfiguration
from app.geometry.camera import CameraMotionEstimator
from app.geometry.pitch_projection import PitchProjectionResult
from app.vision.core import VisionCore


class _IdentityUndistorter:
    def apply(self, frame):
        return frame


class _Tracker:
    def update_with_detections(self, detections):
        return detections


class _ProjectionEngine:
    config = SoccerPitchConfiguration()

    def __init__(self):
        self.pitch_calls = 0

    def update(self, frame, keypoints):
        self.pitch_calls += 1
        return PitchProjectionResult(
            tracking_observations=[],
            projected_keypoints=[],
            homography=np.eye(3, dtype=np.float32),
            homography_status="fresh",
            reprojection_error=0.0,
        )

    def reuse(self, frame):
        return PitchProjectionResult(
            tracking_observations=[],
            projected_keypoints=[],
            homography=np.eye(3, dtype=np.float32),
            homography_status="reused",
            reprojection_error=None,
        )


class _MotionFailureProjectionEngine(_ProjectionEngine):
    def __init__(self):
        super().__init__()
        self.invalidations = 0

    def update(self, frame, keypoints):
        self.pitch_calls += 1
        if self.pitch_calls > 1:
            return PitchProjectionResult(
                tracking_observations=[],
                projected_keypoints=[],
                homography=np.eye(3, dtype=np.float32),
                homography_status="stale",
                reprojection_error=None,
            )
        return PitchProjectionResult(
            tracking_observations=[],
            projected_keypoints=[],
            homography=np.eye(3, dtype=np.float32),
            homography_status="fresh",
            reprojection_error=0.0,
        )

    def invalidate(self):
        self.invalidations += 1


def test_pitch_detection_is_scheduled_and_person_is_unknown():
    detections = sv.Detections(
        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
        tracker_id=np.array([42]),
    )
    projection_engine = _ProjectionEngine()
    core = VisionCore(
        fps=25,
        pitch_detection_interval=5,
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
        projection_engine=projection_engine,
    )
    core._predict_player = lambda frame: detections
    core._predict_pitch = lambda frame: object()

    frames = [core.process(np.zeros((100, 100, 3), dtype=np.uint8), i) for i in range(1, 8)]

    assert projection_engine.pitch_calls == 2
    assert [frame.projection.homography_status for frame in frames] == [
        "fresh", "reused", "reused", "reused", "reused", "fresh", "reused"
    ]
    assert frames[0].color_lookup.tolist() == [4]
    assert np.allclose(frames[0].field_xy[0], [10, 15])


def test_field_coordinates_outside_pitch_are_nan():
    detections = sv.Detections(
        xyxy=np.array([[90, 90, 110, 110]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )
    projection_engine = _ProjectionEngine()
    core = VisionCore(
        enable_pitch=False,
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
        projection_engine=projection_engine,
    )
    core._predict_player = lambda frame: detections

    result = core.process(np.zeros((100, 100, 3), dtype=np.uint8), 1)

    assert np.isnan(result.field_xy).all()


def test_empty_detections_are_safe():
    empty = sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32))
    core = VisionCore(
        enable_pitch=False,
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
    )
    core._predict_player = lambda frame: empty

    result = core.process(np.zeros((32, 32, 3), dtype=np.uint8), 1)

    assert len(result.tracked_detections) == 0
    assert result.field_xy.shape == (0, 2)


def test_camera_motion_forces_pitch_refresh_before_interval() -> None:
    base = np.zeros((240, 320, 3), dtype=np.uint8)
    base[:] = (0, 96, 0)
    cv2.line(base, (20, 30), (300, 70), (255, 255, 255), 3)
    cv2.rectangle(base, (70, 60), (250, 190), (255, 255, 255), 2)
    shifted = cv2.warpAffine(
        base,
        np.float32([[1, 0, 10], [0, 1, 0]]),
        (base.shape[1], base.shape[0]),
    )
    detections = sv.Detections(
        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
        tracker_id=np.array([42]),
    )
    projection_engine = _ProjectionEngine()
    core = VisionCore(
        fps=25,
        pitch_detection_interval=5,
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
        projection_engine=projection_engine,
        camera_motion_estimator=CameraMotionEstimator(threshold_px=6.0),
    )
    core._predict_player = lambda frame: detections
    core._predict_pitch = lambda frame: object()

    core.process(base, 1)
    moved = core.process(shifted, 2)

    assert projection_engine.pitch_calls == 2
    assert core.camera_motion_refresh_count == 1
    assert moved.projection.homography_status == "fresh"


def test_camera_motion_does_not_reuse_old_homography_when_refresh_fails() -> None:
    base = np.zeros((240, 320, 3), dtype=np.uint8)
    base[:] = (0, 96, 0)
    cv2.line(base, (20, 30), (300, 70), (255, 255, 255), 3)
    cv2.rectangle(base, (70, 60), (250, 190), (255, 255, 255), 2)
    shifted = cv2.warpAffine(
        base,
        np.float32([[1, 0, 10], [0, 1, 0]]),
        (base.shape[1], base.shape[0]),
    )
    detections = sv.Detections(
        xyxy=np.array([[5, 5, 15, 15]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
        tracker_id=np.array([42]),
    )
    projection_engine = _MotionFailureProjectionEngine()
    core = VisionCore(
        fps=25,
        pitch_detection_interval=5,
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
        projection_engine=projection_engine,
        camera_motion_estimator=CameraMotionEstimator(threshold_px=6.0),
    )
    core._predict_player = lambda frame: detections
    core._predict_pitch = lambda frame: object()

    core.process(base, 1)
    moved = core.process(shifted, 2)

    assert moved.projection.homography_status == "unavailable"
    assert projection_engine.invalidations == 1
    assert np.isnan(moved.field_xy).all()
