from __future__ import annotations

import cv2
import numpy as np
import pytest
import supervision as sv

from app.config.pitch import SoccerPitchConfiguration
from app.geometry.camera import CameraMotionEstimator
from app.geometry.pitch_projection import PitchProjectionResult
from app.field_registration.perception import PitchPerceptionOutput, StaticPerceptionBackend
from app.field_registration.pitch_model import PitchDimensions, PitchModel
from app.field_registration.tracker import FieldRegistrationCore
from app.field_registration.types import (
    CameraState,
    CameraTrackingStatus,
    PointObservation,
    RegistrationMode,
)
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


def test_field_registration_mode_preserves_legacy_flag_mapping() -> None:
    legacy = VisionCore(
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
        enable_field_registration_v2=False,
    )
    broadcast = VisionCore(
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
        enable_field_registration_v2=True,
    )
    explicit_legacy = VisionCore(
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
        enable_field_registration_v2=True,
        field_registration_mode="legacy",
    )

    assert legacy.field_registration_mode is RegistrationMode.LEGACY
    assert not legacy.enable_field_registration_v2
    assert broadcast.field_registration_mode is RegistrationMode.BROADCAST
    assert broadcast.enable_field_registration_v2
    assert explicit_legacy.field_registration_mode is RegistrationMode.LEGACY
    assert not explicit_legacy.enable_field_registration_v2


def test_v2_field_registration_integrates_without_changing_legacy_projection_units() -> None:
    pitch_model = PitchModel(PitchDimensions(length_m=120.0, width_m=70.0))
    pitch_points = pitch_model.grid(6, 5)
    image_points = pitch_points * np.array([8.0, 5.0]) + np.array([100.0, 100.0])
    observations = tuple(
        PointObservation(str(index), tuple(image), tuple(pitch), 0.95)
        for index, (image, pitch) in enumerate(zip(image_points, pitch_points))
    )
    registration_core = FieldRegistrationCore(
        pitch_model,
        StaticPerceptionBackend([PitchPerceptionOutput(points=observations)]),
    )
    detections = sv.Detections(
        xyxy=np.array([[480.0, 200.0, 520.0, 300.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
        tracker_id=np.array([7]),
    )
    core = VisionCore(
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
        field_registration_core=registration_core,
    )
    core._predict_player = lambda frame: detections

    result = core.process(np.zeros((720, 1280, 3), dtype=np.uint8), 1)

    assert result.projection.homography_status == "relocalized"
    assert result.projection.measurement_usable
    # The V2 core works in metres; VisionCore preserves the legacy centimetre
    # contract until all downstream consumers migrate.
    assert result.field_xy[0] == pytest.approx((5000.0, 4000.0), abs=0.1)
    assert result.pitch_coordinates[0].xy_m == pytest.approx((50.0, 40.0), abs=0.001)
    assert result.pitch_coordinates[0].source == "bbox_bottom"
    assert result.pitch_coordinates[0].sigma_m is not None
    assert result.pitch_coordinates[0].sigma_m > 0.0


def test_v2_rejects_truncated_player_contact_point() -> None:
    pitch_model = PitchModel(PitchDimensions(length_m=120.0, width_m=70.0))
    pitch_points = pitch_model.grid(6, 5)
    image_points = pitch_points * np.array([8.0, 5.0]) + np.array([100.0, 100.0])
    observations = tuple(
        PointObservation(str(index), tuple(image), tuple(pitch), 0.95)
        for index, (image, pitch) in enumerate(zip(image_points, pitch_points))
    )
    registration_core = FieldRegistrationCore(
        pitch_model,
        StaticPerceptionBackend([PitchPerceptionOutput(points=observations)]),
    )
    detections = sv.Detections(
        xyxy=np.array([[480.0, 620.0, 520.0, 720.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        tracker_id=np.array([7]),
    )
    core = VisionCore(
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
        field_registration_core=registration_core,
    )
    core._predict_player = lambda frame: detections

    result = core.process(np.zeros((720, 1280, 3), dtype=np.uint8), 1)

    assert np.isnan(result.field_xy).all()
    assert result.pitch_coordinates[0].xy_m is None
    assert result.pitch_coordinates[0].source == "none"


def test_v2_preserves_predicted_camera_status_while_rejecting_measurement() -> None:
    pitch_model = PitchModel(PitchDimensions(length_m=120.0, width_m=70.0))
    registration_core = FieldRegistrationCore(
        pitch_model,
        StaticPerceptionBackend([]),
    )
    detections = sv.Detections(
        xyxy=np.array([[100.0, 100.0, 140.0, 220.0]], dtype=np.float32),
    )
    core = VisionCore(
        undistorter=_IdentityUndistorter(),
        tracker=_Tracker(),
        field_registration_core=registration_core,
    )
    core._last_camera_state = CameraState(
        status=CameraTrackingStatus.PREDICTED,
        pan_rad=0.0,
        pan_velocity_rad_s=0.0,
        covariance=np.eye(2),
        image_to_pitch=np.eye(3),
        pitch_to_image=np.eye(3),
        confidence=0.4,
    )
    projection = PitchProjectionResult(
        tracking_observations=[],
        projected_keypoints=[],
        homography=np.eye(3),
        homography_status="predicted",
        reprojection_error=None,
        measurement_usable=False,
    )

    field_xy, coordinates = core._field_coordinates(
        detections,
        projection,
        (720, 1280, 3),
    )

    assert np.isnan(field_xy).all()
    assert coordinates[0].xy_m is None
    assert coordinates[0].camera_status == CameraTrackingStatus.PREDICTED
