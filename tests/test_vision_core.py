from __future__ import annotations

import numpy as np
import supervision as sv

from app.config.pitch import SoccerPitchConfiguration
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
