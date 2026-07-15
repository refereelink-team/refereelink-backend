from __future__ import annotations

import numpy as np
import supervision as sv

from app.state.models import BallState, BallStatus, FrameState, PlayerRole, PlayerState
from app.tracking.ball import KinematicBallTracker
from app.vision.ball import BallProcessor


def _detections(x: float, y: float, confidence: float = 0.9) -> sv.Detections:
    return sv.Detections(
        xyxy=np.array([[x - 2, y - 2, x + 2, y + 2]], dtype=np.float32),
        confidence=np.array([confidence], dtype=np.float32),
    )


def test_kinematic_tracker_predicts_then_expires() -> None:
    tracker = KinematicBallTracker(max_prediction_frames=2)

    fresh = tracker.update(np.array([10.0, 20.0]), confidence=0.9, timestamp_s=0.0)
    predicted = tracker.update(None, timestamp_s=0.04)
    expired = tracker.update(None, timestamp_s=0.08)
    unavailable = tracker.update(None, timestamp_s=0.12)

    assert fresh.status == "fresh"
    assert predicted.status == "predicted"
    assert predicted.position is not None
    assert expired.status == "predicted"
    assert unavailable.status == "unavailable"
    assert unavailable.position is None


def test_ball_processor_runs_detector_at_interval_and_predicts_between_calls() -> None:
    calls: list[int] = []

    def detector(frame: np.ndarray) -> sv.Detections:
        calls.append(int(frame[0, 0, 0]))
        return _detections(float(frame[0, 0, 0]), 10.0)

    processor = BallProcessor(
        detection_interval=2,
        max_prediction_frames=3,
        detector=detector,
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)

    first = processor.process(frame, frame_index=1, timestamp_s=0.0)
    second = processor.process(frame, frame_index=2, timestamp_s=0.04)
    third = processor.process(frame, frame_index=3, timestamp_s=0.08)

    assert calls == [0, 0]
    assert first.status == "fresh"
    assert second.status == "fresh"
    assert third.status == "predicted"
    assert processor.detection_count == 2
    assert processor.predicted_frames == 1


def test_ball_state_and_frame_state_serialize_optional_coordinates() -> None:
    state = BallState(status=BallStatus.PREDICTED, image_x=10.0, image_y=20.0, age_frames=1)
    frame = FrameState(
        frame_id=3,
        ball=state,
        players=[
            PlayerState(
                track_id=1,
                role=PlayerRole.UNKNOWN,
                team_id=-1,
                confidence=0.6,
            )
        ],
    )

    data = frame.model_dump(mode="json")

    assert data["ball"]["status"] == "predicted"
    assert data["ball"]["field_x"] is None
    assert data["players"][0]["semantic_status"] == "unknown"
