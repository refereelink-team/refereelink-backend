import numpy as np
import supervision as sv

from app.tracking.ball import BallTracker


def test_ball_tracker_returns_closest_detection_to_recent_centroid() -> None:
    tracker = BallTracker(buffer_size=3)

    tracker.update(sv.Detections(xyxy=np.array([[0, 0, 10, 10]], dtype=np.float32)))
    tracker.update(sv.Detections(xyxy=np.array([[2, 0, 12, 10]], dtype=np.float32)))

    detections = sv.Detections(
        xyxy=np.array(
            [
                [4, 0, 14, 10],
                [100, 100, 110, 110],
            ],
            dtype=np.float32,
        )
    )

    selected = tracker.update(detections)

    assert len(selected) == 1
    np.testing.assert_array_equal(selected.xyxy, np.array([[4, 0, 14, 10]], dtype=np.float32))


def test_ball_tracker_returns_empty_detections_when_input_is_empty() -> None:
    tracker = BallTracker(buffer_size=3)
    detections = sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32))

    selected = tracker.update(detections)

    assert len(selected) == 0
