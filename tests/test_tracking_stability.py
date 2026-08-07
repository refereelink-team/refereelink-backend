from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import supervision as sv

from app.vision.core import VisionCore
from app.vision.entities import TrackEntityManager
from app.vision.display import TrackDisplaySmoother


class _IdentityUndistorter:
    def apply(self, frame):
        return frame


class _SequenceTracker:
    def __init__(self, sequence):
        self.sequence = iter(sequence)

    def update_with_detections(self, detections):
        del detections
        return next(self.sequence)


def _detections(track_ids, boxes):
    return sv.Detections(
        xyxy=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        confidence=np.ones(len(track_ids), dtype=np.float32),
        tracker_id=np.asarray(track_ids, dtype=np.int32),
    )


def test_entity_manager_rebinds_a_new_tracker_id_after_a_short_gap():
    manager = TrackEntityManager(
        max_prediction_gap_frames=2,
        reactivation_window_frames=4,
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    first = manager.update(_detections([7], [[20, 20, 32, 60]]), frame, 1)
    manager.update(_detections([7], [[24, 20, 36, 60]]), frame, 2)
    manager.update(_detections([], []), frame, 3)
    rebound = manager.update(_detections([99], [[30, 20, 42, 60]]), frame, 4)

    entity_id = first.entity_ids[7]
    assert rebound.entity_ids[99] == entity_id
    assert rebound.rebindings == {99: 7}
    assert rebound.statuses[99] == "reactivated"


def test_display_smoother_predicts_motion_and_expires_after_configured_gap():
    smoother = TrackDisplaySmoother(max_missing_frames=2)
    player = SimpleNamespace(
        track_id=7,
        entity_id=3,
        bbox=(10, 10, 20, 40),
        team_id=0,
        team="home",
        role="outfield",
        confidence=0.9,
    )

    smoother.update([player])
    moved = SimpleNamespace(**{**player.__dict__, "bbox": (14, 10, 24, 40)})
    smoother.update([moved])
    predicted = smoother.update([])

    assert len(predicted) == 1
    assert predicted[0].entity_id == 3
    assert predicted[0].track_status == "predicted"
    assert predicted[0].missing_frames == 1
    assert predicted[0].bbox[0] >= 12.79
    assert smoother.update([])[0].missing_frames == 2
    assert smoother.update([]) == []


def test_vision_core_counts_partial_track_gaps_and_recovery():
    observed = _detections([7], [[20, 20, 32, 60]])
    empty = _detections([], [])
    tracker = _SequenceTracker([observed, empty, empty, observed])
    core = VisionCore(
        enable_pitch=False,
        undistorter=_IdentityUndistorter(),
        tracker=tracker,
        max_prediction_gap_frames=2,
    )
    core._predict_player = lambda frame: observed

    # The detector output is irrelevant to this test; the injected tracker
    # supplies the controlled active/missing/recovered sequence.
    frames = [
        core.process(np.zeros((100, 100, 3), dtype=np.uint8), index)
        for index in range(1, 5)
    ]

    assert core.track_id_interruptions == 1
    assert core.track_occlusion_events == 1
    assert core.track_predicted_frames == 2
    assert core.track_recovered_count == 1
    assert core.track_reactivated_count == 1
    assert core.track_id_switches == 0
    assert core.track_max_missing_frames == 2
    assert frames[-1].track_status[7] == "reactivated"
    assert core.track_lifecycle[7] == [
        "detected",
        "occluded",
        "predicted",
        "reactivated",
    ]
    assert core.track_lifecycle_counts["occluded"] == 1
    assert core.track_lifecycle_counts["predicted"] == 1


def test_vision_core_records_id_switch_and_entity_reactivation():
    observed = _detections([7], [[20, 20, 32, 60]])
    empty = _detections([], [])
    rebound = _detections([99], [[20, 20, 32, 60]])
    tracker = _SequenceTracker([observed, empty, rebound])
    core = VisionCore(
        enable_pitch=False,
        undistorter=_IdentityUndistorter(),
        tracker=tracker,
        max_prediction_gap_frames=2,
        reactivation_window_frames=4,
    )
    core._predict_player = lambda frame: observed
    frames = [
        core.process(np.zeros((100, 100, 3), dtype=np.uint8), index)
        for index in range(1, 4)
    ]

    assert frames[-1].rebindings == {99: 7}
    assert core.track_id_switches == 1
    assert core.track_reactivated_count == 1
    assert core.track_lifecycle[99] == ["id_switch", "reactivated"]
