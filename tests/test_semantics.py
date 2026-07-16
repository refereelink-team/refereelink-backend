"""Pure NumPy tests for phase-2 player semantic components."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.classification.online import OnlineTeamClassifier
from app.classification.team import UNKNOWN_TEAM_ID, TeamClassifier
from app.vision.semantics import (
    UNKNOWN_ROLE,
    SemanticObservation,
    TrackSemanticManager,
    TrajectorySemanticManager,
)


def _solid_bgr(color: tuple[int, int, int], height: int = 40, width: int = 24) -> np.ndarray:
    crop = np.zeros((height, width, 3), dtype=np.uint8)
    crop[:] = color
    return crop


def test_unfitted_classifier_and_invalid_crops_fall_back_to_unknown() -> None:
    classifier = TeamClassifier()

    ids, confidences = classifier.predict_with_confidence([np.zeros((2, 2, 3), dtype=np.uint8), np.array([])])

    assert ids.tolist() == [UNKNOWN_TEAM_ID, UNKNOWN_TEAM_ID]
    assert np.all(confidences == 0.0)


def test_color_classifier_separates_upper_body_colours_and_returns_confidence() -> None:
    red = _solid_bgr((0, 0, 230))
    blue = _solid_bgr((220, 0, 0))
    classifier = TeamClassifier(confidence_threshold=0.55).fit([red, blue], labels=[0, 1])

    ids, confidences = classifier.predict_with_confidence([red, blue])

    assert ids.tolist() == [0, 1]
    assert np.all(confidences > 0.55)


def test_ambiguous_low_information_crop_is_unknown() -> None:
    classifier = TeamClassifier().fit(
        [_solid_bgr((0, 0, 230)), _solid_bgr((220, 0, 0))], labels=[0, 1]
    )

    ids, confidences = classifier.predict_with_confidence([_solid_bgr((125, 125, 125))])

    assert ids.tolist() == [UNKNOWN_TEAM_ID]
    assert confidences.tolist() == [0.0]


def test_single_observed_team_is_not_treated_as_a_two_team_classifier() -> None:
    classifier = TeamClassifier().fit([_solid_bgr((0, 0, 230))], labels=[0])

    ids, confidences = classifier.predict_with_confidence([_solid_bgr((0, 0, 230))])

    assert ids.tolist() == [UNKNOWN_TEAM_ID]
    assert confidences.tolist() == [0.0]


def test_labelled_fit_is_persistable_and_unlabelled_fit_is_rejected(tmp_path: Path) -> None:
    crops = [_solid_bgr((0, 0, 230)), _solid_bgr((220, 0, 0))]
    with pytest.raises(ValueError, match="labels are required"):
        TeamClassifier().fit(crops)
    first = TeamClassifier(confidence_threshold=0.55).fit(crops, labels=[0, 1])
    model_path = tmp_path / "team-prototypes.bin"
    first.save(model_path)
    restored = TeamClassifier().load(model_path)

    first_ids = first.predict(crops)
    assert first_ids.tolist() == restored.predict(crops).tolist()


def test_online_classifier_never_fits_during_runtime_warmup() -> None:
    classifier = OnlineTeamClassifier(warmup_frames=2, warmup_stride=2, min_warmup_crops=2)
    red = _solid_bgr((0, 0, 230))
    blue = _solid_bgr((220, 0, 0))

    assert classifier.collect_and_predict(np.empty((1, 1, 3)), [red, blue]).tolist() == [-1, -1]
    assert classifier.collect_and_predict(np.empty((1, 1, 3)), [red, blue]).tolist() == [-1, -1]
    assert not classifier.fitted
    with pytest.raises(ValueError, match="labels are required"):
        classifier.fit([red, blue])
    classifier.fit([red, blue], labels=[0, 1])
    assert classifier.fitted


def test_trajectory_manager_weighted_window_suppresses_one_frame_flip() -> None:
    manager = TrajectorySemanticManager(
        history_size=6,
        recency_decay=0.9,
        stable_threshold=0.55,
        switch_margin=0.12,
    )

    state = manager.update(
        7,
        role="outfield",
        role_confidence=0.95,
        team_id=0,
        team_confidence=0.95,
        frame_index=1,
    )
    assert state.role == "outfield"
    assert state.team_id == 0
    assert state.semantic_status == "stable"

    flipped = manager.update(
        7,
        role="referee",
        role_confidence=0.99,
        team_id=1,
        team_confidence=0.99,
        frame_index=2,
    )
    assert flipped.role == "outfield"
    assert flipped.team_id == 0
    assert manager.semantic_label_switches == 0


def test_trajectory_manager_switches_after_sustained_challenger_evidence() -> None:
    manager = TrajectorySemanticManager(history_size=4, recency_decay=0.8, switch_margin=0.05)
    manager.update(3, role="outfield", role_confidence=0.9, team_id=0, team_confidence=0.9, frame_index=1)

    for frame in range(2, 6):
        state = manager.update(
            3,
            role="goalkeeper",
            role_confidence=0.95,
            team_id=1,
            team_confidence=0.95,
            frame_index=frame,
        )

    assert state.role == "goalkeeper"
    assert state.team_id == 1
    assert state.role_switches == 1
    assert state.team_switches == 1


def test_unknown_evidence_eventually_releases_stale_labels() -> None:
    manager = TrajectorySemanticManager(history_size=3, recency_decay=0.5, stable_threshold=0.55)
    manager.update(4, role="outfield", role_confidence=1.0, team_id=0, team_confidence=1.0, frame_index=1)

    for frame in range(2, 5):
        state = manager.update(
            4,
            role=UNKNOWN_ROLE,
            role_confidence=1.0,
            team_id=UNKNOWN_TEAM_ID,
            team_confidence=1.0,
            frame_index=frame,
        )

    assert state.role == UNKNOWN_ROLE
    assert state.team_id == UNKNOWN_TEAM_ID
    assert state.semantic_status == "unknown"


def test_stale_tracks_are_removed_and_observation_wrapper_sets_frame() -> None:
    manager = TrajectorySemanticManager(max_missing_frames=2)
    state = manager.update_observation(
        9,
        SemanticObservation(role="outfield", role_confidence=0.8, frame_index=10),
    )

    assert state.last_seen_frame == 10
    assert manager.remove_stale(12) == []
    assert manager.remove_stale(13) == [9]
    assert manager.get(9) is None


def test_track_semantic_manager_exposes_stable_frame_level_contract_without_models() -> None:
    manager = TrackSemanticManager()
    frame = np.zeros((80, 120, 3), dtype=np.uint8)
    detections = SimpleNamespace(
        tracker_id=np.asarray([7, 11]),
        xyxy=np.asarray([[10, 10, 30, 70], [50, 10, 80, 70]], dtype=np.float32),
    )

    results = manager.update(frame, detections, frame_index=42)

    assert set(results) == {7, 11}
    for track_id, result in results.items():
        assert result.track_id == track_id
        assert result.role == UNKNOWN_ROLE
        assert result.team_id == UNKNOWN_TEAM_ID
        assert result.role_confidence == 0.0
        assert result.team_confidence == 0.0
        assert result.status == "unknown"
        assert result.as_dict()["status"] == "unknown"
