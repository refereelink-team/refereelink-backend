from __future__ import annotations

import numpy as np

from app.classification.team_calibration.prototypes import build_prototypes
from app.classification.team_calibration.role_predictor import SupervisedRolePrototypeClassifier
from app.classification.team_calibration.types import PlayerRole, TeamLabel, TeamPrototype, TrackFeature
from app.vision.display import TrackDisplaySmoother


def _prototype(team: TeamLabel, role: PlayerRole, feature: list[float]) -> TeamPrototype:
    value = np.asarray(feature, dtype=np.float32)
    return TeamPrototype(
        team=team,
        role=role,
        color_feature=value,
        deep_feature=None,
        track_count=1,
        sample_count=5,
    )


def test_referee_track_is_retained_as_team_none_prototype() -> None:
    track = TrackFeature(
        track_id=9,
        team=TeamLabel.NONE,
        role=PlayerRole.REFEREE,
        color_feature=np.asarray([1.0, 0.0], dtype=np.float32),
        observation_count=5,
        quality_sum=4.0,
    )

    prototypes = build_prototypes([track])

    assert (TeamLabel.NONE, PlayerRole.REFEREE) in prototypes


def test_supervised_role_classifier_selects_goalkeeper_and_referee() -> None:
    prototypes = {
        (TeamLabel.HOME, PlayerRole.OUTFIELD): _prototype(
            TeamLabel.HOME, PlayerRole.OUTFIELD, [1.0, 0.0, 0.0, 0.0]
        ),
        (TeamLabel.AWAY, PlayerRole.OUTFIELD): _prototype(
            TeamLabel.AWAY, PlayerRole.OUTFIELD, [0.0, 1.0, 0.0, 0.0]
        ),
        (TeamLabel.HOME, PlayerRole.GOALKEEPER): _prototype(
            TeamLabel.HOME, PlayerRole.GOALKEEPER, [0.0, 0.0, 1.0, 0.0]
        ),
        (TeamLabel.AWAY, PlayerRole.GOALKEEPER): _prototype(
            TeamLabel.AWAY, PlayerRole.GOALKEEPER, [0.0, 0.0, 1.0, 0.0]
        ),
        (TeamLabel.NONE, PlayerRole.REFEREE): _prototype(
            TeamLabel.NONE, PlayerRole.REFEREE, [0.0, 0.0, 0.0, 1.0]
        ),
    }
    classifier = SupervisedRolePrototypeClassifier(
        prototypes,
        color_weight=1.0,
        deep_weight=0.0,
        min_observations=1,
        min_margin=0.01,
        max_distance=1.0,
    )

    goalkeeper = classifier.predict(
        color_feature=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
        deep_feature=None,
        observation_count=1,
    )
    referee = classifier.predict(
        color_feature=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        deep_feature=None,
        observation_count=1,
    )

    assert goalkeeper.role == PlayerRole.GOALKEEPER
    assert referee.role == PlayerRole.REFEREE


def test_display_smoother_holds_short_detector_gap_without_creating_player() -> None:
    class Player:
        def __init__(self, track_id: int, bbox: tuple[float, float, float, float]) -> None:
            self.track_id = track_id
            self.bbox = bbox
            self.team_id = 0
            self.team = TeamLabel.HOME
            self.role = PlayerRole.OUTFIELD
            self.confidence = 0.9

    smoother = TrackDisplaySmoother(max_missing_frames=2)
    first = smoother.update([Player(7, (10, 10, 20, 40))])
    second = smoother.update([])
    third = smoother.update([])
    fourth = smoother.update([])

    assert [track.track_id for track in first] == [7]
    assert [track.track_id for track in second] == [7]
    assert second[0].missing_frames == 1
    assert [track.track_id for track in third] == [7]
    assert fourth == []
