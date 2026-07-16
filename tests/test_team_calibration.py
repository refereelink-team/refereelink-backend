from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.classification.team_calibration.appearance_features import AppearanceFeatureExtractor
from app.classification.team_calibration.bundle import CalibrationBundle
from app.classification.team_calibration.color_features import ColorFeatureExtractor
from app.classification.team_calibration.predictor import SupervisedPrototypeClassifier
from app.classification.team_calibration.prototypes import build_prototypes
from app.classification.team_calibration.quality import CropQualityAssessor
from app.classification.team_calibration.roi import JerseyROIExtractor
from app.classification.team_calibration.track_features import TrackFeatureBank
from app.classification.team_calibration.types import (
    PlayerRole,
    TeamLabel,
    TrackFeature,
    ValidationReport,
)
from app.classification.team_calibration.validation import CalibrationValidator


def _jersey(color: tuple[int, int, int], *, height: int = 80, width: int = 48) -> np.ndarray:
    image = np.full((height, width, 3), color, dtype=np.uint8)
    # Add deterministic edges so the default blur check does not confuse a
    # clean synthetic jersey with a motion-blurred crop.
    cv2.rectangle(image, (2, 2), (width - 3, height - 3), (255, 255, 255), 2)
    return image


def _track(track_id: int, team: TeamLabel, hue: float) -> TrackFeature:
    color = np.zeros(36, dtype=np.float32)
    color[int(hue)] = 1.0
    return TrackFeature(
        track_id=track_id,
        team=team,
        role=PlayerRole.OUTFIELD,
        color_feature=color,
        deep_feature=np.eye(4, dtype=np.float32)[track_id % 4],
        observation_count=5,
        quality_sum=4.0,
        last_update_frame=track_id,
    )


def test_roi_extractor_uses_central_15_to_65_percent_torso() -> None:
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    extractor = JerseyROIExtractor()
    box = extractor.box(frame.shape, [10, 10, 90, 90])
    assert box is not None
    assert box.y1 == 22
    assert box.y2 == 62
    assert box.width == 60


def test_quality_assessor_rejects_blur_and_accepts_clear_crop() -> None:
    assessor = CropQualityAssessor()
    clear = assessor.assess(_jersey((0, 0, 220)), detection_confidence=0.9)
    blurred = assessor.assess(
        cv2.GaussianBlur(_jersey((0, 0, 220)), (31, 31), 0),
        detection_confidence=0.9,
    )
    assert clear.accepted
    assert not blurred.accepted
    assert "blur_low" in blurred.reasons


def test_hsv_lab_features_are_fixed_size_and_normalized() -> None:
    extractor = ColorFeatureExtractor()
    feature = extractor.extract(_jersey((0, 0, 220)))
    assert feature.shape == (extractor.feature_dim,)
    assert np.isclose(np.linalg.norm(feature), 1.0)


def test_track_bank_aggregates_quality_weighted_features() -> None:
    bank = TrackFeatureBank(ema_alpha=0.5)
    first = np.array([1.0, 0.0], dtype=np.float32)
    second = np.array([0.0, 1.0], dtype=np.float32)
    bank.update(
        3,
        team=TeamLabel.HOME,
        role=PlayerRole.OUTFIELD,
        color_feature=first,
        deep_feature=None,
        quality=1.0,
        frame_index=1,
    )
    state = bank.update(
        3,
        team=TeamLabel.HOME,
        role=PlayerRole.OUTFIELD,
        color_feature=second,
        deep_feature=None,
        quality=1.0,
        frame_index=2,
    )
    assert state.observation_count == 2
    assert state.color_feature is not None
    assert state.color_feature[0] > 0.0
    assert state.color_feature[1] > 0.0


def test_validator_requires_both_labelled_outfield_teams() -> None:
    validator = CalibrationValidator(min_tracks_per_team=2, min_samples_per_track=5)
    report = validator.validate([_track(1, TeamLabel.HOME, 1), _track(2, TeamLabel.HOME, 1)])
    assert not report.passed
    assert "away_outfield_tracks_insufficient" in report.reasons


def test_supervised_classifier_rejects_until_observations_and_margin_are_valid() -> None:
    tracks = [
        _track(1, TeamLabel.HOME, 1),
        _track(2, TeamLabel.HOME, 1),
        _track(3, TeamLabel.AWAY, 20),
        _track(4, TeamLabel.AWAY, 20),
    ]
    prototypes = build_prototypes(tracks, include_goalkeepers=False)
    classifier = SupervisedPrototypeClassifier(prototypes, min_observations=3, max_distance=10.0)
    unknown = classifier.predict(
        color_feature=tracks[0].color_feature,
        deep_feature=tracks[0].deep_feature,
        observation_count=2,
    )
    assert unknown.team == TeamLabel.UNKNOWN
    assert unknown.rejection_reason == "observations_insufficient"
    prediction = classifier.predict(
        color_feature=tracks[0].color_feature,
        deep_feature=tracks[0].deep_feature,
        observation_count=3,
    )
    assert prediction.team == TeamLabel.HOME


def test_bundle_roundtrip_is_pickle_free(tmp_path: Path) -> None:
    tracks = [_track(1, TeamLabel.HOME, 1), _track(2, TeamLabel.AWAY, 20)]
    prototypes = build_prototypes(tracks, include_goalkeepers=False)
    report = ValidationReport(passed=True, goalkeeper_mapping_ready=False)
    bundle = CalibrationBundle("match-1", "camera-1", prototypes, report)
    path = tmp_path / "team.npz"
    bundle.save(path)
    restored = CalibrationBundle.load(path)
    assert restored.bundle_version == "team-calibration-v1"
    assert restored.prototypes.keys() == bundle.prototypes.keys()
    assert np.array_equal(
        restored.prototypes[(TeamLabel.HOME, PlayerRole.OUTFIELD)].color_feature,
        bundle.prototypes[(TeamLabel.HOME, PlayerRole.OUTFIELD)].color_feature,
    )


def test_mobile_net_extractor_supports_injected_fake_model() -> None:
    import torch

    class FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = torch.nn.Conv2d(3, 8, kernel_size=1)
            self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

    extractor = AppearanceFeatureExtractor(device="cpu", model=FakeModel(), pretrained=False)
    result = extractor.extract_batch([_jersey((0, 0, 220)), _jersey((220, 0, 0))])
    assert result.shape == (2, 8)
    assert np.allclose(np.linalg.norm(result, axis=1), 1.0)

