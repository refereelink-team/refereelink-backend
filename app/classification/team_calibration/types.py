from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class PlayerRole(str, Enum):
    """Role and team are separate semantic dimensions."""

    OUTFIELD = "outfield"
    # The alias keeps old callers that used PLAYER working while the wire
    # value and new code use the more precise OUTFIELD name.
    PLAYER = "outfield"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"
    STAFF = "staff"
    UNKNOWN = "unknown"


class TeamLabel(str, Enum):
    HOME = "home"
    AWAY = "away"
    NONE = "none"
    UNKNOWN = "unknown"


class CalibrationLabel(str, Enum):
    HOME_OUTFIELD = "home_outfield"
    AWAY_OUTFIELD = "away_outfield"
    HOME_GOALKEEPER = "home_goalkeeper"
    AWAY_GOALKEEPER = "away_goalkeeper"
    REFEREE = "referee"
    IGNORE = "ignore"

    @property
    def team(self) -> TeamLabel:
        if self in {CalibrationLabel.HOME_OUTFIELD, CalibrationLabel.HOME_GOALKEEPER}:
            return TeamLabel.HOME
        if self in {CalibrationLabel.AWAY_OUTFIELD, CalibrationLabel.AWAY_GOALKEEPER}:
            return TeamLabel.AWAY
        return TeamLabel.NONE

    @property
    def role(self) -> PlayerRole:
        if self in {CalibrationLabel.HOME_OUTFIELD, CalibrationLabel.AWAY_OUTFIELD}:
            return PlayerRole.OUTFIELD
        if self in {CalibrationLabel.HOME_GOALKEEPER, CalibrationLabel.AWAY_GOALKEEPER}:
            return PlayerRole.GOALKEEPER
        if self == CalibrationLabel.REFEREE:
            return PlayerRole.REFEREE
        return PlayerRole.UNKNOWN


@dataclass(frozen=True)
class FeatureObservation:
    track_id: int
    team: TeamLabel
    role: PlayerRole
    color_feature: Optional[np.ndarray]
    deep_feature: Optional[np.ndarray]
    quality: float
    frame_index: int


@dataclass
class TrackFeature:
    track_id: int
    team: TeamLabel
    role: PlayerRole
    color_feature: Optional[np.ndarray] = None
    deep_feature: Optional[np.ndarray] = None
    observation_count: int = 0
    quality_sum: float = 0.0
    last_update_frame: int = -1


@dataclass
class TeamPrototype:
    team: TeamLabel
    role: PlayerRole
    color_feature: Optional[np.ndarray]
    deep_feature: Optional[np.ndarray]
    track_count: int
    sample_count: int
    intra_class_dispersion: float = 0.0
    color_intra_scale: float = 0.0
    color_inter_scale: float = 0.0
    deep_intra_scale: float = 0.0
    deep_inter_scale: float = 0.0


@dataclass(frozen=True)
class TeamPrediction:
    team: TeamLabel = TeamLabel.UNKNOWN
    confidence: float = 0.0
    margin: float = 0.0
    best_distance: float = float("inf")
    observation_count: int = 0
    source: str = "supervised_prototype_fusion"
    rejection_reason: Optional[str] = None


@dataclass(frozen=True)
class RolePrediction:
    """Track-level role prediction from a supervised or model-backed source."""

    role: PlayerRole = PlayerRole.UNKNOWN
    confidence: float = 0.0
    margin: float = 0.0
    best_distance: float = float("inf")
    observation_count: int = 0
    source: str = "supervised_role_prototype"
    rejection_reason: Optional[str] = None


@dataclass
class ValidationReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    home_track_count: int = 0
    away_track_count: int = 0
    home_sample_count: int = 0
    away_sample_count: int = 0
    home_intra_class_dispersion: float = 0.0
    away_intra_class_dispersion: float = 0.0
    inter_class_separation: float = 0.0
    leave_one_track_out_accuracy: Optional[float] = None
    goalkeeper_mapping_ready: bool = False
    referee_mapping_ready: bool = False
    goalkeeper_track_count: int = 0
    referee_track_count: int = 0
    referee_sample_count: int = 0
