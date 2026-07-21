"""Supervised, per-match team calibration components.

The package deliberately contains no unsupervised fitting path.  A runtime
classifier can only produce HOME/AWAY after a labelled calibration bundle has
been validated and loaded.
"""

from app.classification.team_calibration.appearance_features import AppearanceFeatureExtractor
from app.classification.team_calibration.bundle import CalibrationBundle
from app.classification.team_calibration.color_features import ColorFeatureExtractor
from app.classification.team_calibration.predictor import SupervisedPrototypeClassifier
from app.classification.team_calibration.role_predictor import (
    CalibratedRoleClassifier,
    SupervisedRolePrototypeClassifier,
)
from app.classification.team_calibration.prototypes import build_prototypes
from app.classification.team_calibration.quality import CropQualityAssessor, QualityAssessment
from app.classification.team_calibration.roi import JerseyROIExtractor
from app.classification.team_calibration.runtime import TeamAssignmentService
from app.classification.team_calibration.session import CalibrationState, TeamCalibrationSession
from app.classification.team_calibration.track_features import TrackFeatureBank
from app.classification.team_calibration.types import (
    CalibrationLabel,
    FeatureObservation,
    PlayerRole,
    TeamLabel,
    TeamPrediction,
    TeamPrototype,
    RolePrediction,
)

__all__ = [
    "AppearanceFeatureExtractor",
    "CalibrationBundle",
    "CalibrationState",
    "CalibrationLabel",
    "ColorFeatureExtractor",
    "CropQualityAssessor",
    "FeatureObservation",
    "JerseyROIExtractor",
    "PlayerRole",
    "QualityAssessment",
    "SupervisedPrototypeClassifier",
    "SupervisedRolePrototypeClassifier",
    "CalibratedRoleClassifier",
    "TeamAssignmentService",
    "TeamCalibrationSession",
    "TeamLabel",
    "TeamPrediction",
    "TeamPrototype",
    "RolePrediction",
    "TrackFeatureBank",
    "build_prototypes",
]
