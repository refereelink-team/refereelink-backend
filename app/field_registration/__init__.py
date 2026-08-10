"""Constrained soccer-pitch registration primitives.

The package deliberately keeps the camera model and geometry independent from
the neural-network backend.  This makes calibration, synthetic verification,
and deployment backends testable without loading model weights.
"""

from app.field_registration.camera_model import CameraRigProfile
from app.field_registration.initializer import (
    HomographyInitialization,
    HomographyInitializer,
    HomographyInitializerConfig,
)
from app.field_registration.lens import LensCalibration, LensModel, LensUndistorter
from app.field_registration.pitch_model import PitchDimensions, PitchModel, VenueProfile
from app.field_registration.projection import PitchProjector
from app.field_registration.rig_calibration import (
    PanAnchor,
    RigCalibrationResult,
    calibrate_fixed_pan_rig,
)
from app.field_registration.tracker import FieldRegistrationConfig, FieldRegistrationCore
from app.field_registration.types import (
    CameraState,
    CameraTrackingStatus,
    FieldRegistrationFrame,
    LineObservation,
    PitchCoordinate,
    PointObservation,
)

__all__ = [
    "CameraRigProfile",
    "CameraState",
    "CameraTrackingStatus",
    "FieldRegistrationFrame",
    "FieldRegistrationConfig",
    "FieldRegistrationCore",
    "HomographyInitialization",
    "HomographyInitializer",
    "HomographyInitializerConfig",
    "LensCalibration",
    "LensModel",
    "LensUndistorter",
    "LineObservation",
    "PitchCoordinate",
    "PitchDimensions",
    "PitchModel",
    "PitchProjector",
    "PanAnchor",
    "RigCalibrationResult",
    "PointObservation",
    "VenueProfile",
    "calibrate_fixed_pan_rig",
]
