"""Shared detection, tracking and pitch-projection primitives."""

from app.vision.core import VisionCore, VisionFrame
from app.vision.ball import BallProcessor
from app.vision.backends import CallableBackend, DetectorBackend, UltralyticsBackend
from app.vision.semantics import TrackSemanticManager
from app.vision.display import DisplayTrack, TrackDisplaySmoother
from app.vision.entities import EntityUpdate, TrackEntityManager

__all__ = [
    "VisionCore",
    "VisionFrame",
    "BallProcessor",
    "TrackSemanticManager",
    "DisplayTrack",
    "TrackDisplaySmoother",
    "EntityUpdate",
    "TrackEntityManager",
    "CallableBackend",
    "DetectorBackend",
    "UltralyticsBackend",
]
