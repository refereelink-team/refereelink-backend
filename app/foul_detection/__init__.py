"""Foul detection adapter — bridges fouls_far into the app pipeline."""

from app.foul_detection.detector import FoulDetector

__all__ = ["FoulDetector"]
