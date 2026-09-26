"""Tests for FoulDetector cooldown and confidence filtering.

`offside` is no longer required: the detector takes an injected predictor,
and these tests use a mock predictor instead of real model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from app.foul_detection.detector import FoulDetector
from app.foul_detection.types import FoulPrediction


def _make_detector(
    cooldown_frames: int = 25,
    confidence: float = 0.5,
    predicted_confidence: float = 0.9,
) -> FoulDetector:
    predictor = MagicMock()
    predictor.predict.return_value = FoulPrediction(confidence=predicted_confidence)
    return FoulDetector(
        checkpoint_path="dummy.pth",
        device="cpu",
        cooldown_frames=cooldown_frames,
        confidence_threshold=confidence,
        predictor=predictor,
    )


def test_drops_noise_on_ten_second_clip():
    """300 frames of a single foul -> under 15 candidates."""
    detector = _make_detector(cooldown_frames=25)

    candidates = [
        detector.update(np.zeros((720, 1280, 3), dtype=np.uint8), frame_index=i) for i in range(300)
    ]

    accepted = [c for c in candidates if c is not None]
    assert len(accepted) < 15
    assert detector.inference_count > 0


def test_avoids_zero_detections_on_foul_clip():
    """A foul clip must yield at least one candidate."""
    detector = _make_detector(cooldown_frames=10)

    candidates = [
        detector.update(np.zeros((720, 1280, 3), dtype=np.uint8), frame_index=i) for i in range(30)
    ]

    accepted = [c for c in candidates if c is not None]
    assert len(accepted) >= 1


def test_single_foul_clip_yields_exactly_one_candidate():
    """With a long cooldown, a single foul window yields exactly one candidate."""
    detector = _make_detector(cooldown_frames=200)

    candidates = [
        detector.update(np.zeros((720, 1280, 3), dtype=np.uint8), frame_index=i) for i in range(150)
    ]

    accepted = [c for c in candidates if c is not None]
    assert len(accepted) == 1


def test_weak_inference_output_is_filtered():
    """A weak predictor output must be filtered by the confidence threshold."""
    detector = _make_detector(confidence=0.7, predicted_confidence=0.2)

    results = [
        detector.update(np.zeros((720, 1280, 3), dtype=np.uint8), frame_index=i) for i in range(50)
    ]
    assert all(r is None for r in results)
    assert detector.inference_count > 0
