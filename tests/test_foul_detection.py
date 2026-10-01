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
    predictor.predict.return_value = FoulPrediction(
        confidence=predicted_confidence, decision="offence_no_card"
    )
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


def test_no_offence_does_not_emit_or_consume_cooldown():
    predictor = MagicMock()
    predictor.predict.side_effect = [
        FoulPrediction(confidence=0.99, decision="no_offence"),
        FoulPrediction(confidence=0.9, decision="yellow_card"),
    ]
    detector = FoulDetector("unused", window_size=1, stride=1, predictor=predictor)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    assert detector.update(frame, 0) is None
    candidate = detector.update(frame, 1)

    assert candidate is not None
    assert candidate.decision == "yellow_card"


def test_mvit_no_offence_logits_do_not_generate_candidate(monkeypatch):
    import torch

    from app.foul_detection.predictor import MViTFoulPredictor

    model = MagicMock(
        return_value=(torch.tensor([[10.0, 0.0, 0.0, 0.0]]), torch.zeros((1, 8)), None)
    )
    monkeypatch.setattr(MViTFoulPredictor, "_load_model", lambda self, path: model)
    predictor = MViTFoulPredictor("unused")
    monkeypatch.setattr(predictor, "_preprocess", lambda frames: torch.zeros((1, 1, 3, 16, 2, 2)))

    assert predictor.predict([np.zeros((16, 16, 3), dtype=np.uint8)]) is None


def test_default_checkpoint_matches_multiview_setup():
    from app.constants.paths import FOUL_MODEL_PATH
    from app.multiview.inference import DEFAULT_WEIGHTS_PATH

    assert FOUL_MODEL_PATH == str(DEFAULT_WEIGHTS_PATH)
