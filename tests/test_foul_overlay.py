from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from app.foul_detection.overlay import FoulCandidateOverlay
from app.foul_detection.types import FoulPrediction
from app.modes.foul_detection import run_foul_detection


def _candidate() -> FoulPrediction:
    return FoulPrediction(confidence=0.9, decision="yellow_card", action="Pushing")


def test_candidate_is_visible_between_inferences_then_expires() -> None:
    overlay = FoulCandidateOverlay(fps=25)
    frame = np.zeros((200, 320, 3), dtype=np.uint8)

    assert np.any(overlay.annotate(frame, _candidate(), 1))
    assert np.any(overlay.annotate(frame, None, 50))
    assert not np.any(overlay.annotate(frame, None, 51))
    assert not np.any(frame)


def test_no_offence_does_not_create_overlay() -> None:
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    prediction = FoulPrediction(confidence=0.99, decision="no_offence")

    assert not np.any(FoulCandidateOverlay(fps=25).annotate(frame, prediction, 1))


def test_standalone_foul_mode_exports_visible_candidates(monkeypatch) -> None:
    frame = np.zeros((200, 320, 3), dtype=np.uint8)
    detector = MagicMock()
    detector.update.side_effect = [_candidate(), None]
    monkeypatch.setattr("app.modes.foul_detection.FoulDetector", lambda **kwargs: detector)
    monkeypatch.setattr(
        "app.modes.foul_detection.sv.VideoInfo.from_video_path",
        lambda path: SimpleNamespace(fps=25),
    )
    monkeypatch.setattr(
        "app.modes.foul_detection.sv.get_video_frames_generator",
        lambda **kwargs: iter([frame, frame]),
    )

    outputs = list(run_foul_detection("unused", "cpu", "unused", enable_undistortion=False))

    assert len(outputs) == 2
    assert all(np.any(output) for output in outputs)
    assert not np.any(frame)
