"""Visible candidate notifications for standalone video and radar modes."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from app.foul_detection.types import FoulPrediction


class FoulCandidateOverlay:
    def __init__(self, fps: float, duration_seconds: float = 2.0) -> None:
        self._hold_frames = max(1, round(max(fps, 1.0) * duration_seconds))
        self._prediction: Optional[FoulPrediction] = None
        self._expires_frame = -1

    def annotate(
        self, frame: np.ndarray, prediction: Optional[FoulPrediction], frame_index: int
    ) -> np.ndarray:
        if prediction is not None and prediction.decision != "no_offence":
            self._prediction = prediction
            self._expires_frame = frame_index + self._hold_frames
        if self._prediction is None or frame_index >= self._expires_frame:
            return frame

        annotated = frame.copy()
        height, width = annotated.shape[:2]
        font_scale = max(0.3, min(0.7, width / 900.0))
        action = self._prediction.action or "Unknown"
        text = f"FOUL CANDIDATE: {action} ({self._prediction.confidence:.0%})"
        text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(
            annotated,
            (8, 8),
            (min(width - 1, text_size[0] + 24), min(height - 1, text_size[1] + baseline + 24)),
            (20, 20, 150),
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (16, 16 + text_size[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return annotated
