"""
FoulDetector — rolling-window adapter for optional MVFoul clip classification.

Uses :mod:`app.foul_detection.model` so missing ``fouls_far`` / checkpoint does
not break imports. Prefer :class:`AsyncFoulWorker` for live pipelines.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from app.foul_detection.model import (
    FoulPrediction,
    load_mvfoul_model,
    predict_foul_from_frames,
)

DEFAULT_WINDOW_SIZE: int = 24
DEFAULT_STRIDE: int = 8
DEFAULT_INPUT_FPS: float = 25.0
DEFAULT_TARGET_FPS: float = 17.0


class FoulDetector:
    """Rolling-window foul detector (synchronous; prefer async worker live)."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        input_fps: float = DEFAULT_INPUT_FPS,
        target_fps: float = DEFAULT_TARGET_FPS,
    ) -> None:
        self._model = load_mvfoul_model(checkpoint_path, device=device)
        self._device = device
        self._input_fps = input_fps
        self._target_fps = target_fps
        self._stride = stride
        self._buffer: deque = deque(maxlen=window_size)
        self._frames_since_inference: int = 0
        self._latest_prediction: Optional[FoulPrediction] = None

    def update(self, frame: np.ndarray) -> Optional[FoulPrediction]:
        self._buffer.append(frame)
        self._frames_since_inference += 1

        if (
            len(self._buffer) == self._buffer.maxlen
            and self._frames_since_inference >= self._stride
        ):
            frames_array = np.stack(list(self._buffer), axis=0)
            self._latest_prediction = predict_foul_from_frames(
                frames_array,
                self._model,
                input_fps=self._input_fps,
                target_fps=self._target_fps,
                device=self._device,
            )
            self._frames_since_inference = 0

        return self._latest_prediction

    @property
    def latest_prediction(self) -> Optional[FoulPrediction]:
        return self._latest_prediction
