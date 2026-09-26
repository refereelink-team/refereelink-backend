"""Rolling-window foul candidate detector.

The previous implementation depended on `offside.foul_model` from the
external `fouls_far` package, which was never completed and never shipped
with this repository. This version keeps the same public API
(``update(frame, frame_index) -> Optional[FoulPrediction]``) but delegates
inference to an injected :class:`~app.foul_detection.predictor.FoulPredictor`.

Default predictor: :class:`MViTFoulPredictor` (SoccerNet VARS MViT V2 Small,
checkpoint ``assets/weights/14_model.pth.tar``).
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np

from app.foul_detection.predictor import FoulPredictor, MViTFoulPredictor
from app.foul_detection.types import FoulPrediction

DEFAULT_WINDOW_SIZE: int = 24
DEFAULT_STRIDE: int = 8
DEFAULT_INPUT_FPS: float = 25.0
DEFAULT_TARGET_FPS: float = 17.0


class FoulDetector:
    """Rolling-window foul detector.

    Maintains a fixed-length frame buffer. Every ``stride`` frames — once
    the buffer is full — it hands the buffered window to the predictor.
    The most recent prediction is cached and returned when it passes the
    confidence and cooldown filters.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        input_fps: float = DEFAULT_INPUT_FPS,
        target_fps: float = DEFAULT_TARGET_FPS,
        cooldown_frames: int = 25,
        confidence_threshold: float = 0.5,
        predictor: Optional[FoulPredictor] = None,
    ) -> None:
        self._input_fps = input_fps
        self._target_fps = target_fps
        self._stride = stride
        self._buffer: deque = deque(maxlen=window_size)
        self._frames_since_inference: int = 0
        self._latest_prediction: Optional[FoulPrediction] = None
        self._cooldown_frames = cooldown_frames
        self._confidence_threshold = confidence_threshold
        self._last_candidate_frame = -cooldown_frames
        self._predictor = predictor or MViTFoulPredictor(checkpoint_path, device)
        self.inference_count = 0

    def update(self, frame: np.ndarray, frame_index: int) -> Optional[FoulPrediction]:
        """Append a frame, run inference on the stride, apply filters."""
        self._buffer.append(frame)
        self._frames_since_inference += 1

        if (
            len(self._buffer) == self._buffer.maxlen
            and self._frames_since_inference >= self._stride
        ):
            self._latest_prediction = self._predictor.predict(list(self._buffer))
            self.inference_count += 1
            self._frames_since_inference = 0

        prediction = self._latest_prediction
        if prediction is None:
            return None
        if prediction.confidence < self._confidence_threshold:
            return None
        if frame_index - self._last_candidate_frame < self._cooldown_frames:
            return None

        self._last_candidate_frame = frame_index
        return prediction

    @property
    def latest_prediction(self) -> Optional[FoulPrediction]:
        return self._latest_prediction
