"""
FoulDetector — rolling-window adapter that bridges fouls_far into the app pipeline.

Injects the repo root onto sys.path so ``import offside`` resolves without
requiring fouls_far to be installed as a package.
"""

import sys
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# sys.path injection: repo root contains fouls_far/offside/
# This file lives at  app/foul_detection/detector.py
# Repo root is therefore three levels up.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from offside.foul_model import (  # noqa: E402
    FoulPrediction,
    load_mvfoul_model,
    predict_foul_from_frames,
)

DEFAULT_WINDOW_SIZE: int = 24
DEFAULT_STRIDE: int = 8
DEFAULT_INPUT_FPS: float = 25.0
DEFAULT_TARGET_FPS: float = 17.0


class FoulDetector:
    """
    Rolling-window foul detector.

    Maintains a fixed-length frame buffer.  Every ``stride`` frames — once the
    buffer is full — it stacks the buffered frames into a (T, H, W, 3) array
    and runs MVFoul inference.  The most-recent prediction is cached and
    returned on every subsequent call until the next inference window fires.

    Returns ``None`` during the initial warmup period (fewer than
    ``window_size`` frames have been fed in).
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cpu',
        window_size: int = DEFAULT_WINDOW_SIZE,
        stride: int = DEFAULT_STRIDE,
        input_fps: float = DEFAULT_INPUT_FPS,
        target_fps: float = DEFAULT_TARGET_FPS,
    ) -> None:
        """
        Load the MVFoul model and initialise the frame buffer.

        Args:
            checkpoint_path: Path to the ``.pth.tar`` checkpoint file.
            device: Torch device string (``'cpu'``, ``'cuda'``, ``'mps'``, …).
            window_size: Number of frames in each inference window.
            stride: How many frames to advance before the next inference run.
            input_fps: Frame rate of the source video feed.
            target_fps: Frame rate the MVFoul model expects after resampling.
        """
        self._model = load_mvfoul_model(checkpoint_path, device=device)
        self._device = device
        self._input_fps = input_fps
        self._target_fps = target_fps
        self._stride = stride
        # deque auto-discards oldest frame once maxlen is reached
        self._buffer: deque = deque(maxlen=window_size)
        self._frames_since_inference: int = 0
        self._latest_prediction: Optional[FoulPrediction] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> Optional[FoulPrediction]:
        """
        Append *frame* to the rolling buffer and potentially run inference.

        Args:
            frame: BGR ``uint8`` array of shape ``(H, W, 3)``.

        Returns:
            The most-recent :class:`FoulPrediction`, or ``None`` if the buffer
            has not yet accumulated ``window_size`` frames.
        """
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
        """Return the most-recent prediction without triggering a stride check."""
        return self._latest_prediction
