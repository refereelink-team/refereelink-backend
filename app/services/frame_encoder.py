"""Single-writer JPEG cache for browser video delivery."""

from __future__ import annotations

import threading
import time
from typing import Optional

import cv2
import numpy as np


class LatestJpegFrame:
    """Encode each processed frame once and serve the latest JPEG to clients."""

    def __init__(self, quality: int = 75) -> None:
        self.quality = int(np.clip(quality, 30, 100))
        self._lock = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self.frames_encoded = 0
        self.encode_time_ms = 0.0

    def update(self, frame: np.ndarray) -> bool:
        start = time.perf_counter()
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.quality],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if not ok:
            return False
        with self._lock:
            self._jpeg = encoded.tobytes()
            self.frames_encoded += 1
            self.encode_time_ms += elapsed_ms
        return True

    @property
    def latest(self) -> Optional[bytes]:
        with self._lock:
            return self._jpeg

    @property
    def average_encode_time_ms(self) -> float:
        with self._lock:
            return self.encode_time_ms / max(self.frames_encoded, 1)

    def clear(self) -> None:
        with self._lock:
            self._jpeg = None
