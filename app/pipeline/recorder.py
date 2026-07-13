from __future__ import annotations

import logging
import threading
from typing import Optional

import cv2
import numpy as np
import supervision as sv

from app.state.models import SourceStatus
from app.state.store import StateStore

logger = logging.getLogger(__name__)


class VideoRecorder:
    def __init__(
        self,
        target_path: str,
        store: StateStore,
        fps: float = 25.0,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        self._target_path = target_path
        self._sink: Optional[sv.VideoSink] = None
        self._store = store
        self._fps = fps
        self._width = width
        self._height = height
        self._lock = threading.Lock()
        self._active = False

    def start(self) -> None:
        with self._lock:
            if self._active:
                return
            video_info = sv.VideoInfo(
                width=self._width,
                height=self._height,
                fps=int(self._fps),
                total_frames=-1,
            )
            self._sink = sv.VideoSink(self._target_path, video_info)
            self._active = True
            logger.info("VideoRecorder started: %s", self._target_path)

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                return
            if self._sink is not None:
                self._sink.__exit__(None, None, None)
                self._sink = None
            self._active = False
            logger.info("VideoRecorder stopped")

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if self._active and self._sink is not None:
                self._sink.write_frame(frame)

    @property
    def active(self) -> bool:
        return self._active
