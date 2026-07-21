from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.state.store import StateStore

logger = logging.getLogger(__name__)


class VideoRecorder:
    def __init__(
        self,
        target_path: str,
        store: StateStore,
        fps: float = 25.0,
    ) -> None:
        self._target_path = str(Path(target_path).expanduser().resolve())
        self._writer: Optional[cv2.VideoWriter] = None
        self._store = store
        self._fps = max(float(fps), 1.0)
        self._lock = threading.Lock()
        self._active = False
        self._frames_written = 0

    def start(self) -> None:
        with self._lock:
            if self._active:
                return
            Path(self._target_path).parent.mkdir(parents=True, exist_ok=True)
            self._frames_written = 0
            self._active = True
            logger.info("VideoRecorder started: %s", self._target_path)

    def stop(self) -> None:
        with self._lock:
            if not self._active:
                return
            if self._writer is not None:
                self._writer.release()
                self._writer = None
            self._active = False
            logger.info(
                "VideoRecorder stopped: %s (%d frames)",
                self._target_path,
                self._frames_written,
            )

    def write(self, frame: np.ndarray) -> None:
        with self._lock:
            if not self._active:
                return
            image = np.asarray(frame)
            if image.ndim != 3 or image.shape[2] < 3 or image.shape[0] <= 0 or image.shape[1] <= 0:
                logger.warning("VideoRecorder skipped invalid frame with shape=%s", image.shape)
                return
            image = image[..., :3]
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            if self._writer is None:
                height, width = image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    self._target_path,
                    fourcc,
                    self._fps,
                    (int(width), int(height)),
                )
                if not writer.isOpened():
                    logger.error("VideoRecorder could not open output: %s", self._target_path)
                    writer.release()
                    self._active = False
                    return
                self._writer = writer
            self._writer.write(image)
            self._frames_written += 1

    @property
    def active(self) -> bool:
        return self._active

    @property
    def target_path(self) -> str:
        return self._target_path

    @property
    def frames_written(self) -> int:
        return self._frames_written
