from __future__ import annotations

import logging
import threading
from collections import deque
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class PipelineMode(str, Enum):
    REALTIME = "realtime"
    OFFLINE = "offline"


DEFAULT_BUFFER_SIZE = 4


class BoundedFrameBuffer:
    def __init__(
        self,
        maxsize: int = DEFAULT_BUFFER_SIZE,
        mode: PipelineMode = PipelineMode.REALTIME,
    ) -> None:
        self._maxsize = maxsize
        self._mode = mode
        self._buffer: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._dropped_frames = 0
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._closed = False

    def put(self, frame: np.ndarray) -> bool:
        with self._lock:
            if self._closed:
                return False

            if self._mode == PipelineMode.REALTIME:
                while len(self._buffer) >= self._maxsize:
                    self._buffer.popleft()
                    self._dropped_frames += 1
                self._buffer.append(frame)
            else:
                while len(self._buffer) >= self._maxsize and not self._closed:
                    self._not_full.wait()
                if self._closed:
                    return False
                self._buffer.append(frame)
            self._not_empty.notify_all()
            return True

    def get(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        with self._lock:
            if self._closed:
                return None

            if self._mode == PipelineMode.REALTIME:
                if not self._buffer:
                    return None
                result = self._buffer[-1]
                self._buffer.clear()
                self._not_full.notify_all()
                return result
            else:
                while not self._buffer and not self._closed:
                    self._not_empty.wait(timeout=timeout)
                if self._closed and not self._buffer:
                    return None
                result = self._buffer.popleft()
                self._not_full.notify_all()
                return result

    def get_latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if not self._buffer:
                return None
            result = self._buffer[-1]
            self._buffer.clear()
            self._not_full.notify_all()
            return result

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_frames

    @property
    def queue_length(self) -> int:
        with self._lock:
            return len(self._buffer)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def __len__(self) -> int:
        return self.queue_length
