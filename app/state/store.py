from __future__ import annotations

import threading
from collections import deque
from typing import Optional

from app.state.models import (
    FrameState,
    GameEvent,
    MetricsSnapshot,
    PipelineConfig,
    SourceStatus,
)
from app.classification.team_calibration.session import TeamCalibrationSession
from app.services.frame_encoder import LatestJpegFrame


MAX_EVENTS = 500
MAX_LOG_LINES = 200


class StateStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()

        self._latest_frame_state: Optional[FrameState] = None
        self._config = PipelineConfig()
        self._events: deque[GameEvent] = deque(maxlen=MAX_EVENTS)
        self._metrics = MetricsSnapshot()
        self._source_status: SourceStatus = SourceStatus.DISCONNECTED
        self._pipeline_running = False
        self._log_lines: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._latest_raw_frame = None
        self._jpeg_frame = LatestJpegFrame()
        self._team_calibration = TeamCalibrationSession()

    @property
    def latest_frame_state(self) -> Optional[FrameState]:
        with self._lock:
            return self._latest_frame_state

    @latest_frame_state.setter
    def latest_frame_state(self, value: FrameState) -> None:
        with self._lock:
            self._latest_frame_state = value

    @property
    def config(self) -> PipelineConfig:
        with self._lock:
            return self._config

    def update_config(self, updates: dict) -> PipelineConfig:
        with self._lock:
            existing = self._config.model_dump()
            existing.update(updates)
            self._config = PipelineConfig(**existing)
            return self._config

    @property
    def events(self) -> list[GameEvent]:
        with self._lock:
            return list(self._events)

    def add_event(self, event: GameEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def metrics(self) -> MetricsSnapshot:
        with self._lock:
            return self._metrics

    @metrics.setter
    def metrics(self, value: MetricsSnapshot) -> None:
        with self._lock:
            self._metrics = value

    @property
    def source_status(self) -> SourceStatus:
        with self._lock:
            return self._source_status

    @source_status.setter
    def source_status(self, value: SourceStatus) -> None:
        with self._lock:
            self._source_status = value

    @property
    def pipeline_running(self) -> bool:
        with self._lock:
            return self._pipeline_running

    @pipeline_running.setter
    def pipeline_running(self, value: bool) -> None:
        with self._lock:
            self._pipeline_running = value

    @property
    def log_lines(self) -> list[str]:
        with self._lock:
            return list(self._log_lines)

    def append_log(self, message: str) -> None:
        with self._lock:
            self._log_lines.append(message)

    def publish_raw_frame(self, frame) -> None:
        """Publish a raw BGR frame and encode it once for all MJPEG clients."""

        self._latest_raw_frame = frame
        self._jpeg_frame.update(frame)

    @property
    def latest_jpeg_frame(self) -> Optional[bytes]:
        return self._jpeg_frame.latest

    @property
    def jpeg_frames_encoded(self) -> int:
        return self._jpeg_frame.frames_encoded

    @property
    def jpeg_encode_latency_ms(self) -> float:
        return self._jpeg_frame.average_encode_time_ms

    @property
    def team_calibration(self) -> TeamCalibrationSession:
        return self._team_calibration

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "frame_state": (
                    self._latest_frame_state.model_dump()
                    if self._latest_frame_state
                    else None
                ),
                "config": self._config.model_dump(),
                "events": [e.model_dump() for e in self._events],
                "metrics": self._metrics.model_dump(),
                "source_status": self._source_status.value,
                "pipeline_running": self._pipeline_running,
                "log_lines": list(self._log_lines),
                "jpeg_frames_encoded": self._jpeg_frame.frames_encoded,
                "jpeg_encode_latency_ms": self._jpeg_frame.average_encode_time_ms,
                "team_calibration": self._team_calibration.snapshot(),
            }
