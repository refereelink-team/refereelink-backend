"""Async foul classification worker — never blocks the main inference loop."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.events.engine import FoulEventAdapter
from app.state.models import GameEvent

logger = logging.getLogger(__name__)


@dataclass
class FoulJob:
    event_id: str
    frame_id: int
    timestamp: float
    field_x: Optional[float]
    field_y: Optional[float]
    involved_track_ids: list[int]
    label_a: str
    label_b: str
    bbox_union: Optional[tuple[float, float, float, float]]
    frames: np.ndarray  # (T, H, W, 3) BGR uint8, already cropped/padded


@dataclass
class AsyncFoulWorker:
    """Background clip classifier fed by geometry contact triggers."""

    checkpoint_path: str
    device: str = "cpu"
    window_size: int = 24
    max_queue: int = 4
    confidence_threshold: float = 0.48
    input_fps: float = 25.0
    target_fps: float = 17.0
    roi_padding: float = 0.35

    _frame_buffer: deque = field(default_factory=lambda: deque(maxlen=48), init=False)
    _job_queue: queue.Queue = field(default_factory=queue.Queue, init=False)
    _result_queue: queue.Queue = field(default_factory=queue.Queue, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _model: Any = field(default=None, init=False)
    _model_error: Optional[str] = field(default=None, init=False)
    _adapter: FoulEventAdapter = field(init=False)
    inference_count: int = field(default=0, init=False)
    last_latency_ms: float = field(default=0.0, init=False)
    classify_latency_acc_ms: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self._adapter = FoulEventAdapter(confidence_threshold=self.confidence_threshold)
        self._frame_buffer = deque(maxlen=max(self.window_size * 2, 48))
        self._job_queue = queue.Queue(maxsize=self.max_queue)
        self._result_queue = queue.Queue()

    @property
    def queue_length(self) -> int:
        return self._job_queue.qsize()

    @property
    def model_available(self) -> bool:
        return self._model is not None and self._model_error is None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="foul-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._job_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def push_frame(self, frame: np.ndarray) -> None:
        """Keep a short rolling buffer of undistorted frames for clip crops."""
        self._frame_buffer.append(frame.copy())

    def enqueue_from_event(self, event: GameEvent) -> bool:
        """Snapshot recent frames around a geometry foul candidate."""
        if len(self._frame_buffer) < max(4, self.window_size // 2):
            return False
        frames = list(self._frame_buffer)[-self.window_size :]
        bbox = event.evidence.get("bbox_union")
        cropped = [_crop_roi(f, bbox, self.roi_padding) for f in frames]
        try:
            stacked = np.stack(cropped, axis=0)
        except ValueError:
            return False
        foul_details = event.foul_details or {}
        job = FoulJob(
            event_id=event.id,
            frame_id=event.frame_id,
            timestamp=event.timestamp,
            field_x=event.field_x,
            field_y=event.field_y,
            involved_track_ids=list(event.involved_track_ids),
            label_a=str(foul_details.get("label_a", "T?")),
            label_b=str(foul_details.get("label_b", "T?")),
            bbox_union=bbox if isinstance(bbox, tuple) else None,
            frames=stacked,
        )
        try:
            self._job_queue.put_nowait(job)
            return True
        except queue.Full:
            logger.debug("Foul worker queue full; dropping classification job")
            return False

    def drain_results(self) -> list[GameEvent]:
        events: list[GameEvent] = []
        while True:
            try:
                item = self._result_queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                events.append(item)
        return events

    def _ensure_model(self) -> bool:
        if self._model is not None:
            return True
        if self._model_error is not None:
            return False
        try:
            from app.foul_detection.model import load_mvfoul_model

            self._model = load_mvfoul_model(self.checkpoint_path, device=self.device)
            return True
        except Exception as exc:
            self._model_error = str(exc)
            logger.warning("Foul classifier unavailable: %s", exc)
            return False

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._job_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                break
            if not isinstance(job, FoulJob):
                continue
            if not self._ensure_model():
                continue
            started = time.monotonic()
            try:
                from app.foul_detection.model import predict_foul_from_frames

                prediction = predict_foul_from_frames(
                    job.frames,
                    self._model,
                    input_fps=self.input_fps,
                    target_fps=self.target_fps,
                    device=self.device,
                )
            except Exception as exc:
                logger.warning("Foul classification failed: %s", exc)
                continue
            latency_ms = (time.monotonic() - started) * 1000.0
            self.inference_count += 1
            self.last_latency_ms = latency_ms
            self.classify_latency_acc_ms += latency_ms
            event = self._adapter.update(
                prediction,
                frame_id=job.frame_id,
                timestamp=job.timestamp,
                field_xy=(job.field_x, job.field_y)
                if job.field_x is not None and job.field_y is not None
                else None,
                involved_track_ids=job.involved_track_ids,
                label_a=job.label_a,
                label_b=job.label_b,
                source="mvfoul",
                parent_event_id=job.event_id,
            )
            if event is not None:
                self._result_queue.put(event)


def _crop_roi(
    frame: np.ndarray,
    bbox: Any,
    padding: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        # Downscale whole frame to keep async cost bounded.
        scale = 320 / max(h, w)
        if scale >= 1.0:
            return frame
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        import cv2

        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x1, y1, x2, y2 = map(float, bbox)
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * padding, bh * padding
    x1 = max(0, int(x1 - pad_x))
    y1 = max(0, int(y1 - pad_y))
    x2 = min(w, int(x2 + pad_x))
    y2 = min(h, int(y2 + pad_y))
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return frame
    crop = frame[y1:y2, x1:x2]
    import cv2

    return cv2.resize(crop, (224, 224), interpolation=cv2.INTER_AREA)
