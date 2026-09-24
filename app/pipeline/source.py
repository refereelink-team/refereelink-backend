from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from app.field_ingest.frames import CapturedFrame, LatestFrameQueue, utc_timestamp
from app.state.models import SourceStatus
from app.state.store import StateStore

logger = logging.getLogger(__name__)

DEFAULT_RTSP_TIMEOUT_SEC = 5.0
DEFAULT_RECONNECT_DELAY_SEC = 2.0
MAX_RECONNECT_ATTEMPTS = 10


class VideoSource(ABC):
    def __init__(self, store: Optional[StateStore] = None) -> None:
        self._store = store
        self._fps: float = 0.0
        self._frame_count = 0
        self._start_time: Optional[float] = None

    @abstractmethod
    def read(self) -> tuple[bool, Optional[np.ndarray]]: ...

    @abstractmethod
    def release(self) -> None: ...

    @abstractmethod
    def is_opened(self) -> bool: ...

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def _increment_frame(self) -> None:
        self._frame_count += 1
        if self._start_time is None:
            self._start_time = time.monotonic()

    def _update_source_status(self, status: SourceStatus) -> None:
        if self._store is not None:
            self._store.source_status = status

    def capture_timestamp_ms(self) -> float:
        return time.time() * 1000

    def read_packet(self) -> tuple[bool, CapturedFrame | None]:
        ret, frame = self.read()
        if not ret or frame is None:
            return False, None
        return True, CapturedFrame(
            image=frame,
            session_id="",
            stream_epoch=0,
            source_frame_id=self.frame_count,
            t_us=None,
            transport_pts90k=None,
            capture_unix_us=int(time.time() * 1_000_000),
            camera_motion=None,
            pose_missing_reason=None,
            backend_received_at=utc_timestamp(),
        )


class LocalFileSource(VideoSource):
    def __init__(
        self,
        file_path: str,
        store: Optional[StateStore] = None,
    ) -> None:
        super().__init__(store)
        resolved = str(Path(file_path).resolve())
        self._cap = cv2.VideoCapture(resolved)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {resolved}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self._fps <= 0:
            self._fps = 25.0
        self._frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._finished = False
        self._update_source_status(SourceStatus.CONNECTED)
        logger.info(
            "LocalFileSource opened: %s (%.1f fps, %dx%d)",
            resolved,
            self._fps,
            self._frame_width,
            self._frame_height,
        )

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._finished:
            return False, None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            self._finished = True
            self._update_source_status(SourceStatus.DISCONNECTED)
            return False, None
        self._increment_frame()
        return True, frame

    def is_opened(self) -> bool:
        return self._cap.isOpened() and not self._finished

    def release(self) -> None:
        self._cap.release()
        self._update_source_status(SourceStatus.DISCONNECTED)

    @property
    def frame_width(self) -> int:
        return self._frame_width

    @property
    def frame_height(self) -> int:
        return self._frame_height


class RTSPSource(VideoSource):
    def __init__(
        self,
        rtsp_url: str,
        store: Optional[StateStore] = None,
        buffer_size: int = 1,
        low_latency: bool = True,
        reconnect_delay: float = DEFAULT_RECONNECT_DELAY_SEC,
        timeout: float = DEFAULT_RTSP_TIMEOUT_SEC,
    ) -> None:
        super().__init__(store)
        self._rtsp_url = rtsp_url
        self._buffer_size = buffer_size
        self._low_latency = low_latency
        self._reconnect_delay = reconnect_delay
        self._timeout = timeout
        self._reconnect_attempts = 0
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time = 0.0
        self._fps = 25.0
        self._lock = threading.Lock()
        # Defer actual capture opening; the first read() will trigger it.
        # This keeps the constructor non-blocking and lets callers inspect
        # the source without paying the network round-trip.

    def _build_gstreamer_pipeline(self) -> str:
        return (
            f"rtspsrc location={self._rtsp_url} latency=0 buffer-mode=auto ! "
            "rtpjitterbuffer ! "
            "decodebin ! videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def _open_capture(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()

            # Try GStreamer first, then fall back to FFmpeg
            try:
                pipeline = self._build_gstreamer_pipeline()
                self._cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if not self._cap.isOpened():
                    raise RuntimeError("GStreamer pipeline failed")
                logger.info("RTSPSource using GStreamer backend")
            except Exception:
                logger.info("RTSPSource falling back to FFmpeg backend")
                self._cap = cv2.VideoCapture(self._rtsp_url, cv2.CAP_FFMPEG)

            if self._low_latency:
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self._buffer_size)
                # FFmpeg low-latency flags
                import os

                os.environ.setdefault(
                    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                    "rtsp_transport;tcp|analyzeduration;100000|"
                    "probesize;32768|fflags;nobuffer|"
                    "flags;low_delay|max_delay;0",
                )

            self._fps = self._cap.get(cv2.CAP_PROP_FPS)
            if self._fps <= 0:
                self._fps = 25.0
            self._last_frame_time = time.monotonic()

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        with self._lock:
            if self._cap is None:
                try:
                    self._open_capture()
                except Exception:
                    self._attempt_reconnect()
                    return False, None

            if not self._cap.isOpened():
                self._attempt_reconnect()
                return False, None

            start = time.monotonic()
            ret, frame = self._cap.read()

            if not ret or frame is None:
                elapsed = time.monotonic() - start
                if elapsed >= self._timeout:
                    logger.warning("RTSP read timeout after %.1fs", elapsed)
                self._update_source_status(SourceStatus.DISCONNECTED)
                self._attempt_reconnect()
                return False, None

            self._last_frame_time = time.monotonic()
            self._update_source_status(SourceStatus.CONNECTED)
            self._reconnect_attempts = 0
            self._increment_frame()
            return True, frame

    def _attempt_reconnect(self) -> None:
        if self._reconnect_attempts >= MAX_RECONNECT_ATTEMPTS:
            self._update_source_status(SourceStatus.ERROR)
            logger.error("RTSP reconnect failed after %d attempts", MAX_RECONNECT_ATTEMPTS)
            return

        self._reconnect_attempts += 1
        self._update_source_status(SourceStatus.RECONNECTING)
        logger.warning(
            "RTSP reconnecting (attempt %d/%d)...", self._reconnect_attempts, MAX_RECONNECT_ATTEMPTS
        )
        time.sleep(self._reconnect_delay)
        self._open_capture()

    def is_opened(self) -> bool:
        with self._lock:
            return self._cap is not None and self._cap.isOpened()

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
        self._update_source_status(SourceStatus.DISCONNECTED)

    @property
    def last_frame_time(self) -> float:
        return self._last_frame_time

    @property
    def reconnect_attempts(self) -> int:
        return self._reconnect_attempts


class FieldIngestSource(VideoSource):
    """VideoSource adapter for one explicitly bound live field epoch."""

    def __init__(
        self,
        queue: LatestFrameQueue,
        *,
        session_id: str,
        stream_epoch: int,
        fps: float,
        store: Optional[StateStore] = None,
        release_callback: Optional[Callable[[], None]] = None,
        metrics_callback: Optional[Callable[[], dict]] = None,
    ) -> None:
        super().__init__(store)
        self._queue = queue
        self.session_id = session_id
        self.stream_epoch = stream_epoch
        self._fps = fps
        self._release_callback = release_callback
        self._metrics_callback = metrics_callback
        self._released = False
        self._last_packet: CapturedFrame | None = None
        self._update_source_status(SourceStatus.RECONNECTING)

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        ret, packet = self.read_packet()
        return ret, packet.image if packet is not None else None

    def read_packet(self) -> tuple[bool, CapturedFrame | None]:
        if self._released:
            return False, None
        packet = self._queue.get(timeout=0.5)
        if packet is None:
            if not self.is_opened():
                self._update_source_status(SourceStatus.DISCONNECTED)
            return False, None
        self._last_packet = packet
        self._increment_frame()
        self._update_source_status(SourceStatus.CONNECTED)
        return True, packet

    def is_opened(self) -> bool:
        # The receiver leaves "listening" before the joiner flush is queued.
        # EOF is the queue close that follows that flush, so tail frames are
        # still readable after the listener itself has stopped.
        return not self._released and not self._queue.closed

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self._release_callback is not None:
            self._release_callback()
        else:
            self._queue.close()
        self._update_source_status(SourceStatus.DISCONNECTED)

    def capture_timestamp_ms(self) -> float:
        if self._last_packet is not None and self._last_packet.capture_unix_us is not None:
            return self._last_packet.capture_unix_us / 1000.0
        return super().capture_timestamp_ms()

    @property
    def stream_metrics(self) -> dict:
        if self._metrics_callback is None:
            return {}
        try:
            return self._metrics_callback()
        except KeyError:
            # The service removes the epoch as soon as the pipeline releases
            # it.  Metrics emission may race with that cleanup; retain the
            # source identity without resurrecting the runtime.
            return {
                "session_id": self.session_id,
                "stream_epoch": self.stream_epoch,
                "queue_closed": True,
            }


def create_video_source(
    source_path: str,
    store: Optional[StateStore] = None,
) -> VideoSource:
    if source_path.startswith("rtsp://") or source_path.startswith("rtsps://"):
        return RTSPSource(source_path, store=store)
    return LocalFileSource(source_path, store=store)
