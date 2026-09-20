from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class DecodedVideoFrame:
    image: np.ndarray
    transport_pts90k: int | None
    decode_index: int
    received_at: str


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    session_id: str
    stream_epoch: int
    source_frame_id: int | None
    t_us: int | None
    transport_pts90k: int | None
    capture_unix_us: int | None
    camera_motion: dict[str, Any] | None
    pose_missing_reason: str | None
    backend_received_at: str


class LatestFrameQueue:
    """A bounded queue for live inference which always favors the newest frame."""

    def __init__(self, maxsize: int = 2) -> None:
        self._maxsize = max(1, maxsize)
        self._items: deque[CapturedFrame] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._dropped = 0

    def put(self, frame: CapturedFrame) -> bool:
        with self._condition:
            if self._closed:
                return False
            while len(self._items) >= self._maxsize:
                self._items.popleft()
                self._dropped += 1
            self._items.append(frame)
            self._condition.notify()
            return True

    def get(self, timeout: float | None = None) -> CapturedFrame | None:
        with self._condition:
            if not self._items and not self._closed:
                self._condition.wait(timeout=timeout)
            if not self._items:
                return None
            frame = self._items[-1]
            self._dropped += max(0, len(self._items) - 1)
            self._items.clear()
            return frame

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    @property
    def dropped(self) -> int:
        with self._condition:
            return self._dropped

    @property
    def length(self) -> int:
        with self._condition:
            return len(self._items)

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed


class FrameJoiner:
    """Associate decoded video with WSS frame and Core Motion metadata.

    Video and telemetry arrive on independent transports.  The joiner matches
    by transport PTS first and only uses a bounded prior-PTS fallback.  It
    never treats network arrival order as a frame identity.
    """

    def __init__(
        self,
        session_id: str,
        stream_epoch: int,
        *,
        max_samples: int = 512,
        max_pending: int = 4,
        max_wait_ms: int = 50,
        max_pts_delta90k: int = 4_500,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.session_id = session_id
        self.stream_epoch = stream_epoch
        self._max_samples = max(1, max_samples)
        self._frame_by_pts: dict[int, dict[str, Any]] = {}
        self._frame_order: deque[int] = deque()
        self._expired_pts: set[int] = set()
        self._expired_pts_order: deque[int] = deque()
        self._motion_samples: deque[tuple[int, dict[str, Any]]] = deque(
            maxlen=self._max_samples
        )
        self._pending: deque[tuple[DecodedVideoFrame, float]] = deque(maxlen=max_pending)
        self._max_wait_s = max_wait_ms / 1000.0
        self._max_pts_delta90k = max_pts_delta90k
        self._clock = clock
        self.missing_frame_metadata = 0
        self.missing_motion = 0

    def ingest_item(self, item: dict[str, Any], received_at: str) -> list[CapturedFrame]:
        ready = self._flush_expired(received_at)
        item_type = item.get("type")
        if item_type in {"camera_motion", "motion"}:
            t_us = _int_value(item.get("t_us", item.get("tUs")))
            if t_us is not None:
                self._motion_samples.append((t_us, item))
            return ready

        if item_type != "frame":
            return ready
        pts = _int_value(item.get("transport_pts90k"))
        if pts is None:
            return ready
        if pts in self._expired_pts:
            # Metadata that arrives after the 50 ms join deadline must not be
            # reused for a later decoded frame with the same PTS.
            self._expired_pts.discard(pts)
            return ready
        self._frame_by_pts[pts] = item
        self._frame_order.append(pts)
        while len(self._frame_order) > self._max_samples:
            oldest = self._frame_order.popleft()
            self._frame_by_pts.pop(oldest, None)
        remaining: deque[tuple[DecodedVideoFrame, float]] = deque(maxlen=self._pending.maxlen)
        for decoded, deadline in self._pending:
            match = self._take_metadata(decoded.transport_pts90k)
            if match is None:
                remaining.append((decoded, deadline))
            else:
                ready.append(self._build(decoded, match, received_at))
        self._pending = remaining
        ready.extend(self._flush_expired(received_at))
        return ready

    def ingest_decoded(self, decoded: DecodedVideoFrame) -> list[CapturedFrame]:
        ready = self._flush_expired(decoded.received_at)
        match = self._take_metadata(decoded.transport_pts90k)
        if match is not None:
            ready.append(self._build(decoded, match, decoded.received_at))
            return ready
        if len(self._pending) == self._pending.maxlen:
            expired, _ = self._pending.popleft()
            ready.append(self._build(expired, None, expired.received_at))
        self._pending.append((decoded, self._clock() + self._max_wait_s))
        return ready

    def flush(self) -> list[CapturedFrame]:
        ready = [self._build(decoded, None, decoded.received_at) for decoded, _ in self._pending]
        self._pending.clear()
        return ready

    def _flush_expired(self, received_at: str) -> list[CapturedFrame]:
        now = self._clock()
        ready: list[CapturedFrame] = []
        while self._pending and self._pending[0][1] <= now:
            decoded, _ = self._pending.popleft()
            if decoded.transport_pts90k is not None:
                self._expired_pts.add(decoded.transport_pts90k)
                self._expired_pts_order.append(decoded.transport_pts90k)
                while len(self._expired_pts_order) > self._max_samples:
                    self._expired_pts.discard(self._expired_pts_order.popleft())
            ready.append(self._build(decoded, None, received_at))
        return ready

    def _take_metadata(self, pts: int | None) -> dict[str, Any] | None:
        if pts is None:
            return None
        exact = self._frame_by_pts.pop(pts, None)
        if exact is not None:
            return exact
        candidates = [
            (candidate_pts, self._frame_by_pts[candidate_pts])
            for candidate_pts in self._frame_by_pts
            if candidate_pts <= pts and pts - candidate_pts <= self._max_pts_delta90k
        ]
        if not candidates:
            return None
        candidate_pts, candidate = max(candidates, key=lambda pair: pair[0])
        self._frame_by_pts.pop(candidate_pts, None)
        return candidate

    def _build(
        self,
        decoded: DecodedVideoFrame,
        metadata: dict[str, Any] | None,
        received_at: str,
    ) -> CapturedFrame:
        has_metadata = bool(metadata)
        metadata = metadata or {}
        t_us = _int_value(metadata.get("t_us", metadata.get("tUs")))
        motion = self._motion_for(t_us)
        reason: str | None = None
        if not has_metadata:
            reason = "frame_metadata_timeout"
            self.missing_frame_metadata += 1
        elif motion is None:
            reason = "no_motion_within_50ms"
            self.missing_motion += 1
        return CapturedFrame(
            image=decoded.image,
            session_id=self.session_id,
            stream_epoch=self.stream_epoch,
            source_frame_id=_int_value(metadata.get("frame_id", metadata.get("frameId"))),
            t_us=t_us,
            transport_pts90k=decoded.transport_pts90k,
            capture_unix_us=_int_value(
                metadata.get("capture_unix_us", metadata.get("captureUnixUs"))
            ),
            camera_motion=motion,
            pose_missing_reason=reason,
            backend_received_at=received_at,
        )

    def _motion_for(self, t_us: int | None) -> dict[str, Any] | None:
        if t_us is None:
            return None
        eligible = [sample for sample in self._motion_samples if sample[0] <= t_us]
        if not eligible:
            return None
        sample_t_us, sample = eligible[-1]
        if t_us - sample_t_us > 50_000:
            return None
        return sample


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _int_value(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
