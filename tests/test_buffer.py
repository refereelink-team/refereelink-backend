from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from app.pipeline.buffer import BoundedFrameBuffer, PipelineMode


def _make_frame(value: int = 0) -> np.ndarray:
    return np.full((10, 10, 3), value, dtype=np.uint8)


def test_buffer_put_and_get_realtime():
    buf = BoundedFrameBuffer(maxsize=3, mode=PipelineMode.REALTIME)
    f1 = _make_frame(1)
    f2 = _make_frame(2)
    f3 = _make_frame(3)
    buf.put(f1)
    buf.put(f2)
    buf.put(f3)
    assert len(buf) == 3
    result = buf.get()
    assert result is not None
    assert np.array_equal(result, f3)
    assert len(buf) == 0


def test_buffer_drops_oldest_in_realtime_mode():
    buf = BoundedFrameBuffer(maxsize=2, mode=PipelineMode.REALTIME)
    buf.put(_make_frame(1))
    buf.put(_make_frame(2))
    buf.put(_make_frame(3))
    assert buf.dropped_frames == 1
    assert len(buf) == 2


def test_buffer_does_not_drop_in_offline_mode():
    buf = BoundedFrameBuffer(maxsize=2, mode=PipelineMode.OFFLINE)
    buf.put(_make_frame(1))
    buf.put(_make_frame(2))
    t = threading.Thread(target=lambda: buf.put(_make_frame(3)))
    t.start()
    time.sleep(0.05)
    assert len(buf) == 2
    buf.get()
    t.join(timeout=1.0)
    assert len(buf) == 2


def test_buffer_get_returns_latest_only_in_realtime():
    buf = BoundedFrameBuffer(maxsize=5, mode=PipelineMode.REALTIME)
    for i in range(5):
        buf.put(_make_frame(i))
    result = buf.get()
    assert result is not None
    assert result[0, 0, 0] == 4
    assert len(buf) == 0


def test_buffer_close_unblocks_getters():
    buf = BoundedFrameBuffer(maxsize=2, mode=PipelineMode.OFFLINE)
    result = []

    def getter():
        result.append(buf.get(timeout=0.5))

    t = threading.Thread(target=getter)
    t.start()
    time.sleep(0.05)
    buf.close()
    t.join(timeout=1.0)
    assert result == [None]


def test_buffer_get_latest():
    buf = BoundedFrameBuffer(maxsize=5, mode=PipelineMode.REALTIME)
    buf.put(_make_frame(1))
    buf.put(_make_frame(2))
    result = buf.get_latest()
    assert result is not None
    assert result[0, 0, 0] == 2
    assert len(buf) == 0
