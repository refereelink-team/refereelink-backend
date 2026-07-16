from __future__ import annotations

import os
import tempfile
import threading
import time

import cv2
import numpy as np
import pytest

from app.pipeline.source import (
    RTSPSource,
    VideoSource,
    LocalFileSource,
    create_video_source,
)
from app.state.models import SourceStatus
from app.state.store import StateStore


def _create_test_video(path: str, num_frames: int = 30, fps: int = 10,
                       width: int = 64, height: int = 48) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        frame = np.full((height, width, 3), (i * 8) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_local_file_source_reads_frames():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.mp4")
        _create_test_video(path, num_frames=5)
        src = LocalFileSource(path)
        try:
            assert src.is_opened()
            count = 0
            while True:
                ret, frame = src.read()
                if not ret or frame is None:
                    break
                count += 1
            assert count == 5
            assert src.frame_count == 5
        finally:
            src.release()


def test_local_file_source_updates_store_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.mp4")
        _create_test_video(path, num_frames=2)
        store = StateStore()
        src = LocalFileSource(path, store=store)
        try:
            assert store.source_status == SourceStatus.CONNECTED
            src.read()
            src.read()
            ret, _ = src.read()
            assert not ret
            assert store.source_status == SourceStatus.DISCONNECTED
        finally:
            src.release()


def test_local_file_source_invalid_path_raises():
    with pytest.raises(FileNotFoundError):
        LocalFileSource("/nonexistent/path/to/video.mp4")


def test_create_video_source_routes_to_rtsp():
    src = create_video_source("rtsp://192.168.1.100:554/stream")
    try:
        assert isinstance(src, RTSPSource)
        assert not src.is_opened()
    finally:
        src.release()


def test_create_video_source_routes_to_local():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.mp4")
        _create_test_video(path, num_frames=1)
        src = create_video_source(path)
        try:
            assert isinstance(src, LocalFileSource)
        finally:
            src.release()


def test_rtsp_source_constructor_does_not_block():
    """Verify the constructor is non-blocking so the source can be inspected
    without paying the network round-trip."""
    src = RTSPSource(
        "rtsp://127.0.0.1:1/nonexistent",
        reconnect_delay=0.01,
        timeout=0.1,
    )
    try:
        assert src.reconnect_attempts == 0
        assert src.fps > 0
    finally:
        src.release()


def test_rtsp_source_tracks_reconnect_attempts():
    """Verify the reconnect state is reported through the source status."""
    src = RTSPSource(
        "rtsp://127.0.0.1:1/nonexistent",
        reconnect_delay=0.01,
        timeout=0.1,
    )
    try:
        store = StateStore()
        src._store = store
        # Drive reconnect directly without blocking on cv2.VideoCapture.
        src._reconnect_attempts = 0
        src._update_source_status(SourceStatus.RECONNECTING)
        assert store.source_status == SourceStatus.RECONNECTING
        src._update_source_status(SourceStatus.CONNECTED)
        assert store.source_status == SourceStatus.CONNECTED
    finally:
        src.release()


def test_video_source_capture_timestamp_ms():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.mp4")
        _create_test_video(path, num_frames=1)
        src = LocalFileSource(path)
        try:
            ts = src.capture_timestamp_ms()
            assert ts > 0
        finally:
            src.release()
