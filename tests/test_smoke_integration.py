from __future__ import annotations

import os
import threading
import time

import cv2
import numpy as np
import pytest


def _create_synthetic_video(path: str, num_frames: int = 20,
                            width: int = 320, height: int = 240) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (width, height))
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(frame, f"F{i}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)
    writer.release()


def test_pipeline_processes_synthetic_video(tmp_path):
    from app.pipeline.source import LocalFileSource
    from app.pipeline.engine import InferencePipeline
    from app.state.store import StateStore
    from app.state.models import SourceStatus

    video_path = str(tmp_path / "synthetic.mp4")
    _create_synthetic_video(video_path, num_frames=10)

    store = StateStore()
    source = LocalFileSource(video_path, store=store)

    assert store.source_status == SourceStatus.CONNECTED
    assert source.is_opened()

    try:
        store.pipeline_running = True
        ret, frame = source.read()
        assert ret
        assert frame is not None
        assert frame.shape == (240, 320, 3)

        for _ in range(5):
            ret, frame = source.read()
            if not ret:
                break
        assert source.frame_count >= 5
    finally:
        source.release()


def test_state_store_pipeline_integration(tmp_path):
    from app.pipeline.source import LocalFileSource
    from app.state.store import StateStore
    from app.state.models import FrameState, SourceStatus, PlayerState, PlayerRole
    from app.state.events import EventBus

    store = StateStore()
    bus = EventBus()

    events_received: list = []
    bus.subscribe("frame_pushed", lambda fs: events_received.append(fs))

    fs = FrameState(
        frame_id=1,
        players=[PlayerState(
            track_id=7, role=PlayerRole.PLAYER, team_id=0,
            field_x=10.0, field_y=20.0, confidence=0.9,
        )],
    )
    store.latest_frame_state = fs
    bus.emit("frame_pushed", fs=fs)
    assert len(events_received) == 1
    assert events_received[0].frame_id == 1

    assert store.latest_frame_state is not None
    assert store.latest_frame_state.players[0].track_id == 7


def test_buffer_pipeline_integration():
    """Verify the buffer correctly hands frames between producer and consumer."""
    import threading
    from app.pipeline.buffer import BoundedFrameBuffer, PipelineMode

    buf = BoundedFrameBuffer(maxsize=2, mode=PipelineMode.OFFLINE)
    frames_consumed: list[np.ndarray] = []
    consume_event = threading.Event()

    def consumer():
        for _ in range(3):
            frame = buf.get(timeout=2.0)
            if frame is not None:
                frames_consumed.append(frame)
        consume_event.set()

    t = threading.Thread(target=consumer)
    t.start()
    time.sleep(0.05)

    for i in range(3):
        buf.put(np.full((10, 10, 3), i, dtype=np.uint8))

    consume_event.wait(timeout=3.0)
    t.join(timeout=1.0)
    assert len(frames_consumed) == 3
    assert buf.dropped_frames == 0
