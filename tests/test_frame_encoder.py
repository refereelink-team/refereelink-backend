from __future__ import annotations

import numpy as np

from app.services.frame_encoder import LatestJpegFrame
from app.state.store import StateStore


def test_latest_jpeg_frame_encodes_once_and_replaces_previous_frame() -> None:
    encoder = LatestJpegFrame(quality=80)
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    assert encoder.update(frame)
    first = encoder.latest
    assert first is not None
    assert first.startswith(b"\xff\xd8")
    assert encoder.frames_encoded == 1
    assert encoder.average_encode_time_ms >= 0.0

    frame[0, 0] = (0, 0, 255)
    assert encoder.update(frame)
    assert encoder.frames_encoded == 2
    assert encoder.latest != first


def test_state_store_publishes_encoded_frame_for_all_clients() -> None:
    store = StateStore()
    store.publish_raw_frame(np.zeros((16, 16, 3), dtype=np.uint8))

    assert store.latest_jpeg_frame is not None
    assert store.jpeg_frames_encoded == 1
    snapshot = store.snapshot()
    assert snapshot["jpeg_frames_encoded"] == 1
