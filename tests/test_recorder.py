from __future__ import annotations

import cv2
import numpy as np

from app.pipeline.recorder import VideoRecorder
from app.state.store import StateStore


def test_video_recorder_writes_annotated_frames_and_releases_file(tmp_path) -> None:
    target = tmp_path / "debug" / "annotated.mp4"
    recorder = VideoRecorder(str(target), StateStore(), fps=10.0)

    recorder.start()
    for index in range(3):
        frame = np.full((48, 64, 3), index * 40, dtype=np.uint8)
        recorder.write(frame)
    recorder.stop()

    assert target.is_file()
    assert recorder.frames_written == 3
    assert not recorder.active

    capture = cv2.VideoCapture(str(target))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) == 64
    assert int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 48
    frames = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        frames += 1
    capture.release()
    assert frames == 3


def test_video_recorder_ignores_invalid_frames(tmp_path) -> None:
    recorder = VideoRecorder(str(tmp_path / "invalid.mp4"), StateStore(), fps=10.0)

    recorder.start()
    recorder.write(np.empty((0, 0, 3), dtype=np.uint8))
    recorder.stop()

    assert recorder.frames_written == 0
    assert not (tmp_path / "invalid.mp4").exists()
