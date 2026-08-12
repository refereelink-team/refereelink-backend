from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.run_offline_pitch_registration import _read_frame_block


def _video(path: Path, count: int = 18) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (96, 64)
    )
    assert writer.isOpened()
    for index in range(count):
        frame = np.full((64, 96, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_frame_block_decodes_contiguous_long_gop_range(tmp_path) -> None:
    path = tmp_path / "source.mp4"
    _video(path)

    frames = _read_frame_block(path, 5, 14)

    assert len(frames) == 9
    levels = [float(np.mean(frame)) for frame in frames]
    assert levels == sorted(levels)
    assert levels[0] == pytest.approx(50.0, abs=5.0)
    assert levels[-1] == pytest.approx(130.0, abs=5.0)


@pytest.mark.parametrize("start,end", [(-1, 2), (2, 2), (3, 2)])
def test_frame_block_rejects_invalid_range(tmp_path, start: int, end: int) -> None:
    with pytest.raises(ValueError):
        _read_frame_block(tmp_path / "unused.mp4", start, end)
