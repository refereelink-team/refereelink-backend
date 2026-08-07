from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import supervision as sv

from app.classification.team_calibration.clip import (
    CalibrationClipService,
    MAX_CLIP_MS,
    SourceInfo,
    inspect_video,
    validate_clip_range,
)
from app.classification.team_calibration.session import CalibrationState, TeamCalibrationSession
from app.classification.team_calibration.types import CalibrationLabel


def _make_video(path: Path, *, frames: int = 20, fps: float = 10.0) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (96, 64),
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        frame[:, :] = (30 + index, 100, 40)
        cv2.rectangle(frame, (20, 4), (42, 58), (0, 0, 220), -1)
        cv2.line(frame, (0, index % 64), (95, index % 64), (255, 255, 255), 1)
        writer.write(frame)
    writer.release()


class FakeVisionCore:
    def __init__(self, **_kwargs) -> None:
        pass

    def load_models(self) -> None:
        pass

    def process(self, frame: np.ndarray, _frame_index: int):
        detections = sv.Detections(
            xyxy=np.asarray([[20, 4, 42, 58]], dtype=np.float32),
            confidence=np.asarray([0.95], dtype=np.float32),
            class_id=np.asarray([0], dtype=np.int32),
            tracker_id=np.asarray([7], dtype=np.int32),
        )
        return SimpleNamespace(
            undistorted_frame=frame,
            tracked_detections=detections,
        )


def test_inspect_video_reads_duration_fps_and_dimensions(tmp_path: Path) -> None:
    path = tmp_path / "sample.mp4"
    _make_video(path, frames=20, fps=10.0)
    info = inspect_video(path)
    assert isinstance(info, SourceInfo)
    assert info.width == 96
    assert info.height == 64
    assert info.fps == 10.0
    assert 1900 <= info.duration_ms <= 2100


def test_clip_range_rejects_invalid_order_and_length() -> None:
    validate_clip_range(0, 1000, 5000)
    for start, end in ((1000, 1000), (2000, 1000), (-1, 100), (0, MAX_CLIP_MS + 1)):
        try:
            validate_clip_range(start, end, 5000)
        except ValueError:
            pass
        else:
            raise AssertionError(f"range should be rejected: {start}, {end}")


def test_offline_processing_keeps_stable_tracks_and_labels_immediately(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _make_video(source, frames=40, fps=10.0)
    session = TeamCalibrationSession(require_appearance=False, min_samples_per_track=2)
    service = CalibrationClipService(
        session,
        temp_root=tmp_path / "runtime",
        vision_core_factory=FakeVisionCore,
    )

    preview = service.preview(
        source_path=str(source),
        match_id="match-1",
        camera_id="camera-1",
        device="cuda",
    )
    assert preview["state"] == CalibrationState.SOURCE_PREVIEW.value
    service.start_clip(500)
    processing = service.finish_clip(2500)
    assert processing["state"] == CalibrationState.PROCESSING.value

    deadline = time.monotonic() + 5.0
    while session.state == CalibrationState.PROCESSING and time.monotonic() < deadline:
        time.sleep(0.01)
    assert session.state == CalibrationState.REVIEW
    metadata = service.metadata()
    assert metadata["frame_count"] >= 15
    assert metadata["tracks"][0]["track_id"] == 7
    assert metadata["tracks"][0]["observation_count"] == metadata["frame_count"]

    labelled = service.label_track(7, CalibrationLabel.HOME_OUTFIELD)
    assert labelled["tracks"][0]["label"] == CalibrationLabel.HOME_OUTFIELD.value
    assert labelled["tracks"][0]["sample_count"] > 0
    assert service.review_video_path().is_file()


def test_manual_role_label_uses_relaxed_role_sampling(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _make_video(source, frames=40, fps=10.0)
    session = TeamCalibrationSession(require_appearance=False, min_samples_per_track=2)
    service = CalibrationClipService(
        session,
        temp_root=tmp_path / "runtime",
        vision_core_factory=FakeVisionCore,
    )

    service.preview(
        source_path=str(source),
        match_id="match-role",
        camera_id="camera-1",
        device="cuda",
    )
    service.start_clip(500)
    service.finish_clip(2500)
    deadline = time.monotonic() + 5.0
    while session.state == CalibrationState.PROCESSING and time.monotonic() < deadline:
        time.sleep(0.01)

    labelled = service.label_track(7, CalibrationLabel.REFEREE)

    track = next(item for item in labelled["tracks"] if item["track_id"] == 7)
    assert track["role"] == "referee"
    assert track["sample_count"] > 0
