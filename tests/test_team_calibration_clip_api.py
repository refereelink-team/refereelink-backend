from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import supervision as sv
from fastapi.testclient import TestClient

from app.classification.team_calibration.clip import CalibrationClipService
from app.classification.team_calibration.session import TeamCalibrationSession
from app.server.main import app


def _make_video(path: Path) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (96, 64))
    assert writer.isOpened()
    for index in range(40):
        frame = np.full((64, 96, 3), (30 + index, 100, 40), dtype=np.uint8)
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
        return SimpleNamespace(undistorted_frame=frame, tracked_detections=detections)


def test_clip_api_supports_preview_range_and_async_processing(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    _make_video(source)
    store = app.state.store
    previous_session = store._team_calibration
    previous_service = store._team_calibration_clip
    session = TeamCalibrationSession(require_appearance=False, min_samples_per_track=2)
    service = CalibrationClipService(
        session,
        temp_root=tmp_path / "runtime",
        vision_core_factory=FakeVisionCore,
    )
    store._team_calibration = session
    store._team_calibration_clip = service
    try:
        with TestClient(app) as client:
            preview = client.post(
                "/api/team-calibration/source/preview",
                json={"video_source": str(source), "match_id": "m", "camera_id": "c"},
            )
            assert preview.status_code == 200
            assert preview.json()["state"] == "source_preview"
            ranged = client.get(
                "/api/team-calibration/clip/source",
                headers={"Range": "bytes=0-15"},
            )
            assert ranged.status_code == 206
            assert ranged.headers["content-range"].startswith("bytes 0-15/")
            assert len(ranged.content) == 16

            assert client.post("/api/team-calibration/clip/start", json={"start_ms": 500}).status_code == 200
            processing = client.post("/api/team-calibration/clip/finish", json={"end_ms": 2500})
            assert processing.status_code == 200
            assert processing.json()["state"] == "processing"

            deadline = time.monotonic() + 5.0
            status = processing.json()
            while status["state"] == "processing" and time.monotonic() < deadline:
                time.sleep(0.01)
                status = client.get("/api/team-calibration/clip/status").json()
            assert status["state"] == "review"
            metadata = client.get("/api/team-calibration/clip/metadata")
            assert metadata.status_code == 200
            assert metadata.json()["tracks"][0]["track_id"] == 7
            assert client.get("/api/team-calibration/clip/video", headers={"Range": "bytes=0-5"}).status_code == 206
    finally:
        store._team_calibration = previous_session
        store._team_calibration_clip = previous_service
