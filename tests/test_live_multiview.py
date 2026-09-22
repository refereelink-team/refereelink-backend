from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.multiview.live_ingest import supervisor as supervisor_module
from app.multiview.live_ingest.config import LiveCameraConfig, LiveIngestConfig
from app.multiview.live_ingest.coordinator import LiveSliceCoordinator, probe_media
from app.multiview.live_ingest.service import LiveMultiviewService
from app.multiview.live_ingest.store import (
    IndexedSegment,
    LiveMultiviewCaseStore,
    SegmentIndex,
)
from app.multiview.live_ingest.supervisor import LiveIngestSupervisor
from app.multiview.models import CameraRole, CaptureState, EvidenceView, MultiviewCase
from app.multiview.repository import MultiviewCaseRepository
from app.multiview.review_store import MultiviewReviewStore
from app.multiview.service import MultiviewAnalysisService
from app.server.api.multiview import router as multiview_router


def _config(tmp_path: Path, *, buffer_seconds: float = 3.0) -> LiveIngestConfig:
    return LiveIngestConfig(
        cameras=(
            LiveCameraConfig("cam_main", "主机位", CameraRole.MAIN, "rtsp://camera-main/live"),
            LiveCameraConfig("cam_side", "侧机位", CameraRole.SIDE, "rtsp://camera-side/live"),
            LiveCameraConfig(
                "cam_replay", "端线机位", CameraRole.REPLAY, "rtsp://camera-replay/live"
            ),
        ),
        output_root=tmp_path / "live",
        buffer_seconds=buffer_seconds,
        segment_seconds=1.0,
        expected_width=64,
        expected_height=48,
        expected_fps=10.0,
        max_segment_age_seconds=10.0,
    )


def _case(case_id: str = "live-test-case") -> MultiviewCase:
    return MultiviewCase(
        case_id=case_id,
        title="现场多视角复核",
        match_name="固定机位实时采集",
        match_clock="12:34:56",
        event_time_s=3.0,
        zone="待人工确认",
        videos=[
            EvidenceView(camera_id="cam_main", display_name="主机位", role=CameraRole.MAIN),
            EvidenceView(camera_id="cam_side", display_name="侧机位", role=CameraRole.SIDE),
            EvidenceView(camera_id="cam_replay", display_name="端线机位", role=CameraRole.REPLAY),
        ],
    )


def test_ingest_uses_isolated_rtsp_tcp_h264_stream_copy(tmp_path: Path) -> None:
    config = _config(tmp_path)
    command = LiveIngestSupervisor(config).ffmpeg_command(config.cameras[0])

    assert command[command.index("-rtsp_transport") + 1] == "tcp"
    assert command[command.index("-c:v") + 1] == "copy"
    assert command[command.index("-segment_format") + 1] == "mpegts"
    assert command[command.index("-segment_time") + 1] == "1.0"


def test_reconnect_reindexes_new_capture_directory(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    index = SegmentIndex(config.database_path)
    supervisor = LiveIngestSupervisor(config, segment_index=index)
    camera = config.cameras[0]
    commands: list[list[str]] = []

    class ExitedProcess:
        def poll(self) -> int:
            return 1

    monkeypatch.setattr(
        supervisor_module.subprocess,
        "Popen",
        lambda command, **_: commands.append(command) or ExitedProcess(),
    )
    monkeypatch.setattr(
        supervisor_module,
        "probe_media",
        lambda *_args, **_kwargs: {
            "codec_name": "h264",
            "profile": "Main",
            "width": 64,
            "height": 48,
            "fps": 10.0,
            "has_b_frames": 0,
            "duration_s": 1.0,
            "pts_start_s": 0.0,
        },
    )

    supervisor._ensure_camera_process(camera, now_s=0.0)
    first_path = Path(commands[-1][-1]).with_name("segment-0000000000.ts")
    first_path.write_bytes(b"x" * 2048)
    now_s = time.time()
    os.utime(first_path, (now_s - 2.0, now_s - 2.0))
    supervisor._reconcile_camera(camera)

    supervisor._ensure_camera_process(camera, now_s=2.0)
    supervisor._ensure_camera_process(camera, now_s=4.0)
    second_path = Path(commands[-1][-1]).with_name("segment-0000000000.ts")
    second_path.write_bytes(b"x" * 2048)
    os.utime(second_path, (now_s - 1.0, now_s - 1.0))
    supervisor._reconcile_camera(camera)

    assert first_path.parent != second_path.parent
    snapshots = index.freeze_window(
        (camera.camera_id,),
        end_time_s=now_s - 1.0,
        window_seconds=1.9,
        max_gap_s=1.5,
    )
    assert {segment.path for segment in snapshots[camera.camera_id]} == {
        first_path.resolve(),
        second_path.resolve(),
    }


def test_segment_ring_pruning_respects_frozen_window(tmp_path: Path) -> None:
    config = _config(tmp_path)
    index = SegmentIndex(config.database_path)
    paths: list[Path] = []
    for camera in config.cameras:
        for second in range(4):
            path = tmp_path / f"{camera.camera_id}-{second}.ts"
            path.write_bytes(b"segment")
            paths.append(path)
            index.record_segment(
                IndexedSegment(
                    camera_id=camera.camera_id,
                    path=path,
                    start_time_s=float(second),
                    end_time_s=float(second + 1),
                    duration_s=1.0,
                )
            )

    snapshots = index.freeze_window(
        (camera.camera_id for camera in config.cameras),
        end_time_s=4.0,
        window_seconds=3.0,
        max_gap_s=1.5,
    )
    assert all(len(snapshot) == 3 for snapshot in snapshots.values())
    frozen_paths = {segment.path for snapshot in snapshots.values() for segment in snapshot}
    assert set(index.prune(older_than_s=10.0)) == set(paths) - frozen_paths

    index.release_window(snapshots)
    expired = index.prune(older_than_s=10.0)
    assert set(expired) == frozen_paths


def test_prune_waits_for_concurrent_freeze_window(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    index = SegmentIndex(config.database_path)
    paths: list[Path] = []
    for camera in config.cameras:
        for second in range(4):
            path = tmp_path / f"concurrent-{camera.camera_id}-{second}.ts"
            path.write_bytes(b"segment")
            paths.append(path)
            index.record_segment(
                IndexedSegment(
                    camera_id=camera.camera_id,
                    path=path,
                    start_time_s=float(second),
                    end_time_s=float(second + 1),
                    duration_s=1.0,
                )
            )

    entered_freeze = threading.Event()
    release_freeze = threading.Event()
    original_window_segments = index._window_segments

    def paused_window_segments(*args, **kwargs):
        entered_freeze.set()
        assert release_freeze.wait(timeout=5)
        return original_window_segments(*args, **kwargs)

    monkeypatch.setattr(index, "_window_segments", paused_window_segments)

    def freeze() -> dict[str, list[IndexedSegment]]:
        return index.freeze_window(
            (camera.camera_id for camera in config.cameras),
            end_time_s=4.0,
            window_seconds=3.0,
            max_gap_s=1.5,
        )

    def prune() -> list[Path]:
        return index.prune(older_than_s=10.0)

    with ThreadPoolExecutor(max_workers=2) as executor:
        freeze_future = executor.submit(freeze)
        assert entered_freeze.wait(timeout=5)
        prune_future = executor.submit(prune)
        release_freeze.set()
        snapshots = freeze_future.result(timeout=5)
        pruned = prune_future.result(timeout=5)

    frozen_paths = {segment.path for snapshot in snapshots.values() for segment in snapshot}
    assert set(pruned) == set(paths) - frozen_paths


def test_live_status_requires_a_shared_trigger_window(tmp_path: Path) -> None:
    config = _config(tmp_path)
    index = SegmentIndex(config.database_path)
    now_s = time.time()

    def add_segments(
        camera: LiveCameraConfig,
        starts: list[float],
        *,
        prefix: str = "",
    ) -> None:
        for sequence, start_time_s in enumerate(starts):
            path = tmp_path / "shared" / camera.camera_id / f"{prefix}{sequence}.ts"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"segment")
            index.record_segment(
                IndexedSegment(
                    camera_id=camera.camera_id,
                    path=path,
                    start_time_s=start_time_s,
                    end_time_s=start_time_s + 1.0,
                    duration_s=1.0,
                )
            )

    main, side, replay = config.cameras
    add_segments(main, [now_s - 3.0, now_s - 2.0, now_s - 1.0])
    add_segments(side, [now_s - 4.0, now_s - 3.0, now_s - 2.0])
    add_segments(replay, [now_s - 4.0, now_s - 3.0, now_s - 2.0])
    service = LiveMultiviewService(config, segment_index=index)

    skewed = service.status()
    assert skewed["min_buffer_s"] >= config.buffer_seconds
    assert skewed["trigger_ready"] is False

    add_segments(main, [now_s - 4.0], prefix="older-")
    assert service.status()["trigger_ready"] is True


def test_dynamic_cases_merge_without_mutating_static_json(tmp_path: Path) -> None:
    static_path = tmp_path / "cases.json"
    static_payload = [_case("static-demo").model_dump(mode="json")]
    static_path.write_text(json.dumps(static_payload), encoding="utf-8")
    original = static_path.read_text(encoding="utf-8")
    live_store = LiveMultiviewCaseStore(tmp_path / "live.sqlite3")
    live_store.save(_case(), capture_state=CaptureState.READY)

    repository = MultiviewCaseRepository(static_path, live_case_store=live_store)
    listed = {case.case_id: case for case in repository.list_cases()}

    assert set(listed) == {"static-demo", "live-test-case"}
    assert listed["live-test-case"].capture_state == CaptureState.READY
    assert static_path.read_text(encoding="utf-8") == original


def test_live_status_trigger_and_dynamic_case_api(tmp_path: Path) -> None:
    config = _config(tmp_path)
    index = SegmentIndex(config.database_path)
    case_store = LiveMultiviewCaseStore(config.database_path)

    class SuccessfulCoordinator:
        def trigger(self) -> MultiviewCase:
            return case_store.save(_case("live-api-case"), capture_state=CaptureState.READY)

    live_service = LiveMultiviewService(
        config,
        segment_index=index,
        case_store=case_store,
        coordinator=SuccessfulCoordinator(),  # type: ignore[arg-type]
    )
    analysis_service = MultiviewAnalysisService(
        repository=MultiviewCaseRepository(live_case_store=case_store),
        review_store=MultiviewReviewStore(tmp_path / "reviews.sqlite3"),
    )
    api = FastAPI()
    api.state.live_multiview_service = live_service
    api.state.multiview_service = analysis_service
    api.include_router(multiview_router)
    with TestClient(api) as client:
        status = client.get("/api/multiview/live/status")
        assert status.status_code == 200
        assert status.json()["configured"] is True
        assert status.json()["trigger_ready"] is False

        triggered = client.post("/api/multiview/live/trigger")
        assert triggered.status_code == 200
        assert triggered.json() == {
            "case_id": "live-api-case",
            "capture_state": "capture_ready",
        }
        case = client.get("/api/multiview/cases/live-api-case")
        assert case.status_code == 200
        assert case.json()["capture_state"] == "capture_ready"
        assert all("path" not in view for view in case.json()["videos"])


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg and ffprobe are required for local H.264 fixture coverage",
)
def test_three_h264_rings_remux_to_registered_playable_case(tmp_path: Path) -> None:
    config = _config(tmp_path)
    index = SegmentIndex(config.database_path)
    case_store = LiveMultiviewCaseStore(config.database_path)
    start_offsets = {"cam_main": 100.0, "cam_side": 99.95, "cam_replay": 100.02}

    for camera in config.cameras:
        for second in range(4):
            path = tmp_path / "segments" / camera.camera_id / f"{second}.ts"
            path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=size=64x48:rate=10:duration=1",
                    "-c:v",
                    "libx264",
                    "-profile:v",
                    "main",
                    "-g",
                    "10",
                    "-keyint_min",
                    "10",
                    "-sc_threshold",
                    "0",
                    "-bf",
                    "0",
                    "-pix_fmt",
                    "yuv420p",
                    "-f",
                    "mpegts",
                    str(path),
                ],
                check=True,
            )
            media = probe_media(path)
            start_time_s = start_offsets[camera.camera_id] + second
            index.record_segment(
                IndexedSegment(
                    camera_id=camera.camera_id,
                    path=path,
                    start_time_s=start_time_s,
                    end_time_s=start_time_s + float(media["duration_s"]),
                    duration_s=float(media["duration_s"]),
                    pts_start_s=media["pts_start_s"],
                    pts_end_s=(
                        float(media["pts_start_s"]) + float(media["duration_s"])
                        if media["pts_start_s"] is not None
                        else None
                    ),
                    media=media,
                )
            )

    coordinator = LiveSliceCoordinator(
        config,
        index,
        case_store,
        clock=lambda: 104.0,
    )
    case = coordinator.trigger()

    assert case.capture_state == CaptureState.READY
    assert len(case.videos) == 3
    assert case.videos[0].sync_offset_ms == 0
    assert case.videos[1].sync_offset_ms < -900
    assert case_store.get_case(case.case_id) == case
    for view in case.videos:
        assert view.path is not None
        media = probe_media(view.path)
        assert media["codec_name"] == "h264"
        assert media["width"] == 64
        assert media["height"] == 48
        assert media["duration_s"] >= 2.5
    manifest = json.loads((Path(case.videos[0].path).parent / "manifest.json").read_text())
    assert manifest["event_time_s"] == pytest.approx(case.event_time_s)
    assert manifest["views"][1]["sync_offset_ms"] == case.videos[1].sync_offset_ms
