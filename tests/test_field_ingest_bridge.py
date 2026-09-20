from __future__ import annotations

import io
import threading
from pathlib import Path
from uuid import uuid4

import numpy as np
from fastapi.testclient import TestClient

from app.field_ingest.frames import (
    DecodedVideoFrame,
    FrameJoiner,
    LatestFrameQueue,
)
from app.field_ingest.config import FieldIngestSettings
from app.field_ingest.models import FieldSessionRegistration, LiveAllocationRequest
from app.field_ingest.receiver import SRTReceiver
from app.field_ingest.service import FieldIngestService
from app.pipeline.source import FieldIngestSource


def _decoded(pts: int = 9000) -> DecodedVideoFrame:
    return DecodedVideoFrame(
        image=np.zeros((2, 3, 3), dtype=np.uint8),
        transport_pts90k=pts,
        decode_index=0,
        received_at="2026-09-20T00:00:00Z",
    )


def test_frame_joiner_matches_pts_and_prior_motion() -> None:
    joiner = FrameJoiner("session", 2)
    joiner.ingest_item(
        {"type": "camera_motion", "sample_id": 7, "t_us": 100_000, "pitch": 0.2},
        "2026-09-20T00:00:00Z",
    )
    joiner.ingest_item(
        {
            "type": "frame",
            "frame_id": 42,
            "t_us": 120_000,
            "transport_pts90k": 9000,
        },
        "2026-09-20T00:00:00Z",
    )

    ready = joiner.ingest_decoded(_decoded())

    assert len(ready) == 1
    assert ready[0].source_frame_id == 42
    assert ready[0].camera_motion == {
        "type": "camera_motion",
        "sample_id": 7,
        "t_us": 100_000,
        "pitch": 0.2,
    }
    assert ready[0].pose_missing_reason is None


def test_frame_joiner_waits_for_late_metadata_without_arrival_order_join() -> None:
    now = [100.0]
    joiner = FrameJoiner("session", 2, clock=lambda: now[0], max_wait_ms=50)

    assert joiner.ingest_decoded(_decoded(18000)) == []
    now[0] += 0.02
    ready = joiner.ingest_item(
        {"type": "frame", "frame_id": 9, "t_us": 10_000, "transport_pts90k": 18_000},
        "2026-09-20T00:00:00Z",
    )

    assert len(ready) == 1
    assert ready[0].source_frame_id == 9
    assert ready[0].transport_pts90k == 18_000


def test_frame_joiner_marks_missing_metadata_after_deadline() -> None:
    now = [100.0]
    joiner = FrameJoiner("session", 2, clock=lambda: now[0], max_wait_ms=50)
    assert joiner.ingest_decoded(_decoded()) == []

    now[0] += 0.051
    ready = joiner.ingest_decoded(_decoded(18000))

    assert len(ready) == 1
    assert ready[0].source_frame_id is None
    assert ready[0].pose_missing_reason == "frame_metadata_timeout"


def test_frame_joiner_does_not_match_metadata_after_deadline() -> None:
    now = [100.0]
    joiner = FrameJoiner("session", 2, clock=lambda: now[0], max_wait_ms=50)
    assert joiner.ingest_decoded(_decoded()) == []

    now[0] += 0.051
    ready = joiner.ingest_item(
        {"type": "frame", "frame_id": 9, "t_us": 10_000, "transport_pts90k": 9_000},
        "2026-09-20T00:00:00Z",
    )

    assert len(ready) == 1
    assert ready[0].source_frame_id is None
    assert ready[0].pose_missing_reason == "frame_metadata_timeout"


def test_latest_frame_queue_drops_old_frames() -> None:
    queue = LatestFrameQueue(maxsize=2)
    frames = [
        _decoded(pts).image for pts in (1, 2, 3)
    ]
    from app.field_ingest.frames import CapturedFrame

    for index, image in enumerate(frames):
        queue.put(
            CapturedFrame(
                image=image,
                session_id="session",
                stream_epoch=1,
                source_frame_id=index,
                t_us=index,
                transport_pts90k=index,
                capture_unix_us=None,
                camera_motion=None,
                pose_missing_reason=None,
                backend_received_at="2026-09-20T00:00:00Z",
            )
        )

    result = queue.get(timeout=0.01)
    assert result is not None
    assert result.source_frame_id == 2
    assert queue.dropped == 2


def test_field_source_reads_packet_and_releases_queue() -> None:
    queue = LatestFrameQueue(maxsize=1)
    from app.field_ingest.frames import CapturedFrame

    queue.put(
        CapturedFrame(
            image=np.zeros((2, 3, 3), dtype=np.uint8),
            session_id="session",
            stream_epoch=4,
            source_frame_id=8,
            t_us=100,
            transport_pts90k=9000,
            capture_unix_us=1_000_000,
            camera_motion=None,
            pose_missing_reason="no_motion_within_50ms",
            backend_received_at="2026-09-20T00:00:00Z",
        )
    )
    released: list[bool] = []
    source = FieldIngestSource(
        queue,
        session_id="session",
        stream_epoch=4,
        fps=30,
        release_callback=lambda: (released.append(True), queue.close()),
    )

    ret, packet = source.read_packet()
    assert ret is True
    assert packet is not None
    assert packet.source_frame_id == 8
    assert source.frame_count == 1
    assert source.capture_timestamp_ms() == 1000.0

    source.release()
    assert released == [True]
    assert source.is_opened() is False


def test_field_source_metrics_survive_epoch_cleanup() -> None:
    queue = LatestFrameQueue(maxsize=1)

    def metrics_after_release() -> dict:
        raise KeyError("live epoch not found")

    source = FieldIngestSource(
        queue,
        session_id="session",
        stream_epoch=4,
        fps=30,
        metrics_callback=metrics_after_release,
    )

    assert source.stream_metrics == {
        "session_id": "session",
        "stream_epoch": 4,
        "queue_closed": True,
    }


def test_srt_command_has_rawvideo_output_and_preserves_ts(tmp_path: Path) -> None:
    receiver = SRTReceiver(
        ffmpeg_bin="ffmpeg",
        host="100.64.0.2",
        port=10000,
        session_id=str(uuid4()),
        stream_epoch=3,
        stream_token="short-token",
        output_path=tmp_path / "capture.ts",
        latency_ms=200,
        width=640,
        height=360,
    )

    command = receiver.command

    assert "showinfo" in command
    assert "rawvideo" in command
    assert "pipe:1" in command
    assert "640x360" in command
    assert str(tmp_path / "capture.ts") in command
    assert "pbkeylen=16" in command[command.index("-i") + 1]


def test_srt_receiver_delivers_raw_frame_with_showinfo_pts(tmp_path: Path) -> None:
    class FakeProcess:
        pid = 123
        returncode = 0

        def __init__(self) -> None:
            self.stdout = io.BytesIO(bytes([1, 2, 3, 4, 5, 6]))
            self.stderr = io.BytesIO(
                b"[Parsed_showinfo_0] n:   0 pts:      0 pts_time:0.000000\n"
            )

        def poll(self):
            return 0

        def wait(self, timeout=None):
            del timeout
            return 0

        def terminate(self):
            return None

        def kill(self):
            return None

    received = []
    event = threading.Event()
    process = FakeProcess()
    receiver = SRTReceiver(
        ffmpeg_bin="fake-ffmpeg",
        host="100.64.0.2",
        port=10000,
        session_id="session",
        stream_epoch=1,
        stream_token="token",
        output_path=tmp_path / "capture.ts",
        latency_ms=200,
        width=2,
        height=1,
        on_decoded_frame=lambda frame: (received.append(frame), event.set()),
        popen=lambda *args, **kwargs: process,
    )

    receiver.start()
    assert event.wait(timeout=1.0)
    receiver.stop()

    assert len(received) == 1
    assert received[0].transport_pts90k == 0
    assert received[0].image.shape == (1, 2, 3)
    assert receiver.snapshot()["raw_frame_count"] == 1


def test_field_service_bridges_telemetry_and_decoded_frame(tmp_path: Path) -> None:
    class FakeReceiver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.state = "starting"

        def start(self):
            self.state = "listening"

        def stop(self):
            self.state = "released"

        def snapshot(self):
            return {
                "state": self.state,
                "decoded_frame_count": 1,
                "raw_frame_count": 1,
                "first_pts90k": 9000,
                "last_pts90k": 9000,
                "last_error": None,
                "exit_code": None,
                "log_tail": [],
            }

        def emit(self):
            self.kwargs["on_decoded_frame"](_decoded())

    settings = FieldIngestSettings(
        auth_token="token",
        root=tmp_path / "field",
        srt_bind_host="100.64.0.2",
        srt_advertise_host="100.64.0.2",
        srt_port=10000,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
    )
    receiver_ref: list[FakeReceiver] = []

    def receiver_factory(**kwargs):
        receiver = FakeReceiver(**kwargs)
        receiver_ref.append(receiver)
        return receiver

    service = FieldIngestService(
        settings,
        ffmpeg_probe=lambda _: (True, ""),
        receiver_factory=receiver_factory,
    )
    session_id = uuid4()
    service.register(
        FieldSessionRegistration(
            session_id=session_id,
            device_id=uuid4(),
            device_name="Test iPhone",
            capabilities=["srt", "wss", "core_motion"],
        )
    )
    service.allocate_live(session_id, LiveAllocationRequest(profile="720p30"))
    service.handle_message(
        session_id,
        {
            "type": "telemetry_batch",
            "schema_version": "1.0",
            "session_id": str(session_id),
            "stream_epoch": 1,
            "client_sequence": 1,
            "items": [
                {"type": "camera_motion", "sample_id": 1, "t_us": 100_000, "pitch": 0.2},
                {"type": "frame", "frame_id": 42, "t_us": 120_000, "transport_pts90k": 9000},
            ],
        },
    )
    receiver_ref[0].emit()
    source = service.open_frame_source(session_id, 1)
    ret, packet = source.read_packet()

    assert ret is True
    assert packet is not None
    assert packet.source_frame_id == 42
    assert packet.camera_motion is not None
    assert service.frame_stream_status(session_id, 1)["joined_frame_count"] == 1
    source.release()


def test_pipeline_start_accepts_explicit_field_epoch() -> None:
    from app.server.main import app

    class DummySource:
        def release(self):
            return None

    class DummyPipeline:
        recording_status = {"enabled": False, "active": False, "path": None, "frames_written": 0}

        def start(self):
            return None

        def stop(self):
            return None

    previous = {
        "pipeline": app.state.pipeline,
        "field_ingest": app.state.field_ingest,
        "create_pipeline": app.state.create_pipeline,
        "attach": app.state.attach_and_start_pipeline,
        "require_team_calibration": app.state.store.config.require_team_calibration,
    }
    calls: list[dict] = []

    class FakeFieldService:
        def open_frame_source(self, session_id, epoch, store=None):
            del store
            calls.append({"session_id": session_id, "epoch": epoch})
            return DummySource()

    def create_pipeline(video_source=None, **kwargs):
        calls.append({"video_source": video_source, **kwargs})
        return DummyPipeline()

    def attach(pipeline):
        app.state.pipeline = pipeline
        pipeline.start()

    app.state.pipeline = None
    app.state.field_ingest = FakeFieldService()
    app.state.create_pipeline = create_pipeline
    app.state.attach_and_start_pipeline = attach
    app.state.store.update_config({"require_team_calibration": False})
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/pipeline/start",
                json={"field_session_id": "session", "field_stream_epoch": 3},
            )
        assert response.status_code == 200
        assert response.json()["field_stream_epoch"] == 3
        assert calls[0]["video_source"] is None
        assert calls[0]["field_session_id"] == "session"
        assert calls[0]["field_stream_epoch"] == 3
    finally:
        app.state.pipeline = previous["pipeline"]
        app.state.field_ingest = previous["field_ingest"]
        app.state.create_pipeline = previous["create_pipeline"]
        app.state.attach_and_start_pipeline = previous["attach"]
        app.state.store.update_config(
            {"require_team_calibration": previous["require_team_calibration"]}
        )


def test_main_factory_builds_pipeline_from_field_epoch() -> None:
    from app.server import main

    class DummySource:
        fps = 30.0

        def read(self):
            return False, None

        def read_packet(self):
            return False, None

        def release(self):
            return None

        def is_opened(self):
            return True

    class FakeFieldService:
        def __init__(self):
            self.source = DummySource()

        def open_frame_source(self, session_id, epoch, store=None):
            assert session_id == "session"
            assert epoch == 3
            assert store is main._store
            return self.source

    previous = main.app.state.field_ingest
    service = FakeFieldService()
    main.app.state.field_ingest = service
    try:
        pipeline = main.create_pipeline(
            field_session_id="session",
            field_stream_epoch=3,
        )
        assert pipeline._source is service.source
    finally:
        main.app.state.field_ingest = previous
