from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pytest
from fastapi.testclient import TestClient
from jsonschema import validate

from app.field_ingest.config import FieldIngestSettings
from app.field_ingest.frames import CapturedFrame, DecodedVideoFrame, FrameJoiner
from app.field_ingest.models import FieldSessionRegistration, LiveAllocationRequest
from app.field_ingest.receiver import SRTReceiver
from app.field_ingest.server import create_receiver_app
from app.field_ingest.service import FieldIngestService


class FakeReceiver:
    instances: list["FakeReceiver"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.instances.append(self)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def snapshot(self):
        return {
            "state": "listening" if self.started and not self.stopped else "released",
            "decoded_frame_count": 0,
            "first_pts90k": None,
            "last_pts90k": None,
            "last_error": None,
            "exit_code": None,
            "log_tail": [],
        }


def _make_client(tmp_path: Path) -> tuple[TestClient, str]:
    token = "test-token"
    settings = FieldIngestSettings(
        auth_token=token,
        root=tmp_path / "field",
        srt_bind_host="100.64.0.2",
        srt_advertise_host="100.64.0.2",
        srt_port=10000,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
    )
    service = FieldIngestService(
        settings,
        ffmpeg_probe=lambda _: (True, ""),
        receiver_factory=FakeReceiver,
    )
    return TestClient(create_receiver_app(service)), token


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _registration():
    return {
        "schema_version": "1.0",
        "session_id": str(uuid4()),
        "device_id": str(uuid4()),
        "device_name": "Test iPhone",
        "capabilities": ["srt", "wss", "core_motion"],
    }


def test_capabilities_require_auth_and_advertise_receiver(tmp_path):
    client, token = _make_client(tmp_path)
    with client:
        assert client.get("/api/v1/field/capabilities").status_code == 401
        response = client.get("/api/v1/field/capabilities", headers=_headers(token))
        assert response.status_code == 200
        assert response.json()["schema_version"] == "1.0"
        assert response.json()["receiver_available"] is True


def test_srt_receiver_command_is_tailnet_listener_with_pts_preserving_outputs(tmp_path):
    receiver = SRTReceiver(
        ffmpeg_bin="/opt/test/ffmpeg",
        host="100.64.0.2",
        port=10000,
        session_id="session-1",
        stream_epoch=3,
        stream_token="short-token",
        output_path=tmp_path / "epoch.ts",
        latency_ms=200,
    )
    command = receiver.command
    assert command[0] == "/opt/test/ffmpeg"
    assert "mode=listener" in command[command.index("-i") + 1]
    assert "streamid=refereelink%2Fsession-1%2F3%2Fshort-token" in command[command.index("-i") + 1]
    assert "passphrase=short-token" in command[command.index("-i") + 1]
    assert "mpegts" in command
    assert "showinfo" in command


def test_session_registration_is_idempotent_and_conflict_safe(tmp_path):
    client, token = _make_client(tmp_path)
    registration = _registration()
    with client:
        first = client.post("/api/v1/field/sessions", json=registration, headers=_headers(token))
        second = client.post("/api/v1/field/sessions", json=registration, headers=_headers(token))
        changed = {**registration, "device_name": "Other iPhone"}
        conflict = client.post("/api/v1/field/sessions", json=changed, headers=_headers(token))
    assert first.status_code == 201
    assert second.status_code == 200
    assert conflict.status_code == 409


def test_wss_hello_telemetry_ack_and_frame_join(tmp_path):
    client, token = _make_client(tmp_path)
    registration = _registration()
    session_id = registration["session_id"]
    with client:
        assert (
            client.post(
                "/api/v1/field/sessions", json=registration, headers=_headers(token)
            ).status_code
            == 201
        )
        allocation = client.post(
            f"/api/v1/field/sessions/{session_id}/live",
            json={"profile": "720p30", "latency_ms": 200},
            headers=_headers(token),
        )
        assert allocation.status_code == 200
        epoch = allocation.json()["stream_epoch"]
        with client.websocket_connect(
            f"/ws/v1/field/sessions/{session_id}", headers=_headers(token)
        ) as websocket:
            websocket.send_json(
                {
                    "type": "hello",
                    "schema_version": "1.0",
                    "session_id": session_id,
                    "device_id": registration["device_id"],
                }
            )
            assert websocket.receive_json()["type"] == "hello_ack"
            websocket.send_json(
                {
                    "type": "telemetry_batch",
                    "schema_version": "1.0",
                    "session_id": session_id,
                    "stream_epoch": epoch,
                    "client_sequence": 1,
                    "items": [
                        {"type": "camera_motion", "sample_id": 1, "t_us": 100_000, "pitch": 0.1},
                        {
                            "type": "frame",
                            "frame_id": 3,
                            "t_us": 120_000,
                            "transport_pts90k": 10_800,
                        },
                    ],
                }
            )
            assert websocket.receive_json()["type"] == "telemetry_ack"
            websocket.send_json(
                {
                    "type": "telemetry_batch",
                    "schema_version": "1.0",
                    "session_id": session_id,
                    "stream_epoch": epoch,
                    "client_sequence": 3,
                    "items": [],
                }
            )
            gap_ack = websocket.receive_json()
            assert gap_ack["missing_sequences"] == [2]
            assert gap_ack["gap_count"] == 1
        status = client.get(f"/api/v1/field/sessions/{session_id}", headers=_headers(token)).json()
    live = status["live_epochs"][0]
    assert live["frame_count"] == 1
    assert live["motion_count"] == 1
    joined_path = tmp_path / "field" / "sessions" / session_id / "metadata" / "joined-000001.ndjson"
    joined = json.loads(joined_path.read_text().splitlines()[0])
    assert joined["pose"]["age_us"] == 20_000


def test_wss_accepts_telemetry_only_when_srt_is_unavailable(tmp_path):
    token = "test-token"
    settings = FieldIngestSettings(
        auth_token=token,
        root=tmp_path / "field",
        srt_bind_host="100.64.0.2",
        srt_advertise_host="100.64.0.2",
        srt_port=10000,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
    )
    service = FieldIngestService(
        settings,
        ffmpeg_probe=lambda _: (False, "ffmpeg does not advertise the srt protocol"),
        receiver_factory=FakeReceiver,
    )
    client = TestClient(create_receiver_app(service))
    registration = _registration()
    session_id = registration["session_id"]
    with client:
        assert (
            client.post(
                "/api/v1/field/sessions", json=registration, headers=_headers(token)
            ).status_code
            == 201
        )
        with client.websocket_connect(
            f"/ws/v1/field/sessions/{session_id}", headers=_headers(token)
        ) as websocket:
            websocket.send_json(
                {
                    "type": "hello",
                    "schema_version": "1.0",
                    "session_id": session_id,
                    "device_id": registration["device_id"],
                }
            )
            assert websocket.receive_json()["type"] == "hello_ack"
            websocket.send_json(
                {
                    "type": "telemetry_batch",
                    "schema_version": "1.0",
                    "session_id": session_id,
                    "stream_epoch": 0,
                    "client_sequence": 1,
                    "items": [
                        {"type": "camera_motion", "sample_id": 1, "t_us": 10_000},
                        {"type": "frame", "frame_id": 1, "t_us": 20_000},
                    ],
                }
            )
            assert websocket.receive_json()["type"] == "telemetry_ack"
            status = client.get(
                f"/api/v1/field/sessions/{session_id}", headers=_headers(token)
            ).json()
    epoch = status["live_epochs"][0]
    assert epoch["stream_epoch"] == 0
    assert epoch["frame_count"] == 1
    assert epoch["motion_count"] == 1
    assert epoch["receiver"] is None


def test_telemetry_fixture_schema_accepts_canonical_batch():
    schema = json.loads(
        (Path(__file__).parent / "contracts" / "field_telemetry.schema.json").read_text()
    )
    validate(
        {
            "type": "telemetry_batch",
            "schema_version": "1.0",
            "session_id": str(uuid4()),
            "stream_epoch": 1,
            "client_sequence": 4,
            "items": [
                {"type": "frame", "frame_id": 1, "t_us": 20_000, "transport_pts90k": 1_800},
                {"type": "camera_motion", "sample_id": 2, "t_us": 10_000},
            ],
        },
        schema,
    )


def test_pending_eviction_does_not_reuse_late_metadata_as_prior_pts():
    now = [0.0]
    joiner = FrameJoiner(
        "session",
        1,
        max_pending=1,
        max_wait_ms=50_000,
        clock=lambda: now[0],
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)

    def decoded(pts: int, index: int) -> DecodedVideoFrame:
        return DecodedVideoFrame(image, pts, index, "2026-01-01T00:00:00Z")

    assert joiner.ingest_decoded(decoded(1_000, 0)) == []
    evicted = joiner.ingest_decoded(decoded(10_000, 1))
    assert evicted[0].transport_pts90k == 1_000
    assert evicted[0].source_frame_id is None

    joiner.ingest_item(
        {"type": "frame", "frame_id": 7, "t_us": 1_000, "transport_pts90k": 1_000},
        "2026-01-01T00:00:01Z",
    )
    result = joiner.ingest_decoded(decoded(2_000, 2))
    stolen = [frame for frame in result if frame.source_frame_id == 7]
    assert stolen == []


def test_release_live_stops_receiver_without_holding_service_lock(tmp_path):
    callback_finished = threading.Event()

    class CallbackReceiver:
        def __init__(self, **kwargs):
            self.on_decoded_frame = kwargs["on_decoded_frame"]
            self.on_end = kwargs["on_end"]

        def start(self):
            return None

        def snapshot(self):
            return {
                "state": "listening",
                "decoded_frame_count": 0,
                "first_pts90k": None,
                "last_pts90k": None,
                "last_error": None,
                "exit_code": None,
                "log_tail": [],
            }

        def stop(self):
            def _run() -> None:
                self.on_end()
                image = np.zeros((2, 2, 3), dtype=np.uint8)
                self.on_decoded_frame(DecodedVideoFrame(image, 50_000, 1, "2026-01-01T00:00:00Z"))
                callback_finished.set()

            thread = threading.Thread(target=_run)
            thread.start()
            thread.join(timeout=0.4)

    settings = FieldIngestSettings(
        auth_token="test-token",
        root=tmp_path / "field",
        srt_bind_host="100.64.0.2",
        srt_advertise_host="100.64.0.2",
        srt_port=10000,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
    )
    service = FieldIngestService(
        settings,
        ffmpeg_probe=lambda _: (True, ""),
        receiver_factory=CallbackReceiver,
    )
    registration = FieldSessionRegistration(
        session_id=uuid4(),
        device_id=uuid4(),
        device_name="Test iPhone",
        capabilities=["srt", "wss", "core_motion"],
    )
    service.register(registration)
    allocation = service.allocate_live(
        registration.session_id, LiveAllocationRequest(profile="720p30", latency_ms=200)
    )
    key = (str(registration.session_id), allocation.stream_epoch)
    queue = service._epochs[key].frame_queue
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    service._record_decoded_frame(
        key[0],
        key[1],
        DecodedVideoFrame(image, 1_000, 0, "2026-01-01T00:00:00Z"),
    )

    started = time.monotonic()
    service.release_live(UUID(str(registration.session_id)), allocation.stream_epoch)
    elapsed = time.monotonic() - started
    frame = queue.get(timeout=0)

    assert elapsed < 0.2
    assert callback_finished.is_set()
    assert frame is not None
    assert frame.transport_pts90k == 1_000
    service.close()


class _StopRaisesReceiver(FakeReceiver):
    def stop(self):
        super().stop()
        raise RuntimeError("receiver stop failed")


def _registered_live(tmp_path: Path, receiver_factory):
    settings = FieldIngestSettings(
        auth_token="test-token",
        root=tmp_path / "field",
        srt_bind_host="100.64.0.2",
        srt_advertise_host="100.64.0.2",
        srt_port=10000,
        ffmpeg_bin="ffmpeg",
        ffprobe_bin="ffprobe",
    )
    service = FieldIngestService(
        settings,
        ffmpeg_probe=lambda _: (True, ""),
        receiver_factory=receiver_factory,
    )
    registration = FieldSessionRegistration(
        session_id=uuid4(),
        device_id=uuid4(),
        device_name="Test iPhone",
        capabilities=["srt", "wss", "core_motion"],
    )
    service.register(registration)
    allocation = service.allocate_live(
        registration.session_id, LiveAllocationRequest(profile="720p30", latency_ms=200)
    )
    return service, registration, allocation


def test_release_live_releases_epoch_when_receiver_stop_raises(tmp_path):
    service, registration, allocation = _registered_live(tmp_path, _StopRaisesReceiver)
    key = (str(registration.session_id), allocation.stream_epoch)
    runtime = service._epochs[key]
    assert service.store.has_active_epoch() is True
    assert runtime.telemetry_file.closed is False
    assert runtime.join_file.closed is False

    with pytest.raises(RuntimeError, match="receiver stop failed"):
        service.release_live(registration.session_id, allocation.stream_epoch)

    assert key not in service._epochs
    assert runtime.telemetry_file.closed is True
    assert runtime.join_file.closed is True
    assert service.store.has_active_epoch() is False
    assert service.store.get_epoch(registration.session_id, allocation.stream_epoch)["status"] == (
        "released"
    )
    replacement = service.allocate_live(
        registration.session_id, LiveAllocationRequest(profile="720p30", latency_ms=200)
    )
    assert replacement.stream_epoch != allocation.stream_epoch
    with pytest.raises(RuntimeError, match="receiver stop failed"):
        service.release_live(registration.session_id, replacement.stream_epoch)
    assert service.store.has_active_epoch() is False
    service.close()


def test_close_drops_epochs_when_receiver_stop_raises(tmp_path):
    service, registration, allocation = _registered_live(tmp_path, _StopRaisesReceiver)
    key = (str(registration.session_id), allocation.stream_epoch)
    runtime = service._epochs[key]
    assert runtime.telemetry_file.closed is False

    with pytest.raises(RuntimeError, match="receiver stop failed"):
        service.close()

    assert service._epochs == {}
    assert runtime.telemetry_file.closed is True
    assert runtime.join_file.closed is True
    with pytest.raises(sqlite3.ProgrammingError):
        service.store.has_active_epoch()


def test_artifact_hash_and_duplicate_upload(tmp_path):
    client, token = _make_client(tmp_path)
    registration = _registration()
    body = b"field artifact"
    digest = hashlib.sha256(body).hexdigest()
    headers = {**_headers(token), "Content-SHA256": digest}
    with client:
        assert (
            client.post(
                "/api/v1/field/sessions", json=registration, headers=_headers(token)
            ).status_code
            == 201
        )
        first = client.put(
            f"/api/v1/field/sessions/{registration['session_id']}/artifacts/manifest.json",
            content=body,
            headers=headers,
        )
        second = client.put(
            f"/api/v1/field/sessions/{registration['session_id']}/artifacts/manifest.json",
            content=body,
            headers=headers,
        )
        conflict = client.put(
            f"/api/v1/field/sessions/{registration['session_id']}/artifacts/manifest.json",
            content=b"different artifact",
            headers={
                **_headers(token),
                "Content-SHA256": hashlib.sha256(b"different artifact").hexdigest(),
            },
        )
    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["status"] == "already_exists"
    assert conflict.status_code == 409


class _EndedReceiver(FakeReceiver):
    def snapshot(self):
        snap = super().snapshot()
        snap["state"] = "ended"
        return snap


def _tail_frame(session_id: str, epoch: int) -> CapturedFrame:
    return CapturedFrame(
        image=np.zeros((2, 2, 3), dtype=np.uint8),
        session_id=session_id,
        stream_epoch=epoch,
        source_frame_id=7,
        t_us=1_000,
        transport_pts90k=90_000,
        capture_unix_us=1_000,
        camera_motion=None,
        pose_missing_reason=None,
        backend_received_at="2026-01-01T00:00:00Z",
    )


def test_field_source_stays_open_until_queue_closes_after_receiver_ends(tmp_path):
    service, registration, allocation = _registered_live(tmp_path, _EndedReceiver)
    source = service.open_frame_source(registration.session_id, allocation.stream_epoch)
    try:
        assert source.is_opened() is True
        key = (str(registration.session_id), allocation.stream_epoch)
        service._epochs[key].frame_queue.put(
            _tail_frame(str(registration.session_id), allocation.stream_epoch)
        )
        ret, packet = source.read_packet()
        assert ret is True
        assert packet is not None
        assert packet.source_frame_id == 7
        service._epochs[key].frame_queue.close()
        assert source.is_opened() is False
    finally:
        source.release()
        service.close()


def test_failed_pipeline_constructor_releases_field_epoch(tmp_path, monkeypatch):
    from app.server import main as server_main

    service, registration, allocation = _registered_live(tmp_path, FakeReceiver)
    previous_service = server_main.app.state.field_ingest
    server_main.app.state.field_ingest = service

    class _BoomPipeline:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("constructor failed")

    monkeypatch.setattr(server_main, "InferencePipeline", _BoomPipeline)
    try:
        with pytest.raises(RuntimeError, match="constructor failed"):
            server_main.create_pipeline(
                None,
                field_session_id=str(registration.session_id),
                field_stream_epoch=allocation.stream_epoch,
            )
        assert (str(registration.session_id), allocation.stream_epoch) not in service._epochs
        assert service.store.has_active_epoch() is False
        replacement = service.allocate_live(
            registration.session_id, LiveAllocationRequest(profile="720p30", latency_ms=200)
        )
        assert replacement.stream_epoch != allocation.stream_epoch
    finally:
        server_main.app.state.field_ingest = previous_service
        service.close()


def test_failed_pipeline_start_releases_field_epoch(tmp_path):
    from app.server import main as server_main

    service, registration, allocation = _registered_live(tmp_path, FakeReceiver)
    previous_service = server_main.app.state.field_ingest
    previous_pipeline = server_main.app.state.pipeline
    previous_global = server_main._pipeline
    server_main.app.state.field_ingest = service
    server_main._pipeline = None
    server_main.app.state.pipeline = None
    source = service.open_frame_source(registration.session_id, allocation.stream_epoch)

    class _FailStart:
        def start(self) -> None:
            raise RuntimeError("models failed")

        def stop(self) -> None:
            source.release()

    try:
        with pytest.raises(RuntimeError, match="models failed"):
            server_main.attach_and_start_pipeline(_FailStart())  # type: ignore[arg-type]
        assert server_main.app.state.pipeline is None
        assert server_main._pipeline is None
        assert service.store.has_active_epoch() is False
        replacement = service.allocate_live(
            registration.session_id, LiveAllocationRequest(profile="720p30", latency_ms=200)
        )
        assert replacement.stream_epoch != allocation.stream_epoch
    finally:
        server_main.app.state.field_ingest = previous_service
        server_main.app.state.pipeline = previous_pipeline
        server_main._pipeline = previous_global
        server_main._store.pipeline_running = False
        service.close()
