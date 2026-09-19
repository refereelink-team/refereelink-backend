from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from jsonschema import validate

from app.field_ingest.config import FieldIngestSettings
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
