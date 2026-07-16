from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.server.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_get_status(client):
    r = client.get("/api/status")
    assert r.status_code == 200
    data = r.json()
    assert "pipeline_running" in data
    assert "source_status" in data
    assert "metrics" in data


def test_get_config(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    data = r.json()
    assert "mode" in data
    assert "video_source" in data
    assert "device" in data
    assert "enable_ball" in data
    assert "ball_detection_interval" in data
    assert data["inference_backend"] == "auto"


def test_update_config(client):
    r = client.put("/api/config", json={"device": "cuda", "mode": "realtime"})
    assert r.status_code == 200
    data = r.json()
    assert data["device"] == "cuda"
    assert data["mode"] == "realtime"


def test_get_events(client):
    r = client.get("/api/events")
    assert r.status_code == 200
    data = r.json()
    assert "events" in data
    assert "total" in data


def test_pipeline_start_stop(client):
    r = client.post("/api/pipeline/start", json={})
    assert r.status_code == 409
    status = r.json().get("status")
    assert status == "error"
    assert r.json().get("code") == "TEAM_CALIBRATION_REQUIRED"
    r = client.post("/api/pipeline/stop")
    assert r.status_code == 200
    assert r.json().get("status") == "stopped"


def test_video_stream_endpoint_exists():
    # Verify the /video/stream route is registered. We don't actually
    # call it here because the MJPEG generator is an infinite loop and
    # the test client would block.
    from app.server.main import app
    paths = {r.path for r in app.routes if hasattr(r, "path")}
    assert "/video/stream" in paths
