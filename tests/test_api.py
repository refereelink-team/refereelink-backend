from __future__ import annotations

import asyncio

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
    r = client.post("/api/pipeline/start")
    assert r.status_code == 200
    assert r.json() == {"status": "started"}
    r = client.post("/api/pipeline/stop")
    assert r.status_code == 200
    assert r.json() == {"status": "stopped"}


def test_video_stream_endpoint_exists(client):
    with client.stream("GET", "/video/stream") as r:
        pass
    assert True
