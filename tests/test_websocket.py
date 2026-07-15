from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from app.server.main import app
from app.state.models import FrameState, PlayerState, PlayerRole
from app.state.store import StateStore


@pytest.fixture
def client():
    return TestClient(app)


def test_websocket_connects(client):
    with client.websocket_connect("/ws/state") as ws:
        msg = ws.receive_text()
        assert msg
        data = json.loads(msg)
        assert data.get("type") in ("metrics", "frame_state") or "type" in data


def test_websocket_receives_frame_state(client):
    with client.websocket_connect("/ws/state") as ws:
        store: StateStore = app.state.store
        fs = FrameState(
            frame_id=99,
            players=[
                PlayerState(
                    track_id=7, role=PlayerRole.PLAYER, team_id=0,
                    field_x=10.0, field_y=20.0, confidence=0.9,
                )
            ],
        )
        store.latest_frame_state = fs
        msg = ws.receive_text()
        data = json.loads(msg)
        if data.get("type") == "frame_state":
            assert data["frame_id"] == 99
            assert len(data["players"]) == 1


def _receive_until(ws, predicate, timeout: float = 3.0) -> dict:
    import time
    end = time.time() + timeout
    while time.time() < end:
        try:
            msg = ws.receive_text()
            data = json.loads(msg)
            if predicate(data):
                return data
        except Exception:
            continue
    raise AssertionError(f"No matching message received within {timeout}s")


def test_websocket_command_start(client):
    with client.websocket_connect("/ws/state") as ws:
        ws.send_text(json.dumps({"command": "start"}))
        data = _receive_until(ws, lambda d: d.get("type") == "ack")
        assert data["command"] == "start"


def test_websocket_command_update_config(client):
    with client.websocket_connect("/ws/state") as ws:
        ws.send_text(json.dumps({
            "command": "update_config",
            "params": {"device": "cuda"},
        }))
        data = _receive_until(ws, lambda d: d.get("type") == "ack")
        assert data["command"] == "update_config"
