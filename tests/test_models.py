from __future__ import annotations

import json

from app.state.models import (
    FrameState,
    GameEvent,
    HomographyStatus,
    MetricsSnapshot,
    PipelineCommand,
    PipelineConfig,
    PlayerRole,
    PlayerState,
    SourceStatus,
)


def test_player_state_serialization():
    p = PlayerState(
        track_id=7,
        role=PlayerRole.PLAYER,
        team_id=0,
        field_x=31.2,
        field_y=18.7,
        confidence=0.91,
    )
    data = p.model_dump()
    assert data["track_id"] == 7
    assert data["role"] == "outfield"
    assert data["team_id"] == 0
    assert data["team"] == "unknown"
    assert data["team_label"] == "unknown"
    assert data["field_x"] == 31.2
    assert data["confidence"] == 0.91
    roundtrip = PlayerState.model_validate(data)
    assert roundtrip == p


def test_game_event_serialization():
    e = GameEvent(
        event_type="foul",
        confidence=0.85,
        severity="likely",
        timestamp=12.3,
        frame_id=42,
        field_x=1000.0,
        field_y=2000.0,
        foul_details={"offence": "serious", "action": "pushing"},
    )
    data = e.model_dump()
    assert data["event_type"] == "foul"
    assert data["foul_details"]["offence"] == "serious"
    assert data["reviewed"] is False
    roundtrip = GameEvent.model_validate(data)
    assert roundtrip.foul_details == {"offence": "serious", "action": "pushing"}


def test_frame_state_serialization():
    fs = FrameState(
        frame_id=1234,
        capture_timestamp_ms=100000.0,
        processed_timestamp_ms=100120.0,
        processing_fps=18.5,
        homography_status=HomographyStatus.FRESH,
        players=[
            PlayerState(
                track_id=7, role=PlayerRole.PLAYER, team_id=0,
                field_x=31.2, field_y=18.7, confidence=0.91,
            )
        ],
        events=[],
    )
    data = fs.model_dump()
    assert data["type"] == "frame_state"
    assert data["frame_id"] == 1234
    assert data["homography_status"] == "fresh"
    assert len(data["players"]) == 1
    raw = json.dumps(data)
    assert "fresh" in raw


def test_metrics_snapshot_serialization():
    m = MetricsSnapshot(
        processing_fps=22.0,
        input_fps=25.0,
        inference_latency_ms=42.0,
        end_to_end_latency_ms=120.0,
        dropped_frames=3,
        queue_length=1,
        player_count=18,
        source_status=SourceStatus.CONNECTED,
        memory_mb=512.0,
        gpu_memory_mb=2048.0,
    )
    data = m.model_dump()
    assert data["type"] == "metrics"
    assert data["processing_fps"] == 22.0
    assert data["source_status"] == "connected"
    assert data["gpu_memory_mb"] == 2048.0


def test_pipeline_config_serialization():
    c = PipelineConfig(
        mode="realtime",
        video_source="rtsp://192.168.1.100/stream",
        enable_foul_detection=True,
        enable_recording=False,
        device="cuda",
    )
    data = c.model_dump()
    assert data["video_source"] == "rtsp://192.168.1.100/stream"
    assert data["device"] == "cuda"
    assert data["enable_foul_detection"] is True


def test_pipeline_command_serialization():
    c = PipelineCommand(command="start", params={"foo": "bar"})
    data = c.model_dump()
    assert data["command"] == "start"
    assert data["params"] == {"foo": "bar"}


def test_homography_status_values():
    assert HomographyStatus.FRESH.value == "fresh"
    assert HomographyStatus.REUSED.value == "reused"
    assert HomographyStatus.STALE.value == "stale"
    assert HomographyStatus.UNAVAILABLE.value == "unavailable"


def test_official_person_model_uses_unknown_role_and_neutral_team():
    player = PlayerState(
        track_id=7,
        role=PlayerRole.UNKNOWN,
        team_id=-1,
        confidence=0.8,
    )

    assert player.model_dump() == {
        "track_id": 7,
        "role": "unknown",
        "team": "unknown",
        "team_label": "unknown",
        "team_id": -1,
        "field_x": None,
        "field_y": None,
        "confidence": 0.8,
        "role_confidence": 0.0,
        "team_confidence": 0.0,
        "team_rejection_reason": None,
        "bbox": None,
        "semantic_status": "unknown",
        "velocity_x": None,
        "velocity_y": None,
    }


def test_player_state_with_null_field_coordinates():
    p = PlayerState(
        track_id=1, role=PlayerRole.PLAYER, team_id=-1,
        field_x=None, field_y=None, confidence=0.0,
    )
    assert p.field_x is None
    assert p.field_y is None
    data = p.model_dump()
    assert data["field_x"] is None
