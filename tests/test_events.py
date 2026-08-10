from __future__ import annotations

from types import SimpleNamespace

from app.events.engine import EventEngine, EventEngineConfig, FoulEventAdapter
from app.state.models import BallState, BallStatus, FrameState, PlayerRole, PlayerState


def _player(track_id: int, team_id: int, x: float, y: float = 3000.0) -> PlayerState:
    return PlayerState(
        track_id=track_id,
        role=PlayerRole.PLAYER,
        team_id=team_id,
        field_x=x,
        field_y=y,
        field_x_m=x / 100.0,
        field_y_m=y / 100.0,
        field_coordinate_usable=True,
        confidence=0.9,
        team_confidence=0.9,
        role_confidence=0.9,
        semantic_status="stable",
    )


def _frame(frame_id: int, timestamp: float, possession: int | None) -> FrameState:
    return FrameState(
        frame_id=frame_id,
        capture_timestamp_ms=timestamp * 1000.0,
        camera_measurement_usable=True,
        possession_track_id=possession,
        players=[
            _player(1, 0, 1000.0),
            _player(2, 0, 11000.0),
            _player(3, 1, 6000.0),
            _player(4, 1, 7000.0),
        ],
        ball=BallState(
            status=BallStatus.FRESH,
            field_x=8000.0,
            field_y=3000.0,
            velocity_x=100.0,
            velocity_y=0.0,
            confidence=0.9,
        ),
    )


def test_event_engine_emits_possession_change_and_pass_candidate() -> None:
    engine = EventEngine(EventEngineConfig(offside_enabled=False))

    assert engine.update(_frame(1, 0.0, 1)) == []
    events = engine.update(_frame(2, 0.5, 2))

    assert {event.event_type for event in events} == {"possession_change", "pass_candidate"}
    pass_event = next(event for event in events if event.event_type == "pass_candidate")
    assert pass_event.involved_track_ids == [1, 2]
    assert pass_event.evidence["possession_gap_s"] == 0.5


def test_event_engine_emits_offside_candidate_with_known_teams() -> None:
    events = EventEngine().update(_frame(1, 0.0, 1))

    offside = next(event for event in events if event.event_type == "offside_candidate")
    assert offside.involved_track_ids == [2, 3]
    assert offside.evidence["attacking_team"] == 0


def test_event_engine_rejects_offside_when_projection_is_not_safe() -> None:
    frame = _frame(1, 0.0, 1)
    frame.camera_measurement_usable = False

    assert not any(
        event.event_type == "offside_candidate" for event in EventEngine().update(frame)
    )

    frame.camera_measurement_usable = True
    frame.players[1].field_coordinate_usable = False
    assert not any(
        event.event_type == "offside_candidate" for event in EventEngine().update(frame)
    )


def test_event_engine_applies_shot_cooldown() -> None:
    engine = EventEngine(EventEngineConfig(shot_speed_mm_s=100.0, offside_enabled=False))
    first = _frame(1, 1.0, 1)
    first.ball.velocity_x = 1000.0
    second = _frame(2, 1.2, 1)
    second.ball.velocity_x = 1000.0

    assert any(event.event_type == "shot_candidate" for event in engine.update(first))
    assert not any(event.event_type == "shot_candidate" for event in engine.update(second))


def test_foul_adapter_preserves_explainable_details() -> None:
    adapter = FoulEventAdapter(confidence_threshold=0.48)
    event = adapter.update(
        SimpleNamespace(confidence=0.8, offence="high", action="pushing"),
        frame_id=12,
        timestamp=2.4,
        field_xy=(4000.0, 2500.0),
    )

    assert event is not None
    assert event.event_type == "foul_candidate"
    assert event.foul_details == {"offence": "high", "action": "pushing"}
    assert event.evidence["source"] == "mvfoul"
