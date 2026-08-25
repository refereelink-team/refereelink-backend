from __future__ import annotations

from types import SimpleNamespace

from app.events.contact import ContactTrigger, ContactTriggerConfig, format_player_identity
from app.events.engine import EventEngine, EventEngineConfig, FoulEventAdapter
from app.state.models import BallState, BallStatus, FrameState, PlayerRole, PlayerState, TeamLabel


def _player(
    track_id: int,
    team_id: int,
    x: float,
    y: float = 3000.0,
    *,
    velocity_x: float | None = None,
    velocity_y: float | None = None,
) -> PlayerState:
    team = TeamLabel.HOME if team_id == 0 else TeamLabel.AWAY if team_id == 1 else TeamLabel.UNKNOWN
    return PlayerState(
        track_id=track_id,
        role=PlayerRole.PLAYER,
        team=team,
        team_label=team,
        team_id=team_id,
        field_x=x,
        field_y=y,
        confidence=0.9,
        team_confidence=0.9,
        role_confidence=0.9,
        semantic_status="stable",
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        bbox=(float(track_id), 10.0, float(track_id) + 40.0, 80.0),
    )


def _frame(frame_id: int, timestamp: float, possession: int | None) -> FrameState:
    return FrameState(
        frame_id=frame_id,
        capture_timestamp_ms=timestamp * 1000.0,
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
        involved_track_ids=[7, 19],
        label_a="Home T7",
        label_b="Away T19",
    )

    assert event is not None
    assert event.event_type == "foul_candidate"
    assert event.foul_details["offence"] == "high"
    assert event.foul_details["action"] == "pushing"
    assert event.foul_details["summary"] == "Home T7 · pushing · Away T19"
    assert event.involved_track_ids == [7, 19]
    assert event.evidence["source"] == "mvfoul"


def test_format_player_identity() -> None:
    assert format_player_identity(_player(12, 0, 100.0)) == "Home T12"
    assert format_player_identity(_player(34, 1, 100.0)) == "Away T34"


def test_contact_trigger_emits_geometry_foul_candidate() -> None:
    trigger = ContactTrigger(
        ContactTriggerConfig(
            max_distance_mm=2000.0,
            closing_speed_mm_s=1000.0,
            cooldown_s=2.0,
        )
    )
    frame = FrameState(
        frame_id=5,
        capture_timestamp_ms=1000.0,
        players=[
            _player(10, 0, 5000.0, 3500.0, velocity_x=2000.0, velocity_y=0.0),
            _player(20, 1, 5600.0, 3500.0, velocity_x=-2000.0, velocity_y=0.0),
            _player(30, 0, 1000.0, 1000.0),
        ],
    )
    events = trigger.update(frame)
    assert len(events) == 1
    event = events[0]
    assert event.event_type == "foul_candidate"
    assert event.evidence["source"] == "geometry"
    assert set(event.involved_track_ids) == {10, 20}
    assert "Home T10" in event.foul_details["summary"]
    assert "Away T20" in event.foul_details["summary"]


def test_contact_trigger_respects_cooldown() -> None:
    trigger = ContactTrigger(ContactTriggerConfig(max_distance_mm=2000.0, closing_speed_mm_s=500.0))
    frame_a = FrameState(
        frame_id=1,
        capture_timestamp_ms=0.0,
        players=[
            _player(1, 0, 5000.0, 3000.0, velocity_x=1500.0),
            _player(2, 1, 5400.0, 3000.0, velocity_x=-1500.0),
        ],
    )
    frame_b = FrameState(
        frame_id=2,
        capture_timestamp_ms=500.0,
        players=[
            _player(1, 0, 5050.0, 3000.0, velocity_x=1500.0),
            _player(2, 1, 5350.0, 3000.0, velocity_x=-1500.0),
        ],
    )
    assert trigger.update(frame_a)
    assert trigger.update(frame_b) == []
