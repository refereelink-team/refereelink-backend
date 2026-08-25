"""Lightweight event candidate state machine for phase three.

The engine intentionally emits *candidates*.  It uses the stable entities and
field coordinates from phase two, but does not claim that a geometric rule is
the final officiating decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from app.state.models import BallStatus, FrameState, GameEvent, PlayerRole


@dataclass(frozen=True)
class EventEngineConfig:
    possession_cooldown_s: float = 0.75
    pass_max_gap_s: float = 1.5
    shot_speed_mm_s: float = 3500.0
    event_cooldown_s: float = 1.0
    offside_enabled: bool = True


class EventEngine:
    """Generate deduplicated possession, pass, shot and offside candidates."""

    def __init__(self, config: EventEngineConfig = EventEngineConfig()) -> None:
        self.config = config
        self._last_possession: Optional[int] = None
        self._last_possession_timestamp: Optional[float] = None
        self._last_event_timestamp: dict[str, float] = {}

    def reset(self) -> None:
        self._last_possession = None
        self._last_possession_timestamp = None
        self._last_event_timestamp.clear()

    def update(self, frame: FrameState) -> list[GameEvent]:
        timestamp = float(frame.capture_timestamp_ms) / 1000.0
        events: list[GameEvent] = []
        possession = frame.possession_track_id

        if possession is not None and possession != self._last_possession:
            previous = self._last_possession
            gap = (
                timestamp - self._last_possession_timestamp
                if self._last_possession_timestamp is not None
                else None
            )
            if previous is not None:
                events.extend(
                    self._emit(
                        event_type="possession_change",
                        confidence=0.55,
                        timestamp=timestamp,
                        frame=frame,
                        involved=[previous, possession],
                        evidence={"previous_track_id": previous, "new_track_id": possession},
                    )
                )
                if gap is not None and 0.0 <= gap <= self.config.pass_max_gap_s:
                    events.extend(
                        self._emit(
                            event_type="pass_candidate",
                            confidence=0.5,
                            timestamp=timestamp,
                            frame=frame,
                            involved=[previous, possession],
                            evidence={"possession_gap_s": round(gap, 3)},
                        )
                    )
            self._last_possession = possession
            self._last_possession_timestamp = timestamp

        ball = frame.ball
        if ball is not None and ball.status in {BallStatus.FRESH, BallStatus.PREDICTED}:
            speed = None
            if ball.velocity_x is not None and ball.velocity_y is not None:
                speed = float(np.hypot(ball.velocity_x, ball.velocity_y))
            if speed is not None and speed >= self.config.shot_speed_mm_s:
                events.extend(
                    self._emit(
                        event_type="shot_candidate",
                        confidence=float(np.clip(speed / (self.config.shot_speed_mm_s * 2), 0.5, 0.99)),
                        timestamp=timestamp,
                        frame=frame,
                        involved=[possession] if possession is not None else [],
                        evidence={"ball_speed_mm_s": round(speed, 1)},
                    )
                )
            if self.config.offside_enabled:
                events.extend(self._offside_candidates(frame, timestamp))

        return events

    def _offside_candidates(self, frame: FrameState, timestamp: float) -> list[GameEvent]:
        ball = frame.ball
        if ball is None or ball.field_x is None:
            return []
        players = [
            player
            for player in frame.players
            if player.team_id in (0, 1)
            and player.role != PlayerRole.REFEREE
            and player.field_x is not None
        ]
        results: list[GameEvent] = []
        for attacking_team in (0, 1):
            attackers = [p for p in players if p.team_id == attacking_team]
            defenders = [p for p in players if p.team_id != attacking_team]
            if len(defenders) < 2:
                continue
            defenders.sort(key=lambda player: float(player.field_x))
            second_last = (
                defenders[-2] if attacking_team == 0 else defenders[1]
            )
            line_x = float(second_last.field_x)
            for attacker in attackers:
                attacker_x = float(attacker.field_x)
                beyond_line = attacker_x > line_x if attacking_team == 0 else attacker_x < line_x
                beyond_ball = attacker_x > ball.field_x if attacking_team == 0 else attacker_x < ball.field_x
                if not (beyond_line and beyond_ball):
                    continue
                results.extend(
                    self._emit(
                        event_type="offside_candidate",
                        confidence=0.5,
                        timestamp=timestamp,
                        frame=frame,
                        involved=[attacker.track_id, second_last.track_id],
                        evidence={
                            "attacking_team": attacking_team,
                            "attacker_x": attacker_x,
                            "second_last_defender_x": line_x,
                            "ball_x": float(ball.field_x),
                        },
                    )
                )
        return results

    def _emit(
        self,
        *,
        event_type: str,
        confidence: float,
        timestamp: float,
        frame: FrameState,
        involved: list[int],
        evidence: dict[str, Any],
    ) -> list[GameEvent]:
        last = self._last_event_timestamp.get(event_type)
        if last is not None and timestamp - last < self.config.event_cooldown_s:
            return []
        self._last_event_timestamp[event_type] = timestamp
        return [
            GameEvent(
                event_type=event_type,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                severity="candidate",
                timestamp=timestamp,
                frame_id=frame.frame_id,
                field_x=frame.ball.field_x if frame.ball is not None else None,
                field_y=frame.ball.field_y if frame.ball is not None else None,
                involved_track_ids=involved,
                evidence=evidence,
            )
        ]


class FoulEventAdapter:
    """Convert an MVFoul-like prediction object into a wire event."""

    def __init__(self, confidence_threshold: float = 0.48) -> None:
        self.confidence_threshold = float(np.clip(confidence_threshold, 0.0, 1.0))

    def update(
        self,
        prediction: Any,
        *,
        frame_id: int,
        timestamp: float,
        field_xy: Optional[tuple[float, float]] = None,
        involved_track_ids: Optional[list[int]] = None,
        label_a: Optional[str] = None,
        label_b: Optional[str] = None,
        source: str = "mvfoul",
        parent_event_id: Optional[str] = None,
    ) -> Optional[GameEvent]:
        if prediction is None:
            return None
        confidence = _prediction_confidence(prediction)
        if confidence < self.confidence_threshold:
            return None
        details = _prediction_details(prediction)
        action = str(details.get("action") or details.get("label") or "foul")
        left = label_a or "T?"
        right = label_b or "T?"
        summary = f"{left} · {action} · {right}"
        details["summary"] = summary
        if label_a:
            details["label_a"] = label_a
        if label_b:
            details["label_b"] = label_b
        evidence: dict[str, Any] = {"source": source, "summary": summary}
        if parent_event_id:
            evidence["parent_event_id"] = parent_event_id
        return GameEvent(
            event_type="foul_candidate",
            confidence=confidence,
            severity="candidate",
            timestamp=timestamp,
            frame_id=frame_id,
            field_x=field_xy[0] if field_xy else None,
            field_y=field_xy[1] if field_xy else None,
            involved_track_ids=list(involved_track_ids or []),
            foul_details=details,
            evidence=evidence,
        )


def _prediction_confidence(prediction: Any) -> float:
    for name in ("confidence", "probability", "score", "offence_confidence"):
        value = getattr(prediction, name, None)
        if value is not None:
            try:
                return float(np.clip(float(value), 0.0, 1.0))
            except (TypeError, ValueError):
                pass
    return 0.0


def _prediction_details(prediction: Any) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for name in ("offence", "action", "label", "severity"):
        value = getattr(prediction, name, None)
        if value is not None:
            details[name] = value.value if hasattr(value, "value") else value
    return details
