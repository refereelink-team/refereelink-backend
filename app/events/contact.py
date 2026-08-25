"""Lightweight geometric contact trigger for live foul candidates.

Runs on the main inference thread. Complexity is O(n^2) over players with
field coordinates (typically < 25). Emits candidates only; never a final call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.state.models import FrameState, GameEvent, PlayerRole, PlayerState, TeamLabel


def format_player_identity(player: PlayerState) -> str:
    """v1 identity: Home/Away + track_id (no jersey OCR)."""
    team = player.team if isinstance(player.team, TeamLabel) else TeamLabel.UNKNOWN
    try:
        team = TeamLabel(str(getattr(team, "value", team)))
    except ValueError:
        team = TeamLabel.UNKNOWN
    if team == TeamLabel.HOME or player.team_id == 0:
        return f"Home T{player.track_id}"
    if team == TeamLabel.AWAY or player.team_id == 1:
        return f"Away T{player.track_id}"
    return f"T{player.track_id}"


@dataclass
class ContactTriggerConfig:
    max_distance_mm: float = 1800.0
    closing_speed_mm_s: float = 2500.0
    cooldown_s: float = 2.0
    require_opposite_teams: bool = True
    min_confidence: float = 0.45


@dataclass
class ContactCandidate:
    track_a: int
    track_b: int
    label_a: str
    label_b: str
    team_a: int
    team_b: int
    distance_mm: float
    closing_speed_mm_s: float
    field_x: float
    field_y: float
    confidence: float
    bbox_union: Optional[tuple[float, float, float, float]] = None


@dataclass
class ContactTrigger:
    """Detect opposing-player proximity + closing speed as foul candidates."""

    config: ContactTriggerConfig = field(default_factory=ContactTriggerConfig)
    _last_positions: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    _last_emit: dict[tuple[int, int], float] = field(default_factory=dict)

    def reset(self) -> None:
        self._last_positions.clear()
        self._last_emit.clear()

    def update(self, frame: FrameState) -> list[GameEvent]:
        timestamp = float(frame.capture_timestamp_ms) / 1000.0
        candidates = self.find_contacts(frame, timestamp)
        events: list[GameEvent] = []
        for cand in candidates:
            pair = (min(cand.track_a, cand.track_b), max(cand.track_a, cand.track_b))
            last = self._last_emit.get(pair)
            if last is not None and timestamp - last < self.config.cooldown_s:
                continue
            self._last_emit[pair] = timestamp
            summary = f"{cand.label_a} · contact · {cand.label_b}"
            events.append(
                GameEvent(
                    event_type="foul_candidate",
                    confidence=cand.confidence,
                    severity="candidate",
                    timestamp=timestamp,
                    frame_id=frame.frame_id,
                    field_x=cand.field_x,
                    field_y=cand.field_y,
                    involved_track_ids=[cand.track_a, cand.track_b],
                    foul_details={
                        "action": "contact",
                        "offence": "unknown",
                        "summary": summary,
                        "label_a": cand.label_a,
                        "label_b": cand.label_b,
                    },
                    evidence={
                        "source": "geometry",
                        "distance_mm": round(cand.distance_mm, 1),
                        "closing_speed_mm_s": round(cand.closing_speed_mm_s, 1),
                        "bbox_union": cand.bbox_union,
                        "summary": summary,
                    },
                )
            )
        return events

    def find_contacts(
        self, frame: FrameState, timestamp: float
    ) -> list[ContactCandidate]:
        players = [
            p
            for p in frame.players
            if p.field_x is not None
            and p.field_y is not None
            and p.role != PlayerRole.REFEREE
        ]
        velocities = self._estimate_velocities(players, timestamp)
        results: list[ContactCandidate] = []
        for i in range(len(players)):
            for j in range(i + 1, len(players)):
                a, b = players[i], players[j]
                if self.config.require_opposite_teams and not self._opposite_teams(a, b):
                    continue
                ax, ay = float(a.field_x), float(a.field_y)
                bx, by = float(b.field_x), float(b.field_y)
                dist = float(np.hypot(ax - bx, ay - by))
                if dist > self.config.max_distance_mm:
                    continue
                va = velocities.get(a.track_id, (0.0, 0.0))
                vb = velocities.get(b.track_id, (0.0, 0.0))
                mid = np.array([(ax + bx) * 0.5, (ay + by) * 0.5], dtype=np.float64)
                # Positive when players move toward each other along the separation axis.
                sep = np.array([bx - ax, by - ay], dtype=np.float64)
                sep_norm = float(np.linalg.norm(sep))
                if sep_norm < 1e-3:
                    closing = float(np.hypot(*(np.asarray(va) - np.asarray(vb))))
                else:
                    unit = sep / sep_norm
                    rel = np.asarray(va, dtype=np.float64) - np.asarray(vb, dtype=np.float64)
                    closing = float(-np.dot(rel, unit))
                if closing < self.config.closing_speed_mm_s and dist > self.config.max_distance_mm * 0.55:
                    # Allow very close pairs even without strong closing speed.
                    if dist > self.config.max_distance_mm * 0.35:
                        continue
                proximity = 1.0 - dist / max(self.config.max_distance_mm, 1.0)
                speed_term = float(
                    np.clip(closing / max(self.config.closing_speed_mm_s * 2.0, 1.0), 0.0, 1.0)
                )
                confidence = float(
                    np.clip(
                        max(self.config.min_confidence, 0.35 * proximity + 0.45 * speed_term + 0.2),
                        0.0,
                        0.95,
                    )
                )
                results.append(
                    ContactCandidate(
                        track_a=a.track_id,
                        track_b=b.track_id,
                        label_a=format_player_identity(a),
                        label_b=format_player_identity(b),
                        team_a=a.team_id,
                        team_b=b.team_id,
                        distance_mm=dist,
                        closing_speed_mm_s=max(0.0, closing),
                        field_x=float(mid[0]),
                        field_y=float(mid[1]),
                        confidence=confidence,
                        bbox_union=_union_bbox(a.bbox, b.bbox),
                    )
                )
        results.sort(key=lambda c: c.distance_mm)
        return results

    def _estimate_velocities(
        self, players: list[PlayerState], timestamp: float
    ) -> dict[int, tuple[float, float]]:
        velocities: dict[int, tuple[float, float]] = {}
        seen: set[int] = set()
        for player in players:
            seen.add(player.track_id)
            x, y = float(player.field_x), float(player.field_y)
            if player.velocity_x is not None and player.velocity_y is not None:
                velocities[player.track_id] = (float(player.velocity_x), float(player.velocity_y))
            else:
                prev = self._last_positions.get(player.track_id)
                if prev is not None:
                    dt = max(timestamp - prev[2], 1e-3)
                    velocities[player.track_id] = ((x - prev[0]) / dt, (y - prev[1]) / dt)
                else:
                    velocities[player.track_id] = (0.0, 0.0)
            self._last_positions[player.track_id] = (x, y, timestamp)
        stale = [tid for tid in self._last_positions if tid not in seen]
        for tid in stale:
            del self._last_positions[tid]
        return velocities

    @staticmethod
    def _opposite_teams(a: PlayerState, b: PlayerState) -> bool:
        if a.team_id in (0, 1) and b.team_id in (0, 1):
            return a.team_id != b.team_id
        teams = {a.team, b.team}
        return TeamLabel.HOME in teams and TeamLabel.AWAY in teams


def _union_bbox(
    a: Optional[tuple[float, float, float, float]],
    b: Optional[tuple[float, float, float, float]],
) -> Optional[tuple[float, float, float, float]]:
    if a is None or b is None:
        return a or b
    return (
        float(min(a[0], b[0])),
        float(min(a[1], b[1])),
        float(max(a[2], b[2])),
        float(max(a[3], b[3])),
    )
