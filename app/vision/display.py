"""Short-horizon display smoothing for tracked-player overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DisplayTrack:
    track_id: int
    entity_id: int | None
    bbox: tuple[float, float, float, float]
    team_id: int
    team_label: str
    role_label: str
    confidence: float
    missing_frames: int = 0
    track_status: str = "detected"


class TrackDisplaySmoother:
    """Keep an overlay visible across brief detector gaps.

    This state is deliberately display-only.  It must not create synthetic
    players in ``FrameState`` or feed stale crops back into classification.
    """

    def __init__(self, *, ema_alpha: float = 0.65, max_missing_frames: int = 4) -> None:
        if not 0.0 <= ema_alpha < 1.0:
            raise ValueError("ema_alpha must be in [0, 1)")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")
        self.ema_alpha = float(ema_alpha)
        self.max_missing_frames = int(max_missing_frames)
        self._tracks: dict[int, DisplayTrack] = {}
        self._velocities: dict[int, tuple[float, float]] = {}

    @property
    def tracks(self) -> dict[int, DisplayTrack]:
        return dict(self._tracks)

    def update(self, players: Iterable[object]) -> list[DisplayTrack]:
        current_ids: set[int] = set()
        result: list[DisplayTrack] = []
        for player in players:
            track_id = int(player.track_id)
            entity_id = getattr(player, "entity_id", None)
            display_key = int(entity_id) if entity_id is not None else track_id
            current_ids.add(display_key)
            bbox = _bbox(player.bbox)
            previous = self._tracks.get(display_key)
            if previous is not None:
                observed_bbox = _blend(previous.bbox, bbox, self.ema_alpha)
                velocity = _bbox_velocity(previous.bbox, observed_bbox)
            else:
                observed_bbox = bbox
                velocity = (0.0, 0.0)
            if entity_id is None and previous is not None:
                entity_id = previous.entity_id
            display = DisplayTrack(
                track_id=track_id,
                entity_id=int(entity_id) if entity_id is not None else None,
                bbox=observed_bbox,
                team_id=int(getattr(player, "team_id", -1)),
                team_label=str(getattr(getattr(player, "team", "unknown"), "value", getattr(player, "team", "unknown"))),
                role_label=str(getattr(getattr(player, "role", "unknown"), "value", getattr(player, "role", "unknown"))),
                confidence=float(getattr(player, "confidence", 0.0)),
                missing_frames=0,
                track_status="detected",
            )
            self._velocities[display_key] = velocity
            self._tracks[display_key] = display
            result.append(display)

        for display_key, previous in list(self._tracks.items()):
            if display_key in current_ids:
                continue
            missing = previous.missing_frames + 1
            if missing > self.max_missing_frames:
                del self._tracks[display_key]
                self._velocities.pop(display_key, None)
                continue
            velocity = self._velocities.get(display_key, (0.0, 0.0))
            predicted_bbox = _translate(previous.bbox, velocity)
            stale = DisplayTrack(
                track_id=previous.track_id,
                entity_id=previous.entity_id,
                bbox=predicted_bbox,
                team_id=previous.team_id,
                team_label=previous.team_label,
                role_label=previous.role_label,
                confidence=previous.confidence * (0.92 ** missing),
                missing_frames=missing,
                track_status="predicted",
            )
            self._tracks[display_key] = stale
            result.append(stale)
        return sorted(result, key=lambda item: item.track_id)

    def clear(self) -> None:
        self._tracks.clear()
        self._velocities.clear()


def _bbox(value: object) -> tuple[float, float, float, float]:
    values = np.asarray(value, dtype=np.float32).reshape(-1)
    if values.size < 4 or not np.isfinite(values[:4]).all():
        return (0.0, 0.0, 1.0, 1.0)
    return tuple(float(item) for item in values[:4])  # type: ignore[return-value]


def _blend(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
    alpha: float,
) -> tuple[float, float, float, float]:
    return tuple(
        float(alpha * old + (1.0 - alpha) * new)
        for old, new in zip(previous, current)
    )  # type: ignore[return-value]


def _bbox_velocity(
    previous: tuple[float, float, float, float],
    current: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Estimate center-point velocity in pixels per processed frame."""

    previous_center = ((previous[0] + previous[2]) / 2, (previous[1] + previous[3]) / 2)
    current_center = ((current[0] + current[2]) / 2, (current[1] + current[3]) / 2)
    return (current_center[0] - previous_center[0], current_center[1] - previous_center[1])


def _translate(
    bbox: tuple[float, float, float, float],
    velocity: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Move a box by one frame without changing its size."""

    dx, dy = velocity
    return (bbox[0] + dx, bbox[1] + dy, bbox[2] + dx, bbox[3] + dy)
