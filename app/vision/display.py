"""Short-horizon display smoothing for tracked-player overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class DisplayTrack:
    track_id: int
    bbox: tuple[float, float, float, float]
    team_id: int
    team_label: str
    role_label: str
    confidence: float
    missing_frames: int = 0


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

    @property
    def tracks(self) -> dict[int, DisplayTrack]:
        return dict(self._tracks)

    def update(self, players: Iterable[object]) -> list[DisplayTrack]:
        current_ids: set[int] = set()
        result: list[DisplayTrack] = []
        for player in players:
            track_id = int(player.track_id)
            current_ids.add(track_id)
            bbox = _bbox(player.bbox)
            previous = self._tracks.get(track_id)
            if previous is not None:
                bbox = _blend(previous.bbox, bbox, self.ema_alpha)
            display = DisplayTrack(
                track_id=track_id,
                bbox=bbox,
                team_id=int(getattr(player, "team_id", -1)),
                team_label=str(getattr(getattr(player, "team", "unknown"), "value", getattr(player, "team", "unknown"))),
                role_label=str(getattr(getattr(player, "role", "unknown"), "value", getattr(player, "role", "unknown"))),
                confidence=float(getattr(player, "confidence", 0.0)),
                missing_frames=0,
            )
            self._tracks[track_id] = display
            result.append(display)

        for track_id, previous in list(self._tracks.items()):
            if track_id in current_ids:
                continue
            missing = previous.missing_frames + 1
            if missing > self.max_missing_frames:
                del self._tracks[track_id]
                continue
            stale = DisplayTrack(
                track_id=previous.track_id,
                bbox=previous.bbox,
                team_id=previous.team_id,
                team_label=previous.team_label,
                role_label=previous.role_label,
                confidence=previous.confidence,
                missing_frames=missing,
            )
            self._tracks[track_id] = stale
            result.append(stale)
        return sorted(result, key=lambda item: item.track_id)

    def clear(self) -> None:
        self._tracks.clear()


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
