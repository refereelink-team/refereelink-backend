from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from app.classification.team_calibration.types import PlayerRole, TeamLabel, TrackFeature


class TrackFeatureBank:
    """Track-level feature aggregation with quality weighting and EMA."""

    def __init__(self, *, ema_alpha: float = 0.85) -> None:
        if not 0.0 <= ema_alpha < 1.0:
            raise ValueError("ema_alpha must be in [0, 1)")
        self.ema_alpha = float(ema_alpha)
        self._tracks: Dict[int, TrackFeature] = {}

    @property
    def tracks(self) -> dict[int, TrackFeature]:
        return dict(self._tracks)

    def update(
        self,
        track_id: int,
        *,
        team: TeamLabel,
        role: PlayerRole,
        color_feature: Optional[np.ndarray],
        deep_feature: Optional[np.ndarray],
        quality: float,
        frame_index: int,
    ) -> TrackFeature:
        quality = float(np.clip(quality, 0.0, 1.0))
        current = self._tracks.get(track_id)
        if current is None:
            current = TrackFeature(track_id=track_id, team=team, role=role)
            self._tracks[track_id] = current
        current.team = team if team != TeamLabel.UNKNOWN else current.team
        current.role = role if role != PlayerRole.UNKNOWN else current.role
        current.observation_count += 1
        current.quality_sum += quality
        current.last_update_frame = int(frame_index)
        current.color_feature = self._merge(current.color_feature, color_feature, quality)
        current.deep_feature = self._merge(current.deep_feature, deep_feature, quality)
        return current

    def remove_stale(self, frame_index: int, *, max_missing_frames: int = 90) -> list[int]:
        stale = [
            track_id
            for track_id, feature in self._tracks.items()
            if frame_index - feature.last_update_frame > max_missing_frames
        ]
        for track_id in stale:
            self._tracks.pop(track_id, None)
        return stale

    def _merge(
        self,
        previous: Optional[np.ndarray],
        current: Optional[np.ndarray],
        quality: float,
    ) -> Optional[np.ndarray]:
        if current is None or current.size == 0 or quality <= 0.0:
            return previous
        value = np.asarray(current, dtype=np.float32)
        if previous is None or previous.shape != value.shape:
            return value.copy()
        alpha = self.ema_alpha
        merged = alpha * previous + (1.0 - alpha) * quality * value
        norm = float(np.linalg.norm(merged))
        return (merged / norm).astype(np.float32) if norm > 1e-8 else merged.astype(np.float32)
