from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

import numpy as np

from app.classification.team_calibration.types import (
    PlayerRole,
    TeamLabel,
    TeamPrototype,
    TrackFeature,
)


def build_prototypes(
    tracks: Iterable[TrackFeature],
    *,
    include_goalkeepers: bool = True,
) -> dict[tuple[TeamLabel, PlayerRole], TeamPrototype]:
    """Build labelled prototypes from one aggregate per Track.

    A Track contributes once, regardless of how many frames it produced.
    This prevents a long trajectory from dominating the match calibration.
    """

    groups: dict[tuple[TeamLabel, PlayerRole], list[TrackFeature]] = defaultdict(list)
    for track in tracks:
        if track.team not in {TeamLabel.HOME, TeamLabel.AWAY}:
            continue
        if track.role == PlayerRole.GOALKEEPER and not include_goalkeepers:
            continue
        if track.role not in {PlayerRole.OUTFIELD, PlayerRole.GOALKEEPER}:
            continue
        groups[(track.team, track.role)].append(track)

    prototypes: dict[tuple[TeamLabel, PlayerRole], TeamPrototype] = {}
    for key, members in groups.items():
        color = _weighted_mean([member.color_feature for member in members], members)
        deep = _weighted_mean([member.deep_feature for member in members], members, normalize=True)
        dispersion = _dispersion(members, color, deep)
        prototypes[key] = TeamPrototype(
            team=key[0],
            role=key[1],
            color_feature=color,
            deep_feature=deep,
            track_count=len(members),
            sample_count=sum(member.observation_count for member in members),
            intra_class_dispersion=dispersion,
        )

    _populate_scales(prototypes)
    return prototypes


def _weighted_mean(
    values: list[Optional[np.ndarray]],
    members: list[TrackFeature],
    *,
    normalize: bool = False,
) -> Optional[np.ndarray]:
    present = [(value, member) for value, member in zip(values, members) if value is not None and value.size]
    if not present:
        return None
    weights = np.asarray([max(member.quality_sum, 1e-6) for _, member in present], dtype=np.float32)
    matrix = np.stack([value for value, _ in present]).astype(np.float32)
    result = (matrix * (weights / weights.sum())[:, None]).sum(axis=0)
    if normalize:
        norm = float(np.linalg.norm(result))
        if norm > 1e-8:
            result /= norm
    return result.astype(np.float32)


def _dispersion(
    members: list[TrackFeature],
    color: Optional[np.ndarray],
    deep: Optional[np.ndarray],
) -> float:
    distances: list[float] = []
    for member in members:
        if color is not None and member.color_feature is not None:
            distances.append(float(np.linalg.norm(member.color_feature - color)))
        if deep is not None and member.deep_feature is not None:
            distances.append(float(1.0 - np.dot(member.deep_feature, deep)))
    return float(np.mean(distances)) if distances else 0.0


def _populate_scales(prototypes: dict[tuple[TeamLabel, PlayerRole], TeamPrototype]) -> None:
    for role in {PlayerRole.OUTFIELD, PlayerRole.GOALKEEPER}:
        home = prototypes.get((TeamLabel.HOME, role))
        away = prototypes.get((TeamLabel.AWAY, role))
        if home is None or away is None:
            continue
        color_inter = _distance(home.color_feature, away.color_feature)
        deep_inter = _distance(home.deep_feature, away.deep_feature)
        home.color_inter_scale = away.color_inter_scale = color_inter
        home.deep_inter_scale = away.deep_inter_scale = deep_inter


def _distance(first: Optional[np.ndarray], second: Optional[np.ndarray]) -> float:
    if first is None or second is None:
        return 0.0
    return float(np.linalg.norm(first - second))

