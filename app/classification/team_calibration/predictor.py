from __future__ import annotations

from typing import Mapping, Optional

import numpy as np

from app.classification.team_calibration.types import (
    PlayerRole,
    TeamLabel,
    TeamPrediction,
    TeamPrototype,
)


class SupervisedPrototypeClassifier:
    """Fixed fusion classifier backed only by validated labelled prototypes."""

    def __init__(
        self,
        prototypes: Optional[Mapping[tuple[TeamLabel, PlayerRole], TeamPrototype]] = None,
        *,
        color_weight: float = 0.6,
        deep_weight: float = 0.4,
        min_observations: int = 3,
        min_margin: float = 0.10,
        max_distance: float = 0.80,
    ) -> None:
        if color_weight < 0 or deep_weight < 0 or color_weight + deep_weight <= 0:
            raise ValueError("feature weights must be non-negative and not both zero")
        self.prototypes = dict(prototypes or {})
        self.color_weight = float(color_weight)
        self.deep_weight = float(deep_weight)
        self.min_observations = max(1, int(min_observations))
        self.min_margin = float(min_margin)
        self.max_distance = float(max_distance)

    @property
    def ready(self) -> bool:
        return all(
            (team, PlayerRole.OUTFIELD) in self.prototypes
            and self.prototypes[(team, PlayerRole.OUTFIELD)].color_feature is not None
            for team in (TeamLabel.HOME, TeamLabel.AWAY)
        )

    def predict(
        self,
        *,
        color_feature: Optional[np.ndarray],
        deep_feature: Optional[np.ndarray],
        role: PlayerRole = PlayerRole.OUTFIELD,
        observation_count: int = 0,
    ) -> TeamPrediction:
        if observation_count < self.min_observations:
            return self._unknown("observations_insufficient", observation_count)
        if role in {PlayerRole.REFEREE, PlayerRole.STAFF}:
            return self._unknown("role_has_no_team", observation_count)
        if role == PlayerRole.UNKNOWN:
            role = PlayerRole.OUTFIELD
        candidates = [
            self.prototypes.get((TeamLabel.HOME, role)),
            self.prototypes.get((TeamLabel.AWAY, role)),
        ]
        if any(candidate is None for candidate in candidates):
            if role == PlayerRole.GOALKEEPER:
                return self._unknown("goalkeeper_prototype_unavailable", observation_count)
            return self._unknown("outfield_prototype_unavailable", observation_count)
        distances = [
            self._fused_distance(candidate, color_feature, deep_feature)
            for candidate in candidates
        ]
        if not all(np.isfinite(distances)):
            return self._unknown("feature_unavailable", observation_count)
        order = np.argsort(distances)
        best_index = int(order[0])
        best = float(distances[best_index])
        second = float(distances[int(order[1])])
        margin = second - best
        if best > self.max_distance:
            return self._unknown("distance_too_large", observation_count, margin, best)
        if margin < self.min_margin:
            return self._unknown("margin_too_small", observation_count, margin, best)
        confidence = float(np.clip(0.5 + margin / max(second + best + 1e-8, 1e-8), 0.0, 1.0))
        team = (TeamLabel.HOME, TeamLabel.AWAY)[best_index]
        return TeamPrediction(
            team=team,
            confidence=confidence,
            margin=margin,
            best_distance=best,
            observation_count=observation_count,
        )

    def _fused_distance(
        self,
        prototype: TeamPrototype,
        color_feature: Optional[np.ndarray],
        deep_feature: Optional[np.ndarray],
    ) -> float:
        distances: list[float] = []
        weights: list[float] = []
        if color_feature is not None and prototype.color_feature is not None:
            raw = float(np.linalg.norm(color_feature - prototype.color_feature))
            # The prototype stores an inter-class scale.  The class-local
            # dispersion is used as a floor so a single Track cannot make a
            # modality numerically dominate the fusion.
            scale = max(prototype.color_inter_scale, prototype.intra_class_dispersion, 1e-3)
            distances.append(raw / scale)
            weights.append(self.color_weight)
        if deep_feature is not None and prototype.deep_feature is not None:
            normalized = np.asarray(deep_feature, dtype=np.float32)
            norm = float(np.linalg.norm(normalized))
            if norm > 1e-8:
                normalized = normalized / norm
            raw = float(1.0 - np.dot(normalized, prototype.deep_feature))
            scale = max(prototype.deep_inter_scale, prototype.intra_class_dispersion, 1e-3)
            distances.append(raw / scale)
            weights.append(self.deep_weight)
        if not distances:
            return float("inf")
        return float(np.average(distances, weights=weights))

    @staticmethod
    def _unknown(
        reason: str,
        observation_count: int,
        margin: float = 0.0,
        best_distance: float = float("inf"),
    ) -> TeamPrediction:
        return TeamPrediction(
            team=TeamLabel.UNKNOWN,
            confidence=0.0,
            margin=float(margin),
            best_distance=float(best_distance),
            observation_count=observation_count,
            rejection_reason=reason,
        )

