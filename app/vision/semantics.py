"""Trajectory-level role and team semantic state management.

This module is intentionally independent from Pydantic and model runtimes.
The future pipeline can pass role/team observations into
``TrajectorySemanticManager.update`` and map the returned dataclass to its
own wire model.  ``TrackSemanticManager`` below exposes the stable frame-level
adapter intended for the future engine integration.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
import logging
from typing import Any, Deque, Iterable, Mapping, Optional

import numpy as np


logger = logging.getLogger(__name__)


UNKNOWN_ROLE = "unknown"
UNKNOWN_TEAM_ID = -1
KNOWN_ROLES = ("player", "goalkeeper", "referee")
KNOWN_TEAM_IDS = (0, 1)


@dataclass(frozen=True)
class SemanticObservation:
    """A single per-frame semantic prediction for one tracked target."""

    role: str = UNKNOWN_ROLE
    role_confidence: float = 0.0
    team_id: int = UNKNOWN_TEAM_ID
    team_confidence: float = 0.0
    frame_index: Optional[int] = None


@dataclass
class TrackSemanticState:
    """Stable semantic state and private evidence history for one track."""

    track_id: int
    role: str = UNKNOWN_ROLE
    role_confidence: float = 0.0
    team_id: int = UNKNOWN_TEAM_ID
    team_confidence: float = 0.0
    semantic_status: str = "unknown"
    last_seen_frame: Optional[int] = None
    last_update_frame: Optional[int] = None
    role_switches: int = 0
    team_switches: int = 0
    _role_history: Deque[SemanticObservation] = field(default_factory=deque, repr=False)
    _team_history: Deque[SemanticObservation] = field(default_factory=deque, repr=False)

    def as_dict(self) -> dict[str, Any]:
        """Return a wire-model-friendly snapshot without private history."""

        return {
            "track_id": self.track_id,
            "role": self.role,
            "role_confidence": self.role_confidence,
            "team_id": self.team_id,
            "team_confidence": self.team_confidence,
            "semantic_status": self.semantic_status,
            "last_seen_frame": self.last_seen_frame,
            "last_update_frame": self.last_update_frame,
        }


@dataclass(frozen=True)
class SemanticResult:
    """Public per-track semantic result returned by the frame-level adapter."""

    track_id: int
    role: str = UNKNOWN_ROLE
    team_id: int = UNKNOWN_TEAM_ID
    role_confidence: float = 0.0
    team_confidence: float = 0.0
    status: str = "unknown"

    @property
    def semantic_status(self) -> str:
        """Compatibility alias for callers using the state-model name."""

        return self.status

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "role": self.role,
            "team_id": self.team_id,
            "role_confidence": self.role_confidence,
            "team_confidence": self.team_confidence,
            "status": self.status,
        }


class TrajectorySemanticManager:
    """Maintain smoothed role/team labels per ByteTrack trajectory.

    Evidence is kept in a bounded time window.  Each observation contributes
    ``confidence * decay ** age`` to its label score.  A new label must clear
    both the confidence threshold and a hysteresis margin before replacing a
    current known label; this prevents one-frame semantic flicker.
    """

    def __init__(
        self,
        *,
        history_size: int = 12,
        recency_decay: float = 0.90,
        stable_threshold: float = 0.55,
        switch_margin: float = 0.10,
        max_missing_frames: int = 30,
    ) -> None:
        if history_size < 1:
            raise ValueError("history_size must be at least 1")
        if not 0.0 < recency_decay <= 1.0:
            raise ValueError("recency_decay must be in (0, 1]")
        if not 0.5 <= stable_threshold <= 1.0:
            raise ValueError("stable_threshold must be between 0.5 and 1.0")
        if switch_margin < 0.0:
            raise ValueError("switch_margin must be non-negative")
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")
        self.history_size = int(history_size)
        self.recency_decay = float(recency_decay)
        self.stable_threshold = float(stable_threshold)
        self.switch_margin = float(switch_margin)
        self.max_missing_frames = int(max_missing_frames)
        self._states: dict[int, TrackSemanticState] = {}
        self.semantic_label_switches = 0

    @property
    def states(self) -> Mapping[int, TrackSemanticState]:
        return self._states

    def get(self, track_id: int) -> Optional[TrackSemanticState]:
        return self._states.get(int(track_id))

    def update(
        self,
        track_id: int,
        *,
        role: Any = None,
        role_confidence: float = 0.0,
        team_id: Any = None,
        team_confidence: float = 0.0,
        frame_index: Optional[int] = None,
    ) -> TrackSemanticState:
        """Add one observation and return the smoothed state.

        ``None`` or non-positive-confidence fields are treated as absent
        evidence.  Explicit unknown values with positive confidence are
        retained in the window, allowing sustained uncertainty to fall back
        to ``unknown/-1`` instead of keeping stale labels forever.
        """

        key = int(track_id)
        state = self._states.setdefault(key, TrackSemanticState(track_id=key))
        observation_frame = frame_index if frame_index is not None else state.last_update_frame
        observation = SemanticObservation(
            role=normalize_role(role),
            role_confidence=_confidence(role_confidence),
            team_id=normalize_team_id(team_id),
            team_confidence=_confidence(team_confidence),
            frame_index=observation_frame,
        )
        if role is not None and observation.role_confidence > 0.0:
            state._role_history.append(observation)
        if team_id is not None and observation.team_confidence > 0.0:
            state._team_history.append(observation)
        self._trim_history(state._role_history)
        self._trim_history(state._team_history)

        previous_role, previous_team = state.role, state.team_id
        state.role, state.role_confidence = self._resolve(
            state._role_history,
            current=state.role,
            unknown=UNKNOWN_ROLE,
            frame_index=frame_index,
        )
        state.team_id, state.team_confidence = self._resolve(
            state._team_history,
            current=state.team_id,
            unknown=UNKNOWN_TEAM_ID,
            frame_index=frame_index,
        )
        if previous_role != state.role and previous_role != UNKNOWN_ROLE:
            state.role_switches += 1
            self.semantic_label_switches += 1
        if previous_team != state.team_id and previous_team != UNKNOWN_TEAM_ID:
            state.team_switches += 1
            self.semantic_label_switches += 1
        state.last_seen_frame = frame_index if frame_index is not None else state.last_seen_frame
        state.last_update_frame = frame_index if frame_index is not None else state.last_update_frame
        state.semantic_status = _semantic_status(state.role, state.team_id)
        return state

    def update_observation(
        self,
        track_id: int,
        observation: SemanticObservation,
    ) -> TrackSemanticState:
        """Typed convenience wrapper around :meth:`update`."""

        return self.update(
            track_id,
            role=observation.role,
            role_confidence=observation.role_confidence,
            team_id=observation.team_id,
            team_confidence=observation.team_confidence,
            frame_index=observation.frame_index,
        )

    def update_many(
        self,
        observations: Iterable[tuple[int, SemanticObservation]],
        *,
        frame_index: Optional[int] = None,
    ) -> list[TrackSemanticState]:
        """Update multiple tracks, filling a shared frame index when needed."""

        states = []
        for track_id, observation in observations:
            if frame_index is not None and observation.frame_index is None:
                observation = SemanticObservation(
                    role=observation.role,
                    role_confidence=observation.role_confidence,
                    team_id=observation.team_id,
                    team_confidence=observation.team_confidence,
                    frame_index=frame_index,
                )
            states.append(self.update_observation(track_id, observation))
        return states

    def remove_stale(self, frame_index: int, *, max_missing_frames: Optional[int] = None) -> list[int]:
        """Remove tracks not observed for the configured number of frames."""

        limit = self.max_missing_frames if max_missing_frames is None else max_missing_frames
        removed: list[int] = []
        for track_id, state in list(self._states.items()):
            if state.last_seen_frame is None or frame_index - state.last_seen_frame > limit:
                removed.append(track_id)
                del self._states[track_id]
        return removed

    def clear(self) -> None:
        self._states.clear()
        self.semantic_label_switches = 0

    def _trim_history(self, history: Deque[SemanticObservation]) -> None:
        while len(history) > self.history_size:
            history.popleft()

    def _resolve(
        self,
        history: Deque[SemanticObservation],
        *,
        current: Any,
        unknown: Any,
        frame_index: Optional[int],
    ) -> tuple[Any, float]:
        if not history:
            return unknown, 0.0
        scores: dict[Any, float] = defaultdict(float)
        for history_index, observation in enumerate(history):
            age = 0
            if frame_index is not None and observation.frame_index is not None:
                age = max(0, frame_index - observation.frame_index)
            else:
                # Callers may not have a source frame number.  The bounded
                # deque still gives us a reliable newest-to-oldest order.
                age = len(history) - 1 - history_index
            weight = observation.role_confidence if unknown == UNKNOWN_ROLE else observation.team_confidence
            scores[observation.role if unknown == UNKNOWN_ROLE else observation.team_id] += weight * (
                self.recency_decay**age
            )
        total = float(sum(scores.values()))
        if total <= 0.0:
            return unknown, 0.0
        ordered = sorted(scores.items(), key=lambda item: (-item[1], str(item[0])))
        candidate, candidate_score = ordered[0]
        candidate_confidence = float(candidate_score / total)
        current_score = float(scores.get(current, 0.0))

        if current != unknown and candidate != current:
            # Keep the current known label until the challenger is both
            # sufficiently supported and clearly ahead of it.
            if (
                candidate_confidence < self.stable_threshold
                or candidate_score <= current_score + self.switch_margin
            ):
                return current, float(current_score / total)
        if candidate == unknown or candidate_confidence < self.stable_threshold:
            return unknown, candidate_confidence
        return candidate, candidate_confidence


class TrackSemanticManager:
    """Stable frame-level adapter for role/team trajectory semantics.

    The public contract is intentionally small::

        manager.update(frame, detections, frame_index) -> dict[int, SemanticResult]

    ``detections`` may be a ``supervision.Detections`` object or a lightweight
    iterable/mapping exposing ``track_id``/``tracker_id`` and ``xyxy``/``bbox``.
    Optional predictors are dependency-injected.  With no predictors, or when
    a predictor is unavailable/fails, the corresponding result is always
    ``role="unknown"`` or ``team_id=-1`` with zero confidence.
    """

    def __init__(
        self,
        *,
        role_classifier: Any = None,
        team_classifier: Any = None,
        **trajectory_options: Any,
    ) -> None:
        self.role_classifier = role_classifier
        self.team_classifier = team_classifier
        self._trajectory = TrajectorySemanticManager(**trajectory_options)

    @property
    def states(self) -> Mapping[int, TrackSemanticState]:
        return self._trajectory.states

    @property
    def semantic_label_switches(self) -> int:
        return self._trajectory.semantic_label_switches

    def get(self, track_id: int) -> Optional[TrackSemanticState]:
        return self._trajectory.get(track_id)

    def update(
        self,
        frame: Optional[np.ndarray],
        detections: Any,
        frame_index: int,
    ) -> dict[int, SemanticResult]:
        """Update current tracks and return results keyed by ByteTrack ID."""

        items = _detection_items(detections)
        crops = [_crop_from_frame(frame, item[1]) for item in items]
        roles, role_confidences, role_available = self._predict_roles(crops)
        teams, team_confidences, team_available = self._predict_teams(frame, crops)

        results: dict[int, SemanticResult] = {}
        for index, (track_id, _box) in enumerate(items):
            state = self._trajectory.update(
                track_id,
                role=roles[index] if role_available else None,
                role_confidence=role_confidences[index] if role_available else 0.0,
                team_id=teams[index] if team_available else None,
                team_confidence=team_confidences[index] if team_available else 0.0,
                frame_index=int(frame_index),
            )
            # A missing predictor must not leak an earlier cached label into
            # the public result.  This makes model startup/failure safe.
            role = state.role if role_available else UNKNOWN_ROLE
            role_confidence = state.role_confidence if role_available else 0.0
            team_id = state.team_id if team_available else UNKNOWN_TEAM_ID
            team_confidence = state.team_confidence if team_available else 0.0
            results[track_id] = SemanticResult(
                track_id=track_id,
                role=role,
                team_id=team_id,
                role_confidence=role_confidence,
                team_confidence=team_confidence,
                status=_semantic_status(role, team_id),
            )

        self._trajectory.remove_stale(int(frame_index))
        return results

    def clear(self) -> None:
        self._trajectory.clear()

    def _predict_roles(
        self,
        crops: list[np.ndarray],
    ) -> tuple[list[str], list[float], bool]:
        if self.role_classifier is None:
            return [UNKNOWN_ROLE] * len(crops), [0.0] * len(crops), False
        try:
            values, confidences = _predictor_output(self.role_classifier, crops)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            _log_predictor_failure("role", exc)
            return [UNKNOWN_ROLE] * len(crops), [0.0] * len(crops), False
        roles = [_normalize_role_prediction(value) for value in _as_sequence(values, len(crops))]
        return roles, _normalized_confidences(confidences, len(crops)), True

    def _predict_teams(
        self,
        frame: Optional[np.ndarray],
        crops: list[np.ndarray],
    ) -> tuple[list[int], list[float], bool]:
        if self.team_classifier is None:
            return [UNKNOWN_TEAM_ID] * len(crops), [0.0] * len(crops), False
        try:
            if hasattr(self.team_classifier, "collect_and_predict"):
                values = self.team_classifier.collect_and_predict(frame, crops)
                # OnlineTeamClassifier exposes confidence separately; use it
                # after collection so warmup still occurs exactly once.
                if hasattr(self.team_classifier, "predict_with_confidence"):
                    values, confidences = self.team_classifier.predict_with_confidence(crops)
                else:
                    confidences = np.where(np.asarray(values) >= 0, 0.5, 0.0)
            else:
                values, confidences = _predictor_output(self.team_classifier, crops)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            _log_predictor_failure("team", exc)
            return [UNKNOWN_TEAM_ID] * len(crops), [0.0] * len(crops), False
        teams = [normalize_team_id(value) for value in _as_sequence(values, len(crops))]
        return teams, _normalized_confidences(confidences, len(crops)), True


def _detection_items(detections: Any) -> list[tuple[int, Any]]:
    """Extract ``(track_id, box)`` pairs from supervision or simple fakes."""

    if detections is None:
        return []
    tracker_ids = getattr(detections, "tracker_id", None)
    if tracker_ids is not None:
        ids = _as_sequence(tracker_ids, len(tracker_ids))
        boxes = getattr(detections, "xyxy", None)
        boxes_sequence = _as_sequence(boxes, len(ids)) if boxes is not None else [None] * len(ids)
        items = []
        for track_id, box in zip(ids, boxes_sequence):
            normalized_id = _try_track_id(track_id)
            if normalized_id is not None:
                items.append((normalized_id, box))
        return items

    if isinstance(detections, Mapping):
        if "track_id" in detections or "tracker_id" in detections:
            track_id = _try_track_id(detections.get("track_id", detections.get("tracker_id")))
            return (
                [(track_id, detections.get("xyxy", detections.get("bbox")))]
                if track_id is not None
                else []
            )
        items = []
        for key, detection in detections.items():
            track_id = _try_track_id(key)
            if track_id is None:
                continue
            items.append((track_id, _detection_box(detection)))
        return items

    try:
        iterable = list(detections)
    except TypeError:
        return []
    items = []
    for detection in iterable:
        track_id = _try_track_id(_detection_value(detection, "track_id", "tracker_id"))
        if track_id is not None:
            items.append((track_id, _detection_box(detection)))
    return items


def _detection_box(detection: Any) -> Any:
    if isinstance(detection, Mapping):
        return detection.get("xyxy", detection.get("bbox"))
    return getattr(detection, "xyxy", getattr(detection, "bbox", None))


def _detection_value(detection: Any, *names: str) -> Any:
    if isinstance(detection, Mapping):
        for name in names:
            if name in detection:
                return detection[name]
        return None
    for name in names:
        value = getattr(detection, name, None)
        if value is not None:
            return value
    return None


def _try_track_id(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _crop_from_frame(frame: Optional[np.ndarray], box: Any) -> np.ndarray:
    """Return a clamped BGR crop, or an empty crop when no frame/box exists."""

    empty = np.empty((0, 0, 3), dtype=np.uint8)
    if frame is None or box is None:
        return empty
    image = np.asarray(frame)
    coordinates = np.asarray(box).reshape(-1)
    if image.ndim < 2 or coordinates.size < 4 or not np.isfinite(coordinates[:4]).all():
        return empty
    height, width = image.shape[:2]
    x1, y1, x2, y2 = coordinates[:4]
    left = max(0, min(width, int(np.floor(x1))))
    top = max(0, min(height, int(np.floor(y1))))
    right = max(0, min(width, int(np.ceil(x2))))
    bottom = max(0, min(height, int(np.ceil(y2))))
    if right <= left or bottom <= top:
        return empty
    return image[top:bottom, left:right].copy()


def _predictor_output(predictor: Any, crops: list[np.ndarray]) -> tuple[Any, Any]:
    if hasattr(predictor, "predict_with_confidence"):
        result = predictor.predict_with_confidence(crops)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, np.zeros(len(crops), dtype=np.float32)
    if hasattr(predictor, "predict"):
        values = predictor.predict(crops)
        return values, np.where(
            np.asarray(values) != UNKNOWN_TEAM_ID,
            0.5,
            0.0,
        )
    if callable(predictor):
        result = predictor(crops)
        if isinstance(result, tuple) and len(result) == 2:
            return result
        return result, np.zeros(len(crops), dtype=np.float32)
    raise AttributeError("predictor must provide predict_with_confidence, predict, or __call__")


def _as_sequence(values: Any, size: int) -> list[Any]:
    if size == 0:
        return []
    if values is None:
        return [None] * size
    if isinstance(values, np.ndarray):
        sequence = values.reshape(-1).tolist()
    elif isinstance(values, (list, tuple)):
        sequence = list(values)
    else:
        sequence = [values]
    if len(sequence) < size:
        sequence.extend([None] * (size - len(sequence)))
    return sequence[:size]


def _normalized_confidences(values: Any, size: int) -> list[float]:
    return [_confidence(value) for value in _as_sequence(values, size)]


def _normalize_role_prediction(value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        return {0: "player", 1: "goalkeeper", 2: "referee"}.get(int(value), UNKNOWN_ROLE)
    return normalize_role(value)


def _log_predictor_failure(kind: str, exc: Exception) -> None:
    logger.warning("%s semantic predictor unavailable; using UNKNOWN fallback: %s", kind, exc)

    def _predict_roles(
        self,
        crops: list[np.ndarray],
    ) -> tuple[list[str], list[float], bool]:
        if self.role_classifier is None:
            return [UNKNOWN_ROLE] * len(crops), [0.0] * len(crops), False
        try:
            values, confidences = _predictor_output(self.role_classifier, crops)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            _log_predictor_failure("role", exc)
            return [UNKNOWN_ROLE] * len(crops), [0.0] * len(crops), False
        roles = [_normalize_role_prediction(value) for value in values]
        return roles, _normalized_confidences(confidences, len(crops)), True

    def _predict_teams(
        self,
        frame: Optional[np.ndarray],
        crops: list[np.ndarray],
    ) -> tuple[list[int], list[float], bool]:
        if self.team_classifier is None:
            return [UNKNOWN_TEAM_ID] * len(crops), [0.0] * len(crops), False
        try:
            if hasattr(self.team_classifier, "collect_and_predict"):
                values = self.team_classifier.collect_and_predict(frame, crops)
                # OnlineTeamClassifier exposes confidence separately; use it
                # after collection so warmup still occurs exactly once.
                if hasattr(self.team_classifier, "predict_with_confidence"):
                    values, confidences = self.team_classifier.predict_with_confidence(crops)
                else:
                    confidences = np.where(np.asarray(values) >= 0, 0.5, 0.0)
            else:
                values, confidences = _predictor_output(self.team_classifier, crops)
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            _log_predictor_failure("team", exc)
            return [UNKNOWN_TEAM_ID] * len(crops), [0.0] * len(crops), False
        teams = [normalize_team_id(value) for value in _as_sequence(values, len(crops))]
        return teams, _normalized_confidences(confidences, len(crops)), True


def normalize_role(role: Any) -> str:
    """Normalize enum/string role values without importing Pydantic models."""

    if role is None:
        return UNKNOWN_ROLE
    value = getattr(role, "value", role)
    normalized = str(value).strip().lower().replace("_", "")
    aliases = {
        "player": "player",
        "goalkeeper": "goalkeeper",
        "keeper": "goalkeeper",
        "referee": "referee",
        "unknown": UNKNOWN_ROLE,
    }
    return aliases.get(normalized, UNKNOWN_ROLE)


def normalize_team_id(team_id: Any) -> int:
    if team_id is None:
        return UNKNOWN_TEAM_ID
    try:
        candidate = int(team_id)
    except (TypeError, ValueError):
        return UNKNOWN_TEAM_ID
    return candidate if candidate in KNOWN_TEAM_IDS else UNKNOWN_TEAM_ID


def _confidence(value: Any) -> float:
    try:
        return float(np.clip(float(value), 0.0, 1.0))
    except (TypeError, ValueError):
        return 0.0


def _semantic_status(role: str, team_id: int) -> str:
    if role != UNKNOWN_ROLE and team_id != UNKNOWN_TEAM_ID:
        return "stable"
    if role != UNKNOWN_ROLE or team_id != UNKNOWN_TEAM_ID:
        return "partial"
    return "unknown"
