"""Short-gap logical player entity continuity.

ByteTrack IDs are detector/tracker implementation details.  This module keeps
an independent logical entity ID across a short occlusion or a conservative
tracker reactivation.  It intentionally does not synthesize detections and it
never feeds predicted boxes back into the tracker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class EntityUpdate:
    """Result of one logical-entity update."""

    entity_ids: dict[int, int] = field(default_factory=dict)
    statuses: dict[int, str] = field(default_factory=dict)
    rebindings: dict[int, int] = field(default_factory=dict)
    predicted_frames: int = 0
    reactivated_count: int = 0
    fragmentations: int = 0


@dataclass
class _EntityState:
    entity_id: int
    tracker_id: int
    bbox: np.ndarray
    velocity: np.ndarray
    descriptor: np.ndarray | None
    last_frame: int
    missing_frames: int = 0
    status: str = "detected"

    def predicted_bbox(self, gap: int | None = None) -> np.ndarray:
        frames = self.missing_frames if gap is None else gap
        return self.bbox + self.velocity * float(max(frames, 0))


class TrackEntityManager:
    """Conservatively stitch short tracker fragments into logical entities."""

    def __init__(
        self,
        *,
        max_prediction_gap_frames: int = 6,
        reactivation_window_frames: int = 12,
        min_appearance_similarity: float = 0.30,
    ) -> None:
        if max_prediction_gap_frames < 0:
            raise ValueError("max_prediction_gap_frames must be non-negative")
        if reactivation_window_frames < max_prediction_gap_frames:
            raise ValueError(
                "reactivation_window_frames must be >= max_prediction_gap_frames"
            )
        if not 0.0 <= min_appearance_similarity <= 1.0:
            raise ValueError("min_appearance_similarity must be in [0, 1]")
        self.max_prediction_gap_frames = int(max_prediction_gap_frames)
        self.reactivation_window_frames = int(reactivation_window_frames)
        self.min_appearance_similarity = float(min_appearance_similarity)
        self._next_entity_id = 1
        self._states: dict[int, _EntityState] = {}
        self._tracker_to_entity: dict[int, int] = {}

    @property
    def states(self) -> dict[int, _EntityState]:
        return dict(self._states)

    def reset(self) -> None:
        self._next_entity_id = 1
        self._states.clear()
        self._tracker_to_entity.clear()

    def update(
        self,
        detections: Any,
        frame: np.ndarray | None,
        frame_index: int,
    ) -> EntityUpdate:
        tracker_ids, boxes = _detection_items(detections)
        if not tracker_ids:
            return self._advance_missing(frame_index)

        observed_ids = set(tracker_ids)
        entity_ids: dict[int, int] = {}
        statuses: dict[int, str] = {}
        rebindings: dict[int, int] = {}
        used_entities: set[int] = set()
        reactivated_count = 0
        fragmentations = 0

        descriptors = [
            _appearance_descriptor(frame, box)
            for box in boxes
        ]

        # First update tracker IDs that are already attached to an entity.
        unresolved: list[tuple[int, np.ndarray, np.ndarray | None]] = []
        for tracker_id, box, descriptor in zip(tracker_ids, boxes, descriptors):
            entity_id = self._tracker_to_entity.get(tracker_id)
            state = self._states.get(entity_id) if entity_id is not None else None
            if state is None or entity_id in used_entities:
                unresolved.append((tracker_id, box, descriptor))
                continue
            previous_missing = state.missing_frames
            self._update_state(state, tracker_id, box, descriptor, frame_index)
            used_entities.add(entity_id)
            entity_ids[tracker_id] = entity_id
            statuses[tracker_id] = "reactivated" if previous_missing else "detected"
            if previous_missing:
                reactivated_count += 1

        # New tracker IDs may be a reactivation of a recently lost entity.
        for tracker_id, box, descriptor in unresolved:
            candidate = self._best_candidate(
                box=box,
                descriptor=descriptor,
                frame_index=frame_index,
                used_entities=used_entities,
            )
            if candidate is None:
                entity_id = self._create_state(
                    tracker_id, box, descriptor, frame_index
                )
                status = "detected"
            else:
                state = candidate
                old_tracker_id = state.tracker_id
                previous_missing = state.missing_frames
                if old_tracker_id != tracker_id:
                    self._tracker_to_entity.pop(old_tracker_id, None)
                    rebindings[tracker_id] = old_tracker_id
                    status = "reactivated"
                else:
                    status = "detected"
                self._update_state(state, tracker_id, box, descriptor, frame_index)
                entity_id = state.entity_id
                if previous_missing:
                    reactivated_count += 1
            used_entities.add(entity_id)
            entity_ids[tracker_id] = entity_id
            statuses[tracker_id] = status

        predicted_frames = 0
        for entity_id, state in list(self._states.items()):
            if entity_id in used_entities:
                continue
            gap = max(1, frame_index - state.last_frame)
            state.missing_frames = gap
            if gap <= self.max_prediction_gap_frames:
                state.status = "predicted"
                predicted_frames += 1
            elif gap <= self.reactivation_window_frames:
                state.status = "lost"
            else:
                self._tracker_to_entity.pop(state.tracker_id, None)
                del self._states[entity_id]
                fragmentations += 1

        return EntityUpdate(
            entity_ids=entity_ids,
            statuses=statuses,
            rebindings=rebindings,
            predicted_frames=predicted_frames,
            reactivated_count=reactivated_count,
            fragmentations=fragmentations,
        )

    def _advance_missing(self, frame_index: int) -> EntityUpdate:
        predicted_frames = 0
        fragmentations = 0
        for entity_id, state in list(self._states.items()):
            gap = max(1, frame_index - state.last_frame)
            state.missing_frames = gap
            if gap <= self.max_prediction_gap_frames:
                state.status = "predicted"
                predicted_frames += 1
            elif gap <= self.reactivation_window_frames:
                state.status = "lost"
            else:
                self._tracker_to_entity.pop(state.tracker_id, None)
                del self._states[entity_id]
                fragmentations += 1
        return EntityUpdate(
            predicted_frames=predicted_frames,
            fragmentations=fragmentations,
        )

    def _create_state(
        self,
        tracker_id: int,
        box: np.ndarray,
        descriptor: np.ndarray | None,
        frame_index: int,
    ) -> int:
        entity_id = self._next_entity_id
        self._next_entity_id += 1
        self._states[entity_id] = _EntityState(
            entity_id=entity_id,
            tracker_id=tracker_id,
            bbox=box.copy(),
            velocity=np.zeros(4, dtype=np.float32),
            descriptor=descriptor.copy() if descriptor is not None else None,
            last_frame=frame_index,
        )
        self._tracker_to_entity[tracker_id] = entity_id
        return entity_id

    def _update_state(
        self,
        state: _EntityState,
        tracker_id: int,
        box: np.ndarray,
        descriptor: np.ndarray | None,
        frame_index: int,
    ) -> None:
        frame_delta = max(frame_index - state.last_frame, 1)
        observed_velocity = (box - state.bbox) / float(frame_delta)
        state.velocity = 0.65 * state.velocity + 0.35 * observed_velocity
        state.bbox = box.copy()
        if descriptor is not None:
            state.descriptor = descriptor.copy()
        state.last_frame = frame_index
        state.missing_frames = 0
        state.status = "detected"
        state.tracker_id = tracker_id
        self._tracker_to_entity[tracker_id] = state.entity_id

    def _best_candidate(
        self,
        *,
        box: np.ndarray,
        descriptor: np.ndarray | None,
        frame_index: int,
        used_entities: set[int],
    ) -> _EntityState | None:
        candidates: list[tuple[float, _EntityState]] = []
        for entity_id, state in self._states.items():
            if entity_id in used_entities or state.missing_frames <= 0:
                continue
            gap = frame_index - state.last_frame
            if gap < 1 or gap > self.reactivation_window_frames:
                continue
            predicted = state.predicted_bbox(gap)
            distance = _center_distance(box, predicted)
            scale = max(_box_diagonal(predicted), 1.0)
            gate = max(40.0, 1.8 * scale + float(np.linalg.norm(state.velocity[:2])) * gap)
            if distance > gate:
                continue
            similarity = _cosine_similarity(descriptor, state.descriptor)
            if similarity is not None and similarity < self.min_appearance_similarity:
                continue
            spatial_cost = distance / gate
            appearance_cost = 0.0 if similarity is None else 0.25 * (1.0 - similarity)
            candidates.append((spatial_cost + appearance_cost, state))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        best_cost, best_state = candidates[0]
        if best_cost > 0.95:
            return None
        if len(candidates) > 1 and candidates[1][0] - best_cost < 0.15:
            # Ambiguous overlap: do not force a logical rebind.
            return None
        return best_state


def _detection_items(detections: Any) -> tuple[list[int], list[np.ndarray]]:
    tracker_ids = getattr(detections, "tracker_id", None)
    boxes = getattr(detections, "xyxy", None)
    if tracker_ids is None or boxes is None:
        return [], []
    ids = [int(value) for value in np.asarray(tracker_ids).reshape(-1)]
    rows = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    count = min(len(ids), len(rows))
    return ids[:count], [rows[index].copy() for index in range(count)]


def _appearance_descriptor(
    frame: np.ndarray | None,
    box: np.ndarray,
) -> np.ndarray | None:
    if frame is None or frame.ndim != 3 or frame.shape[2] < 3:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(value)) for value in box]
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    if x2 - x1 < 8 or y2 - y1 < 12:
        return None
    crop = frame[y1:y2, x1:x2]
    crop = crop[int(crop.shape[0] * 0.15): int(crop.shape[0] * 0.65)]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (16, 8), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).astype(np.float32)
    descriptor = hsv[..., :2].reshape(-1)
    norm = float(np.linalg.norm(descriptor))
    if norm <= 1e-6:
        return None
    return descriptor / norm


def _cosine_similarity(
    left: np.ndarray | None,
    right: np.ndarray | None,
) -> float | None:
    if left is None or right is None or left.shape != right.shape:
        return None
    return float(np.clip(np.dot(left, right), -1.0, 1.0))


def _center_distance(left: np.ndarray, right: np.ndarray) -> float:
    left_center = np.array([(left[0] + left[2]) / 2, (left[1] + left[3]) / 2])
    right_center = np.array([(right[0] + right[2]) / 2, (right[1] + right[3]) / 2])
    return float(np.linalg.norm(left_center - right_center))


def _box_diagonal(box: np.ndarray) -> float:
    return float(np.linalg.norm([box[2] - box[0], box[3] - box[1]]))


__all__ = ["EntityUpdate", "TrackEntityManager"]
