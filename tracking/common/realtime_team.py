from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

MIN_VALID_PIXELS = 24
MIN_TORSO_AREA = 14 * 14
MIN_FEATURE_CONFIDENCE = 0.18
DEFAULT_UPDATE_CONFIDENCE = 0.30


@dataclass
class ColorFeatureSample:
    feature: np.ndarray
    confidence: float
    valid_pixels: int
    total_pixels: int


@dataclass
class TeamPredictionBatch:
    labels: np.ndarray
    confidences: np.ndarray


@dataclass
class TeamPrototypeModel:
    centers: np.ndarray
    init_stats: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def fit(cls, crops: Iterable[np.ndarray]) -> "TeamPrototypeModel":
        valid_samples: List[ColorFeatureSample] = []
        total_crops = 0
        for crop in crops:
            total_crops += 1
            sample = extract_color_feature(crop)
            if sample is None or sample.confidence < MIN_FEATURE_CONFIDENCE:
                continue
            valid_samples.append(sample)

        stats = {
            "init_total_crops": float(total_crops),
            "init_valid_crops": float(len(valid_samples)),
        }

        if len(valid_samples) < 2:
            fallback = np.array(
                [
                    [0.05, 0.75, 0.80, 0.55, 0.75, 0.02, 0.05, 0.05],
                    [0.60, 0.75, 0.80, 0.45, 0.30, 0.02, 0.05, 0.05],
                ],
                dtype=np.float32,
            )
            stats["init_fallback"] = 1.0
            return cls(centers=fallback, init_stats=stats)

        feature_matrix = np.stack([sample.feature for sample in valid_samples])
        cluster_model = KMeans(n_clusters=2, n_init=10, random_state=0)
        cluster_model.fit(feature_matrix)
        centers = cluster_model.cluster_centers_.astype(np.float32)

        center_hues = centers[:, 0]
        order = np.argsort(center_hues)
        remapped_centers = centers[order]
        return cls(centers=remapped_centers, init_stats=stats)

    def predict_feature(self, feature: np.ndarray) -> Tuple[int, float]:
        distances = np.linalg.norm(self.centers - feature[None, :], axis=1)
        best_idx = int(np.argmin(distances))
        ordered = np.sort(distances)
        margin = float(ordered[1] - ordered[0]) if len(ordered) > 1 else float(ordered[0])
        confidence = float(np.clip(margin / 0.35, 0.0, 1.0))
        return best_idx, confidence

    def predict(self, crops: List[np.ndarray]) -> TeamPredictionBatch:
        labels: List[int] = []
        confidences: List[float] = []
        for crop in crops:
            sample = extract_color_feature(crop)
            if sample is None:
                labels.append(0)
                confidences.append(0.0)
                continue
            label, margin_conf = self.predict_feature(sample.feature)
            labels.append(label)
            confidences.append(float(np.clip(0.55 * sample.confidence + 0.45 * margin_conf, 0.0, 1.0)))
        return TeamPredictionBatch(
            labels=np.asarray(labels, dtype=int),
            confidences=np.asarray(confidences, dtype=np.float32),
        )


def _upper_body_crop(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    y1 = max(0, int(h * 0.08))
    y2 = max(y1 + 1, int(h * 0.55))
    x1 = max(0, int(w * 0.18))
    x2 = max(x1 + 1, int(w * 0.82))
    return crop[y1:y2, x1:x2]


def extract_color_feature(crop: np.ndarray) -> Optional[ColorFeatureSample]:
    torso = _upper_body_crop(crop)
    if torso.size == 0:
        return None

    h, w = torso.shape[:2]
    total_pixels = int(h * w)
    if total_pixels < MIN_TORSO_AREA:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)

    sat_mask = hsv[:, :, 1] > 40
    val_mask = hsv[:, :, 2] > 35
    mask = sat_mask & val_mask
    valid_pixels = int(mask.sum())
    if valid_pixels < MIN_VALID_PIXELS:
        mask = np.ones(hsv.shape[:2], dtype=bool)
        valid_pixels = total_pixels

    hsv_pixels = hsv[mask]
    lab_pixels = lab[mask]
    if hsv_pixels.size == 0 or lab_pixels.size == 0:
        return None

    mean_hsv = hsv_pixels.mean(axis=0)
    mean_lab = lab_pixels.mean(axis=0)
    std_hsv = hsv_pixels.std(axis=0)

    feature = np.array(
        [
            mean_hsv[0] / 180.0,
            mean_hsv[1] / 255.0,
            mean_hsv[2] / 255.0,
            mean_lab[1] / 255.0,
            mean_lab[2] / 255.0,
            std_hsv[0] / 180.0,
            std_hsv[1] / 255.0,
            std_hsv[2] / 255.0,
        ],
        dtype=np.float32,
    )
    confidence = float(np.clip(valid_pixels / max(total_pixels, 1), 0.0, 1.0))
    return ColorFeatureSample(
        feature=feature,
        confidence=confidence,
        valid_pixels=valid_pixels,
        total_pixels=total_pixels,
    )


@dataclass
class TeamAssignmentSmoother:
    decay: float = 0.92
    confidence_boost: float = 1.5
    min_update_confidence: float = 0.35
    switch_margin: float = 0.65
    min_switch_streak: int = 3
    warmup_frames: int = 2
    scores: Dict[int, np.ndarray] = field(default_factory=dict)
    stable_labels: Dict[int, int] = field(default_factory=dict)
    candidate_labels: Dict[int, int] = field(default_factory=dict)
    candidate_streaks: Dict[int, int] = field(default_factory=dict)
    track_ages: Dict[int, int] = field(default_factory=dict)
    low_confidence_streaks: Dict[int, int] = field(default_factory=dict)

    def update(
        self,
        track_ids: List[int],
        raw_labels: np.ndarray,
        confidences: np.ndarray,
    ) -> np.ndarray:
        if len(track_ids) == 0:
            return np.array([], dtype=int)

        smoothed: List[int] = []
        for track_id, label, confidence in zip(track_ids, raw_labels.tolist(), confidences.tolist()):
            score = self.scores.get(track_id)
            if score is None:
                score = np.zeros(2, dtype=np.float32)
            else:
                score = score * self.decay

            track_age = self.track_ages.get(track_id, 0) + 1
            self.track_ages[track_id] = track_age

            is_confident = float(confidence) >= self.min_update_confidence
            if is_confident:
                score[int(label)] += self.confidence_boost * float(confidence)
                self.low_confidence_streaks[track_id] = 0
            else:
                self.low_confidence_streaks[track_id] = self.low_confidence_streaks.get(track_id, 0) + 1

            previous_label = self.stable_labels.get(track_id)
            candidate = int(np.argmax(score))
            sorted_score = np.sort(score)
            margin = float(sorted_score[-1] - sorted_score[-2]) if len(sorted_score) > 1 else float(sorted_score[-1])

            if previous_label is None:
                stable_label = candidate
                self.candidate_labels.pop(track_id, None)
                self.candidate_streaks.pop(track_id, None)
            elif not is_confident:
                stable_label = previous_label
            elif candidate == previous_label:
                stable_label = previous_label
                self.candidate_labels.pop(track_id, None)
                self.candidate_streaks.pop(track_id, None)
            elif margin < self.switch_margin or track_age <= self.warmup_frames:
                stable_label = previous_label
                self.candidate_labels[track_id] = candidate
                self.candidate_streaks[track_id] = 0
            else:
                previous_candidate = self.candidate_labels.get(track_id)
                if previous_candidate == candidate:
                    streak = self.candidate_streaks.get(track_id, 0) + 1
                else:
                    streak = 1
                self.candidate_labels[track_id] = candidate
                self.candidate_streaks[track_id] = streak

                if streak >= self.min_switch_streak:
                    stable_label = candidate
                    self.candidate_labels.pop(track_id, None)
                    self.candidate_streaks.pop(track_id, None)
                else:
                    stable_label = previous_label

            self.scores[track_id] = score
            self.stable_labels[track_id] = stable_label
            smoothed.append(stable_label)
        return np.asarray(smoothed, dtype=int)
