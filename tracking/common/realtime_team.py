from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
from sklearn.cluster import KMeans


MIN_VALID_PIXELS = 24


def _upper_body_crop(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    y1 = max(0, int(h * 0.08))
    y2 = max(y1 + 1, int(h * 0.55))
    x1 = max(0, int(w * 0.18))
    x2 = max(x1 + 1, int(w * 0.82))
    return crop[y1:y2, x1:x2]


def extract_color_feature(crop: np.ndarray) -> Optional[np.ndarray]:
    torso = _upper_body_crop(crop)
    if torso.size == 0:
        return None

    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(torso, cv2.COLOR_BGR2LAB)

    sat_mask = hsv[:, :, 1] > 40
    val_mask = hsv[:, :, 2] > 35
    mask = sat_mask & val_mask
    if int(mask.sum()) < MIN_VALID_PIXELS:
        mask = np.ones(hsv.shape[:2], dtype=bool)

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
    return feature


@dataclass
class TeamPrototypeModel:
    centers: np.ndarray

    @classmethod
    def fit(cls, crops: Iterable[np.ndarray]) -> "TeamPrototypeModel":
        features: List[np.ndarray] = []
        for crop in crops:
            feature = extract_color_feature(crop)
            if feature is not None:
                features.append(feature)

        if len(features) < 2:
            fallback = np.array(
                [
                    [0.05, 0.75, 0.80, 0.55, 0.75, 0.02, 0.05, 0.05],
                    [0.60, 0.75, 0.80, 0.45, 0.30, 0.02, 0.05, 0.05],
                ],
                dtype=np.float32,
            )
            return cls(centers=fallback)

        feature_matrix = np.stack(features)
        cluster_model = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = cluster_model.fit_predict(feature_matrix)
        centers = cluster_model.cluster_centers_.astype(np.float32)

        # 保持 team 0 / team 1 稳定：按 hue 升序排序
        center_hues = centers[:, 0]
        order = np.argsort(center_hues)
        remapped_centers = centers[order]
        return cls(centers=remapped_centers)

    def predict_feature(self, feature: np.ndarray) -> int:
        distances = np.linalg.norm(self.centers - feature[None, :], axis=1)
        return int(np.argmin(distances))

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        labels: List[int] = []
        for crop in crops:
            feature = extract_color_feature(crop)
            if feature is None:
                labels.append(0)
                continue
            labels.append(self.predict_feature(feature))
        return np.asarray(labels, dtype=int)


@dataclass
class TeamAssignmentSmoother:
    decay: float = 0.85
    confidence_boost: float = 1.0
    scores: Dict[int, np.ndarray] = field(default_factory=dict)

    def update(self, track_ids: List[int], raw_labels: np.ndarray) -> np.ndarray:
        if len(track_ids) == 0:
            return np.array([], dtype=int)

        smoothed: List[int] = []
        for track_id, label in zip(track_ids, raw_labels.tolist()):
            score = self.scores.get(track_id)
            if score is None:
                score = np.zeros(2, dtype=np.float32)
            else:
                score = score * self.decay
            score[int(label)] += self.confidence_boost
            self.scores[track_id] = score
            smoothed.append(int(np.argmax(score)))
        return np.asarray(smoothed, dtype=int)
