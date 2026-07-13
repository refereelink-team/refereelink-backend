from __future__ import annotations

import logging
import pickle
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from app.classification.team import TeamClassifier
from app.constants.classes import PLAYER_CLASS_ID

logger = logging.getLogger(__name__)

WARMUP_FRAMES = 120
WARMUP_STRIDE = 15


class OnlineTeamClassifier:
    def __init__(
        self,
        device: str = "cpu",
        warmup_frames: int = WARMUP_FRAMES,
        warmup_stride: int = WARMUP_STRIDE,
        classifier_path: Optional[str] = None,
    ) -> None:
        self._device = device
        self._warmup_frames = warmup_frames
        self._warmup_stride = warmup_stride
        self._classifier = TeamClassifier(device=device)
        self._crops: list[np.ndarray] = []
        self._fitted = False
        self._fitting = False
        self._frame_count = 0
        self._lock = threading.Lock()

        if classifier_path is not None and Path(classifier_path).exists():
            self._load_classifier(classifier_path)

    def collect_and_predict(
        self,
        frame: np.ndarray,
        player_crops: list[np.ndarray],
    ) -> np.ndarray:
        with self._lock:
            self._frame_count += 1

            if not self._fitted and not self._fitting:
                if (
                    self._frame_count % self._warmup_stride == 0
                    and self._frame_count <= self._warmup_frames
                ):
                    for crop in player_crops:
                        if crop.size > 0:
                            self._crops.append(crop.copy())

                if self._frame_count >= self._warmup_frames:
                    self._start_fit()

            if not self._fitted:
                return np.full(len(player_crops), -1, dtype=np.int64)

            return self._classifier.predict(player_crops)

    def _start_fit(self) -> None:
        if len(self._crops) < 10:
            logger.warning("Not enough crops for team classifier (%d collected)", len(self._crops))
            self._fitted = True
            return
        self._fitting = True
        logger.info("Starting async team classifier fit with %d crops", len(self._crops))
        thread = threading.Thread(target=self._run_fit, daemon=True)
        thread.start()

    def _run_fit(self) -> None:
        try:
            self._classifier.fit(list(self._crops))
            with self._lock:
                self._fitted = True
                self._fitting = False
            self._crops.clear()
            logger.info("Team classifier fit complete")
        except Exception as exc:
            logger.error("Team classifier fit failed: %s", exc)
            with self._lock:
                self._fitted = True
                self._fitting = False

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._fitted

    def save(self, path: str) -> None:
        state = {
            "reducer": self._classifier.reducer,
            "cluster_model": self._classifier.cluster_model,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Team classifier state saved to %s", path)

    def _load_classifier(self, path: str) -> None:
        with open(path, "rb") as f:
            state = pickle.load(f)
        self._classifier.reducer = state["reducer"]
        self._classifier.cluster_model = state["cluster_model"]
        self._fitted = True
        logger.info("Team classifier state loaded from %s", path)
