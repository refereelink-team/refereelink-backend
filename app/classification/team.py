from typing import List

import numpy as np


class TeamClassifier:
    """Minimal stub — no heavy SigLIP/UMAP/KMeans."""

    def __init__(self, device: str = "cpu", batch_size: int = 32) -> None:
        self.device = device
        self.batch_size = batch_size

    def fit(self, crops: List[np.ndarray]) -> None:
        pass

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        if len(crops) == 0:
            return np.array([], dtype=np.int64)
        return np.zeros(len(crops), dtype=np.int64)
