from __future__ import annotations

from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch


class AppearanceFeatureExtractor:
    """Batched MobileNetV3-Small native pooled appearance embeddings.

    The extractor returns the native global-average-pooled feature size
    (currently 576 for torchvision's MobileNetV3-Small).  No untrained
    projection head is inserted.
    """

    def __init__(
        self,
        *,
        device: str = "cpu",
        model: Optional[Any] = None,
        pretrained: bool = True,
        batch_size: int = 32,
        input_height: int = 128,
        input_width: int = 64,
    ) -> None:
        self.device = torch.device(device)
        self.batch_size = max(1, int(batch_size))
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self._model = model
        self._pretrained = bool(pretrained)
        self._loaded = False
        self.feature_dim: Optional[int] = None

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> "AppearanceFeatureExtractor":
        if self._model is None:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

            weights = MobileNet_V3_Small_Weights.DEFAULT if self._pretrained else None
            network = mobilenet_v3_small(weights=weights)
            self._model = network
        self._model.eval()
        self._model.to(self.device)
        with torch.inference_mode():
            sample = torch.zeros(
                (1, 3, self.input_height, self.input_width),
                device=self.device,
            )
            features = self._forward_features(sample)
            self.feature_dim = int(features.shape[1])
        self._loaded = True
        return self

    def extract_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.empty((0, self.feature_dim or 0), dtype=np.float32)
        if not self.loaded:
            self.load()
        outputs: list[np.ndarray] = []
        for start in range(0, len(crops), self.batch_size):
            batch = torch.from_numpy(
                np.stack([self._preprocess(crop) for crop in crops[start:start + self.batch_size]])
            ).to(self.device)
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        features = self._forward_features(batch)
                else:
                    features = self._forward_features(batch)
            features = torch.nn.functional.normalize(features.float(), p=2, dim=1)
            outputs.append(features.cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def extract(self, crop: np.ndarray) -> np.ndarray:
        batch = self.extract_batch([crop])
        return batch[0] if len(batch) else np.empty(0, dtype=np.float32)

    def _forward_features(self, batch: torch.Tensor) -> torch.Tensor:
        assert self._model is not None
        # torchvision MobileNetV3 exposes features and avgpool on the full
        # classification model.  Supporting an injected model with a
        # forward_features method keeps tests independent of model weights.
        if hasattr(self._model, "forward_features"):
            features = self._model.forward_features(batch)
        else:
            features = self._model.features(batch)
        if features.ndim == 4:
            if hasattr(self._model, "avgpool"):
                features = self._model.avgpool(features)
            else:
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        return features

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        image = np.asarray(crop)
        if image.ndim < 3 or image.shape[2] < 3:
            image = np.zeros((self.input_height, self.input_width, 3), dtype=np.uint8)
        else:
            image = cv2.resize(
                image[..., :3],
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
        rgb = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray(
            [0.229, 0.224, 0.225], dtype=np.float32
        )
        return np.transpose(rgb, (2, 0, 1)).astype(np.float32)

