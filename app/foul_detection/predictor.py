"""Pluggable foul predictors.

`offside.foul_model` (from the external `fouls_far` package) was never
completed and never shipped with this repository. This module replaces it
with a self-contained predictor protocol plus one concrete implementation
backed by the SoccerNet VARS MViT V2 Small checkpoint, shared with the
multi-view review module.

To swap in a different model (a lighter CNN, a rules-based detector, or a
custom-trained checkpoint), implement `FoulPredictor` and pass an instance
to `FoulDetector(..., predictor=...)`.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional, Protocol

import cv2
import numpy as np
import torch

from app.foul_detection.types import FoulPrediction


_ACTION_LABELS = {
    0: "Tackle",
    1: "Standing Tackle",
    2: "High Leg",
    3: "Holding",
    4: "Pushing",
    5: "Elbowing",
    6: "Challenge",
    7: "Dive",
}
_SEVERITY_LABELS = {
    0: "No Offence",
    1: "Offence + No Card",
    2: "Offence + Yellow Card",
    3: "Offence + Red Card",
}
_CARDS = {0: "none", 1: "none", 2: "yellow", 3: "red"}
_DECISIONS = {
    0: "no_offence",
    1: "offence_no_card",
    2: "yellow_card",
    3: "red_card",
}

# MViT V2 Small expects 16 frames in its temporal window.
_INPUT_FRAMES = 16
_CROP_SIZE = 224
# Kinetics-style normalization used by torchvision MViT weights.
_MEAN = (0.45, 0.45, 0.45)
_STD = (0.225, 0.225, 0.225)


class FoulPredictor(Protocol):
    """Interface every foul predictor must implement."""

    def predict(self, frames: list[np.ndarray]) -> Optional[FoulPrediction]:
        """Consume a rolling window of BGR frames, return a candidate or None."""


class MViTFoulPredictor:
    """SoccerNet VARS MViT V2 Small predictor.

    The checkpoint is the same `14_model.pth.tar` used by the multi-view
    review module. The live foul detector only has a single camera view,
    so the input tensor is replicated across the model's expected four
    attention slots. This is a pragmatic stand-in until a purpose-built
    single-view model is available.
    """

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.checkpoint_path = str(checkpoint_path)
        self._model = self._load_model(checkpoint_path)
        self.last_inference_ms = 0.0

    def _resolve_model_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parent.parent.parent
        return repo_root / "third_party" / "sn-mvfoul" / "VARS model"

    def _load_model(self, checkpoint_path: str) -> torch.nn.Module:
        model_dir = self._resolve_model_dir()
        if not (model_dir / "model.py").is_file():
            raise FileNotFoundError(
                f"SoccerNet VARS model source not found at {model_dir}. "
                "Run `bash tools/setup_mvfoul.sh` to fetch it."
            )
        if str(model_dir) not in sys.path:
            sys.path.insert(0, str(model_dir))

        from model import MVNetwork  # type: ignore[import-not-found]

        model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Foul checkpoint not found: {ckpt_path}")
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        except Exception:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        return model

    def predict(self, frames: list[np.ndarray]) -> Optional[FoulPrediction]:
        if not frames:
            return None
        started = time.perf_counter()
        tensor = self._preprocess(frames).to(self.device)
        # Replicate the single live view across the four attention slots the
        # model was trained with. The aggregation head tolerates this without
        # reshaping the checkpoint.
        tensor = tensor.repeat(1, 4, 1, 1, 1, 1)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    off_logits, act_logits, _attention = self._model(tensor)
            else:
                off_logits, act_logits, _attention = self._model(tensor)
        self.last_inference_ms = (time.perf_counter() - started) * 1000.0

        off_probs = torch.softmax(off_logits.float().reshape(-1, 4), dim=-1)[0]
        act_probs = torch.softmax(act_logits.float().reshape(-1, 8), dim=-1)[0]
        sev_index = int(torch.argmax(off_probs).item())
        act_index = int(torch.argmax(act_probs).item())
        confidence = float((off_probs[sev_index].item() + act_probs[act_index].item()) / 2.0)
        return FoulPrediction(
            confidence=confidence,
            action=_ACTION_LABELS[act_index],
            severity=_SEVERITY_LABELS[sev_index],
            card=_CARDS[sev_index],
            decision=_DECISIONS[sev_index],
            raw_scores={
                "severity_probs": off_probs.detach().cpu().tolist(),
                "action_probs": act_probs.detach().cpu().tolist(),
            },
        )

    def _preprocess(self, frames: list[np.ndarray]) -> torch.Tensor:
        n = len(frames)
        if n >= _INPUT_FRAMES:
            indices = np.linspace(0, n - 1, _INPUT_FRAMES).astype(int)
            picked = [frames[int(i)] for i in indices]
        else:
            picked = list(frames) + [frames[-1]] * (_INPUT_FRAMES - n)

        resized: list[np.ndarray] = []
        for frame in picked:
            image = np.asarray(frame)
            if image.ndim < 3 or image.shape[2] < 3:
                image = np.zeros((_CROP_SIZE, _CROP_SIZE, 3), dtype=np.uint8)
            else:
                image = image[..., :3]
            image = cv2.resize(image, (_CROP_SIZE, _CROP_SIZE))
            resized.append(image)

        arr = np.stack(resized, axis=0)  # (T, H, W, 3) BGR uint8
        # BGR -> RGB, [0, 1], normalize, permute to (C, T, H, W)
        arr = arr[..., ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr.copy()).permute(3, 0, 1, 2)
        mean = torch.tensor(_MEAN).view(3, 1, 1, 1)
        std = torch.tensor(_STD).view(3, 1, 1, 1)
        tensor = (tensor - mean) / std
        # Shape: (1, 1, C, T, H, W) -> batch=1, views=1
        return tensor.unsqueeze(0).unsqueeze(0)


__all__ = ["FoulPredictor", "MViTFoulPredictor"]
