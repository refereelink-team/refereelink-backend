"""Lazy SoccerNet VARS MViT V2 Small multi-view inference.

The external SoccerNet model source and checkpoint remain runtime assets; no
GPL model source or large weights are committed to SC. Configure alternate
locations with SC_MVFOUL_CODE_PATH and SC_MVFOUL_WEIGHTS_PATH.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np

from app.constants.paths import REPO_ROOT_DIR, WEIGHTS_DIR
from app.multiview.labels import ACTION_LABELS, CARDS, DECISIONS, DECISIONS_ZH, SEVERITY_LABELS
from app.multiview.localization import event_prior_window, gradcam_boxes, optical_flow_box

DEFAULT_MODEL_DIR = REPO_ROOT_DIR / "third_party" / "sn-mvfoul" / "VARS model"
DEFAULT_WEIGHTS_PATH = WEIGHTS_DIR / "14_model.pth.tar"
DEFAULT_START_FRAME = 63
DEFAULT_END_FRAME = 87
DEFAULT_FPS = 17


class FoulInferenceError(RuntimeError):
    pass


def configured_model_dir() -> Path:
    return Path(os.environ.get("SC_MVFOUL_CODE_PATH", DEFAULT_MODEL_DIR)).expanduser()


def configured_weights_path() -> Path:
    return Path(os.environ.get("SC_MVFOUL_WEIGHTS_PATH", DEFAULT_WEIGHTS_PATH)).expanduser()


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_prerequisites() -> dict[str, Any]:
    try:
        import torch
        import torchvision  # noqa: F401

        torch_ok = True
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
        torch_error = None
    except Exception as exc:  # pragma: no cover - depends on local torch installation
        torch_ok = False
        cuda_available = False
        gpu_name = None
        torch_error = str(exc)
    model_dir = configured_model_dir()
    weights_path = configured_weights_path()
    code_ready = (model_dir / "model.py").is_file()
    weights_ready = weights_path.is_file() and weights_path.stat().st_size > 1_000_000
    missing: list[str] = []
    if not torch_ok:
        missing.append("torch/torchvision 不可用")
    if not code_ready:
        missing.append("SoccerNet sn-mvfoul VARS model 源码缺失")
    if not weights_ready:
        missing.append("14_model.pth.tar 权重缺失")
    return {
        "ready": torch_ok and code_ready and weights_ready,
        "torch_ok": torch_ok,
        "torch_error": torch_error,
        "cuda_available": cuda_available,
        "gpu_name": gpu_name,
        "model_code_ready": code_ready,
        "weights_ready": weights_ready,
        "model_dir": str(model_dir),
        "weights_path": str(weights_path),
        "missing": missing,
    }


ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ViewTiming:
    source_fps: float
    duration_s: float
    window_start_s: float
    window_end_s: float


class FoulInferenceService:
    def __init__(self, device: str = "auto") -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - dependency is required by pyproject
            raise FoulInferenceError("torch 未安装") from exc
        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise FoulInferenceError("指定了 CUDA，但 torch.cuda.is_available() 为 False")
        self.device = torch.device(device)
        self.start_frame = DEFAULT_START_FRAME
        self.end_frame = DEFAULT_END_FRAME
        self.fps = DEFAULT_FPS
        self._lock = threading.RLock()
        self.checkpoint_hash = checkpoint_sha256(configured_weights_path())
        self._model = self._load_model()
        from torchvision.models.video import MViT_V2_S_Weights

        self._transform = MViT_V2_S_Weights.KINETICS400_V1.transforms()

    def _load_model(self) -> Any:
        torch = self.torch
        model_dir = configured_model_dir()
        weights_path = configured_weights_path()
        if not (model_dir / "model.py").is_file():
            raise FoulInferenceError(f"SoccerNet VARS model 源码不存在：{model_dir}")
        if not weights_path.is_file() or weights_path.stat().st_size <= 1_000_000:
            raise FoulInferenceError(f"官方权重不存在：{weights_path}")
        path_string = str(model_dir)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)
        from model import MVNetwork  # type: ignore[import-not-found]

        model = MVNetwork(net_name="mvit_v2_s", agr_type="attention")
        try:
            checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        except Exception:
            checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        model.load_state_dict(state, strict=False)
        model.to(self.device).eval()
        return model

    @staticmethod
    def _read_video(path: Path) -> tuple[np.ndarray, float]:
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FoulInferenceError(f"视频无法打开：{path.name}")
        source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 25.0
        frames: list[np.ndarray] = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        if len(frames) < 2:
            raise FoulInferenceError(f"视频帧数不足：{path.name}")
        return np.stack(frames), source_fps

    def _load_view(self, path: Path) -> tuple[Any, tuple[int, int], ViewTiming]:
        frames, source_fps = self._read_video(path)
        original_size = (int(frames.shape[2]), int(frames.shape[1]))
        start = min(self.start_frame, max(0, len(frames) - 2))
        end = min(self.end_frame, len(frames))
        if end - start < 2:
            start, end = 0, len(frames)
        window = frames[start:end]
        sampling_factor = max(source_fps / self.fps, 1.0)
        sampled = [
            window[index]
            for index in range(len(window))
            if index % sampling_factor < 1
        ]
        if len(sampled) < 2:
            sampled = list(window)
        tensor = self.torch.from_numpy(np.stack(sampled)).permute(0, 3, 1, 2)
        transformed = self._transform(tensor)
        timing = ViewTiming(
            source_fps=source_fps,
            duration_s=len(frames) / source_fps,
            window_start_s=start / source_fps,
            window_end_s=end / source_fps,
        )
        return transformed, original_size, timing

    def _build_batch(
        self,
        views: Sequence[Path],
        report: ProgressCallback | None,
    ) -> tuple[Any, list[tuple[int, int]], list[ViewTiming], float]:
        started = time.perf_counter()
        tensors: list[Any] = []
        frame_sizes: list[tuple[int, int]] = []
        timings: list[ViewTiming] = []
        for index, path in enumerate(views):
            if report:
                report({"step": "decode_start", "pct": round(index / len(views) * 60), "view": path.name})
            view_started = time.perf_counter()
            tensor, frame_size, timing = self._load_view(path)
            tensors.append(tensor)
            frame_sizes.append(frame_size)
            timings.append(timing)
            if report:
                report({
                    "step": "decode_done",
                    "pct": round((index + 1) / len(views) * 60),
                    "view": path.name,
                    "ms": round((time.perf_counter() - view_started) * 1000, 1),
                })
        time_count = min(int(tensor.shape[1]) for tensor in tensors)
        batch = self.torch.stack([tensor[:, :time_count] for tensor in tensors]).unsqueeze(0)
        return batch, frame_sizes, timings, (time.perf_counter() - started) * 1000

    def analyze_views(
        self,
        view_paths: Sequence[Path],
        event_time_s: float = 3.0,
        progress: ProgressCallback | None = None,
    ) -> dict[str, Any]:
        views = sorted({path.resolve() for path in view_paths}, key=lambda item: item.name.lower())
        if not views:
            raise FoulInferenceError("没有可读取的多视角视频")
        if progress:
            progress({"step": "start", "pct": 0, "views": [item.name for item in views]})
        batch, frame_sizes, timings, preprocess_ms = self._build_batch(views, progress)
        batch = batch.to(self.device, non_blocking=True)
        torch = self.torch
        with self._lock:
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            with torch.inference_mode():
                if self.device.type == "cuda":
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        offence_logits, action_logits, attention = self._model(batch)
                    torch.cuda.synchronize()
                else:
                    offence_logits, action_logits, attention = self._model(batch)
            if torch.isnan(offence_logits).any() or torch.isnan(action_logits).any():
                with torch.inference_mode():
                    offence_logits, action_logits, attention = self._model(batch.float())
            inference_ms = (time.perf_counter() - started) * 1000
            gpu_mem_mb = (
                round(torch.cuda.max_memory_allocated() / 1e6, 1)
                if self.device.type == "cuda"
                else None
            )

        offence_probabilities = torch.softmax(offence_logits.float().reshape(-1, 4), dim=-1)[0]
        action_probabilities = torch.softmax(action_logits.float().reshape(-1, 8), dim=-1)[0]
        offence_index = int(torch.argmax(offence_probabilities).item())
        action_index = int(torch.argmax(action_probabilities).item())
        confidence = float(
            (offence_probabilities[offence_index] + action_probabilities[action_index]).item() / 2.0
        )
        action_candidates = sorted(
            (
                {"label": ACTION_LABELS[index], "confidence": round(float(probability), 4)}
                for index, probability in enumerate(action_probabilities.detach().cpu())
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )[:3]
        severity_candidates = sorted(
            (
                {"label": SEVERITY_LABELS[index], "confidence": round(float(probability), 4)}
                for index, probability in enumerate(offence_probabilities.detach().cpu())
            ),
            key=lambda item: item["confidence"],
            reverse=True,
        )[:3]

        localization_started = time.perf_counter()
        localization_source: str | None = None
        temporal_diagnostics: dict[str, Any]
        if offence_index == 0:
            localization = {}
            temporal_diagnostics = {"reason": "no_offence", "views": {}}
        else:
            try:
                with self._lock:
                    localization, temporal_diagnostics = gradcam_boxes(
                        self._model,
                        batch,
                        offence_index,
                        frame_sizes,
                        [(timing.window_start_s, timing.window_end_s) for timing in timings],
                    )
                for index, box in localization.items():
                    if box.get("active_start_s") is None:
                        box.update(event_prior_window(event_time_s, timings[index].duration_s))
                        temporal_diagnostics["views"][str(index)]["fallback"] = "event_prior"
                    elif timings[index].window_start_s <= event_time_s <= timings[index].window_end_s:
                        anchored_start = min(float(box["active_start_s"]), event_time_s)
                        anchored_end = max(float(box["active_end_s"]), event_time_s)
                        anchor_applied = (
                            anchored_start != float(box["active_start_s"])
                            or anchored_end != float(box["active_end_s"])
                        )
                        box["active_start_s"] = round(anchored_start, 4)
                        box["active_end_s"] = round(anchored_end, 4)
                        temporal_diagnostics["views"][str(index)][
                            "event_anchor_applied"
                        ] = anchor_applied
                fallback_views: list[int] = []
                for index, path in enumerate(views):
                    if index in localization:
                        continue
                    flow_box = optical_flow_box(path, self.start_frame, self.end_frame)
                    if flow_box:
                        flow_box.update(event_prior_window(event_time_s, timings[index].duration_s))
                        localization[index] = flow_box
                        fallback_views.append(index)
                        temporal_diagnostics["views"].setdefault(str(index), {}).update(
                            fallback="optical_flow_event_prior"
                        )
                localization_source = (
                    "gradcam+optical_flow" if fallback_views else "gradcam"
                ) if localization else None
            except Exception as exc:
                localization = {}
                temporal_diagnostics = {
                    "reason": "gradcam_failed",
                    "error": str(exc),
                    "views": {},
                }
                for index, path in enumerate(views):
                    box = optical_flow_box(path, self.start_frame, self.end_frame)
                    if box:
                        box.update(event_prior_window(event_time_s, timings[index].duration_s))
                        localization[index] = box
                        temporal_diagnostics["views"][str(index)] = {
                            "valid": False,
                            "reason": "optical_flow_event_prior",
                        }
                localization_source = "optical_flow" if localization else None
        gradcam_ms = (time.perf_counter() - localization_started) * 1000

        view_attention: list[float] = []
        if attention is not None and getattr(attention, "dim", lambda: 0)() == 2:
            view_attention = [round(float(item), 4) for item in attention[0].detach().float().cpu()]
        return {
            "action": ACTION_LABELS[action_index],
            "severity": SEVERITY_LABELS[offence_index],
            "card": CARDS[offence_index],
            "decision": DECISIONS[offence_index],
            "decision_zh": DECISIONS_ZH[offence_index],
            "confidence": round(confidence, 4),
            "action_candidates": action_candidates,
            "severity_candidates": severity_candidates,
            "checkpoint_hash": self.checkpoint_hash,
            "model": "MViT_V2_S",
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu",
            "inference_ms": round(inference_ms, 1),
            "preprocess_ms": round(preprocess_ms, 1),
            "gradcam_ms": round(gradcam_ms, 1),
            "gpu_mem_mb": gpu_mem_mb,
            "localization": {str(index): value for index, value in localization.items()},
            "localization_source": localization_source,
            "temporal_localization": {
                **temporal_diagnostics,
                "event_time_s": round(float(event_time_s), 4),
                "input_windows": [
                    {
                        "source_fps": round(timing.source_fps, 4),
                        "duration_s": round(timing.duration_s, 4),
                        "start_s": round(timing.window_start_s, 4),
                        "end_s": round(timing.window_end_s, 4),
                    }
                    for timing in timings
                ],
            },
            "view_attention": view_attention,
            "views": [path.name for path in views],
        }
