"""Grad-CAM localization for the SoccerNet VARS MViT model.

The classifier does not output boxes. We back-propagate the selected offence
logit through the last MViT transformer block, reduce the 8x7x7 token grid and
return the largest high-response component. Optical flow is an explicit
fallback and is reported as such to the UI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CROP_SIZE = 224
RESIZE_SHORT = 256
CAM_GRID = (8, 7, 7)
BOX_THRESHOLD = 0.6
TEMPORAL_TOP_FRACTION = 0.2
TEMPORAL_MIN_CONTRAST = 0.15
TEMPORAL_HIGH_THRESHOLD = 0.6
TEMPORAL_LOW_THRESHOLD = 0.35
TEMPORAL_SMOOTHING_KERNEL = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)


def largest_component_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return None
    idx = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    return tuple(int(stats[idx, column]) for column in (
        cv2.CC_STAT_LEFT,
        cv2.CC_STAT_TOP,
        cv2.CC_STAT_WIDTH,
        cv2.CC_STAT_HEIGHT,
    ))


def crop_to_original_percent(
    x: float,
    y: float,
    width: float,
    height: float,
    original_width: int,
    original_height: int,
) -> list[float]:
    scale = RESIZE_SHORT / min(original_width, original_height)
    resized_width = original_width * scale
    resized_height = original_height * scale
    x_original = (x + (resized_width - CROP_SIZE) / 2.0) / scale
    y_original = (y + (resized_height - CROP_SIZE) / 2.0) / scale
    rect = [
        round(100.0 * x_original / original_width, 1),
        round(100.0 * y_original / original_height, 1),
        round(100.0 * width / scale / original_width, 1),
        round(100.0 * height / scale / original_height, 1),
    ]
    rect[0] = max(0.0, min(rect[0], 100.0))
    rect[1] = max(0.0, min(rect[1], 100.0))
    rect[2] = max(1.0, min(rect[2], 100.0 - rect[0]))
    rect[3] = max(1.0, min(rect[3], 100.0 - rect[1]))
    return rect


def temporal_activity_scores(cam: np.ndarray) -> tuple[np.ndarray, bool, str | None]:
    """Reduce a T×H×W attribution volume to a stable normalized timeline."""
    volume = np.maximum(np.asarray(cam, dtype=np.float32), 0.0)
    if volume.ndim != 3 or volume.shape[0] < 1:
        raise ValueError("temporal CAM must have shape T x H x W")
    flattened = volume.reshape(volume.shape[0], -1)
    top_count = max(1, int(np.ceil(flattened.shape[1] * TEMPORAL_TOP_FRACTION)))
    top_values = np.partition(flattened, flattened.shape[1] - top_count, axis=1)[:, -top_count:]
    raw_scores = top_values.mean(axis=1)
    maximum = float(raw_scores.max())
    minimum = float(raw_scores.min())
    if maximum <= 1e-8:
        return np.zeros_like(raw_scores), False, "zero_response"
    contrast = (maximum - minimum) / maximum
    normalized = (raw_scores - minimum) / max(maximum - minimum, 1e-8)
    padded = np.pad(normalized, (1, 1), mode="edge")
    smoothed = np.convolve(padded, TEMPORAL_SMOOTHING_KERNEL, mode="valid")
    smooth_min = float(smoothed.min())
    smooth_max = float(smoothed.max())
    if smooth_max > smooth_min:
        smoothed = (smoothed - smooth_min) / (smooth_max - smooth_min)
    else:
        smoothed = np.zeros_like(smoothed)
    if contrast < TEMPORAL_MIN_CONTRAST:
        return smoothed.astype(np.float32), False, "flat_response"
    return smoothed.astype(np.float32), True, None


def temporal_bins(
    scores: np.ndarray,
    window_start_s: float,
    window_end_s: float,
) -> list[dict[str, float]]:
    values = np.asarray(scores, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return []
    start_s = max(0.0, float(window_start_s))
    end_s = max(start_s, float(window_end_s))
    edges = np.linspace(start_s, end_s, values.size + 1)
    return [
        {
            "start_s": round(float(edges[index]), 4),
            "end_s": round(float(edges[index + 1]), 4),
            "score": round(float(np.clip(score, 0.0, 1.0)), 4),
        }
        for index, score in enumerate(values)
    ]


def select_temporal_window(
    cam: np.ndarray,
    window_start_s: float,
    window_end_s: float,
) -> dict[str, Any]:
    """Select one contiguous high-response interval around the strongest bin."""
    scores, informative, reason = temporal_activity_scores(cam)
    bins = temporal_bins(scores, window_start_s, window_end_s)
    peak_index = int(np.argmax(scores)) if scores.size else 0
    result: dict[str, Any] = {
        "valid": informative,
        "reason": reason,
        "peak_index": peak_index,
        "active_indices": [],
        "temporal_bins": bins,
    }
    if not informative or not bins:
        return result

    left = peak_index
    right = peak_index
    while left > 0 and scores[left - 1] >= TEMPORAL_HIGH_THRESHOLD:
        left -= 1
    while right + 1 < scores.size and scores[right + 1] >= TEMPORAL_HIGH_THRESHOLD:
        right += 1
    while left > 0 and scores[left - 1] >= TEMPORAL_LOW_THRESHOLD:
        left -= 1
    while right + 1 < scores.size and scores[right + 1] >= TEMPORAL_LOW_THRESHOLD:
        right += 1

    if left == right and scores.size > 1:
        if left == 0:
            right = 1
        elif right == scores.size - 1:
            left -= 1
        elif scores[left - 1] >= scores[right + 1]:
            left -= 1
        else:
            right += 1

    if left == 0 and right == scores.size - 1:
        result.update(valid=False, reason="full_window_response")
        return result

    result.update(
        active_indices=list(range(left, right + 1)),
        active_start_s=bins[left]["start_s"],
        active_end_s=bins[right]["end_s"],
        peak_s=round((bins[peak_index]["start_s"] + bins[peak_index]["end_s"]) / 2.0, 4),
    )
    return result


def event_prior_window(
    event_time_s: float,
    duration_s: float | None = None,
    half_width_s: float = 0.5,
) -> dict[str, float | str]:
    peak_s = max(0.0, float(event_time_s))
    start_s = max(0.0, peak_s - half_width_s)
    end_s = peak_s + half_width_s
    if duration_s is not None and duration_s > 0:
        peak_s = min(peak_s, duration_s)
        start_s = min(start_s, duration_s)
        end_s = min(end_s, duration_s)
    return {
        "active_start_s": round(start_s, 4),
        "active_end_s": round(max(start_s, end_s), 4),
        "peak_s": round(peak_s, 4),
        "temporal_source": "event_prior",
    }


def gradcam_boxes(
    model: Any,
    batch: Any,
    offence_index: int,
    frame_sizes: list[tuple[int, int]],
    window_ranges: list[tuple[float, float]],
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    import torch

    backbone = model.mvnetwork.aggregation_model.model
    target_block = backbone.blocks[-1]
    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["activations"] = output[0] if isinstance(output, tuple) else output

    handle = target_block.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            offence_logits, _action_logits, _attention = model(batch)
            score = offence_logits.reshape(-1, 4)[0, offence_index]
            activations = captured["activations"]
            gradients = torch.autograd.grad(score, activations, retain_graph=False)[0]
    finally:
        handle.remove()

    batch_views, token_count, _ = activations.shape
    time_grid, height_grid, width_grid = CAM_GRID
    expected_tokens = 1 + time_grid * height_grid * width_grid
    if token_count != expected_tokens:
        raise RuntimeError(
            f"Unexpected Grad-CAM token count: {token_count}, expected {expected_tokens}"
        )

    weights = gradients.mean(dim=1)
    cam = torch.relu((activations * weights.unsqueeze(1)).sum(dim=-1)[:, 1:])
    cam = cam.reshape(batch_views, time_grid, height_grid, width_grid)
    boxes: dict[int, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "grid": list(CAM_GRID),
        "top_fraction": TEMPORAL_TOP_FRACTION,
        "min_contrast": TEMPORAL_MIN_CONTRAST,
        "high_threshold": TEMPORAL_HIGH_THRESHOLD,
        "low_threshold": TEMPORAL_LOW_THRESHOLD,
        "views": {},
    }
    view_count = min(int(batch.shape[1]), batch_views, len(frame_sizes), len(window_ranges))
    for view_index in range(view_count):
        volume = cam[view_index].detach().float().cpu().numpy()
        window_start_s, window_end_s = window_ranges[view_index]
        temporal = select_temporal_window(volume, window_start_s, window_end_s)
        diagnostics["views"][str(view_index)] = {
            "valid": temporal["valid"],
            "reason": temporal["reason"],
            "peak_index": temporal["peak_index"],
            "active_indices": temporal["active_indices"],
            "temporal_bins": temporal["temporal_bins"],
        }
        active_indices = temporal["active_indices"]
        grid = volume[active_indices].mean(axis=0) if active_indices else volume.mean(axis=0)
        maximum = float(grid.max())
        if maximum <= 0:
            continue
        heat = cv2.resize(grid / maximum, (CROP_SIZE, CROP_SIZE), cv2.INTER_CUBIC)
        bbox = largest_component_bbox(heat >= BOX_THRESHOLD)
        if bbox is None:
            flat_index = int(grid.argmax())
            grid_y, grid_x = divmod(flat_index, width_grid)
            cell = CROP_SIZE / width_grid
            half = 1.5 * cell
            center_x = (grid_x + 0.5) * cell
            center_y = (grid_y + 0.5) * cell
            bbox = (
                int(max(0, center_x - half)),
                int(max(0, center_y - half)),
                int(min(CROP_SIZE, half * 2)),
                int(min(CROP_SIZE, half * 2)),
            )
        x, y, width, height = bbox
        original_width, original_height = frame_sizes[view_index]
        box: dict[str, Any] = {
            "rect": crop_to_original_percent(
                x, y, width, height, original_width, original_height
            ),
            "score": round(float(heat[y : y + height, x : x + width].mean()), 3),
            "source": "gradcam",
            "temporal_bins": temporal["temporal_bins"],
        }
        if temporal["valid"]:
            box.update(
                active_start_s=temporal["active_start_s"],
                active_end_s=temporal["active_end_s"],
                peak_s=temporal["peak_s"],
                temporal_source="gradcam",
            )
        boxes[view_index] = box
    return boxes, diagnostics


def optical_flow_box(
    video_path: Path,
    start_frame: int,
    end_frame: int,
) -> dict[str, Any] | None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    frames: list[np.ndarray] = []
    try:
        index = 0
        while index < end_frame:
            ok, frame = capture.read()
            if not ok:
                break
            if index >= start_frame:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            index += 1
    finally:
        capture.release()
    if len(frames) < 2:
        return None
    first = cv2.resize(frames[0], (320, 180))
    last = cv2.resize(frames[-1], (320, 180))
    flow = cv2.calcOpticalFlowFarneback(first, last, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    maximum = float(magnitude.max())
    if maximum <= 1e-6:
        return None
    bbox = largest_component_bbox(magnitude >= maximum * 0.5)
    if bbox is None:
        return None
    x, y, width, height = bbox
    return {
        "rect": [
            round(100.0 * x / 320, 1),
            round(100.0 * y / 180, 1),
            round(100.0 * width / 320, 1),
            round(100.0 * height / 180, 1),
        ],
        "score": round(float(magnitude[y : y + height, x : x + width].mean() / maximum), 3),
        "source": "optical_flow",
    }
