"""Grad-CAM localization for the SoccerNet VARS MViT model.

The classifier does not output boxes. We back-propagate the selected offence
logit through an MViT transformer block and return the largest high-response
component. Temporal attribution controls *when* evidence is shown; spatial
attribution uses a soft event-time prior so unrelated motion elsewhere in the
model window cannot dominate the displayed region. Optical flow is an explicit
fallback and is reported as such to the UI.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CROP_SIZE = 224
RESIZE_SHORT = 256
CAM_TIME_BINS = 8
BOX_THRESHOLD = 0.6
TEMPORAL_TOP_FRACTION = 0.2
TEMPORAL_MIN_CONTRAST = 0.15
TEMPORAL_HIGH_THRESHOLD = 0.6
TEMPORAL_LOW_THRESHOLD = 0.35
TEMPORAL_MAX_ACTIVE_BINS = 4
TEMPORAL_SMOOTHING_KERNEL = np.asarray([0.25, 0.5, 0.25], dtype=np.float32)
SPATIAL_MIN_CONTRAST = 0.15
VIEW_ATTENTION_MIN_ABSOLUTE = 0.08
VIEW_ATTENTION_MIN_PEAK_RATIO = 0.35
EVENT_SPATIAL_SIGMA_S = 0.24
EVENT_ACTIVITY_FLOOR = 0.35


def infer_cam_grid(token_count: int, time_bins: int = CAM_TIME_BINS) -> tuple[int, int, int]:
    """Infer the T×H×W grid from an MViT token sequence including its CLS token."""
    spatial_tokens, remainder = divmod(int(token_count) - 1, int(time_bins))
    side = math.isqrt(max(0, spatial_tokens))
    if remainder or side * side != spatial_tokens:
        raise RuntimeError(
            f"Unexpected Grad-CAM token count: {token_count}; "
            f"cannot form {time_bins} square temporal grids"
        )
    return time_bins, side, side


def full_window_spatial_grid(cam: np.ndarray) -> np.ndarray:
    """Aggregate space independently from the temporal activation gate."""
    volume = np.maximum(np.asarray(cam, dtype=np.float32), 0.0)
    if volume.ndim != 3 or volume.shape[0] < 1:
        raise ValueError("spatiotemporal CAM must have shape T x H x W")
    return volume.mean(axis=0)


def event_weighted_spatial_grid(
    cam: np.ndarray,
    window_start_s: float,
    window_end_s: float,
    event_time_s: float | None,
    *,
    sigma_s: float = EVENT_SPATIAL_SIGMA_S,
) -> np.ndarray:
    """Aggregate CAM space with a soft prior around the known event time.

    The prior is deliberately soft: every temporal token keeps a non-zero
    contribution, while tokens near the event and with stronger temporal CAM
    activity receive more weight. If the event does not fall inside the model
    window, the stable full-window mean remains the fallback.
    """
    volume = np.maximum(np.asarray(cam, dtype=np.float32), 0.0)
    if volume.ndim != 3 or volume.shape[0] < 1:
        raise ValueError("spatiotemporal CAM must have shape T x H x W")
    start_s = float(window_start_s)
    end_s = float(window_end_s)
    if (
        event_time_s is None
        or not np.isfinite(event_time_s)
        or end_s <= start_s
        or not start_s <= float(event_time_s) <= end_s
    ):
        return full_window_spatial_grid(volume)

    edges = np.linspace(start_s, end_s, volume.shape[0] + 1, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2.0
    sigma = max(float(sigma_s), (end_s - start_s) / volume.shape[0] / 2.0, 1e-6)
    event_weights = np.exp(-0.5 * ((centers - float(event_time_s)) / sigma) ** 2)
    activity_scores, _informative, _reason = temporal_activity_scores(volume)
    activity_weights = EVENT_ACTIVITY_FLOOR + (1.0 - EVENT_ACTIVITY_FLOOR) * activity_scores
    weights = event_weights * activity_weights
    total = float(weights.sum())
    if total <= 1e-8:
        return full_window_spatial_grid(volume)
    weights = weights / total
    return np.tensordot(weights.astype(np.float32), volume, axes=(0, 0))


def attention_gate(
    view_index: int,
    view_attention: list[float] | None,
) -> tuple[bool, float, list[str]]:
    """Reject views whose fusion attention is negligible relative to other views."""
    if not view_attention or view_index >= len(view_attention):
        return True, 1.0, []
    values = np.maximum(np.asarray(view_attention, dtype=np.float32), 0.0)
    peak = float(values.max()) if values.size else 0.0
    value = float(values[view_index])
    threshold = max(VIEW_ATTENTION_MIN_ABSOLUTE, peak * VIEW_ATTENTION_MIN_PEAK_RATIO)
    score = min(1.0, value / max(threshold, 1e-8))
    return value >= threshold, score, ([] if value >= threshold else ["low_view_attention"])


def localization_reliability(
    *,
    view_index: int,
    view_attention: list[float] | None,
    temporal_valid: bool,
    temporal_reason: str | None,
    spatial_grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    crop_size: int = CROP_SIZE,
) -> dict[str, Any]:
    """Classify CAM evidence as normal, cautionary, or hidden.

    A soft warning should not erase useful demo evidence. Only conditions that
    make the localization effectively unusable are hidden; boundary contact,
    broad temporal response, and weak contrast remain visible with a caution
    treatment in the UI.
    """
    attention_valid, attention_score, reasons = attention_gate(view_index, view_attention)
    if not temporal_valid:
        reasons.append(temporal_reason or "unreliable_temporal_response")

    grid = np.maximum(np.asarray(spatial_grid, dtype=np.float32), 0.0)
    maximum = float(grid.max()) if grid.size else 0.0
    median = float(np.median(grid)) if grid.size else 0.0
    spatial_contrast = (maximum - median) / max(maximum, 1e-8)
    if maximum <= 1e-8:
        reasons.append("zero_response")
    elif spatial_contrast < SPATIAL_MIN_CONTRAST:
        reasons.append("low_spatial_contrast")

    x, y, width, height = bbox
    if x <= 0 or y <= 0 or x + width >= crop_size or y + height >= crop_size:
        reasons.append("crop_boundary_contact")

    unique_reasons = list(dict.fromkeys(reasons))
    # Either warning alone can still be useful in a demo. Their conjunction is
    # materially different: the model barely used the view and its activation
    # points outside the observable center crop, which produced the grossly
    # unrelated wide-shot regions seen in CUDA evaluation.
    low_attention_at_boundary = (
        not attention_valid and "crop_boundary_contact" in unique_reasons
    )
    if "zero_response" in unique_reasons or low_attention_at_boundary:
        display_tier = "hidden"
    elif unique_reasons:
        display_tier = "caution"
    else:
        display_tier = "normal"

    temporal_score = 1.0 if temporal_valid else 0.45
    contrast_score = (
        0.0
        if maximum <= 1e-8
        else max(0.4, min(1.0, spatial_contrast / max(SPATIAL_MIN_CONTRAST, 1e-8)))
    )
    boundary_score = 0.6 if "crop_boundary_contact" in unique_reasons else 1.0
    reliability_score = min(attention_score, temporal_score, contrast_score, boundary_score)
    return {
        "display_tier": display_tier,
        "reliable": display_tier == "normal",
        "reliability_score": round(float(np.clip(reliability_score, 0.0, 1.0)), 4),
        "reliability_reasons": unique_reasons,
        "spatial_contrast": round(spatial_contrast, 4),
    }


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

    if left == 0 and right == scores.size - 1:
        result.update(valid=False, reason="full_window_response")
        return result

    # The MViT head only exposes eight temporal tokens. A broad response over
    # most of those tokens is not evidence for a frame-precise interval. Keep
    # the strongest contiguous sub-window around the peak so the UI does not
    # present an almost full-clip activation as a confident localization.
    if right - left + 1 > TEMPORAL_MAX_ACTIVE_BINS:
        first_start = max(left, peak_index - TEMPORAL_MAX_ACTIVE_BINS + 1)
        last_start = min(peak_index, right - TEMPORAL_MAX_ACTIVE_BINS + 1)
        candidates = range(first_start, last_start + 1)
        best_start = max(
            candidates,
            key=lambda start: (
                float(scores[start : start + TEMPORAL_MAX_ACTIVE_BINS].sum()),
                -abs((start + (TEMPORAL_MAX_ACTIVE_BINS - 1) / 2.0) - peak_index),
            ),
        )
        left = best_start
        right = best_start + TEMPORAL_MAX_ACTIVE_BINS - 1

    if left == right and scores.size > 1:
        if left == 0:
            right = 1
        elif right == scores.size - 1:
            left -= 1
        elif scores[left - 1] >= scores[right + 1]:
            left -= 1
        else:
            right += 1

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
    *,
    action_index: int | None = None,
    target_head: str = "offence",
    target_block_index: int = -1,
    view_attention: list[float] | None = None,
    event_time_s: float | None = None,
) -> tuple[dict[int, dict[str, Any]], dict[str, Any]]:
    import torch

    backbone = model.mvnetwork.aggregation_model.model
    target_block = backbone.blocks[target_block_index]
    captured: dict[str, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        captured["activations"] = output[0] if isinstance(output, tuple) else output

    handle = target_block.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            offence_logits, action_logits, _attention = model(batch)
            if target_head == "offence":
                score = offence_logits.reshape(-1, 4)[0, offence_index]
            elif target_head == "action" and action_index is not None:
                score = action_logits.reshape(-1, 8)[0, action_index]
            else:
                raise ValueError(f"Unsupported Grad-CAM target: {target_head}")
            activations = captured["activations"]
            gradients = torch.autograd.grad(score, activations, retain_graph=False)[0]
    finally:
        handle.remove()

    batch_views, token_count, _ = activations.shape
    time_grid, height_grid, width_grid = infer_cam_grid(token_count)

    weights = gradients.mean(dim=1)
    cam = torch.relu((activations * weights.unsqueeze(1)).sum(dim=-1)[:, 1:])
    cam = cam.reshape(batch_views, time_grid, height_grid, width_grid)
    boxes: dict[int, dict[str, Any]] = {}
    diagnostics: dict[str, Any] = {
        "grid": [time_grid, height_grid, width_grid],
        "target_head": target_head,
        "target_block_index": target_block_index,
        "spatial_aggregation": "event_time_weighted",
        "event_time_s": event_time_s,
        "event_spatial_sigma_s": EVENT_SPATIAL_SIGMA_S,
        "event_activity_floor": EVENT_ACTIVITY_FLOOR,
        "top_fraction": TEMPORAL_TOP_FRACTION,
        "min_contrast": TEMPORAL_MIN_CONTRAST,
        "high_threshold": TEMPORAL_HIGH_THRESHOLD,
        "low_threshold": TEMPORAL_LOW_THRESHOLD,
        "max_active_bins": TEMPORAL_MAX_ACTIVE_BINS,
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
        # Use a soft event prior rather than the hard display interval. This
        # preserves spatial stability while preventing unrelated motion in the
        # early/late parts of the clip from dominating the localization.
        grid = event_weighted_spatial_grid(
            volume,
            window_start_s,
            window_end_s,
            event_time_s,
        )
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
        reliability = localization_reliability(
            view_index=view_index,
            view_attention=view_attention,
            temporal_valid=bool(temporal["valid"]),
            temporal_reason=temporal["reason"],
            spatial_grid=grid,
            bbox=bbox,
        )
        diagnostics["views"][str(view_index)].update(reliability)
        original_width, original_height = frame_sizes[view_index]
        box: dict[str, Any] = {
            "rect": crop_to_original_percent(
                x, y, width, height, original_width, original_height
            ),
            "score": round(float(heat[y : y + height, x : x + width].mean()), 3),
            "source": "gradcam",
            "temporal_bins": temporal["temporal_bins"],
            "display_tier": reliability["display_tier"],
            "reliable": reliability["reliable"],
            "reliability_score": reliability["reliability_score"],
            "reliability_reasons": reliability["reliability_reasons"],
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
