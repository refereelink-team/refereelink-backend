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


def gradcam_boxes(
    model: Any,
    batch: Any,
    offence_index: int,
    frame_sizes: list[tuple[int, int]],
) -> dict[int, dict[str, Any]]:
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
    cam = cam.reshape(batch_views, time_grid, height_grid, width_grid).mean(dim=1)
    boxes: dict[int, dict[str, Any]] = {}
    view_count = min(int(batch.shape[1]), batch_views, len(frame_sizes))
    for view_index in range(view_count):
        grid = cam[view_index].detach().float().cpu().numpy()
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
        boxes[view_index] = {
            "rect": crop_to_original_percent(
                x, y, width, height, original_width, original_height
            ),
            "score": round(float(heat[y : y + height, x : x + width].mean()), 3),
            "source": "gradcam",
        }
    return boxes


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
