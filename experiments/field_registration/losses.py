"""Losses shared by field-perception candidate training scripts."""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def heatmap_focal_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    positive = targets >= 0.999
    negative = targets < 0.999
    negative_weights = (1.0 - targets).pow(4)
    positive_loss = -torch.log(probabilities) * (1.0 - probabilities).pow(2) * positive
    negative_loss = (
        -torch.log(1.0 - probabilities)
        * probabilities.pow(2)
        * negative_weights
        * negative
    )
    positive_count = positive.sum().clamp_min(1)
    return (positive_loss.sum() + negative_loss.sum()) / positive_count


def masked_offset_loss(
    predicted_offsets: torch.Tensor,
    target_offsets: torch.Tensor,
    landmark_mask: torch.Tensor,
) -> torch.Tensor:
    expanded_mask = landmark_mask.repeat_interleave(2, dim=1)
    values = functional.smooth_l1_loss(
        predicted_offsets,
        target_offsets,
        reduction="none",
    )
    return (values * expanded_mask).sum() / expanded_mask.sum().clamp_min(1.0)


def semantic_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    foreground_weight: float = 4.0,
) -> torch.Tensor:
    weights = torch.ones(logits.shape[1], device=logits.device, dtype=logits.dtype)
    weights[1:] = foreground_weight
    return functional.cross_entropy(logits, targets, weight=weights)


def dual_head_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    semantic_weight: float = 1.0,
    heatmap_weight: float = 1.0,
    offset_weight: float = 0.25,
) -> tuple[torch.Tensor, dict[str, float]]:
    semantic_logits, landmark_logits, predicted_offsets = outputs
    semantic = semantic_cross_entropy(semantic_logits, batch["semantic_target"])
    heatmap = heatmap_focal_loss(landmark_logits, batch["landmark_heatmaps"])
    offset = masked_offset_loss(
        predicted_offsets,
        batch["landmark_offsets"],
        batch["offset_mask"],
    )
    total = (
        semantic_weight * semantic
        + heatmap_weight * heatmap
        + offset_weight * offset
    )
    return total, {
        "semantic": float(semantic.detach()),
        "heatmap": float(heatmap.detach()),
        "offset": float(offset.detach()),
        "total": float(total.detach()),
    }
