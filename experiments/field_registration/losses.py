"""Losses shared by field-perception candidate training scripts."""

from __future__ import annotations

import torch
from torch.nn import functional as functional


def heatmap_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    visibility: torch.Tensor | None = None,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    positive = targets >= 0.999
    negative = targets < 0.999
    if visibility is None:
        visibility = targets.flatten(2).amax(dim=2) > 0.0
    if visibility.ndim != 2 or visibility.shape != targets.shape[:2]:
        raise ValueError("landmark visibility must have shape (N, C)")
    visible = visibility.to(dtype=torch.bool).unsqueeze(-1).unsqueeze(-1)
    positive = positive & visible
    negative = negative & visible
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


def class_balanced_semantic_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    maximum_class_weight: float = 8.0,
) -> torch.Tensor:
    class_count = logits.shape[1]
    counts = torch.bincount(targets.reshape(-1), minlength=class_count).to(logits)
    present = counts > 0
    frequencies = counts / counts.sum().clamp_min(1.0)
    weights = torch.zeros_like(frequencies)
    weights[present] = torch.rsqrt(frequencies[present].clamp_min(1e-8))
    weights[present] /= weights[present].mean().clamp_min(1e-8)
    weights = weights.clamp(max=maximum_class_weight)
    return functional.cross_entropy(logits, targets, weight=weights)


def semantic_soft_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    probabilities = functional.softmax(logits, dim=1)
    one_hot = functional.one_hot(
        targets,
        num_classes=logits.shape[1],
    ).permute(0, 3, 1, 2).to(probabilities)
    # Thin painted lines are the task; the overwhelmingly large background is
    # already represented in cross entropy and would dominate mean Dice.
    probabilities = probabilities[:, 1:]
    one_hot = one_hot[:, 1:]
    intersection = (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = probabilities.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    present = one_hot.sum(dim=(0, 2, 3)) > 0
    if not bool(present.any()):
        return logits.sum() * 0.0
    dice = (2.0 * intersection[present] + epsilon) / (
        denominator[present] + epsilon
    )
    return 1.0 - dice.mean()


def semantic_distillation_kl(
    student_logits: torch.Tensor,
    teacher_probabilities: torch.Tensor,
    teacher_mask: torch.Tensor,
) -> torch.Tensor:
    if teacher_probabilities.shape != student_logits.shape:
        raise ValueError("teacher semantic probabilities must match student logits")
    if teacher_mask.shape != student_logits.shape[:1] + student_logits.shape[2:]:
        raise ValueError("teacher semantic mask must have shape (N, H, W)")
    normalized_teacher = teacher_probabilities / teacher_probabilities.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-6)
    values = functional.kl_div(
        functional.log_softmax(student_logits, dim=1),
        normalized_teacher,
        reduction="none",
    ).sum(dim=1)
    mask = teacher_mask.to(values)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def landmark_distillation_bce(
    student_logits: torch.Tensor,
    teacher_heatmaps: torch.Tensor,
    teacher_visibility: torch.Tensor,
) -> torch.Tensor:
    if teacher_heatmaps.shape != student_logits.shape:
        raise ValueError("teacher landmark heatmaps must match student logits")
    if teacher_visibility.shape != student_logits.shape[:2]:
        raise ValueError("teacher landmark visibility must have shape (N, C)")
    values = functional.binary_cross_entropy_with_logits(
        student_logits,
        teacher_heatmaps,
        reduction="none",
    )
    mask = teacher_visibility.to(values).unsqueeze(-1).unsqueeze(-1)
    spatial_size = student_logits.shape[-2] * student_logits.shape[-1]
    return (values * mask).sum() / (mask.sum() * spatial_size).clamp_min(1.0)


def dual_head_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    semantic_weight: float = 1.0,
    semantic_dice_weight: float = 1.0,
    heatmap_weight: float = 1.0,
    offset_weight: float = 0.25,
    semantic_distillation_weight: float = 0.0,
    landmark_distillation_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    semantic_logits, landmark_logits, predicted_offsets = outputs
    semantic = class_balanced_semantic_cross_entropy(
        semantic_logits, batch["semantic_target"]
    )
    semantic_dice = semantic_soft_dice_loss(
        semantic_logits, batch["semantic_target"]
    )
    heatmap = heatmap_focal_loss(
        landmark_logits,
        batch["landmark_heatmaps"],
        batch.get("landmark_visibility"),
    )
    offset = masked_offset_loss(
        predicted_offsets,
        batch["landmark_offsets"],
        batch["offset_mask"],
    )
    total = (
        semantic_weight * semantic
        + semantic_dice_weight * semantic_dice
        + heatmap_weight * heatmap
        + offset_weight * offset
    )
    semantic_distillation = semantic_logits.sum() * 0.0
    landmark_distillation = landmark_logits.sum() * 0.0
    if semantic_distillation_weight > 0.0 and {
        "teacher_semantic_probabilities",
        "teacher_semantic_mask",
    } <= batch.keys():
        semantic_distillation = semantic_distillation_kl(
            semantic_logits,
            batch["teacher_semantic_probabilities"],
            batch["teacher_semantic_mask"],
        )
        total = total + semantic_distillation_weight * semantic_distillation
    if landmark_distillation_weight > 0.0 and {
        "teacher_landmark_heatmaps",
        "teacher_landmark_visibility",
    } <= batch.keys():
        landmark_distillation = landmark_distillation_bce(
            landmark_logits,
            batch["teacher_landmark_heatmaps"],
            batch["teacher_landmark_visibility"],
        )
        total = total + landmark_distillation_weight * landmark_distillation
    return total, {
        "semantic": float(semantic.detach()),
        "semantic_dice": float(semantic_dice.detach()),
        "heatmap": float(heatmap.detach()),
        "offset": float(offset.detach()),
        "semantic_distillation": float(semantic_distillation.detach()),
        "landmark_distillation": float(landmark_distillation.detach()),
        "total": float(total.detach()),
    }
