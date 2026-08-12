#!/usr/bin/env python3
"""Train the three-phase MobileNetV3 pitch-perception baseline.

Phase order is fixed:
  1. SoccerNet/manual supervised training (default 30 epochs)
  2. accepted PnLCalib/TVCalib teacher distillation (default 10 epochs)
  3. accepted test1 teacher-consensus domain adaptation (default 5 epochs)

The test2 video has no command-line input by design and cannot enter training.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
import time
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from app.field_registration.models import build_pitch_perception_model
from app.field_registration.pitch_model import PitchModel
from experiments.field_registration.augmentation import PitchTrainingAugmenter
from experiments.field_registration.dataset import PitchRegistrationDataset
from experiments.field_registration.losses import dual_head_loss
from experiments.field_registration.research_assets import sha256_file
from experiments.field_registration.soccernet import SoccerNetCalibrationDataset
from experiments.field_registration.teacher_dataset import TeacherDistillationDataset


@dataclass(frozen=True)
class TrainingPhase:
    name: str
    epochs: int
    learning_rate_scale: float
    semantic_weight: float
    semantic_dice_weight: float
    heatmap_weight: float
    offset_weight: float
    semantic_distillation_weight: float
    landmark_distillation_weight: float


SUPERVISED = TrainingPhase("supervised", 30, 1.0, 1.0, 1.0, 1.0, 0.25, 0.0, 0.0)
DISTILLATION = TrainingPhase("distillation", 10, 0.35, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)
DOMAIN_ADAPTATION = TrainingPhase("domain_adaptation", 5, 0.10, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0)


def _move_batch(
    batch: Mapping[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "frame_index"
    }


def _loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch: dict[str, torch.Tensor],
    phase: TrainingPhase,
) -> tuple[torch.Tensor, dict[str, float]]:
    return dual_head_loss(
        outputs,
        batch,
        semantic_weight=phase.semantic_weight,
        semantic_dice_weight=phase.semantic_dice_weight,
        heatmap_weight=phase.heatmap_weight,
        offset_weight=phase.offset_weight,
        semantic_distillation_weight=phase.semantic_distillation_weight,
        landmark_distillation_weight=phase.landmark_distillation_weight,
    )


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_fp16: bool,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    foreground_intersection = 0
    foreground_union = 0
    pck_hits = 0
    pck_total = 0
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_fp16,
        ):
            outputs = model(batch["image"])
            loss, _ = _loss(outputs, batch, SUPERVISED)
        loss_sum += float(loss)
        semantic_prediction = outputs[0].argmax(dim=1) > 0
        semantic_truth = batch["semantic_target"] > 0
        foreground_intersection += int((semantic_prediction & semantic_truth).sum())
        foreground_union += int((semantic_prediction | semantic_truth).sum())
        predicted_flat = outputs[1].flatten(2).argmax(dim=2)
        target_flat = batch["landmark_heatmaps"].flatten(2)
        visible = batch.get(
            "landmark_visibility",
            (target_flat.max(dim=2).values > 0.5).float(),
        ) > 0.0
        target_indices = target_flat.argmax(dim=2)
        width = outputs[1].shape[-1]
        predicted_xy = torch.stack(
            (predicted_flat % width, predicted_flat // width), dim=-1
        )
        target_xy = torch.stack(
            (target_indices % width, target_indices // width), dim=-1
        )
        distance = torch.linalg.vector_norm((predicted_xy - target_xy).float(), dim=-1)
        pck_hits += int(((distance <= 2.5) & visible).sum())
        pck_total += int(visible.sum())
    return {
        "loss": loss_sum / max(len(loader), 1),
        "foreground_iou": foreground_intersection / max(foreground_union, 1),
        "pck_10px": pck_hits / max(pck_total, 1),
    }


def train_phase(
    model: torch.nn.Module,
    loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    phase: TrainingPhase,
    *,
    use_fp16: bool,
    history: list[dict[str, float | int | str]],
) -> dict[str, float]:
    for group in optimizer.param_groups:
        group["lr"] = float(group["base_lr"]) * phase.learning_rate_scale
    validation: dict[str, float] = {}
    for epoch in range(1, phase.epochs + 1):
        model.train()
        started = time.perf_counter()
        training_loss = 0.0
        for raw_batch in loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                outputs = model(batch["image"])
                loss, _ = _loss(outputs, batch, phase)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_loss += float(loss.detach())
        validation = validate(model, validation_loader, device, use_fp16)
        row: dict[str, float | int | str] = {
            "phase": phase.name,
            "epoch": epoch,
            "phase_epochs": phase.epochs,
            "train_loss": training_loss / max(len(loader), 1),
            "validation_loss": validation["loss"],
            "foreground_iou": validation["foreground_iou"],
            "pck_10px": validation["pck_10px"],
            "elapsed_sec": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
    return validation


def _supervised_dataset(
    *,
    soccernet_index: Path | None,
    manifests: Sequence[Path] | None,
    split: str,
    input_size: tuple[int, int],
    augment: bool,
) -> Dataset:
    augmenter = PitchTrainingAugmenter() if augment else None
    if soccernet_index is not None:
        return SoccerNetCalibrationDataset(
            soccernet_index,
            expected_split=split,
            input_size=input_size,
            augmenter=augmenter,
        )
    if manifests:
        return PitchRegistrationDataset(
            manifests,
            expected_split=split,
            input_size=input_size,
            augmenter=augmenter,
        )
    raise ValueError(f"{split} needs a SoccerNet index or manual manifests")


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    workers: int,
    shuffle: bool,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
    )


def _with_epochs(phase: TrainingPhase, epochs: int) -> TrainingPhase:
    return TrainingPhase(
        phase.name,
        max(int(epochs), 0),
        phase.learning_rate_scale,
        phase.semantic_weight,
        phase.semantic_dice_weight,
        phase.heatmap_weight,
        phase.offset_weight,
        phase.semantic_distillation_weight,
        phase.landmark_distillation_weight,
    )


def _pitch_model(dataset: Dataset) -> PitchModel:
    pitch_model = getattr(dataset, "pitch_model", None)
    if not isinstance(pitch_model, PitchModel):
        raise ValueError("training dataset does not expose pitch dimensions")
    return pitch_model


def _require_matching_geometry(reference: Dataset, candidate: Dataset, name: str) -> None:
    if _pitch_model(candidate).dimensions != _pitch_model(reference).dimensions:
        raise ValueError(f"{name} pitch dimensions differ from supervised data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    train_group = parser.add_mutually_exclusive_group(required=True)
    train_group.add_argument("--train-soccernet-index", type=Path)
    train_group.add_argument("--train-manifests", type=Path, nargs="+")
    validation_group = parser.add_mutually_exclusive_group(required=True)
    validation_group.add_argument("--validation-soccernet-index", type=Path)
    validation_group.add_argument("--validation-manifests", type=Path, nargs="+")
    parser.add_argument("--distillation-index", type=Path)
    parser.add_argument("--domain-adaptation-index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--supervised-epochs", type=int, default=30)
    parser.add_argument("--distillation-epochs", type=int, default=10)
    parser.add_argument("--domain-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--no-pretrained", action="store_true")
    arguments = parser.parse_args()
    if arguments.distillation_epochs > 0 and arguments.distillation_index is None:
        raise ValueError("distillation epochs require --distillation-index")
    if arguments.domain_epochs > 0 and arguments.domain_adaptation_index is None:
        raise ValueError("domain epochs require --domain-adaptation-index")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    device = torch.device(arguments.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    use_fp16 = device.type == "cuda"
    input_size = arguments.input_width, arguments.input_height
    train_dataset = _supervised_dataset(
        soccernet_index=arguments.train_soccernet_index,
        manifests=arguments.train_manifests,
        split="train",
        input_size=input_size,
        augment=True,
    )
    validation_dataset = _supervised_dataset(
        soccernet_index=arguments.validation_soccernet_index,
        manifests=arguments.validation_manifests,
        split="validation",
        input_size=input_size,
        augment=False,
    )
    if train_dataset.vocabulary != validation_dataset.vocabulary:  # type: ignore[attr-defined]
        raise ValueError("training and validation vocabularies differ")
    _require_matching_geometry(train_dataset, validation_dataset, "validation")
    vocabulary = train_dataset.vocabulary  # type: ignore[attr-defined]
    model = build_pitch_perception_model(
        "mobilenet_v3_dual_head",
        vocabulary.semantic_class_count,
        len(vocabulary.landmark_labels),
        pretrained=not arguments.no_pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    for group in optimizer.param_groups:
        group["base_lr"] = arguments.learning_rate
    scaler = torch.amp.GradScaler(device.type, enabled=use_fp16)
    validation_loader = _loader(
        validation_dataset,
        batch_size=arguments.batch_size,
        workers=arguments.workers,
        shuffle=False,
        pin_memory=use_fp16,
    )
    history: list[dict[str, float | int | str]] = []
    phases: list[tuple[TrainingPhase, Dataset]] = [
        (_with_epochs(SUPERVISED, arguments.supervised_epochs), train_dataset)
    ]
    if arguments.distillation_index is not None:
        phases.append(
            (
                _with_epochs(DISTILLATION, arguments.distillation_epochs),
                TeacherDistillationDataset(
                    arguments.distillation_index,
                    expected_split="distillation",
                    input_size=input_size,
                    augmenter=PitchTrainingAugmenter(),
                ),
            )
        )
    if arguments.domain_adaptation_index is not None:
        phases.append(
            (
                _with_epochs(DOMAIN_ADAPTATION, arguments.domain_epochs),
                TeacherDistillationDataset(
                    arguments.domain_adaptation_index,
                    expected_split="domain",
                    input_size=input_size,
                    augmenter=PitchTrainingAugmenter(),
                ),
            )
        )
    validation: dict[str, float] = {}
    for phase, dataset in phases:
        if phase.epochs == 0:
            continue
        if dataset.vocabulary != vocabulary:  # type: ignore[attr-defined]
            raise ValueError(f"{phase.name} vocabulary differs from supervised data")
        _require_matching_geometry(train_dataset, dataset, phase.name)
        loader = _loader(
            dataset,
            batch_size=arguments.batch_size,
            workers=arguments.workers,
            shuffle=True,
            pin_memory=use_fp16,
        )
        validation = train_phase(
            model,
            loader,
            validation_loader,
            device,
            optimizer,
            scaler,
            phase,
            use_fp16=use_fp16,
            history=history,
        )
    if not history:
        raise ValueError("at least one training phase must contain an epoch")

    input_files = [
        value
        for value in (
            arguments.train_soccernet_index,
            arguments.validation_soccernet_index,
            arguments.distillation_index,
            arguments.domain_adaptation_index,
        )
        if value is not None
    ] + list(arguments.train_manifests or ()) + list(arguments.validation_manifests or ())
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "architecture": "mobilenet_v3_dual_head",
            "input_size": list(input_size),
            "semantic_labels": list(vocabulary.semantic_labels),
            "landmark_labels": list(vocabulary.landmark_labels),
            "pitch_dimensions_m": asdict(_pitch_model(train_dataset).dimensions),
            "state_dict": model.state_dict(),
            "validation": validation,
            "training": {
                "seed": arguments.seed,
                "phases": [row.name for row, _ in phases if row.epochs > 0],
                "phase_sample_counts": {
                    row.name: len(dataset)
                    for row, dataset in phases
                    if row.epochs > 0
                },
                "input_hashes": {
                    str(path): sha256_file(path) for path in input_files
                },
                "test2_used": False,
            },
        },
        arguments.output,
    )
    arguments.output.with_suffix(".history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
