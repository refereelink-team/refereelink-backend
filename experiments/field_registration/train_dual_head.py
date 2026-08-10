#!/usr/bin/env python3
"""Train the MobileNetV3 dual-head field-perception baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from app.field_registration.models import build_pitch_perception_model
from experiments.field_registration.dataset import PitchRegistrationDataset
from experiments.field_registration.losses import dual_head_loss


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
        if key != "frame_index"
    }


@torch.inference_mode()
def _validate(
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
            loss, _ = dual_head_loss(outputs, batch)
        loss_sum += float(loss)
        semantic_prediction = outputs[0].argmax(dim=1) > 0
        semantic_truth = batch["semantic_target"] > 0
        foreground_intersection += int((semantic_prediction & semantic_truth).sum())
        foreground_union += int((semantic_prediction | semantic_truth).sum())
        predicted_flat = outputs[1].flatten(2).argmax(dim=2)
        target_flat = batch["landmark_heatmaps"].flatten(2)
        visible = target_flat.max(dim=2).values > 0.5
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--input-width", type=int, default=512)
    parser.add_argument("--input-height", type=int, default=288)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-pretrained", action="store_true")
    arguments = parser.parse_args()

    device = torch.device(arguments.device)
    use_fp16 = device.type == "cuda"
    input_size = (arguments.input_width, arguments.input_height)
    train_dataset = PitchRegistrationDataset(
        arguments.train_manifests,
        expected_split="train",
        input_size=input_size,
    )
    validation_dataset = PitchRegistrationDataset(
        arguments.validation_manifests,
        expected_split="validation",
        input_size=input_size,
    )
    if train_dataset.vocabulary != validation_dataset.vocabulary:
        raise ValueError("training and validation vocabularies differ")
    vocabulary = train_dataset.vocabulary
    assert vocabulary is not None
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
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.workers,
        pin_memory=use_fp16,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.workers,
        pin_memory=use_fp16,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    for epoch in range(1, arguments.epochs + 1):
        model.train()
        started = time.perf_counter()
        training_loss = 0.0
        for raw_batch in train_loader:
            batch = _move_batch(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_fp16,
            ):
                outputs = model(batch["image"])
                loss, _ = dual_head_loss(outputs, batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            training_loss += float(loss.detach())
        validation = _validate(model, validation_loader, device, use_fp16)
        row = {
            "epoch": epoch,
            "train_loss": training_loss / max(len(train_loader), 1),
            "validation_loss": validation["loss"],
            "foreground_iou": validation["foreground_iou"],
            "pck_10px": validation["pck_10px"],
            "elapsed_sec": time.perf_counter() - started,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True))
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            torch.save(
                {
                    "format_version": 1,
                    "architecture": "mobilenet_v3_dual_head",
                    "input_size": list(input_size),
                    "semantic_labels": list(vocabulary.semantic_labels),
                    "landmark_labels": list(vocabulary.landmark_labels),
                    "state_dict": model.state_dict(),
                    "validation": validation,
                },
                arguments.output,
            )
    history_path = arguments.output.with_suffix(".history.json")
    history_path.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
