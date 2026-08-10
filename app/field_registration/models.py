"""Trainable lightweight dual-head field-perception baselines."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as functional
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large


class ConvNormActivation(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.Hardswish(inplace=True),
        )


class MobileNetV3PitchPerception(nn.Module):
    """MobileNetV3-Large with one shared stride-4 decoder and two heads.

    Semantic logits are returned at input resolution for thin-line training.
    Landmark heatmaps and subpixel offsets stay at stride four to keep memory
    bounded.  The 128-D projection discussed for team classification is not
    relevant here; no random embedding layer is introduced.
    """

    output_stride = 4

    def __init__(
        self,
        semantic_class_count: int,
        landmark_count: int,
        *,
        pretrained: bool = True,
        decoder_channels: int = 96,
    ) -> None:
        super().__init__()
        if semantic_class_count < 2 or landmark_count < 1:
            raise ValueError("dual-head model needs foreground classes and landmarks")
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)
        self.features = backbone.features
        self.low_projection = nn.Sequential(
            nn.Conv2d(24, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.Hardswish(inplace=True),
        )
        self.high_projection = nn.Sequential(
            nn.Conv2d(960, decoder_channels, 1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.Hardswish(inplace=True),
        )
        self.decoder = nn.Sequential(
            ConvNormActivation(decoder_channels + 48, decoder_channels),
            ConvNormActivation(decoder_channels, 64),
        )
        self.semantic_head = nn.Conv2d(64, semantic_class_count, 1)
        self.landmark_head = nn.Conv2d(64, landmark_count, 1)
        self.offset_head = nn.Conv2d(64, landmark_count * 2, 1)
        self.semantic_class_count = int(semantic_class_count)
        self.landmark_count = int(landmark_count)

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape (N, 3, H, W)")
        input_size = images.shape[-2:]
        feature = images
        low_feature: torch.Tensor | None = None
        for index, layer in enumerate(self.features):
            feature = layer(feature)
            if index == 3:
                low_feature = feature
        if low_feature is None:
            raise RuntimeError("MobileNetV3 low-resolution feature was not produced")
        low = self.low_projection(low_feature)
        high = self.high_projection(feature)
        high = functional.interpolate(
            high,
            size=low.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        decoded = self.decoder(torch.cat((low, high), dim=1))
        semantic = self.semantic_head(decoded)
        semantic = functional.interpolate(
            semantic,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        return semantic, self.landmark_head(decoded), self.offset_head(decoded)


def build_pitch_perception_model(
    architecture: Literal["mobilenet_v3_dual_head"],
    semantic_class_count: int,
    landmark_count: int,
    *,
    pretrained: bool = True,
) -> nn.Module:
    if architecture != "mobilenet_v3_dual_head":
        raise ValueError(f"unsupported pitch perception architecture: {architecture}")
    return MobileNetV3PitchPerception(
        semantic_class_count,
        landmark_count,
        pretrained=pretrained,
    )
