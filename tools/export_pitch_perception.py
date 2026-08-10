#!/usr/bin/env python3
"""Export a trained pitch-perception checkpoint to ONNX or TensorRT.

The command never manufactures random production weights. TensorRT export is
performed through NVIDIA ``trtexec`` and fails with an actionable capability
report when the SDK is not installed.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import torch

from app.field_registration.models import build_pitch_perception_model


def deployment_capabilities() -> dict[str, object]:
    return {
        "onnx": importlib.util.find_spec("onnx") is not None,
        "onnxruntime": importlib.util.find_spec("onnxruntime") is not None,
        "tensorrt_python": importlib.util.find_spec("tensorrt") is not None,
        "trtexec": shutil.which("trtexec"),
        "cuda": torch.cuda.is_available(),
    }


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise ValueError("unsupported pitch-perception checkpoint format")
    required = {
        "architecture",
        "input_size",
        "semantic_labels",
        "landmark_labels",
        "state_dict",
        "validation",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"checkpoint is incomplete: missing {', '.join(missing)}")
    input_size = payload["input_size"]
    if not isinstance(input_size, (list, tuple)) or len(input_size) != 2:
        raise ValueError("checkpoint input_size must be [width, height]")
    model = build_pitch_perception_model(
        payload["architecture"],
        len(payload["semantic_labels"]) + 1,
        len(payload["landmark_labels"]),
        pretrained=False,
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval().to(device), payload


def build_trtexec_command(
    trtexec_path: str,
    onnx_path: Path,
    engine_path: Path,
    *,
    use_fp16: bool,
    workspace_mib: int,
) -> list[str]:
    command = [
        trtexec_path,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mib}",
        "--skipInference",
    ]
    if use_fp16:
        command.append("--fp16")
    return command


def export_onnx(
    model: torch.nn.Module,
    output_path: Path,
    *,
    input_size: tuple[int, int],
    device: torch.device,
    use_fp16: bool,
) -> None:
    if importlib.util.find_spec("onnx") is None:
        raise RuntimeError("ONNX is unavailable; install it with `uv sync --extra export`")
    width, height = input_size
    dtype = torch.float16 if use_fp16 else torch.float32
    model = model.to(dtype=dtype)
    example = torch.zeros((1, 3, height, width), device=device, dtype=dtype)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        output_path,
        input_names=["images"],
        output_names=["semantic_logits", "landmark_heatmaps", "landmark_offsets"],
        dynamic_axes={
            "images": {0: "batch"},
            "semantic_logits": {0: "batch"},
            "landmark_heatmaps": {0: "batch"},
            "landmark_offsets": {0: "batch"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx

    onnx.checker.check_model(onnx.load(str(output_path)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--format", choices=("onnx", "tensorrt"), default="onnx")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--workspace-mib", type=int, default=2048)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()

    capabilities = deployment_capabilities()
    device = torch.device(arguments.device)
    if arguments.fp16 and device.type != "cuda":
        raise ValueError("FP16 export requires a CUDA device")
    model, payload = load_trained_model(arguments.checkpoint, device)
    input_size = tuple(int(value) for value in payload["input_size"])
    onnx_path = (
        arguments.output
        if arguments.format == "onnx"
        else arguments.output.with_suffix(".onnx")
    )
    export_onnx(
        model,
        onnx_path,
        input_size=input_size,
        device=device,
        use_fp16=arguments.fp16,
    )
    if arguments.format == "tensorrt":
        trtexec_path = capabilities["trtexec"]
        if not isinstance(trtexec_path, str):
            raise RuntimeError(
                "TensorRT export requires NVIDIA trtexec; capability report: "
                + json.dumps(capabilities, sort_keys=True)
            )
        command = build_trtexec_command(
            trtexec_path,
            onnx_path,
            arguments.output,
            use_fp16=arguments.fp16,
            workspace_mib=arguments.workspace_mib,
        )
        subprocess.run(command, check=True)
    report = {
        "status": "ok",
        "format": arguments.format,
        "checkpoint": str(arguments.checkpoint),
        "output": str(arguments.output),
        "input_size": list(input_size),
        "precision": "fp16" if arguments.fp16 else "fp32",
        "validation": payload["validation"],
        "capabilities": capabilities,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.report is not None:
        arguments.report.parent.mkdir(parents=True, exist_ok=True)
        arguments.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
