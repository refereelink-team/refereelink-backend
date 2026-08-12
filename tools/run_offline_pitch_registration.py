#!/usr/bin/env python3
"""Run forward/backward broadcast registration and render a debug MP4.

The command reads a local video twice. Pass ``--pitch-perception-checkpoint``
to run the dual-head student; omitting it intentionally selects the legacy
Ultralytics keypoint adapter as an E0/debug baseline. The saved JSON records
the selected perception backend.
"""

# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

try:
    from tools._bootstrap import ensure_repository_root
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from _bootstrap import ensure_repository_root

ensure_repository_root(__file__)

from app.constants.paths import PITCH_DETECTION_MODEL_PATH
from app.field_registration.offline import (
    BidirectionalCameraSmoother,
    OfflineCameraObservation,
    OfflineSmoothingConfig,
    save_offline_smoothing_result,
)
from app.field_registration.perception import (
    LegacyKeypointPerceptionBackend,
    PitchPerceptionBackend,
)
from app.field_registration.pitch_model import PitchDimensions, PitchModel
from app.field_registration.tracker import FieldRegistrationConfig, FieldRegistrationCore
from app.field_registration.torch_perception import TorchPitchPerceptionBackend
from app.field_registration.types import CameraState, MeasurementTier, RegistrationMode
from app.geometry.pitch_projection import build_pitch_point_references
from app.config.pitch import SoccerPitchConfiguration
from app.vision.backends import UltralyticsBackend


def _pitch_model_and_references() -> tuple[PitchModel, tuple[object, ...]]:
    model = PitchModel(PitchDimensions())
    dimensions = model.dimensions
    metric_config = SoccerPitchConfiguration(
        length=int(round(dimensions.length_m * 100.0)),
        width=int(round(dimensions.width_m * 100.0)),
        penalty_box_length=int(round(dimensions.penalty_area_depth_m * 100.0)),
        penalty_box_width=int(round(dimensions.penalty_area_width_m * 100.0)),
        goal_box_length=int(round(dimensions.goal_area_depth_m * 100.0)),
        goal_box_width=int(round(dimensions.goal_area_width_m * 100.0)),
        centre_circle_radius=int(round(dimensions.centre_circle_radius_m * 100.0)),
        penalty_spot_distance=int(round(dimensions.penalty_spot_distance_m * 100.0)),
    )
    return model, tuple(build_pitch_point_references(metric_config))


class LegacyPitchPredictor:
    def __init__(self, model_path: str, device: str, imgsz: int) -> None:
        self.backend = UltralyticsBackend(model_path, device=device)
        self.imgsz = imgsz
        self.fp16 = device.startswith("cuda") and torch.cuda.is_available()

    def __call__(self, frame: np.ndarray):
        result = self.backend.predict(
            frame,
            imgsz=self.imgsz,
            half=self.fp16,
        )[0]
        import supervision as sv

        return sv.KeyPoints.from_ultralytics(result)


def _core(
    backend: PitchPerceptionBackend,
    pitch_model: PitchModel,
    *,
    fps: float,
    semantic_interval: int,
) -> FieldRegistrationCore:
    return FieldRegistrationCore(
        pitch_model,
        backend,
        config=FieldRegistrationConfig(
            fps=fps,
            normal_semantic_interval=semantic_interval,
            stable_semantic_interval=max(semantic_interval * 2, 1),
            registration_mode=RegistrationMode.BROADCAST,
        ),
    )


def _video_metadata(path: Path) -> tuple[int, int, float, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(path)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if min(width, height, count) <= 0 or fps <= 0.0:
        raise ValueError("source video reports invalid metadata")
    return width, height, fps, count


def _forward_pass(
    source: Path,
    core: FieldRegistrationCore,
    *,
    frame_limit: int | None,
) -> tuple[list[OfflineCameraObservation], list[tuple[int, int]]]:
    capture = cv2.VideoCapture(str(source))
    observations: list[OfflineCameraObservation] = []
    boundaries: list[tuple[int, int]] = []
    index = 0
    while frame_limit is None or index < frame_limit:
        ok, frame = capture.read()
        if not ok:
            break
        result = core.process(frame, index)
        state = result.camera_state
        observations.append(
            OfflineCameraObservation(
                index,
                state.shot_id,
                (frame.shape[1], frame.shape[0]),
                state,
            )
        )
        if not boundaries or boundaries[-1][1] != state.shot_id:
            boundaries.append((index, state.shot_id))
        index += 1
    capture.release()
    return observations, boundaries


def _shot_ranges(observations: list[OfflineCameraObservation]) -> list[tuple[int, int, int]]:
    ranges: list[tuple[int, int, int]] = []
    start = 0
    while start < len(observations):
        shot_id = observations[start].shot_id
        end = start + 1
        while end < len(observations) and observations[end].shot_id == shot_id:
            end += 1
        ranges.append((start, end, shot_id))
        start = end
    return ranges


def _read_frame_block(source: Path, start: int, end: int) -> list[np.ndarray]:
    """Decode one half-open frame range sequentially.

    Seeking every frame is unreliable for long-GOP MP4 files.  A block seek
    normally lands at the preceding keyframe and OpenCV decodes to ``start``.
    If that backend path fails, a sequential decode from frame zero is slower
    but deterministic and only retains the requested block in memory.
    """

    if start < 0 or end <= start:
        raise ValueError("frame block must satisfy 0 <= start < end")

    def decode(*, seek: bool) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise FileNotFoundError(source)
        if seek:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start)
            index = start
        else:
            index = 0
        frames: list[np.ndarray] = []
        while index < end:
            ok, frame = capture.read()
            if not ok or frame is None:
                capture.release()
                return []
            if index >= start:
                frames.append(frame)
            index += 1
        capture.release()
        return frames

    expected = end - start
    frames = decode(seek=True)
    if len(frames) != expected:
        frames = decode(seek=False)
    if len(frames) != expected:
        raise RuntimeError(f"could not decode source frame block [{start}, {end})")
    return frames


def _backward_pass(
    source: Path,
    forward: list[OfflineCameraObservation],
    backend: PitchPerceptionBackend,
    pitch_model: PitchModel,
    *,
    fps: float,
    semantic_interval: int,
    chunk_frames: int = 240,
) -> list[OfflineCameraObservation]:
    if chunk_frames <= 0:
        raise ValueError("backward chunk size must be positive")
    output: list[OfflineCameraObservation] = []
    for start, end, shot_id in _shot_ranges(forward):
        core = _core(
            backend,
            pitch_model,
            fps=fps,
            semantic_interval=semantic_interval,
        )
        reverse_index = 0
        chunk_end = end
        while chunk_end > start:
            chunk_start = max(start, chunk_end - chunk_frames)
            frames = _read_frame_block(source, chunk_start, chunk_end)
            for offset in range(len(frames) - 1, -1, -1):
                source_index = chunk_start + offset
                frame = frames[offset]
                result = core.process(frame, reverse_index)
                state = result.camera_state
                if state.shot_id != 0:
                    # A detector-triggered false cut inside an already segmented
                    # forward Shot cannot be allowed to merge across boundaries.
                    core.reset()
                    result = core.process(frame, 0)
                    state = result.camera_state
                    reverse_index = 0
                state = CameraState(**{**state.__dict__, "shot_id": shot_id})
                output.append(
                    OfflineCameraObservation(
                        source_index,
                        shot_id,
                        (frame.shape[1], frame.shape[0]),
                        state,
                    )
                )
                reverse_index += 1
            chunk_end = chunk_start
    return output


def _draw_pitch_overlay(
    frame: np.ndarray,
    state: CameraState,
    pitch_model: PitchModel,
    frame_index: int,
) -> np.ndarray:
    output = frame.copy()
    color = {
        MeasurementTier.SAFE: (80, 230, 100),
        MeasurementTier.PREVIEW: (0, 190, 255),
        MeasurementTier.UNAVAILABLE: (80, 80, 220),
    }[state.measurement_tier]
    if state.pitch_to_image is not None:
        for points in pitch_model.semantic_elements(curve_samples=72).values():
            homogeneous = np.column_stack((points, np.ones(len(points))))
            projected = (state.pitch_to_image @ homogeneous.T).T
            valid = np.abs(projected[:, 2]) > 1e-9
            image = np.full((len(points), 2), np.nan)
            image[valid] = projected[valid, :2] / projected[valid, 2, None]
            finite = np.all(np.isfinite(image), axis=1)
            pixels = np.rint(image[finite]).astype(np.int32)
            if len(pixels) >= 2:
                cv2.polylines(
                    output,
                    [pixels.reshape(-1, 1, 2)],
                    False,
                    color,
                    2,
                    cv2.LINE_AA,
                )
    label = (
        f"f={frame_index} shot={state.shot_id} "
        f"{state.measurement_tier.value.upper()} {state.status.value.upper()}"
    )
    cv2.rectangle(output, (8, 8), (min(570, output.shape[1] - 8), 42), (12, 12, 12), -1)
    cv2.putText(
        output,
        label,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    return output


def _render(
    source: Path,
    output: Path,
    states: tuple[CameraState, ...],
    pitch_model: PitchModel,
    fps: float,
    image_size: tuple[int, int],
) -> None:
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        image_size,
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot open output video: {output}")
    for index, state in enumerate(states):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(_draw_pitch_overlay(frame, state, pitch_model, index))
    capture.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-registration", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--pitch-model-path", default=PITCH_DETECTION_MODEL_PATH)
    parser.add_argument("--pitch-perception-checkpoint", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--semantic-interval", type=int, default=5)
    parser.add_argument("--backward-chunk-frames", type=int, default=240)
    parser.add_argument("--max-frames", type=int)
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    output_video = arguments.output_video.resolve()
    report_path = arguments.report.resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps, source_frames = _video_metadata(source)
    pitch_model, references = _pitch_model_and_references()
    if arguments.pitch_perception_checkpoint is not None:
        backend: PitchPerceptionBackend = TorchPitchPerceptionBackend.from_checkpoint(
            arguments.pitch_perception_checkpoint,
            pitch_model,
            device=arguments.device,
            use_fp16=arguments.device.startswith("cuda"),
        )
        perception_name = "mobilenet_v3_dual_head"
    else:
        predictor = LegacyPitchPredictor(
            arguments.pitch_model_path, arguments.device, arguments.imgsz
        )
        backend = LegacyKeypointPerceptionBackend(
            predictor=predictor,
            references=references,
            minimum_confidence=0.35,
        )
        perception_name = "legacy_32_keypoint_adapter"
    started = time.perf_counter()
    forward_core = _core(
        backend,
        pitch_model,
        fps=fps,
        semantic_interval=arguments.semantic_interval,
    )
    forward, boundaries = _forward_pass(
        source, forward_core, frame_limit=arguments.max_frames
    )
    backward = _backward_pass(
        source,
        forward,
        backend,
        pitch_model,
        fps=fps,
        semantic_interval=arguments.semantic_interval,
        chunk_frames=arguments.backward_chunk_frames,
    )
    result = BidirectionalCameraSmoother(
        config=OfflineSmoothingConfig(fps=fps)
    ).smooth(forward, backward)
    json_path, npz_path = save_offline_smoothing_result(
        result, arguments.output_registration.resolve()
    )
    _render(source, output_video, result.states, pitch_model, fps, (width, height))
    elapsed = time.perf_counter() - started
    tier_counts = Counter(state.measurement_tier.value for state in result.states)
    status_counts = Counter(state.status.value for state in result.states)
    report = {
        "format_version": 1,
        "source": str(source),
        "source_frames": source_frames,
        "processed_frames": len(result.states),
        "source_fps": fps,
        "device": arguments.device,
        "elapsed_sec": elapsed,
        "effective_source_fps": len(result.states) / max(elapsed, 1e-9),
        "forward_semantic_inferences": forward_core.semantic_inference_count,
        "shot_boundaries": boundaries,
        "shot_count": result.shot_count,
        "smoothed_frame_count": result.smoothed_frame_count,
        "preview_only_frame_count": result.preview_only_frame_count,
        "measurement_tiers": dict(tier_counts),
        "camera_statuses": dict(status_counts),
        "output_video": str(output_video),
        "registration_json": str(json_path),
        "registration_npz": str(npz_path),
        "perception_backend": perception_name,
        "accuracy_valid": False,
        "notes": [
            "The MP4 is a projection-stability diagnostic, not accuracy ground truth.",
            "SAFE/PREVIEW accuracy requires independent manual or public test labels.",
            "The trained dual-head student is not selected until SoccerNet/test2 gates pass.",
        ],
    }
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
