#!/usr/bin/env python3
"""Run forward/backward broadcast registration and render a debug MP4.

The command reads a local video twice. It uses the existing Ultralytics pitch
checkpoint through the V2 perception adapter until a real dual-head student
checkpoint is available. The saved JSON explicitly records this limitation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from app.constants.paths import PITCH_DETECTION_MODEL_PATH
from app.field_registration.offline import (
    BidirectionalCameraSmoother,
    OfflineCameraObservation,
    OfflineSmoothingConfig,
    save_offline_smoothing_result,
)
from app.field_registration.perception import LegacyKeypointPerceptionBackend
from app.field_registration.pitch_model import PitchDimensions, PitchModel
from app.field_registration.tracker import FieldRegistrationConfig, FieldRegistrationCore
from app.field_registration.types import CameraState, MeasurementTier, RegistrationMode
from app.geometry.pitch_projection import PitchProjectionEngine
from app.config.pitch import SoccerPitchConfiguration
from app.vision.backends import UltralyticsBackend


def _pitch_model_and_references() -> tuple[PitchModel, tuple[object, ...]]:
    engine = PitchProjectionEngine(SoccerPitchConfiguration(), fps=25.0)
    config = engine.config
    model = PitchModel(
        PitchDimensions(
            length_m=config.length / 100.0,
            width_m=config.width / 100.0,
            penalty_area_depth_m=config.penalty_box_length / 100.0,
            penalty_area_width_m=config.penalty_box_width / 100.0,
            goal_area_depth_m=config.goal_box_length / 100.0,
            goal_area_width_m=config.goal_box_width / 100.0,
            centre_circle_radius_m=config.centre_circle_radius / 100.0,
            penalty_spot_distance_m=config.penalty_spot_distance / 100.0,
        )
    )
    return model, tuple(engine.references)


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
    predictor: LegacyPitchPredictor,
    pitch_model: PitchModel,
    references: tuple[object, ...],
    *,
    fps: float,
    semantic_interval: int,
) -> FieldRegistrationCore:
    backend = LegacyKeypointPerceptionBackend(
        predictor=predictor,
        references=references,
        minimum_confidence=0.35,
    )
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


def _read_frame(capture: cv2.VideoCapture, index: int) -> np.ndarray:
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok or frame is None:
        raise RuntimeError(f"could not read source frame {index}")
    return frame


def _backward_pass(
    source: Path,
    forward: list[OfflineCameraObservation],
    predictor: LegacyPitchPredictor,
    pitch_model: PitchModel,
    references: tuple[object, ...],
    *,
    fps: float,
    semantic_interval: int,
) -> list[OfflineCameraObservation]:
    capture = cv2.VideoCapture(str(source))
    output: list[OfflineCameraObservation] = []
    for start, end, shot_id in _shot_ranges(forward):
        core = _core(
            predictor,
            pitch_model,
            references,
            fps=fps,
            semantic_interval=semantic_interval,
        )
        reverse_index = 0
        for source_index in range(end - 1, start - 1, -1):
            frame = _read_frame(capture, source_index)
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
    capture.release()
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--semantic-interval", type=int, default=5)
    parser.add_argument("--max-frames", type=int)
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    output_video = arguments.output_video.resolve()
    report_path = arguments.report.resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    width, height, fps, source_frames = _video_metadata(source)
    predictor = LegacyPitchPredictor(
        arguments.pitch_model_path, arguments.device, arguments.imgsz
    )
    pitch_model, references = _pitch_model_and_references()
    started = time.perf_counter()
    forward_core = _core(
        predictor,
        pitch_model,
        references,
        fps=fps,
        semantic_interval=arguments.semantic_interval,
    )
    forward, boundaries = _forward_pass(
        source, forward_core, frame_limit=arguments.max_frames
    )
    backward = _backward_pass(
        source,
        forward,
        predictor,
        pitch_model,
        references,
        fps=fps,
        semantic_interval=arguments.semantic_interval,
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
        "perception_backend": "legacy_32_keypoint_adapter",
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
