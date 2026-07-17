"""Shared person detection, ByteTrack and pitch-coordinate pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import supervision as sv
import torch

from app.config.pitch import SoccerPitchConfiguration
from app.constants.classes import UNKNOWN_COLOR_ID
from app.constants.paths import (
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_DETECTION_MODEL_PATH,
)
from app.geometry.camera import (
    CameraMotionEstimator,
    CameraUndistorter,
    build_undistorter,
)
from app.geometry.pitch_projection import PitchProjectionEngine, PitchProjectionResult


def _empty_detections() -> sv.Detections:
    return sv.Detections(xyxy=np.empty((0, 4), dtype=np.float32))


@dataclass
class VisionFrame:
    undistorted_frame: np.ndarray
    detections: sv.Detections
    tracked_detections: sv.Detections
    projection: PitchProjectionResult
    field_xy: np.ndarray
    color_lookup: np.ndarray
    person_only: bool

    @property
    def homography_status(self) -> str:
        return self.projection.homography_status


class VisionCore:
    """One reusable detection/tracking/projection implementation.

    The default player model is an official YOLOv11 COCO checkpoint and is
    therefore intentionally treated as a person-only detector.  Role-aware
    custom checkpoints can be enabled later without changing the output
    shape by setting ``person_only=False``.
    """

    def __init__(
        self,
        device: str = "cpu",
        fps: float = 25.0,
        player_model_path: str = PLAYER_DETECTION_MODEL_PATH,
        pitch_model_path: str = PITCH_DETECTION_MODEL_PATH,
        camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
        enable_undistortion: bool = True,
        calibration_alpha: float = 0.0,
        pitch_detection_interval: int = 5,
        imgsz: int = 640,
        enable_player: bool = True,
        enable_pitch: bool = True,
        person_only: bool = True,
        undistorter: Optional[CameraUndistorter] = None,
        player_model: object = None,
        pitch_model: object = None,
        tracker: Optional[object] = None,
        projection_engine: Optional[PitchProjectionEngine] = None,
        inference_backend: str = "auto",
        camera_motion_threshold_px: float = 6.0,
        camera_motion_estimator: Optional[CameraMotionEstimator] = None,
        track_activation_threshold: float = 0.25,
        track_lost_buffer: int = 45,
        track_matching_threshold: float = 0.8,
        track_minimum_consecutive_frames: int = 2,
    ) -> None:
        self.device = device
        self.fps = max(float(fps), 1.0)
        self.player_model_path = player_model_path
        self.pitch_model_path = pitch_model_path
        self.pitch_detection_interval = max(int(pitch_detection_interval), 1)
        self.imgsz = max(int(imgsz), 32)
        self.enable_player = enable_player
        self.enable_pitch = enable_pitch
        self.person_only = person_only
        self.inference_backend = inference_backend
        self.camera_motion_refresh_count = 0
        self._player_model = player_model
        self._pitch_model = pitch_model
        self._tracker = (
            tracker
            if tracker is not None
            else sv.ByteTrack(
                track_activation_threshold=float(track_activation_threshold),
                lost_track_buffer=max(int(track_lost_buffer), 1),
                minimum_matching_threshold=float(track_matching_threshold),
                frame_rate=self.fps,
                minimum_consecutive_frames=max(int(track_minimum_consecutive_frames), 1),
            )
        )
        self._projection_engine = (
            projection_engine
            if projection_engine is not None
            else PitchProjectionEngine(config=SoccerPitchConfiguration(), fps=self.fps)
        )
        self._last_pitch_detection_frame: Optional[int] = None
        self._use_fp16 = device.startswith("cuda") and torch.cuda.is_available()
        self.frames_processed = 0
        self.player_inference_count = 0
        self.pitch_detection_count = 0
        self.pitch_reuse_count = 0
        self.homography_available_count = 0
        self.track_id_interruptions = 0
        self._previous_track_ids: set[int] = set()
        self.player_inference_time_ms = 0.0
        self.pitch_inference_time_ms = 0.0
        self._undistorter = undistorter or build_undistorter(
            calibration_path=camera_calibration_path,
            enabled=enable_undistortion,
            alpha=calibration_alpha,
        )
        self._camera_motion_estimator = camera_motion_estimator or CameraMotionEstimator(
            threshold_px=camera_motion_threshold_px
        )

    @property
    def projection_engine(self) -> PitchProjectionEngine:
        return self._projection_engine

    def load_models(self) -> None:
        from app.vision.backends import UltralyticsBackend

        if self.enable_player and self._player_model is None:
            backend = None if self.inference_backend == "auto" else self.inference_backend
            self._player_model = UltralyticsBackend(
                self.player_model_path,
                backend=backend,
                device=self.device,
            )
        if self.enable_pitch and self._pitch_model is None:
            backend = None if self.inference_backend == "auto" else self.inference_backend
            self._pitch_model = UltralyticsBackend(
                self.pitch_model_path,
                backend=backend,
                device=self.device,
            )

    def _run_model(self, model: object, frame: np.ndarray):
        if hasattr(model, "predict"):
            try:
                return model.predict(frame, imgsz=self.imgsz, half=self._use_fp16)
            except TypeError:
                try:
                    return model.predict(frame, imgsz=self.imgsz)
                except TypeError:
                    return model.predict(frame)
        try:
            return model(
                frame,
                imgsz=self.imgsz,
                verbose=False,
                half=self._use_fp16,
            )
        except TypeError:
            try:
                return model(frame, imgsz=self.imgsz, verbose=False)
            except TypeError:
                return model(frame)

    def _predict_player(self, frame: np.ndarray) -> sv.Detections:
        if not self.enable_player:
            return _empty_detections()
        if self._player_model is None:
            raise RuntimeError("Player model has not been loaded")
        result = self._run_model(self._player_model, frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        if not self.person_only or len(detections) == 0:
            return detections
        if detections.class_id is None:
            return detections
        return detections[detections.class_id == 0]

    def _predict_pitch(self, frame: np.ndarray) -> sv.KeyPoints:
        if self._pitch_model is None:
            raise RuntimeError("Pitch model has not been loaded")
        result = self._run_model(self._pitch_model, frame)[0]
        return sv.KeyPoints.from_ultralytics(result)

    def _should_detect_pitch(self, frame_index: int) -> bool:
        return (
            self._last_pitch_detection_frame is None
            or frame_index - self._last_pitch_detection_frame
            >= self.pitch_detection_interval
        )

    def _projection_for_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        force_refresh: bool = False,
    ) -> PitchProjectionResult:
        if not self.enable_pitch:
            return PitchProjectionResult(
                tracking_observations=[],
                projected_keypoints=[],
                homography=None,
                homography_status="unavailable",
                reprojection_error=None,
            )
        if force_refresh or self._should_detect_pitch(frame_index):
            start = time.perf_counter()
            keypoints = self._predict_pitch(frame)
            self.pitch_inference_time_ms += (time.perf_counter() - start) * 1000
            self.pitch_detection_count += 1
            self._last_pitch_detection_frame = frame_index
            projection = self._projection_engine.update(frame=frame, keypoints=keypoints)
            if force_refresh and projection.homography_status != "fresh":
                invalidate = getattr(self._projection_engine, "invalidate", None)
                if callable(invalidate):
                    invalidate()
                return PitchProjectionResult(
                    tracking_observations=[],
                    projected_keypoints=[],
                    homography=None,
                    homography_status="unavailable",
                    reprojection_error=None,
                )
            return projection
        self.pitch_reuse_count += 1
        return self._projection_engine.reuse(frame=frame)

    def _field_coordinates(
        self,
        detections: sv.Detections,
        projection: PitchProjectionResult,
    ) -> np.ndarray:
        field_xy = np.full((len(detections), 2), np.nan, dtype=np.float32)
        if projection.homography is None or len(detections) == 0:
            return field_xy

        image_xy = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        ).astype(np.float32)
        try:
            transformed = cv2.perspectiveTransform(
                image_xy.reshape(-1, 1, 2), projection.homography
            ).reshape(-1, 2)
        except cv2.error:
            return field_xy

        config = self._projection_engine.config
        valid = (
            np.isfinite(transformed).all(axis=1)
            & (transformed[:, 0] >= 0)
            & (transformed[:, 0] <= config.length)
            & (transformed[:, 1] >= 0)
            & (transformed[:, 1] <= config.width)
        )
        field_xy[valid] = transformed[valid]
        return field_xy

    def process(self, frame: np.ndarray, frame_index: int) -> VisionFrame:
        undistorted_frame = self._undistorter.apply(frame)
        start = time.perf_counter()
        detections = self._predict_player(undistorted_frame)
        self.player_inference_time_ms += (time.perf_counter() - start) * 1000
        self.player_inference_count += 1
        tracked_detections = self._tracker.update_with_detections(detections)
        current_track_ids = (
            {int(track_id) for track_id in tracked_detections.tracker_id}
            if tracked_detections.tracker_id is not None
            else set()
        )
        if self._previous_track_ids and self._previous_track_ids.isdisjoint(current_track_ids):
            self.track_id_interruptions += 1
        self._previous_track_ids = current_track_ids
        motion = self._camera_motion_estimator.measure(undistorted_frame)
        force_pitch_refresh = bool(motion is not None and motion.requires_refresh)
        if force_pitch_refresh:
            self.camera_motion_refresh_count += 1
        projection = self._projection_for_frame(
            undistorted_frame,
            frame_index,
            force_refresh=force_pitch_refresh,
        )
        if projection.homography_status == "fresh":
            self._camera_motion_estimator.mark_reference(undistorted_frame)
        field_xy = self._field_coordinates(tracked_detections, projection)
        self.frames_processed += 1
        if projection.available:
            self.homography_available_count += 1
        color_lookup = np.full(
            len(tracked_detections),
            UNKNOWN_COLOR_ID if self.person_only else 0,
            dtype=np.int64,
        )
        return VisionFrame(
            undistorted_frame=undistorted_frame,
            detections=detections,
            tracked_detections=tracked_detections,
            projection=projection,
            field_xy=field_xy,
            color_lookup=color_lookup,
            person_only=self.person_only,
        )
