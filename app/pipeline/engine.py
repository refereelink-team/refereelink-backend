from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
import supervision as sv
import torch

from app.classification.online import OnlineTeamClassifier
from app.constants.classes import (
    GOALKEEPER_CLASS_ID,
    PLAYER_CLASS_ID,
    REFEREE_CLASS_ID,
)
from app.constants.paths import PITCH_DETECTION_MODEL_PATH, PLAYER_DETECTION_MODEL_PATH
from app.config.pitch import SoccerPitchConfiguration
from app.geometry.pitch_projection import PitchProjectionEngine
from app.pipeline.buffer import BoundedFrameBuffer, PipelineMode
from app.pipeline.source import VideoSource
from app.state.models import (
    FrameState,
    HomographyStatus,
    MetricsSnapshot,
    PlayerRole,
    PlayerState,
    SourceStatus,
)
from app.state.store import StateStore

logger = logging.getLogger(__name__)

METRICS_INTERVAL_SEC = 1.0
CONFIG = SoccerPitchConfiguration()


def _map_role(class_id: int) -> PlayerRole:
    if class_id == GOALKEEPER_CLASS_ID:
        return PlayerRole.GOALKEEPER
    if class_id == REFEREE_CLASS_ID:
        return PlayerRole.REFEREE
    return PlayerRole.PLAYER


def _map_homography_status(status: str) -> HomographyStatus:
    try:
        return HomographyStatus(status)
    except ValueError:
        return HomographyStatus.UNAVAILABLE


class InferencePipeline:
    def __init__(
        self,
        source: VideoSource,
        store: StateStore,
        device: str = "cpu",
        mode: PipelineMode = PipelineMode.REALTIME,
    ) -> None:
        self._source = source
        self._store = store
        self._device = device
        self._mode = mode
        self._running = False
        self._thread: Optional[threading.Thread] = None

        self._buffer = BoundedFrameBuffer(maxsize=4, mode=mode)
        self._dropped_frames = 0

        self._metrics_start = 0.0
        self._metrics_frames = 0
        self._metrics_last = 0.0
        self._last_metrics_emit = 0.0
        self._latency_acc_ms = 0.0
        self._end_to_end_acc_ms = 0.0

        if device.startswith("cuda"):
            self._use_fp16 = torch.cuda.is_available()
        else:
            self._use_fp16 = False

        self._player_model = None
        self._pitch_model = None
        self._projection_engine: Optional[PitchProjectionEngine] = None
        self._team_classifier = OnlineTeamClassifier(device=device)
        self._tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._store.pipeline_running = True
        self._load_models()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="pipeline")
        self._thread.start()
        logger.info("InferencePipeline started (device=%s, fp16=%s)", self._device, self._use_fp16)

    def stop(self) -> None:
        self._running = False
        self._store.pipeline_running = False
        self._buffer.close()
        if self._source.is_opened():
            self._source.release()
        logger.info("InferencePipeline stopped")

    def _load_models(self) -> None:
        from ultralytics import YOLO

        logger.info("Loading player detection model")
        self._player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=self._device)
        logger.info("Loading pitch detection model")
        self._pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=self._device)
        self._projection_engine = PitchProjectionEngine(
            config=CONFIG, fps=max(self._source.fps, 1.0)
        )

    def _run_loop(self) -> None:
        self._metrics_start = time.monotonic()
        self._metrics_last = self._metrics_start
        self._last_metrics_emit = self._metrics_start

        while self._running:
            capture_ts = self._source.capture_timestamp_ms()
            ret, frame = self._source.read()

            if not ret or frame is None:
                if not self._source.is_opened():
                    logger.info("Video source ended")
                    self._store.source_status = SourceStatus.DISCONNECTED
                    break
                time.sleep(0.01)
                continue

            inference_start = time.monotonic()
            frame_state = self._process_frame(frame, capture_ts)
            inference_end = time.monotonic()

            if frame_state is not None:
                inference_latency = (inference_end - inference_start) * 1000
                frame_state.processed_timestamp_ms = time.time() * 1000
                end_to_end = frame_state.processed_timestamp_ms - capture_ts

                self._metrics_frames += 1
                self._latency_acc_ms += inference_latency
                self._end_to_end_acc_ms += end_to_end

                self._store.latest_frame_state = frame_state

            now = time.monotonic()
            if now - self._last_metrics_emit >= METRICS_INTERVAL_SEC:
                self._emit_metrics()
                self._last_metrics_emit = now

    def _process_frame(
        self, frame: np.ndarray, capture_timestamp_ms: float
    ) -> Optional[FrameState]:
        with torch.inference_mode():
            pitch_result = self._pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

            player_result = self._player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            detections = self._tracker.update_with_detections(detections)

        projection = self._projection_engine.update(frame=frame, keypoints=keypoints)  # type: ignore[arg-type]

        player_states: list[PlayerState] = []
        if projection.homography is not None and len(detections) > 0:
            xy = detections.get_anchors_coordinates(
                anchor=sv.Position.BOTTOM_CENTER
            ).astype(np.float32)
            transformed = cv2.perspectiveTransform(
                xy.reshape(-1, 1, 2), projection.homography
            )
            if transformed is not None:
                transformed = transformed.reshape(-1, 2)
                for idx in range(len(detections)):
                    tracker_id = int(detections.tracker_id[idx]) if detections.tracker_id is not None else idx
                    player_states.append(PlayerState(
                        track_id=int(tracker_id),
                        role=_map_role(int(detections.class_id[idx])) if detections.class_id is not None else PlayerRole.PLAYER,
                        team_id=0,
                        field_x=float(transformed[idx][0]),
                        field_y=float(transformed[idx][1]),
                        confidence=float(detections.confidence[idx]) if detections.confidence is not None else 0.0,
                    ))
            else:
                # perspectiveTransform failed — fall back to no projection
                for idx in range(len(detections)):
                    tracker_id = int(detections.tracker_id[idx]) if detections.tracker_id is not None else idx
                    player_states.append(PlayerState(
                        track_id=int(tracker_id),
                        role=_map_role(int(detections.class_id[idx])) if detections.class_id is not None else PlayerRole.PLAYER,
                        team_id=0,
                        confidence=float(detections.confidence[idx]) if detections.confidence is not None else 0.0,
                    ))
        else:
            for idx in range(len(detections)):
                tracker_id = int(detections.tracker_id[idx]) if detections.tracker_id is not None else idx
                player_states.append(PlayerState(
                    track_id=int(tracker_id),
                    role=_map_role(int(detections.class_id[idx])) if detections.class_id is not None else PlayerRole.PLAYER,
                    team_id=0,
                    confidence=float(detections.confidence[idx]) if detections.confidence is not None else 0.0,
                ))

        # Draw detection boxes and tracker IDs on the frame for the MJPEG stream.
        annotated_frame = frame.copy()
        n = len(detections)
        if n > 0:
            for i in range(n):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else i
                class_id = int(detections.class_id[i]) if detections.class_id is not None else 0

                if class_id == PLAYER_CLASS_ID:
                    color = (0, 255, 0)
                elif class_id == GOALKEEPER_CLASS_ID:
                    color = (255, 0, 0)
                elif class_id == REFEREE_CLASS_ID:
                    color = (0, 255, 255)
                else:
                    color = (255, 255, 255)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{tracker_id}"
                cv2.putText(
                    annotated_frame, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

        self._store._latest_raw_frame = annotated_frame  # type: ignore[attr-defined]

        elapsed = time.monotonic() - self._metrics_start
        current_fps = self._metrics_frames / max(elapsed, 0.001)

        return FrameState(
            frame_id=self._source.frame_count,
            capture_timestamp_ms=capture_timestamp_ms,
            processing_fps=current_fps,
            homography_status=_map_homography_status(projection.homography_status),
            players=player_states,
            events=[],
        )

    def _emit_metrics(self) -> None:
        elapsed = time.monotonic() - self._metrics_start
        fps = self._metrics_frames / max(elapsed, 0.001)
        avg_lat = self._latency_acc_ms / max(self._metrics_frames, 1)
        avg_e2e = self._end_to_end_acc_ms / max(self._metrics_frames, 1)

        memory_mb = 0.0
        try:
            import psutil
            memory_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        except Exception:
            pass

        gpu_mem_mb = None
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_mb = mem_info.used / (1024 * 1024)
        except Exception:
            pass

        metrics = MetricsSnapshot(
            processing_fps=round(fps, 1),
            input_fps=round(self._source.fps, 1),
            inference_latency_ms=round(avg_lat, 1),
            end_to_end_latency_ms=round(avg_e2e, 1),
            dropped_frames=self._buffer.dropped_frames,
            queue_length=len(self._buffer),
            player_count=len(self._store.latest_frame_state.players)
            if self._store.latest_frame_state else 0,
            source_status=self._store.source_status,
            memory_mb=round(memory_mb, 1),
            gpu_memory_mb=round(gpu_mem_mb, 1) if gpu_mem_mb is not None else None,
        )
        self._store.metrics = metrics


def detection_confidence(detection: Any) -> float:
    try:
        if hasattr(detection, 'confidence') and len(detection) > 2:
            return float(detection.confidence[0]) if detection.confidence is not None else 0.0
        return 0.0
    except (IndexError, TypeError):
        return 0.0
