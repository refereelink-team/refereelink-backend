from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

import cv2
import numpy as np
import torch

from app.constants.paths import (
    BALL_DETECTION_MODEL_PATH,
    CAMERA_CALIBRATION_PATH,
    PITCH_DETECTION_MODEL_PATH,
    PLAYER_DETECTION_MODEL_PATH,
    ROLE_DETECTION_MODEL_PATH,
    TEAM_CLASSIFIER_PATH,
    FOUL_MODEL_PATH,
)
from app.events.engine import EventEngine, FoulEventAdapter
from app.pipeline.buffer import BoundedFrameBuffer, PipelineMode
from app.pipeline.source import VideoSource
from app.state.models import (
    BallState,
    BallStatus,
    FrameState,
    HomographyStatus,
    MetricsSnapshot,
    PlayerRole,
    PlayerState,
    SourceStatus,
)
from app.state.store import StateStore
from app.vision.core import VisionCore
from app.vision.ball import BallProcessor

logger = logging.getLogger(__name__)

METRICS_INTERVAL_SEC = 1.0
POSSESSION_DISTANCE_MM = 900.0


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
        player_model_path: str = PLAYER_DETECTION_MODEL_PATH,
        pitch_model_path: str = PITCH_DETECTION_MODEL_PATH,
        camera_calibration_path: Optional[str] = CAMERA_CALIBRATION_PATH,
        enable_undistortion: bool = True,
        calibration_alpha: float = 0.0,
        pitch_detection_interval: int = 5,
        imgsz: int = 640,
        ball_model_path: str = BALL_DETECTION_MODEL_PATH,
        enable_ball: bool = True,
        ball_detection_interval: int = 2,
        ball_max_prediction_frames: int = 8,
        role_model_path: str = ROLE_DETECTION_MODEL_PATH,
        team_classifier_path: Optional[str] = TEAM_CLASSIFIER_PATH,
        role_detection_interval: int = 3,
        team_classification_interval: int = 5,
        semantic_manager: Optional[object] = None,
        inference_backend: str = "auto",
        enable_foul_detection: bool = False,
        foul_checkpoint_path: Optional[str] = None,
        foul_confidence_threshold: float = 0.48,
        foul_detector: Optional[object] = None,
    ) -> None:
        self._source = source
        self._store = store
        self._device = device
        self._mode = mode
        self._player_model_path = player_model_path
        self._pitch_model_path = pitch_model_path
        self._camera_calibration_path = camera_calibration_path
        self._enable_undistortion = enable_undistortion
        self._calibration_alpha = calibration_alpha
        self._pitch_detection_interval = pitch_detection_interval
        self._imgsz = imgsz
        self._ball_model_path = ball_model_path
        self._enable_ball = enable_ball
        self._ball_detection_interval = ball_detection_interval
        self._ball_max_prediction_frames = ball_max_prediction_frames
        self._role_model_path = role_model_path
        self._team_classifier_path = team_classifier_path
        self._role_detection_interval = role_detection_interval
        self._team_classification_interval = team_classification_interval
        self._inference_backend = inference_backend
        self._enable_foul_detection = enable_foul_detection
        self._foul_checkpoint_path = foul_checkpoint_path or FOUL_MODEL_PATH
        self._foul_confidence_threshold = foul_confidence_threshold
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

        self._vision_core: Optional[VisionCore] = None
        self._ball_processor: Optional[BallProcessor] = None
        self._semantic_manager = semantic_manager
        self._semantic_interval = max(
            1, min(int(role_detection_interval), int(team_classification_interval))
        )
        self._semantic_last_frame: Optional[int] = None
        self._semantic_results: dict[int, object] = {}
        self.semantic_inference_count = 0
        self._event_engine = EventEngine()
        self._foul_detector = foul_detector
        self._foul_adapter = FoulEventAdapter(confidence_threshold=foul_confidence_threshold)
        self.foul_inference_count = 0
        self._previous_ball_field_xy: Optional[np.ndarray] = None
        self._previous_ball_timestamp_s: Optional[float] = None

    def start(self) -> None:
        if self._running:
            return
        self._load_models()
        self._running = True
        self._store.pipeline_running = True
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
        logger.info("Loading shared vision core")
        self._vision_core = VisionCore(
            device=self._device,
            fps=max(self._source.fps, 1.0),
            player_model_path=self._player_model_path,
            pitch_model_path=self._pitch_model_path,
            camera_calibration_path=self._camera_calibration_path,
            enable_undistortion=self._enable_undistortion,
            calibration_alpha=self._calibration_alpha,
            pitch_detection_interval=self._pitch_detection_interval,
            imgsz=self._imgsz,
            inference_backend=self._inference_backend,
        )
        self._vision_core.load_models()
        if self._semantic_manager is None:
            from app.classification.online import OnlineTeamClassifier
            from app.vision.role import UltralyticsRoleClassifier
            from app.vision.semantics import TrackSemanticManager

            role_classifier = UltralyticsRoleClassifier(
                model_path=self._role_model_path,
                device=self._device,
                imgsz=self._imgsz,
            )
            if not role_classifier.load():
                role_classifier = None
            team_classifier = OnlineTeamClassifier(
                device=self._device,
                classifier_path=self._team_classifier_path,
            )
            self._semantic_manager = TrackSemanticManager(
                role_classifier=role_classifier,
                team_classifier=team_classifier,
            )
        if self._enable_ball:
            self._ball_processor = BallProcessor(
                model_path=self._ball_model_path,
                device=self._device,
                imgsz=self._imgsz,
                detection_interval=self._ball_detection_interval,
                max_prediction_frames=self._ball_max_prediction_frames,
                inference_backend=self._inference_backend,
            )
            self._ball_processor.load_model()
        if self._enable_foul_detection and self._foul_detector is None:
            try:
                from app.foul_detection.detector import FoulDetector

                self._foul_detector = FoulDetector(
                    checkpoint_path=self._foul_checkpoint_path,
                    device=self._device,
                )
            except Exception as exc:
                logger.warning("Foul detector unavailable; continuing without foul events: %s", exc)
                self._foul_detector = None

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
        if self._vision_core is None:
            raise RuntimeError("VisionCore is not loaded")

        with torch.inference_mode():
            vision_frame = self._vision_core.process(frame, self._source.frame_count)

        detections = vision_frame.tracked_detections
        projection = vision_frame.projection

        ball_state = self._process_ball(
            frame=vision_frame.undistorted_frame,
            frame_index=self._source.frame_count,
            projection=projection,
            timestamp_s=capture_timestamp_ms / 1000.0,
        )
        foul_event = self._process_foul(
            frame=vision_frame.undistorted_frame,
            frame_id=self._source.frame_count,
            timestamp_s=capture_timestamp_ms / 1000.0,
            ball_state=ball_state,
        )

        player_states: list[PlayerState] = []
        semantic_results = self._update_semantics(
            frame=vision_frame.undistorted_frame,
            detections=detections,
            frame_index=self._source.frame_count,
        )
        for idx in range(len(detections)):
            tracker_id = (
                int(detections.tracker_id[idx])
                if detections.tracker_id is not None
                else idx
            )
            field_xy = vision_frame.field_xy[idx]
            has_field_xy = bool(np.isfinite(field_xy).all())
            semantic = semantic_results.get(tracker_id)
            role = PlayerRole.UNKNOWN
            team_id = -1
            role_confidence = 0.0
            team_confidence = 0.0
            semantic_status = "unknown"
            if semantic is not None:
                try:
                    role_value = getattr(semantic, "role", "unknown")
                    role = PlayerRole(str(getattr(role_value, "value", role_value)))
                except ValueError:
                    role = PlayerRole.UNKNOWN
                try:
                    candidate_team = int(getattr(semantic, "team_id", -1))
                    team_id = candidate_team if candidate_team in (0, 1) else -1
                except (TypeError, ValueError):
                    team_id = -1
                role_confidence = float(getattr(semantic, "role_confidence", 0.0))
                team_confidence = float(getattr(semantic, "team_confidence", 0.0))
                semantic_status = str(
                    getattr(semantic, "semantic_status", getattr(semantic, "status", "unknown"))
                )
            player_states.append(
                PlayerState(
                    track_id=tracker_id,
                    role=role,
                    team_id=team_id,
                    field_x=float(field_xy[0]) if has_field_xy else None,
                    field_y=float(field_xy[1]) if has_field_xy else None,
                    confidence=(
                        float(detections.confidence[idx])
                        if detections.confidence is not None
                        else 0.0
                    ),
                    role_confidence=role_confidence,
                    team_confidence=team_confidence,
                    semantic_status=semantic_status,
                )
            )

        # Draw detection boxes and tracker IDs on the frame for the MJPEG stream.
        annotated_frame = vision_frame.undistorted_frame.copy()
        n = len(detections)
        if n > 0:
            for i in range(n):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                tracker_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else i
                team_id = player_states[i].team_id if i < len(player_states) else -1
                color = {0: (147, 20, 255), 1: (255, 191, 0)}.get(team_id, (128, 128, 128))

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{tracker_id}"
                cv2.putText(
                    annotated_frame, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

        if ball_state.image_x is not None and ball_state.image_y is not None:
            ball_center = (int(round(ball_state.image_x)), int(round(ball_state.image_y)))
            ball_color = (0, 215, 255) if ball_state.status == BallStatus.FRESH else (180, 180, 180)
            cv2.circle(annotated_frame, ball_center, 7, ball_color, 2)
            cv2.putText(
                annotated_frame,
                f"ball:{ball_state.status.value}",
                (ball_center[0] + 8, ball_center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                ball_color,
                1,
            )

        publish_frame = getattr(self._store, "publish_raw_frame", None)
        if publish_frame is not None:
            publish_frame(annotated_frame)
        else:
            self._store._latest_raw_frame = annotated_frame  # type: ignore[attr-defined]

        elapsed = time.monotonic() - self._metrics_start
        current_fps = self._metrics_frames / max(elapsed, 0.001)

        frame_state = FrameState(
            frame_id=self._source.frame_count,
            capture_timestamp_ms=capture_timestamp_ms,
            processing_fps=current_fps,
            homography_status=_map_homography_status(projection.homography_status),
            players=player_states,
            ball=ball_state,
            possession_track_id=self._find_possession_track_id(player_states, ball_state),
            events=[],
        )
        event_engine = getattr(self, "_event_engine", None)
        if event_engine is not None:
            frame_state.events = event_engine.update(frame_state)
        if foul_event is not None:
            frame_state.events.append(foul_event)
        for event in frame_state.events:
            self._store.add_event(event)
        return frame_state

    def _process_foul(
        self,
        *,
        frame: np.ndarray,
        frame_id: int,
        timestamp_s: float,
        ball_state: BallState,
    ):
        foul_detector = getattr(self, "_foul_detector", None)
        if foul_detector is None:
            return None
        try:
            prediction = foul_detector.update(frame)
            self.foul_inference_count += 1
            field_xy = (
                (ball_state.field_x, ball_state.field_y)
                if ball_state.field_x is not None and ball_state.field_y is not None
                else None
            )
            foul_adapter = getattr(self, "_foul_adapter", FoulEventAdapter())
            return foul_adapter.update(
                prediction,
                frame_id=frame_id,
                timestamp=timestamp_s,
                field_xy=field_xy,
            )
        except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
            logger.warning("Foul prediction failed; skipping event: %s", exc)
            return None

    def _update_semantics(
        self,
        frame: np.ndarray,
        detections: Any,
        frame_index: int,
    ) -> dict[int, object]:
        if self._semantic_manager is None:
            return {}
        if (
            self._semantic_last_frame is None
            or frame_index - self._semantic_last_frame >= self._semantic_interval
        ):
            try:
                self._semantic_results = self._semantic_manager.update(
                    frame=frame,
                    detections=detections,
                    frame_index=frame_index,
                )
                self.semantic_inference_count += 1
                self._semantic_last_frame = frame_index
            except (AttributeError, TypeError, ValueError, RuntimeError) as exc:
                logger.warning("Semantic prediction failed; using previous state: %s", exc)
        current_ids = {
            int(track_id)
            for track_id in detections.tracker_id
        } if detections.tracker_id is not None else set()
        return {
            track_id: result
            for track_id, result in self._semantic_results.items()
            if track_id in current_ids
        }

    def _process_ball(
        self,
        frame: np.ndarray,
        frame_index: int,
        projection: Any,
        timestamp_s: float,
    ) -> BallState:
        if self._ball_processor is None:
            return BallState()

        estimate = self._ball_processor.process(
            frame=frame,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )
        if estimate.position is None:
            self._previous_ball_field_xy = None
            self._previous_ball_timestamp_s = None
            return BallState(
                status=BallStatus.UNAVAILABLE,
                confidence=estimate.confidence,
                age_frames=estimate.age_frames,
            )

        image_xy = estimate.position.astype(np.float32)
        field_xy: Optional[np.ndarray] = None
        if projection.homography is not None:
            try:
                field_xy = cv2.perspectiveTransform(
                    image_xy.reshape(1, 1, 2), projection.homography
                ).reshape(2)
            except cv2.error:
                field_xy = None

        valid_field = bool(
            field_xy is not None
            and np.isfinite(field_xy).all()
            and 0 <= field_xy[0] <= self._vision_core.projection_engine.config.length
            and 0 <= field_xy[1] <= self._vision_core.projection_engine.config.width
        )
        if not valid_field:
            field_xy = None
            self._previous_ball_field_xy = None
            self._previous_ball_timestamp_s = None

        field_velocity: Optional[np.ndarray] = None
        if field_xy is not None and self._previous_ball_field_xy is not None:
            dt = max(timestamp_s - (self._previous_ball_timestamp_s or timestamp_s), 1e-3)
            field_velocity = (field_xy - self._previous_ball_field_xy) / dt
        if field_xy is not None:
            self._previous_ball_field_xy = field_xy.copy()
            self._previous_ball_timestamp_s = timestamp_s

        status = BallStatus(estimate.status)
        return BallState(
            status=status,
            image_x=float(image_xy[0]),
            image_y=float(image_xy[1]),
            field_x=float(field_xy[0]) if field_xy is not None else None,
            field_y=float(field_xy[1]) if field_xy is not None else None,
            velocity_x=float(field_velocity[0]) if field_velocity is not None else None,
            velocity_y=float(field_velocity[1]) if field_velocity is not None else None,
            confidence=estimate.confidence,
            age_frames=estimate.age_frames,
        )

    @staticmethod
    def _find_possession_track_id(
        players: list[PlayerState], ball: BallState
    ) -> Optional[int]:
        if ball.field_x is None or ball.field_y is None:
            return None
        ball_xy = np.array([ball.field_x, ball.field_y], dtype=np.float32)
        candidates = [
            player
            for player in players
            if player.field_x is not None and player.field_y is not None
        ]
        if not candidates:
            return None
        distances = [
            float(np.linalg.norm(ball_xy - np.array([p.field_x, p.field_y], dtype=np.float32)))
            for p in candidates
        ]
        best_index = int(np.argmin(distances))
        return candidates[best_index].track_id if distances[best_index] <= POSSESSION_DISTANCE_MM else None

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

        vision_core = self._vision_core
        processed = vision_core.frames_processed if vision_core is not None else 0
        pitch_calls = vision_core.pitch_detection_count if vision_core is not None else 0
        reused = vision_core.pitch_reuse_count if vision_core is not None else 0
        available = vision_core.homography_available_count if vision_core is not None else 0
        player_calls = vision_core.player_inference_count if vision_core is not None else 0
        player_time = vision_core.player_inference_time_ms if vision_core is not None else 0.0
        pitch_time = vision_core.pitch_inference_time_ms if vision_core is not None else 0.0
        track_interruptions = (
            vision_core.track_id_interruptions if vision_core is not None else 0
        )
        ball_processor = self._ball_processor
        ball_calls = ball_processor.detection_count if ball_processor is not None else 0
        ball_predicted = ball_processor.predicted_frames if ball_processor is not None else 0
        ball_available = ball_processor.available_frames if ball_processor is not None else 0
        semantic_switches = (
            int(getattr(self._semantic_manager, "semantic_label_switches", 0))
            if self._semantic_manager is not None
            else 0
        )

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
            player_inference_latency_ms=round(player_time / max(player_calls, 1), 1),
            pitch_inference_latency_ms=round(pitch_time / max(pitch_calls, 1), 1),
            pitch_detection_count=pitch_calls,
            homography_reuse_ratio=round(reused / max(processed, 1), 3),
            homography_available_ratio=round(available / max(processed, 1), 3),
            track_id_interruptions=track_interruptions,
            semantic_inference_count=self.semantic_inference_count,
            semantic_label_switches=semantic_switches,
            ball_detection_count=ball_calls,
            ball_predicted_frames=ball_predicted,
            ball_available_ratio=round(ball_available / max(processed, 1), 3),
            jpeg_frames_encoded=int(getattr(self._store, "jpeg_frames_encoded", 0)),
            jpeg_encode_latency_ms=round(
                float(getattr(self._store, "jpeg_encode_latency_ms", 0.0)), 3
            ),
            foul_inference_count=self.foul_inference_count,
        )
        self._store.metrics = metrics


def detection_confidence(detection: Any) -> float:
    try:
        if hasattr(detection, 'confidence') and len(detection) > 2:
            return float(detection.confidence[0]) if detection.confidence is not None else 0.0
        return 0.0
    except (IndexError, TypeError):
        return 0.0
