from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.state.models import SourceStatus
from app.state.store import StateStore

router = APIRouter()
logger = logging.getLogger(__name__)


def get_store(request: Request) -> StateStore:
    return request.app.state.store


class PipelineStartPayload(BaseModel):
    """Payload for POST /api/pipeline/start.

    Any of the fields can be omitted. When ``video_source`` is provided
    and no pipeline is currently attached, the server builds a new
    InferencePipeline bound to that source and starts it. When omitted,
    the call falls back to the existing pipeline (if any).
    """
    video_source: Optional[str] = None
    device: Optional[str] = None
    inference_backend: Optional[str] = None
    enable_foul_detection: Optional[bool] = None
    foul_checkpoint_path: Optional[str] = None
    foul_confidence_threshold: Optional[float] = None
    player_model_path: Optional[str] = None
    pitch_model_path: Optional[str] = None
    camera_calibration_path: Optional[str] = None
    enable_undistortion: Optional[bool] = None
    calibration_alpha: Optional[float] = None
    pitch_detection_interval: Optional[int] = None
    imgsz: Optional[int] = None
    ball_model_path: Optional[str] = None
    enable_ball: Optional[bool] = None
    ball_detection_interval: Optional[int] = None
    ball_max_prediction_frames: Optional[int] = None
    role_model_path: Optional[str] = None
    team_classifier_path: Optional[str] = None
    role_detection_interval: Optional[int] = None
    team_classification_interval: Optional[int] = None


@router.post("/api/pipeline/start")
async def pipeline_start(request: Request, payload: PipelineStartPayload) -> dict:
    store = get_store(request)
    create_pipeline = getattr(request.app.state, "create_pipeline", None)
    attach_and_start = getattr(request.app.state, "attach_and_start_pipeline", None)
    current = getattr(request.app.state, "pipeline", None)

    device = payload.device or store.config.device or "cpu"
    config_requested = any(
        value is not None
        for value in (
            payload.player_model_path,
            payload.pitch_model_path,
            payload.camera_calibration_path,
            payload.enable_undistortion,
            payload.calibration_alpha,
            payload.pitch_detection_interval,
            payload.imgsz,
            payload.ball_model_path,
            payload.enable_ball,
            payload.ball_detection_interval,
            payload.ball_max_prediction_frames,
            payload.role_model_path,
            payload.team_classifier_path,
            payload.role_detection_interval,
            payload.team_classification_interval,
            payload.inference_backend,
            payload.enable_foul_detection,
            payload.foul_checkpoint_path,
            payload.foul_confidence_threshold,
        )
    )

    if current is not None and (
        payload.video_source is None
        and payload.device is None
        and payload.inference_backend is None
        and payload.enable_foul_detection is None
        and payload.foul_checkpoint_path is None
        and payload.foul_confidence_threshold is None
        and payload.player_model_path is None
        and payload.pitch_model_path is None
        and payload.camera_calibration_path is None
        and payload.enable_undistortion is None
        and payload.calibration_alpha is None
        and payload.pitch_detection_interval is None
        and payload.imgsz is None
        and payload.ball_model_path is None
        and payload.enable_ball is None
        and payload.ball_detection_interval is None
        and payload.ball_max_prediction_frames is None
        and payload.role_model_path is None
        and payload.team_classifier_path is None
        and payload.role_detection_interval is None
        and payload.team_classification_interval is None
    ):
        # Resume existing pipeline.
        store.pipeline_running = True
        try:
            current.start()
        except Exception:
            pass
        return {"status": "started", "mode": "resume"}

    requested_source = payload.video_source or store.config.video_source
    if requested_source and (
        current is None
        or payload.video_source != store.config.video_source
        or config_requested
    ):
        # Build a new pipeline with the requested source.
        try:
            if create_pipeline is None or attach_and_start is None:
                return {"status": "error", "detail": "pipeline factory not registered"}
            pipeline = create_pipeline(
                requested_source,
                device=device,
                inference_backend=payload.inference_backend or store.config.inference_backend,
                enable_foul_detection=(
                    payload.enable_foul_detection
                    if payload.enable_foul_detection is not None
                    else store.config.enable_foul_detection
                ),
                foul_checkpoint_path=payload.foul_checkpoint_path,
                foul_confidence_threshold=(
                    payload.foul_confidence_threshold
                    if payload.foul_confidence_threshold is not None
                    else store.config.foul_confidence_threshold
                ),
                player_model_path=payload.player_model_path or store.config.player_model_path,
                pitch_model_path=payload.pitch_model_path or store.config.pitch_model_path,
                camera_calibration_path=(
                    payload.camera_calibration_path
                    if payload.camera_calibration_path is not None
                    else store.config.camera_calibration_path
                ),
                enable_undistortion=(
                    payload.enable_undistortion
                    if payload.enable_undistortion is not None
                    else store.config.enable_undistortion
                ),
                calibration_alpha=(
                    payload.calibration_alpha
                    if payload.calibration_alpha is not None
                    else store.config.calibration_alpha
                ),
                pitch_detection_interval=(
                    payload.pitch_detection_interval
                    if payload.pitch_detection_interval is not None
                    else store.config.pitch_detection_interval
                ),
                imgsz=payload.imgsz if payload.imgsz is not None else store.config.imgsz,
                ball_model_path=payload.ball_model_path or store.config.ball_model_path,
                enable_ball=(
                    payload.enable_ball
                    if payload.enable_ball is not None
                    else store.config.enable_ball
                ),
                ball_detection_interval=(
                    payload.ball_detection_interval
                    if payload.ball_detection_interval is not None
                    else store.config.ball_detection_interval
                ),
                ball_max_prediction_frames=(
                    payload.ball_max_prediction_frames
                    if payload.ball_max_prediction_frames is not None
                    else store.config.ball_max_prediction_frames
                ),
                role_model_path=payload.role_model_path or store.config.role_model_path,
                team_classifier_path=(
                    payload.team_classifier_path
                    if payload.team_classifier_path is not None
                    else store.config.team_classifier_path
                ),
                role_detection_interval=(
                    payload.role_detection_interval
                    if payload.role_detection_interval is not None
                    else store.config.role_detection_interval
                ),
                team_classification_interval=(
                    payload.team_classification_interval
                    if payload.team_classification_interval is not None
                    else store.config.team_classification_interval
                ),
            )
            attach_and_start(pipeline)
            store.update_config({
                "video_source": requested_source,
                "device": device,
                "inference_backend": payload.inference_backend or store.config.inference_backend,
                "foul_confidence_threshold": (
                    payload.foul_confidence_threshold
                    if payload.foul_confidence_threshold is not None
                    else store.config.foul_confidence_threshold
                ),
                "player_model_path": payload.player_model_path or store.config.player_model_path,
                "pitch_model_path": payload.pitch_model_path or store.config.pitch_model_path,
                "camera_calibration_path": (
                    payload.camera_calibration_path
                    if payload.camera_calibration_path is not None
                    else store.config.camera_calibration_path
                ),
                "enable_undistortion": (
                    payload.enable_undistortion
                    if payload.enable_undistortion is not None
                    else store.config.enable_undistortion
                ),
                "calibration_alpha": (
                    payload.calibration_alpha
                    if payload.calibration_alpha is not None
                    else store.config.calibration_alpha
                ),
                "pitch_detection_interval": (
                    payload.pitch_detection_interval
                    if payload.pitch_detection_interval is not None
                    else store.config.pitch_detection_interval
                ),
                "imgsz": payload.imgsz if payload.imgsz is not None else store.config.imgsz,
                "ball_model_path": payload.ball_model_path or store.config.ball_model_path,
                "enable_ball": (
                    payload.enable_ball
                    if payload.enable_ball is not None
                    else store.config.enable_ball
                ),
                "ball_detection_interval": (
                    payload.ball_detection_interval
                    if payload.ball_detection_interval is not None
                    else store.config.ball_detection_interval
                ),
                "ball_max_prediction_frames": (
                    payload.ball_max_prediction_frames
                    if payload.ball_max_prediction_frames is not None
                    else store.config.ball_max_prediction_frames
                ),
                "role_model_path": payload.role_model_path or store.config.role_model_path,
                "team_classifier_path": (
                    payload.team_classifier_path
                    if payload.team_classifier_path is not None
                    else store.config.team_classifier_path
                ),
                "role_detection_interval": (
                    payload.role_detection_interval
                    if payload.role_detection_interval is not None
                    else store.config.role_detection_interval
                ),
                "team_classification_interval": (
                    payload.team_classification_interval
                    if payload.team_classification_interval is not None
                    else store.config.team_classification_interval
                ),
                "enable_foul_detection": bool(payload.enable_foul_detection) if payload.enable_foul_detection is not None else store.config.enable_foul_detection,
            })
            return {"status": "started", "mode": "new", "video_source": requested_source}
        except FileNotFoundError as exc:
            store.source_status = SourceStatus.ERROR
            return {"status": "error", "detail": f"video source not found: {exc}"}
        except Exception as exc:
            logger.exception("Failed to start pipeline with source=%s", requested_source)
            store.source_status = SourceStatus.ERROR
            return {"status": "error", "detail": str(exc)}

    if current is not None:
        store.pipeline_running = True
        try:
            current.start()
        except Exception:
            pass
        return {"status": "started", "mode": "resume"}

    return {"status": "error", "detail": "no video_source provided and no existing pipeline to resume"}


@router.post("/api/pipeline/stop")
async def pipeline_stop(request: Request) -> dict:
    store = get_store(request)
    stop_and_clear = getattr(request.app.state, "stop_and_clear_pipeline", None)
    if stop_and_clear is not None:
        stop_and_clear()
        return {"status": "stopped", "mode": "cleared"}
    store.pipeline_running = False
    store.source_status = SourceStatus.DISCONNECTED
    return {"status": "stopped"}
