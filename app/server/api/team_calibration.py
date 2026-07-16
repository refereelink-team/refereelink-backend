from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from app.classification.team_calibration.session import TeamCalibrationSession
from app.classification.team_calibration.types import CalibrationLabel
from app.state.store import StateStore

router = APIRouter()
logger = logging.getLogger(__name__)


class CalibrationStartPayload(BaseModel):
    match_id: str
    camera_id: str = "default"
    video_source: Optional[str] = None
    bundle_path: Optional[str] = None
    device: Optional[str] = None


class CalibrationLabelPayload(BaseModel):
    track_id: int
    label: CalibrationLabel


def _session(request: Request) -> TeamCalibrationSession:
    store: StateStore = request.app.state.store
    return store.team_calibration


@router.get("/api/team-calibration")
async def get_team_calibration(request: Request) -> dict:
    return _session(request).snapshot()


@router.post("/api/team-calibration/start")
async def start_team_calibration(
    request: Request,
    payload: CalibrationStartPayload,
) -> dict:
    store: StateStore = request.app.state.store
    session = store.team_calibration
    source = payload.video_source or store.config.video_source
    if not source:
        raise HTTPException(status_code=400, detail="video_source is required for calibration")
    create_pipeline = getattr(request.app.state, "create_pipeline", None)
    attach_and_start = getattr(request.app.state, "attach_and_start_pipeline", None)
    if create_pipeline is None or attach_and_start is None:
        raise HTTPException(status_code=503, detail="pipeline factory not registered")

    session.start(
        match_id=payload.match_id,
        camera_id=payload.camera_id,
        bundle_path=payload.bundle_path,
        device=payload.device or store.config.device,
    )
    try:
        pipeline = create_pipeline(
            source,
            device=payload.device or store.config.device,
            inference_backend=store.config.inference_backend,
            enable_foul_detection=False,
            player_model_path=store.config.player_model_path,
            pitch_model_path=store.config.pitch_model_path,
            camera_calibration_path=store.config.camera_calibration_path,
            enable_undistortion=store.config.enable_undistortion,
            calibration_alpha=store.config.calibration_alpha,
            pitch_detection_interval=store.config.pitch_detection_interval,
            imgsz=store.config.imgsz,
            ball_model_path=store.config.ball_model_path,
            enable_ball=False,
            ball_detection_interval=store.config.ball_detection_interval,
            ball_max_prediction_frames=store.config.ball_max_prediction_frames,
            role_model_path=store.config.role_model_path,
            team_calibration_path=None,
            role_detection_interval=store.config.role_detection_interval,
            team_classification_interval=store.config.team_classification_interval,
            calibration_session=session,
        )
        attach_and_start(pipeline)
        store.update_config({"video_source": source, "device": payload.device or store.config.device})
    except Exception as exc:
        logger.exception("Failed to start team calibration")
        session.reset()
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return session.snapshot()


@router.post("/api/team-calibration/label")
async def label_team_track(
    request: Request,
    payload: CalibrationLabelPayload,
) -> dict:
    try:
        return _session(request).label_track(payload.track_id, payload.label)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/api/team-calibration/validate")
async def validate_team_calibration(request: Request) -> dict:
    try:
        return _session(request).validate()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/api/team-calibration/reset")
async def reset_team_calibration(request: Request) -> dict:
    return _session(request).reset()

