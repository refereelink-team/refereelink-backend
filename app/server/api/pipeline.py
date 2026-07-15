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
    enable_foul_detection: Optional[bool] = None
    foul_checkpoint_path: Optional[str] = None


@router.post("/api/pipeline/start")
async def pipeline_start(request: Request, payload: PipelineStartPayload) -> dict:
    store = get_store(request)
    create_pipeline = getattr(request.app.state, "create_pipeline", None)
    attach_and_start = getattr(request.app.state, "attach_and_start_pipeline", None)
    current = getattr(request.app.state, "pipeline", None)

    device = payload.device or store.config.device or "cpu"
    foul_ckpt = payload.foul_checkpoint_path

    if current is not None and (payload.video_source is None
                                and payload.device is None
                                and payload.enable_foul_detection is None):
        # Resume existing pipeline.
        store.pipeline_running = True
        try:
            current.start()
        except Exception:
            pass
        return {"status": "started", "mode": "resume"}

    if payload.video_source is not None and (current is None or payload.video_source != store.config.video_source):
        # Build a new pipeline with the requested source.
        try:
            if create_pipeline is None or attach_and_start is None:
                return {"status": "error", "detail": "pipeline factory not registered"}
            pipeline = create_pipeline(
                payload.video_source,
                device=device,
                foul_checkpoint_path=foul_ckpt,
            )
            if payload.enable_foul_detection is not None:
                pipeline._enable_foul = payload.enable_foul_detection  # type: ignore[attr-defined]
            attach_and_start(pipeline)
            store.update_config({
                "video_source": payload.video_source,
                "device": device,
                "enable_foul_detection": bool(payload.enable_foul_detection) if payload.enable_foul_detection is not None else store.config.enable_foul_detection,
            })
            return {"status": "started", "mode": "new", "video_source": payload.video_source}
        except FileNotFoundError as exc:
            store.source_status = SourceStatus.ERROR
            return {"status": "error", "detail": f"video source not found: {exc}"}
        except Exception as exc:
            logger.exception("Failed to start pipeline with source=%s", payload.video_source)
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