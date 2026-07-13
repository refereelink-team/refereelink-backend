from __future__ import annotations

from fastapi import APIRouter, Request

from app.state.store import StateStore

router = APIRouter()


def get_store(request: Request) -> StateStore:
    return request.app.state.store


@router.post("/api/pipeline/start")
async def pipeline_start(request: Request) -> dict:
    store = get_store(request)
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is not None:
        pipeline.start()
    store.pipeline_running = True
    return {"status": "started"}


@router.post("/api/pipeline/stop")
async def pipeline_stop(request: Request) -> dict:
    store = get_store(request)
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is not None:
        pipeline.stop()
    store.pipeline_running = False
    return {"status": "stopped"}
