from __future__ import annotations

from fastapi import APIRouter, Request

from app.state.models import PipelineConfig
from app.state.store import StateStore

router = APIRouter()


def get_store(request: Request) -> StateStore:
    return request.app.state.store


@router.get("/api/status")
async def get_status(request: Request) -> dict:
    store = get_store(request)
    return {
        "pipeline_running": store.pipeline_running,
        "source_status": store.source_status.value,
        "metrics": store.metrics.model_dump(),
        "team_classifier_ready": getattr(
            getattr(request.app.state, "pipeline", None),
            "_team_classifier",
            None,
        ) is not None,
    }


@router.get("/api/config")
async def get_config(request: Request) -> dict:
    store = get_store(request)
    return store.config.model_dump()


@router.put("/api/config")
async def update_config(request: Request, updates: dict) -> dict:
    store = get_store(request)
    new_config = store.update_config(updates)
    return new_config.model_dump()
