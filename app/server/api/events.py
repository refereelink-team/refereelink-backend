from __future__ import annotations

from fastapi import APIRouter, Request

from app.state.store import StateStore

router = APIRouter()


def get_store(request: Request) -> StateStore:
    return request.app.state.store


@router.get("/api/events")
async def get_events(request: Request, limit: int = 50, offset: int = 0) -> dict:
    store = get_store(request)
    all_events = store.events
    paginated = all_events[offset:offset + limit]
    return {
        "events": [e.model_dump() for e in paginated],
        "total": len(all_events),
    }
