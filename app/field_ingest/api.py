from __future__ import annotations

import re
from typing import Any
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse

from app.field_ingest.models import (
    FieldSessionRegistration,
    LiveAllocationRequest,
    SCHEMA_VERSION,
)
from app.field_ingest.service import FieldIngestService

_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _bearer(value: str | None) -> str | None:
    if not value:
        return None
    scheme, _, token = value.partition(" ")
    return token if scheme.lower() == "bearer" and token else None


def _require_http_auth(request: Request, service: FieldIngestService) -> None:
    if not service.settings.auth_token:
        raise HTTPException(status_code=503, detail="field ingest authentication is not configured")
    if not service.authorize(_bearer(request.headers.get("authorization"))):
        raise HTTPException(status_code=401, detail="invalid field ingest bearer token")


def create_router(service: FieldIngestService) -> APIRouter:
    router = APIRouter(tags=["field-ingest"])

    @router.get("/api/v1/field/capabilities")
    async def capabilities(request: Request) -> dict[str, Any]:
        _require_http_auth(request, service)
        return service.capabilities().model_dump(mode="json", by_alias=True)

    @router.post("/api/v1/field/sessions", status_code=status.HTTP_201_CREATED)
    async def register_session(
        request: Request, registration: FieldSessionRegistration
    ) -> JSONResponse:
        _require_http_auth(request, service)
        try:
            result, data = service.register(registration)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if result == "conflict":
            raise HTTPException(
                status_code=409, detail="session_id already has a different registration"
            )
        return JSONResponse(
            status_code=200 if result == "same" else status.HTTP_201_CREATED,
            content={
                "schema_version": SCHEMA_VERSION,
                "session_id": str(registration.session_id),
                "device_id": str(registration.device_id),
                "status": data.get("status", "registered"),
            },
        )

    @router.get("/api/v1/field/sessions/{session_id}")
    async def session_status(request: Request, session_id: UUID) -> dict[str, Any]:
        _require_http_auth(request, service)
        try:
            return service.status(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    @router.post("/api/v1/field/sessions/{session_id}/live")
    async def allocate_live(
        request: Request,
        session_id: UUID,
        allocation_request: LiveAllocationRequest,
    ) -> dict[str, Any]:
        _require_http_auth(request, service)
        try:
            allocation = service.allocate_live(session_id, allocation_request)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return allocation.model_dump(mode="json", by_alias=True)

    @router.delete("/api/v1/field/sessions/{session_id}/live/{epoch}")
    async def release_live(request: Request, session_id: UUID, epoch: int) -> dict[str, Any]:
        _require_http_auth(request, service)
        service.release_live(session_id, epoch)
        return {"status": "released", "session_id": str(session_id), "stream_epoch": epoch}

    @router.get("/api/v1/field/sessions/{session_id}/artifacts")
    async def list_artifacts(request: Request, session_id: UUID) -> dict[str, Any]:
        _require_http_auth(request, service)
        try:
            return {"session_id": str(session_id), "artifacts": service.artifacts(session_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc

    @router.put("/api/v1/field/sessions/{session_id}/artifacts/{artifact_id}")
    async def upload_artifact(
        request: Request, session_id: UUID, artifact_id: str
    ) -> dict[str, Any]:
        _require_http_auth(request, service)
        if not _ARTIFACT_ID_RE.fullmatch(artifact_id):
            raise HTTPException(status_code=400, detail="invalid artifact_id")
        content_length = request.headers.get("content-length")
        content_sha256 = request.headers.get("content-sha256")
        if content_length is None or content_sha256 is None:
            raise HTTPException(
                status_code=411, detail="Content-Length and Content-SHA256 are required"
            )
        try:
            length = int(content_length)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid Content-Length") from exc
        try:
            result, data = service.save_artifact(
                session_id,
                artifact_id,
                await request.body(),
                content_sha256,
                length,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        except OverflowError as exc:
            raise HTTPException(status_code=413, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        if result == "conflict":
            raise HTTPException(status_code=409, detail="artifact_id already has a different hash")
        return {**data, "status": "already_exists" if result == "same" else "stored"}

    @router.post("/api/v1/field/sessions/{session_id}/complete")
    async def complete_session(request: Request, session_id: UUID) -> dict[str, Any]:
        _require_http_auth(request, service)
        try:
            service.complete(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="session not found") from exc
        return {"status": "complete", "session_id": str(session_id)}

    @router.websocket("/ws/v1/field/sessions/{session_id}")
    async def telemetry_socket(websocket: WebSocket, session_id: UUID) -> None:
        if not service.settings.auth_token or not service.authorize(
            _bearer(websocket.headers.get("authorization"))
        ):
            await websocket.close(code=1008, reason="invalid field ingest bearer token")
            return
        if service.store.get_session(session_id) is None:
            await websocket.close(code=1008, reason="session not found")
            return
        await websocket.accept()
        try:
            while True:
                message = await websocket.receive_json()
                try:
                    response = service.handle_message(session_id, message)
                except (KeyError, ValueError) as exc:
                    await websocket.send_json(
                        {"type": "error", "code": "INVALID_MESSAGE", "detail": str(exc)}
                    )
                    continue
                await websocket.send_json(response)
        except WebSocketDisconnect:
            service.release_telemetry(session_id, 0)
            return

    return router
