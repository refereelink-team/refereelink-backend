from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.multiview.models import MultiviewAnalyzeRequest
from app.multiview.service import MultiviewAnalysisService

router = APIRouter(prefix="/api/multiview", tags=["multiview"])


def _service(request: Request) -> MultiviewAnalysisService:
    return request.app.state.multiview_service  # type: ignore[attr-defined]


def _case_payload(service: MultiviewAnalysisService, case) -> dict:
    payload = case.model_dump(mode="json")
    payload.pop("scripted_result", None)
    for view_payload, view in zip(payload["videos"], case.videos):
        resolved = service.repository.resolve_view_media(case.case_id, view.camera_id)
        view_payload.pop("path", None)
        view_payload.pop("preview_path", None)
        view_payload["media_url"] = (
            f"/api/multiview/cases/{case.case_id}/media/{view.camera_id}"
            if resolved is not None
            else None
        )
        view_payload["media_kind"] = (
            "video" if resolved is not None and resolved[1].startswith("video/") else "image"
        )
    return payload


@router.get("/cases")
def list_cases(request: Request) -> dict:
    service = _service(request)
    cases = service.repository.list_cases()
    return {"count": len(cases), "cases": [_case_payload(service, case) for case in cases]}


@router.get("/cases/{case_id}")
def get_case(case_id: str, request: Request) -> dict:
    service = _service(request)
    case = service.repository.get_case(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    return _case_payload(service, case)


@router.get("/cases/{case_id}/media/{camera_id}")
def get_case_media(case_id: str, camera_id: str, request: Request) -> FileResponse:
    resolved = _service(request).repository.resolve_view_media(case_id, camera_id)
    if resolved is None:
        raise HTTPException(status_code=404, detail="Evidence media not found")
    path, media_type = resolved
    return FileResponse(path, media_type=media_type, headers={"Accept-Ranges": "bytes"})


@router.get("/status")
def get_status(request: Request) -> dict:
    return _service(request).status()


@router.post("/analyze")
def analyze(payload: MultiviewAnalyzeRequest, request: Request) -> dict:
    return _service(request).analyze(payload.case_id, payload.device).model_dump(mode="json")
