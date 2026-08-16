from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from app.multiview.models import ExplanationRequest, MultiviewAnalyzeRequest, ReviewUpdateRequest
from app.multiview.review_store import ReviewRevisionConflict
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
    review = service.get_review(case.case_id)
    if review is not None:
        payload["review_state"] = review.review_state.value
        payload["review_revision"] = review.revision
    else:
        payload["review_revision"] = 0
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


@router.get("/cases/{case_id}/review")
def get_review(case_id: str, request: Request) -> dict:
    service = _service(request)
    if service.repository.get_case(case_id) is None:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    review = service.get_review(case_id)
    analysis = (
        service.review_store.get_analysis(review.analysis_id)
        if review is not None and review.analysis_id
        else service.review_store.latest_analysis(case_id)
    )
    return {
        "review": review.model_dump(mode="json") if review else None,
        "analysis": analysis.model_dump(mode="json") if analysis else None,
    }


@router.put("/cases/{case_id}/review")
def put_review(case_id: str, payload: ReviewUpdateRequest, request: Request) -> dict:
    try:
        record = _service(request).update_review(
            case_id=case_id,
            expected_revision=payload.expected_revision,
            facts=payload.facts,
            analysis_id=payload.analysis_id,
            review_state=payload.review_state,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ReviewRevisionConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "REVIEW_REVISION_CONFLICT",
                "expected_revision": exc.expected,
                "current_revision": exc.current,
            },
        ) from exc
    return {"review": record.model_dump(mode="json")}


@router.get("/cases/{case_id}/review/history")
def get_review_history(case_id: str, request: Request) -> dict:
    service = _service(request)
    if service.repository.get_case(case_id) is None:
        raise HTTPException(status_code=404, detail=f"Case not found: {case_id}")
    history = service.review_history(case_id)
    return {
        "count": len(history),
        "history": [record.model_dump(mode="json") for record in history],
    }


@router.post("/cases/{case_id}/explanation")
def explain_review(case_id: str, payload: ExplanationRequest, request: Request) -> dict:
    try:
        explanation = _service(request).explain_review(
            case_id,
            payload.revision,
            use_llm=payload.use_llm,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Review revision not found") from exc
    return explanation.model_dump(mode="json")
