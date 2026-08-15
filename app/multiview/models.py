from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class CameraRole(str, Enum):
    MAIN = "main"
    SIDE = "side"
    REPLAY = "replay"
    OTHER = "other"


class ReviewState(str, Enum):
    PENDING = "pending"
    REVIEWED = "reviewed"
    ARCHIVED = "archived"
    UNCERTAIN = "uncertain"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceView(BaseModel):
    camera_id: str
    display_name: str
    role: CameraRole = CameraRole.OTHER
    path: str | None = None
    preview_path: str | None = None
    sync_offset_ms: int = 0
    quality: str = "清晰"


class ScriptedResult(BaseModel):
    decision: str
    decision_zh: str
    action: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    card: Literal["none", "yellow", "red"] = "none"
    localization: dict[str, dict[str, Any]] = Field(default_factory=dict)
    view_attention: list[float] = Field(default_factory=list)


class MultiviewCase(BaseModel):
    case_id: str
    title: str
    match_name: str
    match_clock: str
    description: str = ""
    event_time_s: float = Field(default=0.0, ge=0.0)
    review_state: ReviewState = ReviewState.PENDING
    risk_level: RiskLevel = RiskLevel.MEDIUM
    zone: str = "未知区域"
    videos: list[EvidenceView] = Field(min_length=1)
    scripted_result: ScriptedResult | None = None
    evidence_notes: list[str] = Field(default_factory=list)


class MultiviewAnalyzeRequest(BaseModel):
    case_id: str
    device: str = "auto"


class MultiviewDecision(BaseModel):
    event_id: str
    case_id: str
    timestamp: float
    decision: str
    decision_zh: str
    action: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    card: Literal["none", "yellow", "red"] = "none"
    mode: Literal["model", "scripted"]
    model: str | None = None
    device: str | None = None
    inference_ms: float | None = None
    preprocess_ms: float | None = None
    gradcam_ms: float | None = None
    gpu_mem_mb: float | None = None
    localization: dict[str, dict[str, Any]] = Field(default_factory=dict)
    localization_source: str | None = None
    view_attention: list[float] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)


class MultiviewAnalyzeResponse(BaseModel):
    status: Literal["ok", "error"]
    message: str
    decision: MultiviewDecision | None = None
