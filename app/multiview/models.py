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


class CaptureState(str, Enum):
    READY = "capture_ready"
    FAILED = "capture_failed"


class RiskLevel(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EvidenceSource(str, Enum):
    MODEL = "model"
    HUMAN = "human"
    GEOMETRY = "geometry"
    RULE = "rule"


class TeamLabel(str, Enum):
    HOME = "home"
    AWAY = "away"
    UNKNOWN = "unknown"


class DefendsSide(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    UNKNOWN = "unknown"


class ContactRegion(str, Enum):
    UPPER_BODY = "upper_body"
    LOWER_BODY = "lower_body"
    HEAD = "head"
    UNKNOWN = "unknown"


class ChallengeIntensity(str, Enum):
    CARELESS = "careless"
    RECKLESS = "reckless"
    EXCESSIVE_FORCE = "excessive_force"
    UNKNOWN = "unknown"


class TacticalImpact(str, Enum):
    NONE = "none"
    SPA = "spa"
    DOGSO = "dogso"
    UNKNOWN = "unknown"


class AssessmentStatus(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    UNSUPPORTED = "unsupported"


class RestartType(str, Enum):
    PLAY_ON = "play_on"
    DIRECT_FREE_KICK = "direct_free_kick"
    INDIRECT_FREE_KICK = "indirect_free_kick"
    PENALTY = "penalty"
    PREVIOUS_RESTART = "previous_restart"
    UNKNOWN = "unknown"


class SanctionType(str, Enum):
    NONE = "none"
    YELLOW_CARD = "yellow_card"
    RED_CARD = "red_card"
    PENDING = "pending"


class EvidenceValue(BaseModel):
    value: Any | None = None
    source: EvidenceSource = EvidenceSource.HUMAN
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confirmed: bool = False


class FoulLocation(BaseModel):
    x_m: float = Field(ge=0.0, le=105.0)
    y_m: float = Field(ge=0.0, le=68.0)
    source: Literal["human", "vision"] = "human"
    confirmed: bool = True


class LocationGeometry(BaseModel):
    half: Literal["left", "right", "center"]
    zone: str
    penalty_area_side: Literal["left", "right"] | None = None
    in_penalty_area: bool
    in_offender_own_penalty_area: bool | None = None
    distance_to_left_goal_m: float
    distance_to_right_goal_m: float


class FoulFacts(BaseModel):
    offence_confirmed: EvidenceValue = Field(default_factory=EvidenceValue)
    action: EvidenceValue = Field(default_factory=EvidenceValue)
    offender_team: EvidenceValue = Field(default_factory=EvidenceValue)
    victim_team: EvidenceValue = Field(default_factory=EvidenceValue)
    ball_in_play: EvidenceValue = Field(default_factory=EvidenceValue)
    contact: EvidenceValue = Field(default_factory=EvidenceValue)
    contact_region: EvidenceValue = Field(default_factory=EvidenceValue)
    intensity: EvidenceValue = Field(default_factory=EvidenceValue)
    attempt_to_play_ball: EvidenceValue = Field(default_factory=EvidenceValue)
    tactical_impact: EvidenceValue = Field(default_factory=EvidenceValue)
    location: FoulLocation | None = None
    home_defends_side: EvidenceValue = Field(default_factory=EvidenceValue)


class RuleTraceEntry(BaseModel):
    rule_id: str
    law: str
    section: str
    facts_used: list[str] = Field(default_factory=list)
    result: str
    priority: int = 0
    law_excerpt: str = ""


class RuleAssessment(BaseModel):
    status: AssessmentStatus
    restart: RestartType = RestartType.UNKNOWN
    sanction: SanctionType = SanctionType.PENDING
    ruleset_version: str = "IFAB_2026_27"
    rule_trace: list[RuleTraceEntry] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    geometry: LocationGeometry | None = None
    explanation_template: str = ""


class ReviewRecord(BaseModel):
    case_id: str
    analysis_id: str | None = None
    revision: int = Field(ge=1)
    facts: FoulFacts
    assessment: RuleAssessment
    review_state: ReviewState = ReviewState.PENDING
    created_at: str
    updated_at: str


class ReviewUpdateRequest(BaseModel):
    expected_revision: int = Field(default=0, ge=0)
    analysis_id: str | None = None
    facts: FoulFacts
    review_state: ReviewState = ReviewState.PENDING


class ExplanationRequest(BaseModel):
    revision: int | None = Field(default=None, ge=1)
    use_llm: bool = True


class ExplanationResponse(BaseModel):
    case_id: str
    revision: int
    source: Literal["template", "local_llm"]
    summary: str
    restart: RestartType
    sanction: SanctionType
    rule_ids: list[str] = Field(default_factory=list)
    fallback_reason: str | None = None


class CandidateScore(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)


class EvidenceView(BaseModel):
    camera_id: str
    display_name: str
    role: CameraRole = CameraRole.OTHER
    path: str | None = None
    preview_path: str | None = None
    sync_offset_ms: int = 0
    quality: str = "清晰"


class TemporalBin(BaseModel):
    start_s: float = Field(ge=0.0)
    end_s: float = Field(ge=0.0)
    score: float = Field(ge=0.0, le=1.0)


class LocalizationBox(BaseModel):
    rect: tuple[float, float, float, float]
    score: float = Field(ge=0.0)
    source: str
    active_start_s: float | None = Field(default=None, ge=0.0)
    active_end_s: float | None = Field(default=None, ge=0.0)
    peak_s: float | None = Field(default=None, ge=0.0)
    temporal_bins: list[TemporalBin] = Field(default_factory=list)
    temporal_source: Literal["gradcam", "event_prior"] | None = None
    display_tier: Literal["normal", "caution", "hidden"] = "normal"
    reliable: bool = True
    reliability_score: float = Field(default=1.0, ge=0.0, le=1.0)
    reliability_reasons: list[str] = Field(default_factory=list)


class ScriptedResult(BaseModel):
    decision: str
    decision_zh: str
    action: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    card: Literal["none", "yellow", "red"] = "none"
    localization: dict[str, LocalizationBox] = Field(default_factory=dict)
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
    capture_state: CaptureState | None = None


class MultiviewAnalyzeRequest(BaseModel):
    case_id: str
    device: str = "auto"


class MultiviewDecision(BaseModel):
    analysis_id: str
    event_id: str
    case_id: str
    timestamp: float
    decision: str
    decision_zh: str
    action: str
    severity: str
    confidence: float = Field(ge=0.0, le=1.0)
    action_candidates: list[CandidateScore] = Field(default_factory=list)
    severity_candidates: list[CandidateScore] = Field(default_factory=list)
    checkpoint_hash: str | None = None
    ruleset_compatible: bool = True
    card: Literal["none", "yellow", "red"] = "none"
    suggested_intensity: str | None = None
    mode: Literal["model", "scripted"]
    model: str | None = None
    device: str | None = None
    inference_ms: float | None = None
    preprocess_ms: float | None = None
    gradcam_ms: float | None = None
    gpu_mem_mb: float | None = None
    localization: dict[str, LocalizationBox] = Field(default_factory=dict)
    localization_source: str | None = None
    view_attention: list[float] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)


class MultiviewAnalyzeResponse(BaseModel):
    status: Literal["ok", "error"]
    message: str
    decision: MultiviewDecision | None = None
