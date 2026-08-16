from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from app.multiview.inference import FoulInferenceError, FoulInferenceService, model_prerequisites
from app.multiview.explanation import GuardedExplanationWriter
from app.multiview.localization import event_prior_window
from app.multiview.models import (
    AssessmentStatus,
    FoulFacts,
    ExplanationResponse,
    MultiviewAnalyzeResponse,
    MultiviewDecision,
    ReviewRecord,
    ReviewState,
)
from app.multiview.repository import MultiviewCaseRepository
from app.multiview.review_store import MultiviewReviewStore, new_analysis_id
from app.multiview.rules import IFABRuleEngine, PHYSICAL_ACTIONS


class MultiviewAnalysisService:
    def __init__(
        self,
        repository: MultiviewCaseRepository | None = None,
        review_store: MultiviewReviewStore | None = None,
    ) -> None:
        self.repository = repository or MultiviewCaseRepository()
        self.review_store = review_store or MultiviewReviewStore()
        self.rule_engine = IFABRuleEngine()
        self.explanation_writer = GuardedExplanationWriter()
        self._lock = threading.Lock()
        self._analyzers: dict[str, FoulInferenceService] = {}
        self._load_errors: dict[str, str] = {}

    def status(self) -> dict[str, Any]:
        status = model_prerequisites()
        status["loaded_devices"] = sorted(self._analyzers)
        status["load_errors"] = dict(self._load_errors)
        status["fallback_available"] = any(
            case.scripted_result is not None for case in self.repository.list_cases()
        )
        status["mode"] = "model" if status["ready"] else "scripted_fallback"
        return status

    def _analyzer(self, device: str) -> FoulInferenceService:
        normalized = device or "auto"
        with self._lock:
            existing = self._analyzers.get(normalized)
            if existing is not None:
                return existing
            try:
                analyzer = FoulInferenceService(device=normalized)
            except Exception as exc:
                self._load_errors[normalized] = str(exc)
                raise
            self._analyzers[normalized] = analyzer
            self._load_errors.pop(normalized, None)
            return analyzer

    @staticmethod
    def _localization_with_temporal_prior(case, localization: dict[str, Any]) -> dict[str, dict[str, Any]]:
        views = {view.camera_id: view for view in case.videos}
        prepared: dict[str, dict[str, Any]] = {}
        for camera_id, raw_box in localization.items():
            box = raw_box.model_dump(mode="json") if hasattr(raw_box, "model_dump") else dict(raw_box)
            view = views.get(camera_id)
            if box.get("active_start_s") is None or box.get("temporal_source") == "event_prior":
                offset_s = (view.sync_offset_ms if view is not None else 0) / 1000.0
                box.update(event_prior_window(case.event_time_s + offset_s))
            prepared[camera_id] = box
        return prepared

    def analyze(self, case_id: str, device: str = "auto") -> MultiviewAnalyzeResponse:
        case = self.repository.get_case(case_id)
        if case is None:
            return MultiviewAnalyzeResponse(status="error", message=f"案例不存在：{case_id}")
        video_paths = self.repository.available_video_paths(case)
        prerequisites = model_prerequisites()
        if prerequisites["ready"] and len(video_paths) == len(case.videos):
            try:
                raw = self._analyzer(device).analyze_views(
                    video_paths,
                    event_time_s=case.event_time_s,
                )
                views_by_name = {
                    self.repository.resolve_path(view.path).name: view.camera_id
                    for view in case.videos
                    if self.repository.resolve_path(view.path) is not None
                }
                localization = self._localization_with_temporal_prior(case, {
                    views_by_name.get(raw["views"][int(index)], raw["views"][int(index)]): box
                    for index, box in raw.get("localization", {}).items()
                    if int(index) < len(raw.get("views", []))
                })
                decision = MultiviewDecision(
                    analysis_id=new_analysis_id(case.case_id),
                    event_id=f"MVF-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                    case_id=case.case_id,
                    timestamp=case.event_time_s,
                    decision=raw["decision"],
                    decision_zh=raw["decision_zh"],
                    action=raw["action"],
                    severity=raw["severity"],
                    confidence=raw["confidence"],
                    action_candidates=raw.get("action_candidates", []),
                    severity_candidates=raw.get("severity_candidates", []),
                    checkpoint_hash=raw.get("checkpoint_hash"),
                    ruleset_compatible=raw["action"].strip().lower() in PHYSICAL_ACTIONS | {"dive"},
                    card=raw["card"],
                    mode="model",
                    model=raw["model"],
                    device=raw["device"],
                    inference_ms=raw["inference_ms"],
                    preprocess_ms=raw["preprocess_ms"],
                    gradcam_ms=raw["gradcam_ms"],
                    gpu_mem_mb=raw["gpu_mem_mb"],
                    localization=localization,
                    localization_source=raw.get("localization_source"),
                    view_attention=raw.get("view_attention", []),
                    detail=raw,
                )
                self.review_store.save_analysis(decision)
                return MultiviewAnalyzeResponse(
                    status="ok", message="MViT_V2_S 多视角分析与 Grad-CAM 定位完成", decision=decision
                )
            except FoulInferenceError as exc:
                self._load_errors[device] = str(exc)

        scripted = case.scripted_result
        if scripted is None:
            missing = "；".join(prerequisites["missing"]) or "案例视频不完整"
            return MultiviewAnalyzeResponse(
                status="error", message=f"真实模型链路不可用，且案例没有演示结果：{missing}"
            )
        decision = MultiviewDecision(
            analysis_id=new_analysis_id(case.case_id),
            event_id=f"SCRIPT-{case.case_id}",
            case_id=case.case_id,
            timestamp=case.event_time_s,
            decision=scripted.decision,
            decision_zh=scripted.decision_zh,
            action=scripted.action,
            severity=scripted.severity,
            confidence=scripted.confidence,
            action_candidates=[{"label": scripted.action, "confidence": scripted.confidence}],
            severity_candidates=[{"label": scripted.severity, "confidence": scripted.confidence}],
            ruleset_compatible=scripted.action.strip().lower() in PHYSICAL_ACTIONS | {"dive"},
            card=scripted.card,
            mode="scripted",
            localization=self._localization_with_temporal_prior(case, scripted.localization),
            localization_source="scripted",
            view_attention=scripted.view_attention,
            detail={"missing": prerequisites["missing"]},
        )
        self.review_store.save_analysis(decision)
        return MultiviewAnalyzeResponse(
            status="ok",
            message="演示数据（非模型输出）；部署官方 VARS 源码与权重后自动切换真实 CUDA 推理",
            decision=decision,
        )

    def get_review(self, case_id: str) -> ReviewRecord | None:
        return self.review_store.latest_review(case_id)

    def review_history(self, case_id: str) -> list[ReviewRecord]:
        return self.review_store.review_history(case_id)

    def explain_review(
        self,
        case_id: str,
        revision: int | None,
        *,
        use_llm: bool,
    ) -> ExplanationResponse:
        record = (
            self.review_store.get_review_revision(case_id, revision)
            if revision is not None
            else self.review_store.latest_review(case_id)
        )
        if record is None:
            raise KeyError(case_id)
        analysis = (
            self.review_store.get_analysis(record.analysis_id)
            if record.analysis_id is not None
            else None
        )
        return self.explanation_writer.explain(record, analysis, use_llm=use_llm)

    @staticmethod
    def _fact_value(facts: FoulFacts, name: str) -> Any | None:
        item = getattr(facts, name)
        return item.value if item.confirmed else None

    def _model_conflicts(
        self,
        facts: FoulFacts,
        analysis: MultiviewDecision | None,
    ) -> list[str]:
        if analysis is None:
            return []
        conflicts: list[str] = []
        action = self._fact_value(facts, "action")
        if action is not None and str(action).strip().lower() != analysis.action.strip().lower():
            conflicts.append(f"人工动作“{action}”与模型建议“{analysis.action}”不同")
        offence = self._fact_value(facts, "offence_confirmed")
        model_says_offence = analysis.card != "none" or "No Offence" not in analysis.severity
        if offence is not None and bool(offence) != model_says_offence:
            conflicts.append("人工犯规确认与模型犯规建议不同")
        return conflicts

    def update_review(
        self,
        *,
        case_id: str,
        expected_revision: int,
        facts: FoulFacts,
        analysis_id: str | None,
        review_state: ReviewState,
    ) -> ReviewRecord:
        if self.repository.get_case(case_id) is None:
            raise KeyError(case_id)
        analysis = self.review_store.get_analysis(analysis_id) if analysis_id else None
        if analysis is not None and analysis.case_id != case_id:
            raise ValueError("analysis_id does not belong to this case")
        if analysis is None and analysis_id is None:
            analysis = self.review_store.latest_analysis(case_id)
            analysis_id = analysis.analysis_id if analysis is not None else None
        assessment = self.rule_engine.assess(facts)
        assessment.conflicts.extend(self._model_conflicts(facts, analysis))
        if assessment.conflicts and assessment.status.value == "complete":
            assessment.status = AssessmentStatus.INCOMPLETE
        return self.review_store.append_review(
            case_id=case_id,
            expected_revision=expected_revision,
            facts=facts,
            assessment=assessment,
            analysis_id=analysis_id,
            review_state=review_state,
        )
