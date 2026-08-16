from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

from app.multiview.inference import FoulInferenceError, FoulInferenceService, model_prerequisites
from app.multiview.localization import event_prior_window
from app.multiview.models import MultiviewAnalyzeResponse, MultiviewDecision
from app.multiview.repository import MultiviewCaseRepository
from app.multiview.review_store import MultiviewReviewStore, new_analysis_id


class MultiviewAnalysisService:
    def __init__(
        self,
        repository: MultiviewCaseRepository | None = None,
        review_store: MultiviewReviewStore | None = None,
    ) -> None:
        self.repository = repository or MultiviewCaseRepository()
        self.review_store = review_store or MultiviewReviewStore()
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
