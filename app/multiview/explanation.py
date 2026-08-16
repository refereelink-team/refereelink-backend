from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from app.multiview.models import ExplanationResponse, MultiviewDecision, ReviewRecord


class GuardedExplanationWriter:
    def __init__(
        self,
        url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        self.url = (url if url is not None else os.environ.get("SC_EXPLANATION_LLM_URL", "")).strip()
        self.model = (
            model if model is not None else os.environ.get("SC_EXPLANATION_LLM_MODEL", "Qwen3-4B-Q4_K_M")
        ).strip()
        self.timeout_s = float(
            timeout_s if timeout_s is not None else os.environ.get("SC_EXPLANATION_LLM_TIMEOUT_S", "10")
        )

    @staticmethod
    def _endpoint(base_url: str) -> str:
        value = base_url.rstrip("/")
        if value.endswith("/chat/completions"):
            return value
        if value.endswith("/v1"):
            return f"{value}/chat/completions"
        return f"{value}/v1/chat/completions"

    @staticmethod
    def _confirmed_facts(record: ReviewRecord) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for name, item in record.facts:
            if name == "location":
                if item is not None and item.confirmed:
                    payload[name] = item.model_dump(mode="json")
                continue
            if item.confirmed:
                payload[name] = item.value
        return payload

    def _payload(self, record: ReviewRecord, analysis: MultiviewDecision | None) -> dict[str, Any]:
        assessment = record.assessment
        rule_ids = [item.rule_id for item in assessment.rule_trace]
        evidence = None
        if analysis is not None:
            evidence = {
                "action": analysis.action,
                "severity": analysis.severity,
                "confidence": analysis.confidence,
            }
        system = (
            "你是足球辅助判罚文案整理器。只能复述输入中的已确认事实和确定性规则结果；"
            "不得新增球员身份、位置、动作、牌级、重启方式或规则。输出严格 JSON。"
        )
        content = {
            "confirmed_facts": self._confirmed_facts(record),
            "model_evidence_not_ground_truth": evidence,
            "canonical_assessment": {
                "status": assessment.status.value,
                "restart": assessment.restart.value,
                "sanction": assessment.sanction.value,
                "rule_ids": rule_ids,
                "template": assessment.explanation_template,
            },
        }
        schema = {
            "name": "officiating_explanation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "maxLength": 800},
                    "restart": {"type": "string", "enum": [assessment.restart.value]},
                    "sanction": {"type": "string", "enum": [assessment.sanction.value]},
                    "rule_ids": {
                        "type": "array",
                        "items": {"type": "string", "enum": rule_ids or ["NONE"]},
                    },
                },
                "required": ["summary", "restart", "sanction", "rule_ids"],
                "additionalProperties": False,
            },
        }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_schema", "json_schema": schema},
        }

    def _request(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            self._endpoint(self.url),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            body = json.loads(response.read().decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
        return json.loads(content)

    @staticmethod
    def _validate(output: dict[str, Any], record: ReviewRecord) -> str:
        assessment = record.assessment
        if output.get("restart") != assessment.restart.value:
            raise ValueError("LLM changed restart")
        if output.get("sanction") != assessment.sanction.value:
            raise ValueError("LLM changed sanction")
        allowed_rules = {item.rule_id for item in assessment.rule_trace}
        if not set(output.get("rule_ids", [])).issubset(allowed_rules):
            raise ValueError("LLM introduced unknown rule")
        summary = str(output.get("summary", "")).strip()
        if not summary or len(summary) > 800:
            raise ValueError("LLM summary is empty or too long")

        sanction_words = {
            "yellow_card": "黄牌",
            "red_card": "红牌",
            "none": "不出牌",
        }
        expected_sanction = sanction_words.get(assessment.sanction.value)
        for value in sanction_words.values():
            if value in summary and value != expected_sanction:
                raise ValueError("LLM summary contradicts sanction")

        restart_words = {
            "play_on": "继续比赛",
            "direct_free_kick": "直接任意球",
            "indirect_free_kick": "间接任意球",
            "penalty": "点球",
            "previous_restart": "维持原恢复方式",
        }
        expected_restart = restart_words.get(assessment.restart.value)
        for value in restart_words.values():
            if value in summary and value != expected_restart:
                raise ValueError("LLM summary contradicts restart")
        for unsupported in ("越位", "手球", "优势原则"):
            if unsupported in summary:
                raise ValueError("LLM introduced an unsupported fact")
        return summary

    def explain(
        self,
        record: ReviewRecord,
        analysis: MultiviewDecision | None,
        *,
        use_llm: bool = True,
    ) -> ExplanationResponse:
        rule_ids = [item.rule_id for item in record.assessment.rule_trace]
        fallback_reason = None
        if use_llm and self.url:
            try:
                output = self._request(self._payload(record, analysis))
                summary = self._validate(output, record)
                return ExplanationResponse(
                    case_id=record.case_id,
                    revision=record.revision,
                    source="local_llm",
                    summary=summary,
                    restart=record.assessment.restart,
                    sanction=record.assessment.sanction,
                    rule_ids=rule_ids,
                )
            except (KeyError, TypeError, ValueError, OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
                fallback_reason = f"{type(exc).__name__}: {exc}"
        elif use_llm:
            fallback_reason = "SC_EXPLANATION_LLM_URL is not configured"

        return ExplanationResponse(
            case_id=record.case_id,
            revision=record.revision,
            source="template",
            summary=record.assessment.explanation_template,
            restart=record.assessment.restart,
            sanction=record.assessment.sanction,
            rule_ids=rule_ids,
            fallback_reason=fallback_reason,
        )
