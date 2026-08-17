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
        """ollama 原生 /api/chat 端点。

        OpenAI 兼容端点（/v1/chat/completions）会忽略顶层 think 参数，
        思考模型会额外产生数百 thinking tokens 拖慢生成；原生端点支持
        "think": false 彻底关闭思考，故统一走原生端点。
        """
        value = base_url.rstrip("/")
        if value.endswith("/api/chat"):
            return value
        if value.endswith("/chat/completions"):
            value = value[: -len("/chat/completions")]
        if value.endswith("/v1"):
            value = value[:-3]
        return f"{value}/api/chat"

    def _native_base(self) -> str:
        value = self.url.rstrip("/")
        if value.endswith("/chat/completions"):
            value = value[: -len("/chat/completions")]
        if value.endswith("/v1"):
            value = value[:-3]
        return value

    def keep_hot(self, timeout_s: float = 2.0) -> None:
        """让模型常驻推理服务内存（keep_alive=-1），避免每次解释都冷启动装载。

        解释路径上用短超时（模型已常驻时立即返回；未装载时由后续正式请求完成装载）；
        服务启动预热时用长超时，等待首次装载完成。
        """
        if not self.url:
            return
        payload = {"model": self.model, "keep_alive": -1, "messages": []}
        request = urllib.request.Request(
            f"{self._native_base()}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                response.read()
        except (OSError, urllib.error.URLError, ValueError):
            pass

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
            "不得新增球员身份、位置、动作、牌级、重启方式或规则。"
            "summary 必须是 1-2 句、不超过 80 字的精简判罚说明，依次包含：犯规方与动作、"
            "判罚依据（引用 canonical_assessment.rules 中的 law 与 rule_id，"
            "如'依据 Law 12（L12-INTENSITY-RECKLESS）'）、重启方式、纪律处罚。"
            "同一结论只表述一次，禁止重复、禁止拆成多句、禁止添加修饰性描述。"
            "只输出一个 JSON 对象，不包含 markdown 或任何其他文字，字段为："
            f"summary（字符串）、restart（只能取 \"{assessment.restart.value}\"）、"
            f"sanction（只能取 \"{assessment.sanction.value}\"）、"
            f"rule_ids（数组，只能从 {rule_ids} 中选取）。"
        )
        content = {
            "confirmed_facts": self._confirmed_facts(record),
            "model_evidence_not_ground_truth": evidence,
            "canonical_assessment": {
                "status": assessment.status.value,
                "restart": assessment.restart.value,
                "sanction": assessment.sanction.value,
                "rules": [
                    {
                        "rule_id": item.rule_id,
                        "law": item.law,
                        "section": item.section,
                        "result": item.result,
                    }
                    for item in assessment.rule_trace
                ],
                "template": assessment.explanation_template,
            },
        }
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(content, ensure_ascii=False)},
            ],
            "stream": False,
            "options": {"temperature": 0.1},
            # 关闭思考模式：gemma 思考版会先输出数百 token 思考内容，把耗时从 ~1s 拖到 ~10s；
            # 仅原生 /api/chat 端点支持该参数
            "think": False,
            # 不传 max_tokens：当前 ollama + gemma 组合下会导致空输出；长度由提示词约束
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
        content = body["message"]["content"].strip()
        # 容错：部分模型会给 JSON 包上 markdown 代码块围栏
        if content.startswith("```"):
            content = content.strip("`")
            content = content.removeprefix("json").strip()
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
        if expected_restart and expected_restart not in summary:
            raise ValueError("LLM summary omitted restart")
        # play_on 的模板措辞为"不作纪律处罚"，不强制出现"不出牌"字样
        if expected_sanction and assessment.restart.value != "play_on" and expected_sanction not in summary:
            raise ValueError("LLM summary omitted sanction")
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
                self.keep_hot()
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
