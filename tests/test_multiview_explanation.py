from __future__ import annotations

import json

from app.multiview.explanation import GuardedExplanationWriter
from app.multiview.models import (
    AssessmentStatus,
    FoulFacts,
    RestartType,
    ReviewRecord,
    RuleAssessment,
    RuleTraceEntry,
    SanctionType,
)


def record() -> ReviewRecord:
    return ReviewRecord(
        case_id="case-1",
        revision=2,
        facts=FoulFacts(),
        assessment=RuleAssessment(
            status=AssessmentStatus.COMPLETE,
            restart=RestartType.DIRECT_FREE_KICK,
            sanction=SanctionType.YELLOW_CARD,
            rule_trace=[
                RuleTraceEntry(
                    rule_id="L12-RECKLESS",
                    law="Law 12",
                    section="Reckless challenge",
                    result="黄牌",
                )
            ],
            explanation_template="模板解释",
        ),
        created_at="2026-08-16T00:00:00+00:00",
        updated_at="2026-08-16T00:00:00+00:00",
    )


def test_explanation_uses_template_without_configured_llm() -> None:
    result = GuardedExplanationWriter(url="").explain(record(), None, use_llm=True)
    assert result.source == "template"
    assert result.summary == "模板解释"
    assert "not configured" in (result.fallback_reason or "")


def test_explanation_payload_carries_rule_details_for_citation() -> None:
    writer = GuardedExplanationWriter(url="http://127.0.0.1:8080")
    payload = writer._payload(record(), None)
    content = json.loads(payload["messages"][1]["content"])
    rules = content["canonical_assessment"]["rules"]
    assert rules[0]["rule_id"] == "L12-RECKLESS"
    assert rules[0]["law"] == "Law 12"
    # max_tokens 在当前 ollama + gemma 组合下会导致空输出，不得下发
    assert "max_tokens" not in payload
    assert "ctx_length" not in payload
    # 思考模式会产生数百 token 冗余输出拖慢生成，必须关闭
    assert payload["think"] is False
    assert payload["stream"] is False


def test_explanation_targets_native_chat_endpoint() -> None:
    # OpenAI 兼容端点忽略 think 参数，必须路由到支持 think 的原生端点
    assert (
        GuardedExplanationWriter._endpoint("http://127.0.0.1:11434/v1")
        == "http://127.0.0.1:11434/api/chat"
    )
    assert (
        GuardedExplanationWriter._endpoint("http://127.0.0.1:11434/v1/chat/completions")
        == "http://127.0.0.1:11434/api/chat"
    )
    assert (
        GuardedExplanationWriter._endpoint("http://127.0.0.1:11434")
        == "http://127.0.0.1:11434/api/chat"
    )


def test_explanation_accepts_only_matching_llm_conclusion(monkeypatch) -> None:
    writer = GuardedExplanationWriter(url="http://127.0.0.1:8080")
    monkeypatch.setattr(
        writer,
        "_request",
        lambda payload: {
            "summary": "依据已确认事实，应判直接任意球并出示黄牌。",
            "restart": "direct_free_kick",
            "sanction": "yellow_card",
            "rule_ids": ["L12-RECKLESS"],
        },
    )
    result = writer.explain(record(), None)
    assert result.source == "local_llm"
    assert result.fallback_reason is None


def test_explanation_rejects_llm_attempt_to_change_card(monkeypatch) -> None:
    writer = GuardedExplanationWriter(url="http://127.0.0.1:8080")
    monkeypatch.setattr(
        writer,
        "_request",
        lambda payload: {
            "summary": "应判直接任意球并出示红牌。",
            "restart": "direct_free_kick",
            "sanction": "red_card",
            "rule_ids": ["L12-RECKLESS"],
        },
    )
    result = writer.explain(record(), None)
    assert result.source == "template"
    assert result.summary == "模板解释"
    assert "changed sanction" in (result.fallback_reason or "")


def test_explanation_rejects_llm_summary_missing_restart(monkeypatch) -> None:
    writer = GuardedExplanationWriter(url="http://127.0.0.1:8080")
    monkeypatch.setattr(
        writer,
        "_request",
        lambda payload: {
            "summary": "已确认犯规，出示黄牌。",
            "restart": "direct_free_kick",
            "sanction": "yellow_card",
            "rule_ids": ["L12-RECKLESS"],
        },
    )
    result = writer.explain(record(), None)
    assert result.source == "template"
    assert "omitted restart" in (result.fallback_reason or "")


def test_explanation_rejects_llm_summary_missing_sanction(monkeypatch) -> None:
    writer = GuardedExplanationWriter(url="http://127.0.0.1:8080")
    monkeypatch.setattr(
        writer,
        "_request",
        lambda payload: {
            "summary": "已确认犯规，判给直接任意球。",
            "restart": "direct_free_kick",
            "sanction": "yellow_card",
            "rule_ids": ["L12-RECKLESS"],
        },
    )
    result = writer.explain(record(), None)
    assert result.source == "template"
    assert "omitted sanction" in (result.fallback_reason or "")
