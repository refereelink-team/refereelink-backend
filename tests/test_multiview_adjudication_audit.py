from __future__ import annotations

import time
from pathlib import Path

from app.multiview.models import (
    AssessmentStatus,
    EvidenceSource,
    EvidenceValue,
    FoulFacts,
    FoulLocation,
    ReviewState,
)
from app.multiview.review_store import MultiviewReviewStore
from app.multiview.rules import IFABRuleEngine


def human(value) -> EvidenceValue:
    return EvidenceValue(value=value, source=EvidenceSource.HUMAN, confirmed=True)


def audited_facts() -> FoulFacts:
    return FoulFacts(
        offence_confirmed=human(True),
        action=human("Tackle"),
        offender_team=human("home"),
        victim_team=human("away"),
        ball_in_play=human(True),
        contact=human(True),
        contact_region=human("lower_body"),
        intensity=human("reckless"),
        tactical_impact=human("none"),
        location=FoulLocation(x_m=30.0, y_m=34.0),
        home_defends_side=human("left"),
    )


def p95(values: list[float]) -> float:
    return sorted(values)[max(0, int(len(values) * 0.95) - 1)]


def test_rule_engine_and_sqlite_append_meet_latency_gates(tmp_path) -> None:
    engine = IFABRuleEngine()
    facts = audited_facts()
    rule_durations: list[float] = []
    for _ in range(200):
        started = time.perf_counter()
        assessment = engine.assess(facts)
        rule_durations.append(time.perf_counter() - started)

    store = MultiviewReviewStore(tmp_path / "reviews.sqlite3")
    save_durations: list[float] = []
    for revision in range(20):
        started = time.perf_counter()
        store.append_review(
            case_id="audit-case",
            expected_revision=revision,
            facts=facts,
            assessment=assessment,
            analysis_id=None,
            review_state=ReviewState.REVIEWED,
        )
        save_durations.append(time.perf_counter() - started)

    assert p95(rule_durations) < 0.010
    assert p95(save_durations) < 0.050
    assert [item.revision for item in store.review_history("audit-case")] == list(range(20, 0, -1))


def test_incomplete_assessment_never_claims_penalty_without_direction() -> None:
    facts = audited_facts()
    facts.location = FoulLocation(x_m=10.0, y_m=34.0)
    facts.home_defends_side = EvidenceValue()
    assessment = IFABRuleEngine().assess(facts)

    assert assessment.status is AssessmentStatus.INCOMPLETE
    assert assessment.restart.value == "unknown"
    assert "home_defends_side" in assessment.missing_facts
    assert "点球" not in assessment.explanation_template


def test_frontend_source_does_not_restore_hardcoded_pitch_players() -> None:
    page = Path("web/src/pages/MultiviewReviewPage.tsx").read_text(encoding="utf-8")
    pitch = Path("web/src/components/multiview/FoulLocationPitch.tsx").read_text(encoding="utf-8")

    assert "className=\"home\"" not in page
    assert "className=\"away\"" not in page
    assert "className=\"motion\"" not in page
    assert "FoulLocationPitch" in page
    assert 'aria-label="点击或拖动以设置犯规位置"' in pitch
