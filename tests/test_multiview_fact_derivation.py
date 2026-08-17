from __future__ import annotations

from app.multiview.models import (
    AssessmentStatus,
    EvidenceSource,
    EvidenceValue,
    FoulFacts,
    FoulLocation,
    MultiviewDecision,
    RestartType,
    ReviewState,
    SanctionType,
)
from app.multiview.review_store import MultiviewReviewStore
from app.multiview.service import MultiviewAnalysisService, model_severity_to_intensity


def confirmed(value, source: EvidenceSource = EvidenceSource.HUMAN) -> EvidenceValue:
    return EvidenceValue(value=value, source=source, confirmed=True)


def minimal_facts(action: str = "Tackle", offender: str = "home") -> FoulFacts:
    """Complete reviewer answers without contact or victim fields.

    The simplified review form no longer asks these; the service must
    derive them before running the rule engine.
    """
    return FoulFacts(
        offence_confirmed=confirmed(True),
        action=confirmed(action),
        offender_team=confirmed(offender),
        ball_in_play=confirmed(True),
        intensity=confirmed("reckless"),
        tactical_impact=confirmed("none"),
        location=FoulLocation(x_m=30.0, y_m=34.0),
        home_defends_side=confirmed("left"),
    )


def test_derivation_infers_contact_from_physical_action() -> None:
    derived = MultiviewAnalysisService._derive_missing_facts(minimal_facts())
    assert derived.contact.value is True
    assert derived.contact.source is EvidenceSource.RULE
    assert derived.contact.confirmed is True


def test_derivation_marks_dive_as_no_contact() -> None:
    derived = MultiviewAnalysisService._derive_missing_facts(minimal_facts(action="Dive"))
    assert derived.contact.value is False
    assert derived.contact.source is EvidenceSource.RULE


def test_derivation_infers_victim_team() -> None:
    derived = MultiviewAnalysisService._derive_missing_facts(minimal_facts(offender="home"))
    assert derived.victim_team.value == "away"
    assert derived.victim_team.source is EvidenceSource.RULE


def test_derivation_never_overrides_human_contact() -> None:
    facts = minimal_facts()
    facts.contact = confirmed(False)
    derived = MultiviewAnalysisService._derive_missing_facts(facts)
    assert derived.contact.value is False
    assert derived.contact.source is EvidenceSource.HUMAN


def test_derivation_repairs_stale_same_team_victim() -> None:
    # 表单没有受害方入口，旧记录里的同队值会误触发冲突，必须按犯规方重推导
    facts = minimal_facts(offender="home")
    facts.victim_team = confirmed("home")
    derived = MultiviewAnalysisService._derive_missing_facts(facts)
    assert derived.victim_team.value == "away"
    assert derived.victim_team.source is EvidenceSource.RULE


def test_review_without_contact_completes_via_derivation(tmp_path) -> None:
    service = MultiviewAnalysisService(
        review_store=MultiviewReviewStore(tmp_path / "reviews.sqlite3"),
    )
    case = service.repository.list_cases()[0]
    record = service.update_review(
        case_id=case.case_id,
        expected_revision=0,
        facts=minimal_facts(),
        analysis_id=None,
        review_state=ReviewState.PENDING,
    )
    assert record.assessment.status is AssessmentStatus.COMPLETE
    assert record.assessment.restart is RestartType.DIRECT_FREE_KICK
    assert record.assessment.sanction is SanctionType.YELLOW_CARD
    assert record.facts.contact.confirmed is True


def test_dive_review_uses_simulation_branch_via_derivation(tmp_path) -> None:
    service = MultiviewAnalysisService(
        review_store=MultiviewReviewStore(tmp_path / "reviews.sqlite3"),
    )
    case = service.repository.list_cases()[0]
    record = service.update_review(
        case_id=case.case_id,
        expected_revision=0,
        facts=minimal_facts(action="Dive"),
        analysis_id=None,
        review_state=ReviewState.PENDING,
    )
    assert record.assessment.status is AssessmentStatus.COMPLETE
    assert record.assessment.restart is RestartType.INDIRECT_FREE_KICK
    assert record.assessment.sanction is SanctionType.YELLOW_CARD


def test_human_override_of_model_keeps_complete_with_warning(tmp_path) -> None:
    service = MultiviewAnalysisService(
        review_store=MultiviewReviewStore(tmp_path / "reviews.sqlite3"),
    )
    case = service.repository.list_cases()[0]
    analysis = MultiviewDecision(
        analysis_id="analysis-override-test",
        event_id="EVT-1",
        case_id=case.case_id,
        timestamp=0.0,
        decision="Offence + Yellow Card",
        decision_zh="犯规 + 黄牌",
        action="Pushing",
        severity="Offence + Yellow Card",
        confidence=0.8,
        card="yellow",
        mode="scripted",
    )
    service.review_store.save_analysis(analysis)
    record = service.update_review(
        case_id=case.case_id,
        expected_revision=0,
        facts=minimal_facts(action="Tackle"),
        analysis_id=analysis.analysis_id,
        review_state=ReviewState.PENDING,
    )
    # The human is the final authority: the override stays visible as a
    # warning but never blocks a complete assessment.
    assert record.assessment.status is AssessmentStatus.COMPLETE
    assert any("模型建议" in conflict for conflict in record.assessment.conflicts)


def test_severity_to_intensity_yellow_dominant_is_reckless() -> None:
    candidates = [
        {"label": "Offence + Yellow Card", "confidence": 0.5888},
        {"label": "No Offence", "confidence": 0.219},
        {"label": "Offence + Red Card", "confidence": 0.0994},
    ]
    assert model_severity_to_intensity(candidates, "yellow") == "reckless"


def test_severity_to_intensity_red_mass_escalates_to_excessive_force() -> None:
    candidates = [
        {"label": "Offence + Red Card", "confidence": 0.6},
        {"label": "Offence + Yellow Card", "confidence": 0.3},
        {"label": "Offence + No Card", "confidence": 0.1},
    ]
    assert model_severity_to_intensity(candidates, "red") == "excessive_force"


def test_severity_to_intensity_no_card_is_careless() -> None:
    candidates = [
        {"label": "Offence + No Card", "confidence": 0.7},
        {"label": "Offence + Yellow Card", "confidence": 0.2},
    ]
    assert model_severity_to_intensity(candidates, "none") == "careless"


def test_severity_to_intensity_empty_candidates_falls_back_to_card() -> None:
    assert model_severity_to_intensity([], "yellow") == "reckless"
    assert model_severity_to_intensity([], "red") == "excessive_force"
    assert model_severity_to_intensity([], "none") == "careless"


def test_severity_to_intensity_no_offence_only_falls_back_to_card() -> None:
    candidates = [{"label": "No Offence", "confidence": 0.79}]
    assert model_severity_to_intensity(candidates, "none") == "careless"
