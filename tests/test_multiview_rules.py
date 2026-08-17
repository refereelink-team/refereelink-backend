from __future__ import annotations

import pytest

from app.multiview.models import (
    AssessmentStatus,
    EvidenceSource,
    EvidenceValue,
    FoulFacts,
    FoulLocation,
    RestartType,
    SanctionType,
)
from app.multiview.rules import IFABRuleEngine


def human(value) -> EvidenceValue:
    return EvidenceValue(value=value, source=EvidenceSource.HUMAN, confirmed=True)


def contact_facts(
    *,
    x_m: float,
    intensity: str,
    tactical: str = "none",
    attempt: bool | None = None,
    ball_in_play: bool = True,
) -> FoulFacts:
    return FoulFacts(
        offence_confirmed=human(True),
        action=human("Tackle"),
        offender_team=human("home"),
        victim_team=human("away"),
        ball_in_play=human(ball_in_play),
        contact=human(True),
        contact_region=human("lower_body"),
        intensity=human(intensity),
        attempt_to_play_ball=human(attempt) if attempt is not None else EvidenceValue(),
        tactical_impact=human(tactical),
        location=FoulLocation(x_m=x_m, y_m=34.0),
        home_defends_side=human("left"),
    )


@pytest.mark.parametrize(
    ("facts", "restart", "sanction"),
    [
        (contact_facts(x_m=30, intensity="reckless"), RestartType.DIRECT_FREE_KICK, SanctionType.YELLOW_CARD),
        (contact_facts(x_m=10, intensity="careless"), RestartType.PENALTY, SanctionType.NONE),
        (contact_facts(x_m=30, intensity="careless", tactical="dogso"), RestartType.DIRECT_FREE_KICK, SanctionType.RED_CARD),
        (contact_facts(x_m=10, intensity="careless", tactical="dogso", attempt=True), RestartType.PENALTY, SanctionType.YELLOW_CARD),
        (contact_facts(x_m=10, intensity="careless", tactical="dogso", attempt=False), RestartType.PENALTY, SanctionType.RED_CARD),
        (contact_facts(x_m=10, intensity="excessive_force"), RestartType.PENALTY, SanctionType.RED_CARD),
    ],
)
def test_ifab_contact_foul_truth_table(facts, restart, sanction) -> None:
    assessment = IFABRuleEngine().assess(facts)
    assert assessment.status is AssessmentStatus.COMPLETE
    assert assessment.restart is restart
    assert assessment.sanction is sanction
    assert assessment.rule_trace


def test_simulation_is_indirect_free_kick_and_yellow() -> None:
    facts = FoulFacts(
        offence_confirmed=human(True),
        action=human("Dive"),
        ball_in_play=human(True),
    )
    assessment = IFABRuleEngine().assess(facts)
    assert assessment.status is AssessmentStatus.COMPLETE
    assert assessment.restart is RestartType.INDIRECT_FREE_KICK
    assert assessment.sanction is SanctionType.YELLOW_CARD


def test_rule_trace_entries_carry_law_excerpt() -> None:
    assessment = IFABRuleEngine().assess(contact_facts(x_m=30, intensity="reckless"))
    assert assessment.rule_trace
    for entry in assessment.rule_trace:
        assert entry.law_excerpt, f"{entry.rule_id} 缺少法条摘译"
    intensity_entry = next(
        item for item in assessment.rule_trace if item.rule_id == "L12-INTENSITY-RECKLESS"
    )
    assert "鲁莽犯规予以警告" in intensity_entry.law_excerpt


def test_no_foul_is_play_on() -> None:
    assessment = IFABRuleEngine().assess(FoulFacts(offence_confirmed=human(False)))
    assert assessment.status is AssessmentStatus.COMPLETE
    assert assessment.restart is RestartType.PLAY_ON
    assert assessment.sanction is SanctionType.NONE


def test_ball_out_keeps_previous_restart_but_allows_discipline() -> None:
    assessment = IFABRuleEngine().assess(
        contact_facts(x_m=30, intensity="reckless", ball_in_play=False)
    )
    assert assessment.restart is RestartType.PREVIOUS_RESTART
    assert assessment.sanction is SanctionType.YELLOW_CARD


def test_missing_direction_does_not_guess_penalty() -> None:
    facts = contact_facts(x_m=10, intensity="careless")
    facts.home_defends_side = EvidenceValue()
    assessment = IFABRuleEngine().assess(facts)
    assert assessment.status is AssessmentStatus.INCOMPLETE
    assert assessment.restart is RestartType.UNKNOWN
    assert "home_defends_side" in assessment.missing_facts


def test_missing_location_returns_partial_discipline_only() -> None:
    facts = contact_facts(x_m=30, intensity="reckless")
    facts.location = None
    assessment = IFABRuleEngine().assess(facts)
    assert assessment.status is AssessmentStatus.INCOMPLETE
    assert assessment.restart is RestartType.UNKNOWN
    assert assessment.sanction is SanctionType.YELLOW_CARD
    assert "location" in assessment.missing_facts
