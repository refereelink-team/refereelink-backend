from __future__ import annotations

import pytest

from app.multiview.geometry import analyze_location, defending_side_for_team
from app.multiview.models import (
    AssessmentStatus,
    DefendsSide,
    FoulFacts,
    FoulLocation,
    MultiviewDecision,
    ReviewState,
    RuleAssessment,
    TeamLabel,
)
from app.multiview.review_store import MultiviewReviewStore, ReviewRevisionConflict


def test_pitch_geometry_recognizes_penalty_areas_and_team_direction() -> None:
    left_box = FoulLocation(x_m=11.0, y_m=34.0)
    home_geometry = analyze_location(left_box, TeamLabel.HOME, DefendsSide.LEFT)
    away_geometry = analyze_location(left_box, TeamLabel.AWAY, DefendsSide.LEFT)

    assert home_geometry.zone == "左侧禁区"
    assert home_geometry.in_penalty_area is True
    assert home_geometry.in_offender_own_penalty_area is True
    assert away_geometry.in_offender_own_penalty_area is False
    assert defending_side_for_team(TeamLabel.AWAY, DefendsSide.LEFT) is DefendsSide.RIGHT


def test_pitch_geometry_does_not_guess_own_penalty_area_without_direction() -> None:
    geometry = analyze_location(
        FoulLocation(x_m=94.0, y_m=34.0),
        TeamLabel.HOME,
        DefendsSide.UNKNOWN,
    )
    assert geometry.in_penalty_area is True
    assert geometry.penalty_area_side == "right"
    assert geometry.in_offender_own_penalty_area is None


def test_review_store_is_append_only_and_detects_revision_conflicts(tmp_path) -> None:
    store = MultiviewReviewStore(tmp_path / "reviews.sqlite3")
    facts = FoulFacts(location=FoulLocation(x_m=45.0, y_m=30.0))
    assessment = RuleAssessment(status=AssessmentStatus.INCOMPLETE)

    first = store.append_review(
        case_id="case-1",
        expected_revision=0,
        facts=facts,
        assessment=assessment,
        analysis_id=None,
        review_state=ReviewState.PENDING,
    )
    second = store.append_review(
        case_id="case-1",
        expected_revision=1,
        facts=facts,
        assessment=assessment,
        analysis_id=None,
        review_state=ReviewState.REVIEWED,
    )

    assert first.revision == 1
    assert second.revision == 2
    assert store.latest_review("case-1") == second
    assert [item.revision for item in store.review_history("case-1")] == [2, 1]

    with pytest.raises(ReviewRevisionConflict) as conflict:
        store.append_review(
            case_id="case-1",
            expected_revision=1,
            facts=facts,
            assessment=assessment,
            analysis_id=None,
            review_state=ReviewState.ARCHIVED,
        )
    assert conflict.value.current == 2


def test_review_store_persists_analysis_snapshot(tmp_path) -> None:
    store = MultiviewReviewStore(tmp_path / "reviews.sqlite3")
    decision = MultiviewDecision(
        analysis_id="analysis-1",
        event_id="event-1",
        case_id="case-1",
        timestamp=3.0,
        decision="Yellow Card",
        decision_zh="黄牌",
        action="Tackle",
        severity="Offence + Yellow Card",
        confidence=0.8,
        card="yellow",
        mode="scripted",
    )

    store.save_analysis(decision)

    assert store.get_analysis("analysis-1") == decision
    assert store.latest_analysis("case-1") == decision
