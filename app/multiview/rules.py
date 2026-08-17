from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.multiview.geometry import analyze_location
from app.multiview.models import (
    AssessmentStatus,
    ChallengeIntensity,
    DefendsSide,
    EvidenceValue,
    FoulFacts,
    RestartType,
    RuleAssessment,
    RuleTraceEntry,
    SanctionType,
    TacticalImpact,
    TeamLabel,
)

RULESET_VERSION = "IFAB_2026_27"
PHYSICAL_ACTIONS = {
    "tackle",
    "standing tackle",
    "high leg",
    "holding",
    "pushing",
    "elbowing",
    "challenge",
}

# 各规则条目对应的《足球竞赛规则》原文摘译，供复核界面展示以增强权威性
LAW_EXCERPTS = {
    "L12-NO-OFFENCE": "Law 12：未发生犯规，比赛继续。",
    "L12-SIMULATION": "Law 12：试图以假装被犯规（simulation）欺骗裁判员的球员，必须予以警告。",
    "L12-BALL-OUT": "Law 12：比赛停止后发生的犯规，不另行改变恢复方式，但纪律处罚照常执行。",
    "L12-CONTACT-RESTART": "Law 12：以草率、鲁莽或使用过分力量的方式对对方队员犯规，判罚直接任意球；犯规发生在犯规方本方禁区内，判罚点球（Law 14）。",
    "L12-EXCESSIVE-FORCE": "Law 12：使用过分力量或危及对方队员安全的抢截，必须作为严重犯规罚令出场。",
    "L12-DOGSO-PENALTY-ATTEMPT": "Law 12：在本方禁区内以争抢球为目的的犯规破坏明显得分机会，判点球并予以警告。",
    "L12-DOGSO-PENALTY-NO-ATTEMPT": "Law 12：破坏对方明显得分机会的犯规，必须罚令出场；禁区内犯规判点球。",
    "L12-DOGSO-OUTSIDE": "Law 12：在禁区外以犯规破坏对方明显得分机会，必须罚令出场。",
    "L12-SPA-PENALTY-ATTEMPT": "Law 12：在禁区内以争抢球为目的的犯规阻止有威胁进攻，已判点球时不另行警告。",
    "L12-SPA": "Law 12：以犯规阻止或干扰有威胁的进攻，予以警告。",
    "L12-INTENSITY": "Law 12：草率犯规不作纪律处罚；鲁莽犯规予以警告；使用过分力量必须罚令出场。",
    "SCOPE-UNSUPPORTED-ACTION": "该动作尚未纳入本系统首版规则集，需人工裁判判定。",
    "SCOPE-NON-CONTACT": "本系统首版规则集仅覆盖已确认身体接触的犯规场景。",
}


def _law_excerpt(rule_id: str) -> str:
    if rule_id in LAW_EXCERPTS:
        return LAW_EXCERPTS[rule_id]
    # L12-INTENSITY-CARELESS / -RECKLESS 等动态条目共用强度条款摘译
    if rule_id.startswith("L12-INTENSITY-"):
        return LAW_EXCERPTS["L12-INTENSITY"]
    return ""


@dataclass(frozen=True)
class ConfirmedFact:
    value: Any | None
    known: bool


def _confirmed(item: EvidenceValue) -> ConfirmedFact:
    if not item.confirmed or item.value is None:
        return ConfirmedFact(None, False)
    value = item.value.value if hasattr(item.value, "value") else item.value
    if isinstance(value, str) and value.lower() in {"", "unknown"}:
        return ConfirmedFact(None, False)
    return ConfirmedFact(value, True)


def _trace(
    rule_id: str,
    law: str,
    section: str,
    facts_used: list[str],
    result: str,
    priority: int,
    law_excerpt: str = "",
) -> RuleTraceEntry:
    return RuleTraceEntry(
        rule_id=rule_id,
        law=law,
        section=section,
        facts_used=facts_used,
        result=result,
        priority=priority,
        law_excerpt=law_excerpt or _law_excerpt(rule_id),
    )


def _team_label(value: Any | None) -> str:
    return {"home": "HOME", "away": "AWAY"}.get(str(value).lower(), "未知球队")


def _restart_label(value: RestartType) -> str:
    return {
        RestartType.PLAY_ON: "继续比赛",
        RestartType.DIRECT_FREE_KICK: "直接任意球",
        RestartType.INDIRECT_FREE_KICK: "间接任意球",
        RestartType.PENALTY: "点球",
        RestartType.PREVIOUS_RESTART: "维持原恢复方式",
        RestartType.UNKNOWN: "待确认",
    }[value]


def _sanction_label(value: SanctionType) -> str:
    return {
        SanctionType.NONE: "不出牌",
        SanctionType.YELLOW_CARD: "黄牌",
        SanctionType.RED_CARD: "红牌",
        SanctionType.PENDING: "待确认",
    }[value]


class IFABRuleEngine:
    ruleset_version = RULESET_VERSION

    def assess(self, facts: FoulFacts) -> RuleAssessment:
        missing: list[str] = []
        conflicts: list[str] = []
        trace: list[RuleTraceEntry] = []
        geometry = None

        offence = _confirmed(facts.offence_confirmed)
        if not offence.known:
            return self._result(
                status=AssessmentStatus.INCOMPLETE,
                restart=RestartType.UNKNOWN,
                sanction=SanctionType.PENDING,
                missing=["offence_confirmed"],
                facts=facts,
            )
        if not bool(offence.value):
            trace.append(
                _trace(
                    "L12-NO-OFFENCE",
                    "Law 12",
                    "Fouls and Misconduct",
                    ["offence_confirmed"],
                    "人工确认未发生犯规，继续比赛且不作纪律处罚。",
                    100,
                )
            )
            return self._result(
                status=AssessmentStatus.COMPLETE,
                restart=RestartType.PLAY_ON,
                sanction=SanctionType.NONE,
                trace=trace,
                facts=facts,
            )

        action = _confirmed(facts.action)
        ball_in_play = _confirmed(facts.ball_in_play)
        intensity = _confirmed(facts.intensity)
        tactical = _confirmed(facts.tactical_impact)
        contact = _confirmed(facts.contact)
        offender = _confirmed(facts.offender_team)
        victim = _confirmed(facts.victim_team)
        home_side = _confirmed(facts.home_defends_side)

        action_key = str(action.value).strip().lower() if action.known else ""
        if not action.known:
            missing.append("action")

        if offender.known and victim.known and offender.value == victim.value:
            conflicts.append("犯规方与受害方不能是同一球队")
        if contact.known and not bool(contact.value) and _confirmed(facts.contact_region).known:
            conflicts.append("已确认无接触，但同时填写了接触部位")

        if action_key == "dive":
            if not ball_in_play.known:
                missing.append("ball_in_play")
            restart = (
                RestartType.INDIRECT_FREE_KICK
                if ball_in_play.known and bool(ball_in_play.value)
                else RestartType.PREVIOUS_RESTART
                if ball_in_play.known
                else RestartType.UNKNOWN
            )
            sanction = SanctionType.YELLOW_CARD
            trace.append(
                _trace(
                    "L12-SIMULATION",
                    "Law 12",
                    "Cautions for unsporting behaviour",
                    ["offence_confirmed", "action", "ball_in_play"],
                    "模拟行为应予警告；比赛中以间接任意球恢复。",
                    90,
                )
            )
            status = self._status(missing, conflicts)
            return self._result(
                status=status,
                restart=restart,
                sanction=sanction,
                missing=missing,
                conflicts=conflicts,
                trace=trace,
                facts=facts,
            )

        if action.known and action_key not in PHYSICAL_ACTIONS:
            trace.append(
                _trace(
                    "SCOPE-UNSUPPORTED-ACTION",
                    "Law 12",
                    "Implementation scope",
                    ["action"],
                    f"动作 {action.value} 尚未纳入首版确定性规则集。",
                    100,
                )
            )
            return self._result(
                status=AssessmentStatus.UNSUPPORTED,
                restart=RestartType.UNKNOWN,
                sanction=SanctionType.PENDING,
                trace=trace,
                facts=facts,
            )

        if not contact.known:
            missing.append("contact")
        elif not bool(contact.value):
            return self._result(
                status=AssessmentStatus.UNSUPPORTED,
                restart=RestartType.UNKNOWN,
                sanction=SanctionType.PENDING,
                conflicts=conflicts,
                trace=[
                    _trace(
                        "SCOPE-NON-CONTACT",
                        "Law 12",
                        "Implementation scope",
                        ["action", "contact"],
                        "首版规则仅覆盖已确认身体接触的 MVFoul 动作，非接触场景需人工判定。",
                        100,
                    )
                ],
                facts=facts,
            )
        if not ball_in_play.known:
            missing.append("ball_in_play")
        if not intensity.known:
            missing.append("intensity")
        if not tactical.known:
            missing.append("tactical_impact")

        if facts.location is not None and facts.location.confirmed:
            geometry = analyze_location(
                facts.location,
                offender.value if offender.known else TeamLabel.UNKNOWN,
                home_side.value if home_side.known else DefendsSide.UNKNOWN,
            )
        else:
            missing.append("location")

        restart = RestartType.UNKNOWN
        if ball_in_play.known and not bool(ball_in_play.value):
            restart = RestartType.PREVIOUS_RESTART
            trace.append(
                _trace(
                    "L12-BALL-OUT",
                    "Law 12",
                    "Disciplinary action after play has stopped",
                    ["ball_in_play"],
                    "球已不在比赛中，不因该行为另行判任意球或点球；纪律处罚仍可执行。",
                    95,
                )
            )
        elif ball_in_play.known and geometry is not None:
            if not geometry.in_penalty_area:
                restart = RestartType.DIRECT_FREE_KICK
            elif geometry.in_offender_own_penalty_area is True:
                restart = RestartType.PENALTY
            elif geometry.in_offender_own_penalty_area is False:
                restart = RestartType.DIRECT_FREE_KICK
            else:
                if not offender.known:
                    missing.append("offender_team")
                if not home_side.known:
                    missing.append("home_defends_side")
            if restart in {RestartType.DIRECT_FREE_KICK, RestartType.PENALTY}:
                trace.append(
                    _trace(
                        "L12-CONTACT-RESTART",
                        "Law 12 / Law 13 / Law 14",
                        "Direct free kick and penalty-kick location",
                        ["ball_in_play", "contact", "location", "offender_team", "home_defends_side"],
                        f"已确认接触犯规，按犯规位置以{_restart_label(restart)}恢复。",
                        60,
                    )
                )

        intensity_value = str(intensity.value).lower() if intensity.known else ""
        sanction = SanctionType.PENDING
        if intensity_value == ChallengeIntensity.CARELESS.value:
            sanction = SanctionType.NONE
        elif intensity_value == ChallengeIntensity.RECKLESS.value:
            sanction = SanctionType.YELLOW_CARD
        elif intensity_value == ChallengeIntensity.EXCESSIVE_FORCE.value:
            sanction = SanctionType.RED_CARD
            trace.append(
                _trace(
                    "L12-EXCESSIVE-FORCE",
                    "Law 12",
                    "Serious foul play / excessive force",
                    ["intensity"],
                    "使用过分力量危及对方安全，应出示红牌。",
                    100,
                )
            )

        tactical_value = str(tactical.value).lower() if tactical.known else ""
        attempt = _confirmed(facts.attempt_to_play_ball)
        if tactical_value == TacticalImpact.DOGSO.value:
            if restart is RestartType.PENALTY:
                if not attempt.known:
                    missing.append("attempt_to_play_ball")
                    if sanction is not SanctionType.RED_CARD:
                        sanction = SanctionType.PENDING
                elif bool(attempt.value):
                    if sanction is not SanctionType.RED_CARD:
                        sanction = SanctionType.YELLOW_CARD
                    trace.append(
                        _trace(
                            "L12-DOGSO-PENALTY-ATTEMPT",
                            "Law 12",
                            "DOGSO in the penalty area",
                            ["tactical_impact", "location", "attempt_to_play_ball"],
                            "本方禁区内 DOGSO 且尝试争抢球：判点球并警告；严重犯规的红牌优先。",
                            85,
                        )
                    )
                else:
                    sanction = SanctionType.RED_CARD
                    trace.append(
                        _trace(
                            "L12-DOGSO-PENALTY-NO-ATTEMPT",
                            "Law 12",
                            "DOGSO in the penalty area",
                            ["tactical_impact", "location", "attempt_to_play_ball"],
                            "本方禁区内 DOGSO 且未尝试争抢球：判点球并罚令出场。",
                            90,
                        )
                    )
            elif restart is RestartType.DIRECT_FREE_KICK:
                sanction = SanctionType.RED_CARD
                trace.append(
                    _trace(
                        "L12-DOGSO-OUTSIDE",
                        "Law 12",
                        "Denying a goal or obvious goal-scoring opportunity",
                        ["tactical_impact", "location"],
                        "DOGSO 发生在犯规方本方禁区外：直接任意球并罚令出场。",
                        90,
                    )
                )
            elif sanction is not SanctionType.RED_CARD:
                sanction = SanctionType.PENDING
        elif tactical_value == TacticalImpact.SPA.value:
            if restart is RestartType.PENALTY:
                if not attempt.known:
                    missing.append("attempt_to_play_ball")
                    if sanction is not SanctionType.RED_CARD:
                        sanction = SanctionType.PENDING
                elif bool(attempt.value) and sanction is SanctionType.NONE:
                    trace.append(
                        _trace(
                            "L12-SPA-PENALTY-ATTEMPT",
                            "Law 12",
                            "Stopping a promising attack in the penalty area",
                            ["tactical_impact", "location", "attempt_to_play_ball"],
                            "本方禁区内判点球且尝试争抢球时，SPA 不另行警告。",
                            75,
                        )
                    )
                elif sanction is not SanctionType.RED_CARD:
                    sanction = SanctionType.YELLOW_CARD
            elif restart is RestartType.DIRECT_FREE_KICK and sanction is not SanctionType.RED_CARD:
                sanction = SanctionType.YELLOW_CARD
                trace.append(
                    _trace(
                        "L12-SPA",
                        "Law 12",
                        "Stopping a promising attack",
                        ["tactical_impact", "location"],
                        "阻止有希望的进攻，应予警告。",
                        75,
                    )
                )

        if intensity.known and intensity_value != ChallengeIntensity.EXCESSIVE_FORCE.value:
            trace.append(
                _trace(
                    f"L12-INTENSITY-{intensity_value.upper()}",
                    "Law 12",
                    "Careless, reckless or using excessive force",
                    ["intensity"],
                    f"动作强度为 {intensity_value}，纪律处罚下限为{_sanction_label(sanction)}。",
                    50,
                )
            )

        missing = list(dict.fromkeys(missing))
        status = self._status(missing, conflicts)
        return self._result(
            status=status,
            restart=restart,
            sanction=sanction,
            missing=missing,
            conflicts=conflicts,
            trace=sorted(trace, key=lambda item: item.priority, reverse=True),
            geometry=geometry,
            facts=facts,
        )

    @staticmethod
    def _status(missing: list[str], conflicts: list[str]) -> AssessmentStatus:
        return AssessmentStatus.INCOMPLETE if missing or conflicts else AssessmentStatus.COMPLETE

    def _result(
        self,
        *,
        status: AssessmentStatus,
        restart: RestartType,
        sanction: SanctionType,
        facts: FoulFacts,
        missing: list[str] | None = None,
        conflicts: list[str] | None = None,
        trace: list[RuleTraceEntry] | None = None,
        geometry=None,
    ) -> RuleAssessment:
        assessment = RuleAssessment(
            status=status,
            restart=restart,
            sanction=sanction,
            ruleset_version=self.ruleset_version,
            rule_trace=trace or [],
            missing_facts=missing or [],
            conflicts=conflicts or [],
            geometry=geometry,
        )
        assessment.explanation_template = self._explain(facts, assessment)
        return assessment

    @staticmethod
    def _explain(facts: FoulFacts, assessment: RuleAssessment) -> str:
        offender = _confirmed(facts.offender_team)
        action = _confirmed(facts.action)
        team = _team_label(offender.value) if offender.known else "未知球队"
        action_text = str(action.value) if action.known else "待确认动作"
        if assessment.status is AssessmentStatus.UNSUPPORTED:
            return "该场景超出首版规则集覆盖范围，需要人工裁判直接判定。"
        if assessment.restart is RestartType.PLAY_ON:
            return "人工确认未发生犯规，辅助结论为继续比赛且不作纪律处罚。"
        conclusion = f"辅助结论：{_restart_label(assessment.restart)}，{_sanction_label(assessment.sanction)}。"
        context = f"已确认事件：{team} 球员实施 {action_text}。"
        if assessment.status is AssessmentStatus.INCOMPLETE:
            missing = "、".join(assessment.missing_facts) or "存在事实冲突"
            return f"{context}{conclusion} 当前仍为部分结论，待补充：{missing}"
        return f"{context}{conclusion}"
