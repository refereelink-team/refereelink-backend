from __future__ import annotations

ACTION_LABELS = {
    0: "Tackle",
    1: "Standing Tackle",
    2: "High Leg",
    3: "Holding",
    4: "Pushing",
    5: "Elbowing",
    6: "Challenge",
    7: "Dive",
}

SEVERITY_LABELS = {
    0: "No Offence",
    1: "Offence + No Card",
    2: "Offence + Yellow Card",
    3: "Offence + Red Card",
}

CARDS = {0: "none", 1: "none", 2: "yellow", 3: "red"}
DECISIONS = {0: "no_offence", 1: "offence_no_card", 2: "yellow_card", 3: "red_card"}
DECISIONS_ZH = {
    0: "未检测到犯规",
    1: "建议口头警告/不亮牌",
    2: "建议黄牌",
    3: "建议红牌",
}
