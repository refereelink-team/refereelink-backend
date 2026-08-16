from __future__ import annotations

import math

from app.multiview.models import DefendsSide, FoulLocation, LocationGeometry, TeamLabel

FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0
PENALTY_AREA_DEPTH_M = 16.5
PENALTY_AREA_HALF_WIDTH_M = 20.16
GOAL_CENTER_Y_M = FIELD_WIDTH_M / 2.0
PENALTY_AREA_MIN_Y_M = GOAL_CENTER_Y_M - PENALTY_AREA_HALF_WIDTH_M
PENALTY_AREA_MAX_Y_M = GOAL_CENTER_Y_M + PENALTY_AREA_HALF_WIDTH_M


def _coerce_team(value: object) -> TeamLabel:
    if isinstance(value, TeamLabel):
        return value
    try:
        return TeamLabel(str(value))
    except ValueError:
        return TeamLabel.UNKNOWN


def _coerce_side(value: object) -> DefendsSide:
    if isinstance(value, DefendsSide):
        return value
    try:
        return DefendsSide(str(value))
    except ValueError:
        return DefendsSide.UNKNOWN


def defending_side_for_team(
    team: object,
    home_defends_side: object,
) -> DefendsSide:
    normalized_team = _coerce_team(team)
    home_side = _coerce_side(home_defends_side)
    if normalized_team is TeamLabel.UNKNOWN or home_side is DefendsSide.UNKNOWN:
        return DefendsSide.UNKNOWN
    if normalized_team is TeamLabel.HOME:
        return home_side
    return DefendsSide.RIGHT if home_side is DefendsSide.LEFT else DefendsSide.LEFT


def analyze_location(
    location: FoulLocation,
    offender_team: object = TeamLabel.UNKNOWN,
    home_defends_side: object = DefendsSide.UNKNOWN,
) -> LocationGeometry:
    x_m = float(location.x_m)
    y_m = float(location.y_m)
    if x_m < FIELD_LENGTH_M / 2.0 - 0.25:
        half = "left"
    elif x_m > FIELD_LENGTH_M / 2.0 + 0.25:
        half = "right"
    else:
        half = "center"

    within_penalty_width = PENALTY_AREA_MIN_Y_M <= y_m <= PENALTY_AREA_MAX_Y_M
    penalty_area_side = None
    if within_penalty_width and x_m <= PENALTY_AREA_DEPTH_M:
        penalty_area_side = "left"
    elif within_penalty_width and x_m >= FIELD_LENGTH_M - PENALTY_AREA_DEPTH_M:
        penalty_area_side = "right"

    defending_side = defending_side_for_team(offender_team, home_defends_side)
    in_own_penalty_area: bool | None
    if defending_side is DefendsSide.UNKNOWN:
        in_own_penalty_area = None
    else:
        in_own_penalty_area = penalty_area_side == defending_side.value

    if penalty_area_side == "left":
        zone = "左侧禁区"
    elif penalty_area_side == "right":
        zone = "右侧禁区"
    elif half == "center":
        zone = "中线区域"
    else:
        zone = "左半场" if half == "left" else "右半场"

    return LocationGeometry(
        half=half,
        zone=zone,
        penalty_area_side=penalty_area_side,
        in_penalty_area=penalty_area_side is not None,
        in_offender_own_penalty_area=in_own_penalty_area,
        distance_to_left_goal_m=round(math.hypot(x_m, y_m - GOAL_CENTER_Y_M), 2),
        distance_to_right_goal_m=round(
            math.hypot(FIELD_LENGTH_M - x_m, y_m - GOAL_CENTER_Y_M), 2
        ),
    )
