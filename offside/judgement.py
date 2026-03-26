from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from projection.coords import (
    HALF_PITCH_LENGTH_M,
    HALF_PITCH_WIDTH_M,
    field_meter_center_to_map_pixel,
)

# Metric field geometry (center-origin, meters).
FIELD_RECT = [
    -HALF_PITCH_LENGTH_M,
    -HALF_PITCH_WIDTH_M,
    HALF_PITCH_LENGTH_M,
    HALF_PITCH_WIDTH_M,
]
PENALTY_BOX_DEPTH_M = 16.5
PENALTY_BOX_HALF_WIDTH_M = 40.32 / 2.0
PENALTY_BOX_LEFT = [
    FIELD_RECT[0],
    -PENALTY_BOX_HALF_WIDTH_M,
    FIELD_RECT[0] + PENALTY_BOX_DEPTH_M,
    PENALTY_BOX_HALF_WIDTH_M,
]
PENALTY_BOX_RIGHT = [
    FIELD_RECT[2] - PENALTY_BOX_DEPTH_M,
    -PENALTY_BOX_HALF_WIDTH_M,
    FIELD_RECT[2],
    PENALTY_BOX_HALF_WIDTH_M,
]
GOAL_LINE_L_X = FIELD_RECT[0]
GOAL_LINE_R_X = FIELD_RECT[2]
GOAL_Y_RANGE = [-(7.32 / 2.0), (7.32 / 2.0)]

TEAM_GOAL_SIDE = {
    "RED": "LEFT",
    "BLUE": "RIGHT",
}


def set_team_goal_sides(red_goal_side: str = "LEFT", blue_goal_side: str = "RIGHT") -> None:
    if red_goal_side not in ("LEFT", "RIGHT"):
        raise ValueError("red_goal_side must be 'LEFT' or 'RIGHT'")
    if blue_goal_side not in ("LEFT", "RIGHT"):
        raise ValueError("blue_goal_side must be 'LEFT' or 'RIGHT'")

    global TEAM_GOAL_SIDE
    TEAM_GOAL_SIDE = {
        "RED": red_goal_side,
        "BLUE": blue_goal_side,
    }


def check_judgement(
    wx: float,
    wy: float,
    team: str,
    is_ball: bool = False,
) -> Tuple[str, Tuple[int, int, int]]:
    status_msg = ""
    alert_color = (255, 255, 255)
    padding_m = 0.8

    if (
        wx < FIELD_RECT[0] - padding_m
        or wx > FIELD_RECT[2] + padding_m
        or wy < FIELD_RECT[1] - padding_m
        or wy > FIELD_RECT[3] + padding_m
    ):
        return "OUT OF BOUNDS", (0, 0, 255)

    if is_ball:
        if wx < GOAL_LINE_L_X and GOAL_Y_RANGE[0] <= wy <= GOAL_Y_RANGE[1]:
            return "GOAL!!! (Left)", (0, 255, 0)
        if wx > GOAL_LINE_R_X and GOAL_Y_RANGE[0] <= wy <= GOAL_Y_RANGE[1]:
            return "GOAL!!! (Right)", (0, 255, 0)

    in_left = (
        PENALTY_BOX_LEFT[0] <= wx <= PENALTY_BOX_LEFT[2]
        and PENALTY_BOX_LEFT[1] <= wy <= PENALTY_BOX_LEFT[3]
    )
    in_right = (
        PENALTY_BOX_RIGHT[0] <= wx <= PENALTY_BOX_RIGHT[2]
        and PENALTY_BOX_RIGHT[1] <= wy <= PENALTY_BOX_RIGHT[3]
    )
    if in_left or in_right:
        pass

    return status_msg, alert_color


def compute_offside_line(
    players_map: List[dict],
    defender_team: str,
    direction: str,
) -> Optional[float]:
    def_xs = sorted([p["x"] for p in players_map if p["team"] == defender_team])
    if len(def_xs) < 2:
        return None
    if direction == "LEFT":
        return def_xs[1]
    return def_xs[-2]


def get_attacker_defender_direction(
    players_map: List[dict],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    reds = [p for p in players_map if p["team"] == "RED"]
    blues = [p for p in players_map if p["team"] == "BLUE"]
    if len(reds) < 2 or len(blues) < 2:
        return None, None, None

    max_x = float(FIELD_RECT[2])

    def goal_x_for_side(side: str) -> float:
        return -max_x if side == "LEFT" else max_x

    red_own_goal = goal_x_for_side(TEAM_GOAL_SIDE.get("RED", "LEFT"))
    blue_own_goal = goal_x_for_side(TEAM_GOAL_SIDE.get("BLUE", "RIGHT"))

    red_center_x = float(np.mean([p["x"] for p in reds]))
    blue_center_x = float(np.mean([p["x"] for p in blues]))

    red_forward = abs(red_center_x - red_own_goal)
    blue_forward = abs(blue_center_x - blue_own_goal)

    if red_forward > blue_forward:
        attacker_team, defender_team = "RED", "BLUE"
        direction = "RIGHT" if TEAM_GOAL_SIDE.get("RED", "LEFT") == "LEFT" else "LEFT"
    else:
        attacker_team, defender_team = "BLUE", "RED"
        direction = "RIGHT" if TEAM_GOAL_SIDE.get("BLUE", "RIGHT") == "LEFT" else "LEFT"

    return attacker_team, defender_team, direction


def draw_offside_on_map(
    display_map: np.ndarray,
    players_map: List[dict],
    attacker_team: str,
    defender_team: str,
    direction: str,
) -> None:
    offside_x = compute_offside_line(players_map, defender_team, direction)
    if offside_x is None:
        return

    x_top, y_top = field_meter_center_to_map_pixel(offside_x, FIELD_RECT[1])
    x_bottom, y_bottom = field_meter_center_to_map_pixel(offside_x, FIELD_RECT[3])
    cv2.line(
        display_map,
        (int(x_top), int(y_top)),
        (int(x_bottom), int(y_bottom)),
        (0, 255, 255),
        2,
    )
    cv2.putText(
        display_map,
        "OFFSIDE LINE",
        (int(x_top) + 5, max(15, int(y_top) + 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    is_offside = (lambda x: x < offside_x) if direction == "LEFT" else (lambda x: x > offside_x)
    for p in players_map:
        if p["team"] != attacker_team or not is_offside(p["x"]):
            continue
        px, py = field_meter_center_to_map_pixel(float(p["x"]), float(p["y"]))
        cv2.circle(display_map, (int(px), int(py)), 18, (0, 255, 255), 3)
        cv2.putText(
            display_map,
            "!! OFFSIDE !!",
            (int(px) - 20, int(py) - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
