# -*- coding: utf-8 -*-
"""
越位与出界判定逻辑（从 v_vars_final 移植）。
输入源改为追踪后的 Tracklets：每帧传入带 track_id、地图坐标、阵营的球员与球。
"""
from typing import List, Tuple, Optional
import cv2
import numpy as np


# --- 场地几何常量（与 v_vars_final 一致，可按战术板比例修改）---
FIELD_RECT = [0, 0, 800, 533]  # [x_min, y_min, x_max, y_max]
PENALTY_BOX_LEFT = [0, 100, 132, 433]
PENALTY_BOX_RIGHT = [668, 100, 800, 433]
GOAL_LINE_L_X = 0
GOAL_LINE_R_X = 800
GOAL_Y_RANGE = [220, 313]

# 默认球门方位配置：RED 守左门，BLUE 守右门。
# 不调用 set_team_goal_sides 时使用该默认值。
TEAM_GOAL_SIDE = {
    "RED": "LEFT",
    "BLUE": "RIGHT",
}


def set_team_goal_sides(red_goal_side: str = "LEFT", blue_goal_side: str = "RIGHT") -> None:
    """
    配置红/蓝两队球门所在边（LEFT / RIGHT）。

    示例：
        from offside.judgement import set_team_goal_sides
        set_team_goal_sides(red_goal_side="RIGHT", blue_goal_side="LEFT")
    """
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
    wx: float, wy: float, team: str, is_ball: bool = False
) -> Tuple[str, Tuple[int, int, int]]:
    """
    几何判罚引擎：出界、门线、禁区。
    wx, wy: 映射后的战术板坐标
    team: "RED" | "BLUE" | "BALL"
    is_ball: 是否为球（用于门线技术判定）
    返回 (状态文案, BGR 颜色)
    """
    status_msg = ""
    alert_color = (255, 255, 255)
    padding = 5

    # 1. 出界
    if (
        wx < FIELD_RECT[0] - padding
        or wx > FIELD_RECT[2] + padding
        or wy < FIELD_RECT[1] - padding
        or wy > FIELD_RECT[3] + padding
    ):
        return "OUT OF BOUNDS", (0, 0, 255)

    # 2. 门线技术（仅球）
    if is_ball:
        if wx < GOAL_LINE_L_X and GOAL_Y_RANGE[0] <= wy <= GOAL_Y_RANGE[1]:
            return "GOAL!!! (Left)", (0, 255, 0)
        if wx > GOAL_LINE_R_X and GOAL_Y_RANGE[0] <= wy <= GOAL_Y_RANGE[1]:
            return "GOAL!!! (Right)", (0, 255, 0)

    # 3. 禁区（可扩展为画圈等）
    in_left = (
        PENALTY_BOX_LEFT[0] <= wx <= PENALTY_BOX_LEFT[2]
        and PENALTY_BOX_LEFT[1] <= wy <= PENALTY_BOX_LEFT[3]
    )
    in_right = (
        PENALTY_BOX_RIGHT[0] <= wx <= PENALTY_BOX_RIGHT[2]
        and PENALTY_BOX_RIGHT[1] <= wy <= PENALTY_BOX_RIGHT[3]
    )
    if in_left or in_right:
        pass  # 仅后台记录或按需返回 "IN BOX"

    return "", alert_color


def compute_offside_line(
    players_map: List[dict],
    defender_team: str,
    direction: str,
) -> Optional[float]:
    """
    根据防守方球员地图坐标计算越位线 x 坐标（排除门将：取倒数第二人）。
    players_map: 列表，每项含 'team', 'x', 'y'
    defender_team: "RED" | "BLUE"
    direction: "LEFT" | "RIGHT" 表示进攻方向
    返回 offside_x 或 None
    """
    def_xs = sorted(
        [p["x"] for p in players_map if p["team"] == defender_team]
    )
    if len(def_xs) < 2:
        return None
    if direction == "LEFT":
        return def_xs[1]  # 排除最左门将
    return def_xs[-2]  # 排除最右门将


def get_attacker_defender_direction(
    players_map: List[dict],
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    根据双方“离自家球门的远近”判定进攻方、防守方与进攻方向。

    思路：
    - 先根据 TEAM_GOAL_SIDE 确定每队自家球门所在门线（LEFT=0, RIGHT=FIELD_RECT[2]）；
    - 计算各自球员中心 x 坐标到自家门线的平均距离（越远说明整体越压上）；
    - 距离自家门线更远的一方视为进攻方，direction 为其进攻方向（LEFT/RIGHT）。
    """
    reds = [p for p in players_map if p["team"] == "RED"]
    blues = [p for p in players_map if p["team"] == "BLUE"]
    if len(reds) < 2 or len(blues) < 2:
        return None, None, None

    max_x = float(FIELD_RECT[2])

    def goal_x_for_side(side: str) -> float:
        return 0.0 if side == "LEFT" else max_x

    # 自家球门位置
    red_own_goal = goal_x_for_side(TEAM_GOAL_SIDE.get("RED", "LEFT"))
    blue_own_goal = goal_x_for_side(TEAM_GOAL_SIDE.get("BLUE", "RIGHT"))

    # 球员中心 x
    red_center_x = float(np.mean([p["x"] for p in reds]))
    blue_center_x = float(np.mean([p["x"] for p in blues]))

    # 离自家门线的“前压距离”
    red_forward = abs(red_center_x - red_own_goal)
    blue_forward = abs(blue_center_x - blue_own_goal)

    if red_forward > blue_forward:
        attacker_team, defender_team = "RED", "BLUE"
        # 红队进攻方向：远离自家门线那一侧
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
    """在战术板图上绘制越位线并标出越位球员。原地修改 display_map。"""
    offside_x = compute_offside_line(players_map, defender_team, direction)
    if offside_x is None:
        return
    h = display_map.shape[0]
    cv2.line(
        display_map,
        (int(offside_x), 0),
        (int(offside_x), h),
        (0, 255, 255),
        2,
    )
    cv2.putText(
        display_map,
        "OFFSIDE LINE",
        (int(offside_x) + 5, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    is_offside = (lambda x: x < offside_x) if direction == "LEFT" else (lambda x: x > offside_x)
    for p in players_map:
        if p["team"] == attacker_team and is_offside(p["x"]):
            cv2.circle(
                display_map,
                (int(p["x"]), int(p["y"])),
                18,
                (0, 255, 255),
                3,
            )
            cv2.putText(
                display_map,
                "!! OFFSIDE !!",
                (int(p["x"]) - 20, int(p["y"]) - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
