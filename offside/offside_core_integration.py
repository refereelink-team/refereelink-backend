from __future__ import annotations

"""
与 `core` 状态中台的适配层。

作用：
- 将本模块中的 Tracklet 列表，转换为统一的 `FrameState` / `PlayerState` / `BallState`；
- 在检测到越位线时，按约定构造 `OffsideQuery` 并写入 `GameStateManager`。

用法示例（伪代码）：

    from core import GameStateManager
    from offside.offside_core_integration import process_frame_with_core

    state = GameStateManager()
    frame_id = 0
    while True:
        ...
        tracklets = run_detection_and_tracking(...)
        process_frame_with_core(
            frame_id=frame_id,
            video_ts=None,
            source="OFFSIDE_MODULE",
            tracklets=tracklets,
            game_state=state,
        )
        frame_id += 1
"""

import time
from typing import List, Optional, Tuple

from core import (
    BallState,
    FrameState,
    GameStateManager,
    OffsideQuery,
    PlayerState,
    Team,
)

from projection.modeling import ProjectedTracklet
from offside.judgement import (
    compute_offside_line,
    get_attacker_defender_direction,
)


def _map_team_str_to_enum(team: str) -> Team:
    """将本模块内部使用的阵营字符串映射到 `core.Team`。"""
    if team == "RED":
        return Team.HOME
    if team == "BLUE":
        return Team.AWAY
    if team == "WHITE":
        return Team.REFEREE
    return Team.UNKNOWN


def _build_frame_state_from_tracklets(
    frame_id: int,
    video_ts: Optional[float],
    source: str,
    tracklets: List[ProjectedTracklet],
) -> FrameState:
    """根据当前帧的 Tracklet 列表构造标准的 `FrameState`。"""
    timestamp = time.time()

    players: dict[int, PlayerState] = {}
    ball_state: Optional[BallState] = None

    for t in tracklets:
        x1, y1, x2, y2 = t.xyxy
        pixel_x = float((x1 + x2) / 2.0)
        pixel_y = float(y2)

        if t.team == "BALL":
            # 足球走 `BallState`，不占用 player_id。
            ball_state = BallState(
                pixel_x=pixel_x,
                pixel_y=pixel_y,
                field_x=float(t.map_x),
                field_y=float(t.map_y),
                speed=0.0,
                confidence=float(t.confidence),
            )
            continue

        team_enum = _map_team_str_to_enum(t.team)
        players[int(t.track_id)] = PlayerState(
            player_id=int(t.track_id),
            team=team_enum,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            field_x=float(t.map_x),
            field_y=float(t.map_y),
            speed=0.0,
            confidence=float(t.confidence),
            jersey_number=None,
        )

    return FrameState(
        frame_id=frame_id,
        timestamp=timestamp,
        video_ts=video_ts,
        players=players,
        ball=ball_state,
        source=source,
    )


def _maybe_build_offside_query(
    frame_id: int,
    frame_ts: float,
    tracklets: List[ProjectedTracklet],
    query_id: Optional[int] = None,
) -> Optional[OffsideQuery]:
    """
    基于当前帧的 Tracklet，按 `judgement` 逻辑尝试构造一次越位查询。

    - 根据 RED/BLUE 球员中心位置推断进攻方、防守方与进攻方向；
    - 由防守方后场球员计算出越位线 x（地图坐标）；
    - 若存在越位球员，则 result=True 并记录一名代表性的 offending_player_field。
    """
    players_map = [
        {"id": int(t.track_id), "team": t.team, "x": float(t.map_x), "y": float(t.map_y)}
        for t in tracklets
        if t.team in ("RED", "BLUE")
    ]
    attacker_team, defender_team, direction = get_attacker_defender_direction(
        players_map
    )
    if not attacker_team or not defender_team or not direction:
        return None

    offside_x = compute_offside_line(players_map, defender_team, direction)
    if offside_x is None:
        return None

    # 找出一名“最典型”的越位球员，写入 offending_player_field。
    if direction == "LEFT":
        is_offside = lambda x: x < offside_x
        candidates = sorted(
            [p for p in players_map if p["team"] == attacker_team and is_offside(p["x"])],
            key=lambda p: p["x"],
        )
    else:
        is_offside = lambda x: x > offside_x
        candidates = sorted(
            [p for p in players_map if p["team"] == attacker_team and is_offside(p["x"])],
            key=lambda p: p["x"],
            reverse=True,
        )

    offending_player_field: Optional[Tuple[float, float]] = None
    if candidates:
        offending_player_field = (
            float(candidates[0]["x"]),
            float(candidates[0]["y"]),
        )

    has_offside = bool(candidates)
    details = (
        f"auto_offside_query attacker={attacker_team} "
        f"defender={defender_team} direction={direction}"
    )

    return OffsideQuery(
        query_id=int(query_id if query_id is not None else frame_id),
        key_frame_id=frame_id,
        timestamp=frame_ts,
        result=has_offside,
        details=details,
        offending_player_field=offending_player_field,
        offside_line_field_x=float(offside_x),
        offside_line_pixel_x=None,
    )


def process_frame_with_core(
    frame_id: int,
    video_ts: Optional[float],
    source: str,
    tracklets: List[ProjectedTracklet],
    game_state: GameStateManager,
    emit_offside_query: bool = True,
    offside_query_id: Optional[int] = None,
) -> None:
    """
    将一帧 Tracklet 写入 `GameStateManager`：
    - 构造并调用 `update_frame(FrameState)`；
    - 可选地基于当前帧自动生成一次 `OffsideQuery` 并写入。
    """
    frame_state = _build_frame_state_from_tracklets(
        frame_id=frame_id,
        video_ts=video_ts,
        source=source,
        tracklets=tracklets,
    )
    game_state.update_frame(frame_state)

    if not emit_offside_query:
        return

    query = _maybe_build_offside_query(
        frame_id=frame_id,
        frame_ts=frame_state.timestamp,
        tracklets=tracklets,
        query_id=offside_query_id,
    )
    if query is not None:
        game_state.add_offside_query(query)

