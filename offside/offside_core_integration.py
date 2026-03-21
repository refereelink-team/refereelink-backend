from __future__ import annotations

"""
与 `core` 状态中台的 packet-first 适配层。

作用：
- 将本模块中的投影结果转换为统一的 `FramePacket` / `ProjectedObject`；
- 通过 `GameStateManager.update_packet(...)` 维护 packet 与 frame 快照；
- 在检测到越位线时，按约定构造 `OffsideQuery` 并写入 `GameStateManager`。

用法示例（伪代码）：

    from core import GameStateManager
    from offside.offside_core_integration import process_frame_with_core

    state = GameStateManager()
    frame_id = 0
    while True:
        ...
        projected_objects = build_projected_objects(...)
        process_frame_with_core(
            frame_id=frame_id,
            video_ts=None,
            source="OFFSIDE_MODULE",
            tracklets=projected_objects,
            game_state=state,
        )
        frame_id += 1
"""

import time
from typing import Iterable, Optional, Tuple

from core import FramePacket, GameStateManager, ObjectTrack, OffsideQuery, ProjectedObject

from offside.judgement import compute_offside_line, get_attacker_defender_direction


ProjectedTracklet = ProjectedObject


def _build_packet_from_tracklets(
    frame_id: int,
    video_ts: Optional[float],
    source: str,
    tracklets: Iterable[ProjectedTracklet],
) -> FramePacket:
    """根据当前帧的投影结果构造标准 `FramePacket`。"""
    timestamp = time.time()
    projected_objects = [
        ProjectedObject(
            track_id=int(t.track_id),
            class_id=int(t.class_id),
            xyxy=tuple(int(v) for v in t.xyxy),
            confidence=float(t.confidence),
            map_x=float(t.map_x),
            map_y=float(t.map_y),
            team=t.team,
        )
        for t in tracklets
    ]
    tracked_objects = [
        ObjectTrack(
            track_id=obj.track_id,
            class_id=obj.class_id,
            xyxy=obj.xyxy,
            confidence=obj.confidence,
            team=obj.team,
        )
        for obj in projected_objects
    ]
    return FramePacket(
        frame_id=frame_id,
        timestamp=timestamp,
        video_ts=video_ts,
        source_id=source,
        tracked_objects=tracked_objects,
        projection_tracklets=projected_objects,
        debug_info={"offside_projection_count": len(projected_objects)},
    )


def _maybe_build_offside_query(
    frame_id: int,
    frame_ts: float,
    tracklets: Iterable[ProjectedTracklet],
    query_id: Optional[int] = None,
) -> Optional[OffsideQuery]:
    """
    基于当前帧的投影结果，按 `judgement` 逻辑尝试构造一次越位查询。

    - 根据 RED/BLUE 球员中心位置推断进攻方、防守方与进攻方向；
    - 由防守方后场球员计算出越位线 x（地图坐标）；
    - 若存在越位球员，则 result=True 并记录一名代表性的 offending_player_field。
    """
    players_map = [
        {"id": int(t.track_id), "team": t.team, "x": float(t.map_x), "y": float(t.map_y)}
        for t in tracklets
        if t.team in ("RED", "BLUE")
    ]
    attacker_team, defender_team, direction = get_attacker_defender_direction(players_map)
    if not attacker_team or not defender_team or not direction:
        return None

    offside_x = compute_offside_line(players_map, defender_team, direction)
    if offside_x is None:
        return None

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
    tracklets: list[ProjectedTracklet],
    game_state: GameStateManager,
    emit_offside_query: bool = True,
    offside_query_id: Optional[int] = None,
) -> FramePacket:
    """
    将一帧投影结果写入 `GameStateManager`：
    - 构造并调用 `update_packet(FramePacket)`；
    - 可选地基于当前帧自动生成一次 `OffsideQuery` 并写入。
    """
    packet = _build_packet_from_tracklets(
        frame_id=frame_id,
        video_ts=video_ts,
        source=source,
        tracklets=tracklets,
    )
    game_state.update_packet(packet)

    if not emit_offside_query:
        return packet

    query = _maybe_build_offside_query(
        frame_id=frame_id,
        frame_ts=packet.timestamp,
        tracklets=tracklets,
        query_id=offside_query_id,
    )
    if query is not None:
        game_state.add_offside_query(query)
    return packet
