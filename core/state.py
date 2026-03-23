from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from .packet import FramePacket


class Team(Enum):
    """队伍枚举。"""

    HOME = "HOME"
    AWAY = "AWAY"
    REFEREE = "REFEREE"
    UNKNOWN = "UNKNOWN"


@dataclass
class PlayerState:
    """单个球员在某一帧上的完整状态信息。"""

    player_id: int
    team: Team
    pixel_x: float
    pixel_y: float
    field_x: float = 0.0
    field_y: float = 0.0
    speed: float = 0.0
    confidence: float = 0.0
    jersey_number: Optional[int] = None


@dataclass
class BallState:
    """球在某一帧的状态。"""

    pixel_x: float
    pixel_y: float
    field_x: float = 0.0
    field_y: float = 0.0
    speed: float = 0.0
    confidence: float = 0.0


@dataclass
class FrameState:
    """单帧的稳定业务状态快照。"""

    frame_id: int
    timestamp: float
    video_ts: Optional[float]
    players: Dict[int, PlayerState] = field(default_factory=dict)
    ball: Optional[BallState] = None
    source: str = "default"


@dataclass
class FoulEvent:
    event_id: int
    frame_id: int
    timestamp: float
    foul_type: str
    severity: str
    location_field: Tuple[float, float]
    involved_players: List[int]
    camera_source: str
    offending_player_field: Optional[Tuple[float, float]] = None


@dataclass
class OffsideQuery:
    query_id: int
    key_frame_id: int
    timestamp: float
    result: Optional[bool] = None
    details: str = ""
    offending_player_field: Optional[Tuple[float, float]] = None
    offside_line_field_x: Optional[float] = None
    offside_line_pixel_x: Optional[float] = None


_TEAM_BY_LABEL = {
    "RED": Team.HOME,
    "BLUE": Team.AWAY,
    "WHITE": Team.REFEREE,
    "BALL": Team.UNKNOWN,
}


def frame_state_from_packet(packet: FramePacket) -> FrameState:
    """从统一逐帧结果包中抽取稳定状态快照。"""

    if packet.players:
        players = dict(packet.players)
    else:
        players = {}
        for track in packet.tracked_objects:
            if track.team == "BALL":
                continue
            x1, y1, x2, y2 = track.xyxy
            players[int(track.track_id)] = PlayerState(
                player_id=int(track.track_id),
                team=_TEAM_BY_LABEL.get(track.team, Team.UNKNOWN),
                pixel_x=float((x1 + x2) / 2.0),
                pixel_y=float(y2),
                confidence=float(track.confidence),
            )

    ball = packet.ball
    if ball is None:
        for track in packet.tracked_objects:
            if track.team != "BALL":
                continue
            x1, y1, x2, y2 = track.xyxy
            ball = BallState(
                pixel_x=float((x1 + x2) / 2.0),
                pixel_y=float(y2),
                confidence=float(track.confidence),
            )
            break

    return FrameState(
        frame_id=packet.frame_id,
        timestamp=packet.timestamp,
        video_ts=packet.video_ts,
        players=players,
        ball=ball,
        source=packet.source_id,
    )
