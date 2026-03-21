from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from .state import BallState, PlayerState


@dataclass
class ObjectTrack:
    """统一的像素空间目标跟踪结果。"""

    track_id: int
    class_id: int
    xyxy: Tuple[int, int, int, int]
    confidence: float
    team: str


@dataclass
class ProjectedObject:
    """统一的球场平面投影结果。"""

    track_id: int
    class_id: int
    xyxy: Tuple[int, int, int, int]
    confidence: float
    map_x: float
    map_y: float
    team: str


@dataclass
class FrameMetrics:
    """逐帧处理指标。"""

    total_ms: float = 0.0
    detect_ms: float = 0.0
    track_ms: float = 0.0
    classify_ms: float = 0.0
    project_ms: float = 0.0
    render_ms: float = 0.0
    persist_ms: float = 0.0
    queue_lag_ms: float = 0.0
    fps: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)


@dataclass
class FramePacket:
    """统一的逐帧中间结果包。"""

    frame_id: int
    timestamp: float
    video_ts: Optional[float] = None
    source_id: str = "default"
    raw_frame: Optional[np.ndarray] = None
    annotated_frame: Optional[np.ndarray] = None
    tracked_objects: List[ObjectTrack] = field(default_factory=list)
    players: Dict[int, PlayerState] = field(default_factory=dict)
    ball: Optional[BallState] = None
    projection_tracklets: List[ProjectedObject] = field(default_factory=list)
    projection_frame: Optional[np.ndarray] = None
    events: List[Any] = field(default_factory=list)
    metrics: FrameMetrics = field(default_factory=FrameMetrics)
    debug_info: Dict[str, Any] = field(default_factory=dict)
