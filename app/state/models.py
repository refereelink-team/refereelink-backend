from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class HomographyStatus(str, Enum):
    FRESH = "fresh"
    STALE = "stale"
    UNAVAILABLE = "unavailable"


class PlayerRole(str, Enum):
    PLAYER = "player"
    GOALKEEPER = "goalkeeper"
    REFEREE = "referee"


class SourceStatus(str, Enum):
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class PlayerState(BaseModel):
    track_id: int
    role: PlayerRole
    team_id: int
    field_x: Optional[float] = None
    field_y: Optional[float] = None
    confidence: float


class GameEvent(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    event_type: str
    confidence: float
    severity: str
    timestamp: float
    frame_id: int
    field_x: Optional[float] = None
    field_y: Optional[float] = None
    reviewed: bool = False
    foul_details: Optional[dict[str, Any]] = None


class FrameState(BaseModel):
    type: str = "frame_state"
    frame_id: int
    capture_timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000)
    processed_timestamp_ms: float = Field(default_factory=lambda: time.time() * 1000)
    processing_fps: float = 0.0
    homography_status: HomographyStatus = HomographyStatus.UNAVAILABLE
    players: list[PlayerState] = Field(default_factory=list)
    events: list[GameEvent] = Field(default_factory=list)


class MetricsSnapshot(BaseModel):
    type: str = "metrics"
    processing_fps: float = 0.0
    input_fps: float = 0.0
    inference_latency_ms: float = 0.0
    end_to_end_latency_ms: float = 0.0
    dropped_frames: int = 0
    queue_length: int = 0
    player_count: int = 0
    source_status: SourceStatus = SourceStatus.DISCONNECTED
    memory_mb: float = 0.0
    gpu_memory_mb: Optional[float] = None


class PipelineConfig(BaseModel):
    mode: str = "realtime"
    video_source: str = ""
    enable_foul_detection: bool = False
    enable_recording: bool = False
    target_video_path: str = ""
    show_keypoints: bool = True
    show_tracking_boxes: bool = True
    show_2d_projection: bool = True
    foul_confidence_threshold: float = 0.48
    device: str = "cpu"


class PipelineCommand(BaseModel):
    command: str
    params: dict[str, Any] = Field(default_factory=dict)
