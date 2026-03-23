from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable, Deque, Dict, List, Optional, Tuple

from .packet import FramePacket
from .state import FoulEvent, FrameState, OffsideQuery, frame_state_from_packet


class GameStateManager:
    """全局比赛状态管理器。"""

    def __init__(self, max_history: int = 900) -> None:
        self._lock = threading.RLock()
        self._max_history: int = max_history

        self._current_frame: Optional[FrameState] = None
        self._frame_history: Deque[FrameState] = deque(maxlen=max_history)
        self._frame_index: Dict[int, FrameState] = {}

        self._current_packet: Optional[FramePacket] = None
        self._packet_history: Deque[FramePacket] = deque(maxlen=max_history)
        self._packet_index: Dict[int, FramePacket] = {}

        self._foul_events: List[FoulEvent] = []
        self._offside_queries: List[OffsideQuery] = []

        self._frame_callbacks: List[Callable[[FrameState], None]] = []
        self._packet_callbacks: List[Callable[[FramePacket], None]] = []
        self._foul_callbacks: List[Callable[[FoulEvent], None]] = []

        self._total_frames: int = 0
        self._start_time: Optional[float] = None

    def _prune_indexes(self) -> None:
        if self._frame_history:
            earliest_frame_id = self._frame_history[0].frame_id
            obsolete_frame_ids = [fid for fid in self._frame_index.keys() if fid < earliest_frame_id]
            for fid in obsolete_frame_ids:
                self._frame_index.pop(fid, None)
        if self._packet_history:
            earliest_packet_id = self._packet_history[0].frame_id
            obsolete_packet_ids = [fid for fid in self._packet_index.keys() if fid < earliest_packet_id]
            for fid in obsolete_packet_ids:
                self._packet_index.pop(fid, None)

    def update_frame(self, frame: FrameState) -> None:
        with self._lock:
            if self._start_time is None:
                self._start_time = time.time()
            self._current_frame = frame
            self._frame_history.append(frame)
            self._frame_index[frame.frame_id] = frame
            self._total_frames += 1
            self._prune_indexes()
            callbacks = list(self._frame_callbacks)
        for cb in callbacks:
            try:
                cb(frame)
            except Exception:
                pass

    def update_packet(
        self,
        packet: FramePacket,
        frame_state: Optional[FrameState] = None,
    ) -> FrameState:
        resolved_frame = frame_state or frame_state_from_packet(packet)
        with self._lock:
            if self._start_time is None:
                self._start_time = time.time()
            self._current_packet = packet
            self._packet_history.append(packet)
            self._packet_index[packet.frame_id] = packet
            self._current_frame = resolved_frame
            self._frame_history.append(resolved_frame)
            self._frame_index[resolved_frame.frame_id] = resolved_frame
            self._total_frames += 1
            self._prune_indexes()
            packet_callbacks = list(self._packet_callbacks)
            frame_callbacks = list(self._frame_callbacks)
        for cb in packet_callbacks:
            try:
                cb(packet)
            except Exception:
                pass
        for cb in frame_callbacks:
            try:
                cb(resolved_frame)
            except Exception:
                pass
        return resolved_frame

    def add_foul_event(self, event: FoulEvent) -> None:
        with self._lock:
            self._foul_events.append(event)
            callbacks = list(self._foul_callbacks)
        for cb in callbacks:
            try:
                cb(event)
            except Exception:
                pass

    def add_offside_query(self, query: OffsideQuery) -> None:
        with self._lock:
            self._offside_queries.append(query)

    def get_current_frame(self) -> Optional[FrameState]:
        with self._lock:
            return self._current_frame

    def get_current_packet(self) -> Optional[FramePacket]:
        with self._lock:
            return self._current_packet

    def get_recent_frames(self, n: int) -> List[FrameState]:
        if n <= 0:
            return []
        with self._lock:
            return list(self._frame_history)[-n:]

    def get_recent_packets(self, n: int) -> List[FramePacket]:
        if n <= 0:
            return []
        with self._lock:
            return list(self._packet_history)[-n:]

    def get_frame_by_id(self, frame_id: int) -> Optional[FrameState]:
        with self._lock:
            return self._frame_index.get(frame_id)

    def get_packet_by_id(self, frame_id: int) -> Optional[FramePacket]:
        with self._lock:
            return self._packet_index.get(frame_id)

    def get_frames_range(self, start_id: int, end_id: int) -> List[FrameState]:
        if end_id < start_id:
            return []
        with self._lock:
            ids = sorted(self._frame_index.keys())
            return [self._frame_index[fid] for fid in ids if start_id <= fid <= end_id]

    def get_player_trajectory(self, player_id: int, n_frames: int) -> List[Tuple[float, float]]:
        if n_frames <= 0:
            return []
        with self._lock:
            trajectory_rev: List[Tuple[float, float]] = []
            count = 0
            for frame in reversed(self._frame_history):
                player = frame.players.get(player_id)
                if player is not None:
                    trajectory_rev.append((player.field_x, player.field_y))
                    count += 1
                    if count >= n_frames:
                        break
        trajectory_rev.reverse()
        return trajectory_rev

    def get_foul_events(self, last_n: int) -> List[FoulEvent]:
        if last_n <= 0:
            return []
        with self._lock:
            return self._foul_events[-last_n:]

    def get_offside_queries(self) -> List[OffsideQuery]:
        with self._lock:
            return list(self._offside_queries)

    def get_stats(self) -> Dict[str, float]:
        with self._lock:
            total_frames = self._total_frames
            buffered_frames = len(self._frame_history)
            buffered_packets = len(self._packet_history)
            foul_events = len(self._foul_events)
            start_time = self._start_time
        now = time.time()
        runtime_sec = max(0.0, now - start_time) if start_time is not None else 0.0
        estimated_fps = float(total_frames) / runtime_sec if runtime_sec > 0 else 0.0
        return {
            "total_frames": float(total_frames),
            "buffered_frames": float(buffered_frames),
            "buffered_packets": float(buffered_packets),
            "foul_events": float(foul_events),
            "runtime_sec": runtime_sec,
            "estimated_fps": estimated_fps,
        }

    def on_frame(self, callback: Callable[[FrameState], None]) -> None:
        with self._lock:
            self._frame_callbacks.append(callback)

    def on_packet(self, callback: Callable[[FramePacket], None]) -> None:
        with self._lock:
            self._packet_callbacks.append(callback)

    def on_foul(self, callback: Callable[[FoulEvent], None]) -> None:
        with self._lock:
            self._foul_callbacks.append(callback)
