"""
Parallel processing pipeline for real-time tracking and projection.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional

from .packet import FramePacket


@dataclass
class FrameBuffer:
    """Thread-safe frame buffer for producer-consumer pattern."""

    maxsize: int = 3
    get_timeout: float = 1.0
    _buffer: queue.Queue = field(init=False)
    _closed: bool = False

    def __post_init__(self) -> None:
        self._buffer = queue.Queue(maxsize=self.maxsize)

    def put(self, packet: FramePacket, timeout: float = 1.0) -> None:
        if self._closed:
            return
        try:
            self._buffer.put(packet, timeout=timeout)
        except queue.Full:
            pass

    def get(self, timeout: Optional[float] = None) -> Optional[FramePacket]:
        if self._closed and self._buffer.empty():
            return None
        try:
            return self._buffer.get(timeout=timeout or self.get_timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed = True

    def qsize(self) -> int:
        return self._buffer.qsize()

    def is_empty(self) -> bool:
        return self._buffer.empty()


class ParallelPipeline:
    """Parallel processing pipeline with frame buffer for real-time display."""

    def __init__(
        self,
        max_buffer_size: int = 3,
        producer_callback: Optional[Callable[[], Iterator[FramePacket]]] = None,
    ):
        self.buffer = FrameBuffer(maxsize=max_buffer_size)
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._producer_callback = producer_callback
        self._packets: List[FramePacket] = []
        self._lock = threading.Lock()

    def start_producer(
        self,
        source: str,
        device: str,
        is_camera: bool = False,
    ) -> None:
        from tracking.main import run_player_team_classification_packets

        def producer():
            try:
                for packet in run_player_team_classification_packets(
                    source_video_path=source,
                    device=device,
                    is_camera=is_camera,
                ):
                    if self._stop_event.is_set():
                        break
                    self.buffer.put(packet, timeout=0.5)
            except Exception as e:
                print(f"Producer error: {e}")
            finally:
                self.buffer.close()

        self._producer_thread = threading.Thread(target=producer, daemon=True)
        self._producer_thread.start()

    def start_consumer(
        self,
        project_callback: Callable[[FramePacket], FramePacket],
    ) -> None:
        def consumer():
            while not self._stop_event.is_set():
                packet = self.buffer.get(timeout=0.1)
                if packet is None:
                    if self.buffer._closed:
                        break
                    continue
                processed_packet = project_callback(packet)
                with self._lock:
                    self._packets.append(processed_packet)
                    if len(self._packets) > 10:
                        self._packets = self._packets[-10:]

        self._consumer_thread = threading.Thread(target=consumer, daemon=True)
        self._consumer_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._producer_thread:
            self._producer_thread.join(timeout=2.0)
        if self._consumer_thread:
            self._consumer_thread.join(timeout=2.0)

    def get_latest_packet(self) -> Optional[FramePacket]:
        with self._lock:
            if self._packets:
                return self._packets[-1]
        return None

    def get_packets(self) -> List[FramePacket]:
        with self._lock:
            return list(self._packets)


def create_parallel_pipeline(
    source: str,
    device: str,
    is_camera: bool = False,
    project_callback: Optional[Callable[[FramePacket], FramePacket]] = None,
    field_map_path: str = "field_map.png",
    calib_backend: str = "nbjw",
    dynamic: bool = False,
    recalib_interval: int = 10,
    calibration: str = "",
    debug: bool = False,
    use_prev_homography: bool = True,
) -> ParallelPipeline:
    """Create and start a parallel processing pipeline."""
    from projection.sn_projection_backend import create_projection_engine
    from projection.visualization import (
        build_projected_objects,
        load_field_map,
        render_projection_frame,
    )

    pipeline = ParallelPipeline(max_buffer_size=3)

    field_img = load_field_map(field_map_path)
    engine = create_projection_engine(
        calib_backend=calib_backend,
        dynamic=dynamic or calib_backend in {"nbjw", "pnl"},
        recalib_interval=recalib_interval,
        calibration_path=calibration,
        field_path=field_map_path,
        debug=debug,
        use_prev_homography=use_prev_homography,
    )

    def _sync_players(packet: FramePacket) -> None:
        if not packet.players or not packet.projection_tracklets:
            return
        projected_by_id = {
            int(t.track_id): t
            for t in packet.projection_tracklets
            if t.team != "BALL"
        }
        for player_id, player_state in packet.players.items():
            projected = projected_by_id.get(int(player_id))
            if projected is None:
                continue
            player_state.field_x = float(projected.map_x)
            player_state.field_y = float(projected.map_y)

    def default_project_callback(packet: FramePacket) -> FramePacket:
        h_adapter = engine.update(packet.raw_frame)
        packet.projection_tracklets = build_projected_objects(
            packet.tracked_objects,
            homography=h_adapter,
        )
        _sync_players(packet)
        packet.projection_frame = render_projection_frame(
            tracked_objects=packet.tracked_objects,
            field_img=field_img,
            homography=h_adapter,
        )
        return packet

    callback = project_callback or default_project_callback
    pipeline.start_producer(source, device, is_camera)
    pipeline.start_consumer(callback)
    return pipeline
