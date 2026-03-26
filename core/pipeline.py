"""
Parallel processing pipeline for real-time tracking and projection.

This module provides a frame buffer and parallel processing capability that allows
tracking and projection to run concurrently, enabling real-time display of both views.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional
from core import FramePacket
import numpy as np


@dataclass
class FrameBuffer:
    """Thread-safe frame buffer for producer-consumer pattern.

    Attributes:
        maxsize: Maximum number of frames to buffer (prevents memory buildup)
        get_timeout: Timeout for getting frames from buffer
    """

    _buffer: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=3))
    get_timeout: float = 1.0
    _closed: bool = False

    def put(self, packet: FramePacket, timeout: float = 1.0) -> None:
        """Add a packet to the buffer (blocking)."""
        if self._closed:
            return
        try:
            self._buffer.put(packet, timeout=timeout)
        except queue.Full:
            # Drop frame if buffer is full (prevents backing up)
            pass

    def get(self, timeout: Optional[float] = None) -> Optional[FramePacket]:
        """Get a packet from the buffer (blocking)."""
        if self._closed and self._buffer.empty():
            return None
        try:
            return self._buffer.get(timeout=timeout or self.get_timeout)
        except queue.Empty:
            return None

    def close(self) -> None:
        """Signal that no more frames will be produced."""
        self._closed = True

    def qsize(self) -> int:
        """Get approximate buffer size."""
        return self._buffer.qsize()

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._buffer.empty()


class ParallelPipeline:
    """Parallel processing pipeline with frame buffer for real-time display.

    This pipeline allows tracking (producer) and projection (consumer) to run
    concurrently, enabling real-time display of both views.
    """

    def __init__(
        self,
        max_buffer_size: int = 3,
        producer_callback: Optional[Callable[[], Iterator[FramePacket]]] = None,
    ):
        self.buffer = FrameBuffer(maxsize=max_buffer_size)
        self._producer_thread: Optional[threading.Thread] = None
        self._consumer_thread: Optional[thread.Thread] = None
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
        """Start the tracking producer in a separate thread."""
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
        """Start the projection consumer in a separate thread."""

        def consumer():
            while not self._stop_event.is_set():
                packet = self.buffer.get(timeout=0.1)
                if packet is None:
                    if self.buffer._closed:
                        break
                    continue
                # Apply projection to the packet
                processed_packet = project_callback(packet)
                with self._lock:
                    self._packets.append(processed_packet)
                    # Keep only last N packets
                    if len(self._packets) > 10:
                        self._packets = self._packets[-10:]

        self._consumer_thread = threading.Thread(target=consumer, daemon=True)
        self._consumer_thread.start()

    def stop(self) -> None:
        """Stop all threads and processing."""
        self._stop_event.set()
        if self._producer_thread:
            self._producer_thread.join(timeout=2.0)
        if self._consumer_thread:
            self._consumer_thread.join(timeout=2.0)

    def get_latest_packet(self) -> Optional[FramePacket]:
        """Get the latest processed packet."""
        with self._lock:
            if self._packets:
                return self._packets[-1]
        return None

    def get_packets(self) -> List[FramePacket]:
        """Get all processed packets."""
        with self._lock:
            return list(self._packets)


def create_parallel_pipeline(
    source: str,
    device: str,
    is_camera: bool = False,
    project_callback: Optional[Callable[[FramePacket], FramePacket]] = None,
) -> ParallelPipeline:
    """Create and start a parallel processing pipeline.

    Args:
        source: Video path or camera index
        device: Device to run inference on (cuda, mps, cpu)
        is_camera: Whether source is a camera
        project_callback: Optional callback to process each frame

    Returns:
        Started ParallelPipeline instance
    """
    from projection.visualization import build_projected_objects, render_projection_frame

    pipeline = ParallelPipeline(max_buffer_size=3)

    # Default projection callback if not provided
    def default_project_callback(packet: FramePacket) -> FramePacket:
        from projection.visualization import load_field_map
        import cv2

        field_img = load_field_map("field_map.png")
        packet.projection_tracklets = build_projected_objects(packet.tracked_objects)
        packet.projection_frame = render_projection_frame(
            tracked_objects=packet.tracked_objects,
            field_img=field_img,
        )
        return packet

    callback = project_callback or default_project_callback

    pipeline.start_producer(source, device, is_camera)
    pipeline.start_consumer(callback)

    return pipeline
