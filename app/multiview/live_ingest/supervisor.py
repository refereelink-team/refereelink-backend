from __future__ import annotations

import logging
import subprocess
import threading
import time
from pathlib import Path

from app.multiview.live_ingest.config import LiveCameraConfig, LiveIngestConfig
from app.multiview.live_ingest.coordinator import probe_media, validate_h264_media
from app.multiview.live_ingest.store import IndexedSegment, SegmentIndex

logger = logging.getLogger(__name__)


class LiveIngestSupervisor:
    """The isolated RTSP/TCP H.264 stream-copy process for all three cameras."""

    def __init__(
        self,
        config: LiveIngestConfig,
        *,
        segment_index: SegmentIndex | None = None,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        poll_seconds: float = 0.25,
    ) -> None:
        self.config = config
        self.segment_index = segment_index or SegmentIndex(config.database_path)
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.poll_seconds = poll_seconds
        self._processes: dict[str, subprocess.Popen[bytes]] = {}
        self._seen_paths: set[Path] = set()
        self._next_restart_at: dict[str, float] = {}

    def run_forever(self, stop_event: threading.Event | None = None) -> None:
        stop_event = stop_event or threading.Event()
        self.config.ring_root.mkdir(parents=True, exist_ok=True)
        try:
            while not stop_event.is_set():
                now_s = time.time()
                for camera in self.config.cameras:
                    self._ensure_camera_process(camera, now_s)
                    self._reconcile_camera(camera)
                self._prune(now_s)
                stop_event.wait(self.poll_seconds)
        finally:
            self.stop()

    def stop(self) -> None:
        for process in self._processes.values():
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 5
        for process in self._processes.values():
            timeout = max(0.0, deadline - time.monotonic())
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
        self._processes.clear()

    def ffmpeg_command(self, camera: LiveCameraConfig) -> list[str]:
        camera_dir = self.config.ring_root / camera.camera_id
        return [
            self.ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-rtsp_transport",
            "tcp",
            "-i",
            camera.rtsp_url,
            "-map",
            "0:v:0",
            "-an",
            "-c:v",
            "copy",
            "-f",
            "segment",
            "-segment_format",
            "mpegts",
            "-segment_time",
            str(self.config.segment_seconds),
            "-reset_timestamps",
            "1",
            "-break_non_keyframes",
            "0",
            str(camera_dir / "segment-%010d.ts"),
        ]

    def _ensure_camera_process(self, camera: LiveCameraConfig, now_s: float) -> None:
        existing = self._processes.get(camera.camera_id)
        if existing is not None and existing.poll() is None:
            return
        if existing is not None:
            self._processes.pop(camera.camera_id, None)
            self.segment_index.set_camera_error(camera.camera_id, "FFmpeg 采集进程已退出，正在重连")
            self.segment_index.increment_reconnect(camera.camera_id)
            self._next_restart_at[camera.camera_id] = now_s + 2.0
        if now_s < self._next_restart_at.get(camera.camera_id, 0.0):
            return

        camera_dir = self.config.ring_root / camera.camera_id
        camera_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._processes[camera.camera_id] = subprocess.Popen(
                self.ffmpeg_command(camera),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            self.segment_index.set_camera_error(camera.camera_id, "无法启动 FFmpeg 采集进程")
            self._next_restart_at[camera.camera_id] = now_s + 5.0

    def _reconcile_camera(self, camera: LiveCameraConfig) -> None:
        camera_dir = self.config.ring_root / camera.camera_id
        if not camera_dir.exists():
            return
        minimum_size = 1024
        minimum_age_s = max(0.1, self.config.segment_seconds * 0.1)
        for path in sorted(camera_dir.glob("*.ts")):
            resolved = path.resolve()
            if resolved in self._seen_paths:
                continue
            try:
                stat = resolved.stat()
            except OSError:
                continue
            if stat.st_size < minimum_size or time.time() - stat.st_mtime < minimum_age_s:
                continue
            try:
                media = probe_media(resolved, ffprobe_bin=self.ffprobe_bin)
                parameter_error = validate_h264_media(media, self.config)
                if parameter_error:
                    self.segment_index.set_camera_error(camera.camera_id, parameter_error)
                    self._seen_paths.add(resolved)
                    continue
                end_time_s = stat.st_mtime
                duration_s = float(media["duration_s"])
                self.segment_index.record_segment(
                    IndexedSegment(
                        camera_id=camera.camera_id,
                        path=resolved,
                        start_time_s=end_time_s - duration_s,
                        end_time_s=end_time_s,
                        duration_s=duration_s,
                        pts_start_s=media.get("pts_start_s"),
                        pts_end_s=(
                            float(media["pts_start_s"]) + duration_s
                            if media.get("pts_start_s") is not None
                            else None
                        ),
                        media=media,
                    )
                )
                self._seen_paths.add(resolved)
            except RuntimeError:
                self.segment_index.set_camera_error(camera.camera_id, "无法读取摄像机 GOP 分段")
                self._seen_paths.add(resolved)

    def _prune(self, now_s: float) -> None:
        retention_s = self.config.buffer_seconds + self.config.segment_seconds * 2
        for path in self.segment_index.prune(older_than_s=now_s - retention_s):
            try:
                path.unlink(missing_ok=True)
                self._seen_paths.discard(path.resolve())
            except OSError:
                logger.warning("Unable to delete expired multiview segment: %s", path.name)
