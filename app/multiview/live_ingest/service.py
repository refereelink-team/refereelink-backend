from __future__ import annotations

import time
from pathlib import Path

from app.multiview.live_ingest.config import (
    DEFAULT_OUTPUT_ROOT,
    LiveIngestConfig,
    try_load_live_ingest_config,
)
from app.multiview.live_ingest.coordinator import LiveSliceCoordinator
from app.multiview.live_ingest.store import LiveMultiviewCaseStore, SegmentIndex
from app.multiview.models import MultiviewCase


class LiveMultiviewService:
    """Read local ring health and turn a healthy three-way snapshot into a case."""

    def __init__(
        self,
        config: LiveIngestConfig | None,
        *,
        config_error: str | None = None,
        segment_index: SegmentIndex | None = None,
        case_store: LiveMultiviewCaseStore | None = None,
        coordinator: LiveSliceCoordinator | None = None,
    ) -> None:
        self.config = config
        self.config_error = config_error
        database_path = (
            config.database_path if config is not None else DEFAULT_OUTPUT_ROOT / "live.sqlite3"
        )
        self.segment_index = segment_index or SegmentIndex(database_path)
        self.case_store = case_store or LiveMultiviewCaseStore(database_path)
        self.coordinator = coordinator or (
            LiveSliceCoordinator(config, self.segment_index, self.case_store)
            if config is not None
            else None
        )

    @classmethod
    def from_environment(cls, path: str | Path | None = None) -> LiveMultiviewService:
        config, config_error = try_load_live_ingest_config(path)
        return cls(config, config_error=config_error)

    def status(self) -> dict:
        if self.config is None:
            return {
                "configured": False,
                "ready": False,
                "trigger_ready": False,
                "buffer_target_s": 30.0,
                "min_buffer_s": 0.0,
                "cameras": [],
                "message": self.config_error or "实时多机位采集未配置",
            }

        now_s = time.time()
        cameras = []
        buffers = []
        for camera in self.config.cameras:
            health = self.segment_index.camera_health(camera.camera_id)
            latest_age_s = (
                max(0.0, now_s - health.latest_segment_at_s)
                if health.latest_segment_at_s is not None
                else None
            )
            online = bool(
                health.online
                and latest_age_s is not None
                and latest_age_s <= self.config.max_segment_age_seconds
            )
            buffer_s = self.segment_index.buffer_seconds(
                camera.camera_id,
                max_gap_s=self.config.segment_seconds * 1.5,
            )
            buffers.append(buffer_s)
            cameras.append(
                {
                    "camera_id": camera.camera_id,
                    "display_name": camera.display_name,
                    "role": camera.role.value,
                    "online": online,
                    "latest_segment_age_s": round(latest_age_s, 3)
                    if latest_age_s is not None
                    else None,
                    "buffer_seconds": round(buffer_s, 3),
                    "reconnect_count": health.reconnect_count,
                    "last_error": health.last_error,
                }
            )
        min_buffer_s = min(buffers, default=0.0)
        ready = all(camera["online"] for camera in cameras)
        trigger_ready = ready and min_buffer_s >= self.config.buffer_seconds
        if trigger_ready:
            message = "三路缓冲已就绪，可进入复核"
        elif not ready:
            message = "等待三路固定机位全部在线"
        else:
            message = "正在积累可回放缓冲"
        return {
            "configured": True,
            "ready": ready,
            "trigger_ready": trigger_ready,
            "buffer_target_s": self.config.buffer_seconds,
            "min_buffer_s": round(min_buffer_s, 3),
            "cameras": cameras,
            "message": message,
        }

    def trigger(self) -> MultiviewCase:
        if self.coordinator is None:
            raise RuntimeError(self.config_error or "实时多机位采集未配置")
        return self.coordinator.trigger()
