from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FieldIngestSettings:
    auth_token: str | None
    root: Path
    srt_bind_host: str | None
    srt_advertise_host: str | None
    srt_port: int
    ffmpeg_bin: str
    ffprobe_bin: str
    max_artifact_bytes: int = 8 * 1024 * 1024
    max_telemetry_items: int = 256
    max_joiner_samples: int = 512

    @classmethod
    def from_env(cls) -> "FieldIngestSettings":
        root = Path(os.getenv("REFEREELINK_FIELD_INGEST_ROOT", "/tmp/refereelink-field-ingest"))
        return cls(
            auth_token=os.getenv("REFEREELINK_FIELD_TOKEN") or None,
            root=root,
            srt_bind_host=os.getenv("REFEREELINK_SRT_BIND_HOST") or None,
            srt_advertise_host=os.getenv("REFEREELINK_SRT_ADVERTISE_HOST") or None,
            srt_port=int(os.getenv("REFEREELINK_SRT_PORT", "10000")),
            ffmpeg_bin=os.getenv("REFEREELINK_FFMPEG_BIN", "ffmpeg"),
            ffprobe_bin=os.getenv("REFEREELINK_FFPROBE_BIN", "ffprobe"),
        )
