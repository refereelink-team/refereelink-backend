from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from app.constants.paths import REPO_ROOT_DIR
from app.multiview.models import CameraRole

DEFAULT_OUTPUT_ROOT = REPO_ROOT_DIR / "var" / "multiview"
_CAMERA_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{1,63}$")


class LiveIngestConfigError(ValueError):
    """Raised for configuration errors safe to expose through local status."""


@dataclass(frozen=True)
class LiveCameraConfig:
    camera_id: str
    display_name: str
    role: CameraRole
    rtsp_url: str


@dataclass(frozen=True)
class LiveIngestConfig:
    cameras: tuple[LiveCameraConfig, ...]
    output_root: Path
    buffer_seconds: float = 30.0
    segment_seconds: float = 1.0
    expected_width: int = 1920
    expected_height: int = 1080
    expected_fps: float = 30.0
    max_segment_age_seconds: float = 5.0

    @property
    def database_path(self) -> Path:
        return self.output_root / "live.sqlite3"

    @property
    def ring_root(self) -> Path:
        return self.output_root / "ring"

    @property
    def cases_root(self) -> Path:
        return self.output_root / "cases"


def config_path_from_environment() -> Path | None:
    raw_path = os.environ.get("SC_MULTIVIEW_LIVE_CONFIG")
    return Path(raw_path).expanduser() if raw_path else None


def load_live_ingest_config(path: str | Path | None = None) -> LiveIngestConfig:
    config_path = Path(path).expanduser() if path is not None else config_path_from_environment()
    if config_path is None:
        raise LiveIngestConfigError("未配置 SC_MULTIVIEW_LIVE_CONFIG")
    if not config_path.is_file():
        raise LiveIngestConfigError("实时多机位配置文件不存在")

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LiveIngestConfigError("实时多机位配置文件无法读取") from exc
    if not isinstance(payload, dict):
        raise LiveIngestConfigError("实时多机位配置必须是 JSON 对象")

    cameras_payload = payload.get("cameras")
    if not isinstance(cameras_payload, list) or len(cameras_payload) != 3:
        raise LiveIngestConfigError("实时多机位配置必须包含恰好 3 台摄像机")
    cameras = tuple(_parse_camera(item) for item in cameras_payload)
    camera_ids = [camera.camera_id for camera in cameras]
    if len(set(camera_ids)) != len(camera_ids):
        raise LiveIngestConfigError("摄像机 ID 必须唯一")

    output_raw = os.environ.get("SC_MULTIVIEW_LIVE_ROOT") or payload.get("output_root")
    output_root = Path(output_raw).expanduser() if output_raw else DEFAULT_OUTPUT_ROOT
    if not output_root.is_absolute():
        output_root = (config_path.parent / output_root).resolve()

    return LiveIngestConfig(
        cameras=cameras,
        output_root=output_root,
        buffer_seconds=_positive_float(payload.get("buffer_seconds", 30), "buffer_seconds"),
        segment_seconds=_positive_float(payload.get("segment_seconds", 1), "segment_seconds"),
        expected_width=_positive_int(payload.get("expected_width", 1920), "expected_width"),
        expected_height=_positive_int(payload.get("expected_height", 1080), "expected_height"),
        expected_fps=_positive_float(payload.get("expected_fps", 30), "expected_fps"),
        max_segment_age_seconds=_positive_float(
            payload.get("max_segment_age_seconds", 5),
            "max_segment_age_seconds",
        ),
    )


def try_load_live_ingest_config(
    path: str | Path | None = None,
) -> tuple[LiveIngestConfig | None, str | None]:
    try:
        return load_live_ingest_config(path), None
    except LiveIngestConfigError as exc:
        return None, str(exc)


def _parse_camera(raw: Any) -> LiveCameraConfig:
    if not isinstance(raw, dict):
        raise LiveIngestConfigError("摄像机配置必须是对象")
    camera_id = str(raw.get("camera_id", "")).strip()
    if not _CAMERA_ID_PATTERN.fullmatch(camera_id):
        raise LiveIngestConfigError("摄像机 ID 只能使用小写字母、数字、连字符和下划线")
    display_name = str(raw.get("display_name", "")).strip()
    if not display_name:
        raise LiveIngestConfigError("摄像机必须有显示名称")
    try:
        role = CameraRole(str(raw.get("role", CameraRole.OTHER.value)))
    except ValueError as exc:
        raise LiveIngestConfigError("摄像机角色无效") from exc
    rtsp_url = str(raw.get("rtsp_url", "")).strip()
    parsed = urlparse(rtsp_url)
    if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
        raise LiveIngestConfigError("摄像机必须使用有效的 RTSP 地址")
    return LiveCameraConfig(
        camera_id=camera_id,
        display_name=display_name,
        role=role,
        rtsp_url=rtsp_url,
    )


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LiveIngestConfigError(f"{name} 必须是正数") from exc
    if parsed <= 0:
        raise LiveIngestConfigError(f"{name} 必须是正数")
    return parsed


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LiveIngestConfigError(f"{name} 必须是正整数") from exc
    if parsed <= 0:
        raise LiveIngestConfigError(f"{name} 必须是正整数")
    return parsed
