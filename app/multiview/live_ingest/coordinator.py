from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.multiview.live_ingest.config import LiveIngestConfig
from app.multiview.live_ingest.store import (
    IndexedSegment,
    InsufficientLiveBuffer,
    LiveMultiviewCaseStore,
    SegmentIndex,
)
from app.multiview.models import CaptureState, EvidenceView, MultiviewCase, RiskLevel


class LiveCaptureUnavailable(RuntimeError):
    """The buffer is not healthy enough to create a complete case."""


class LiveCaptureFailed(RuntimeError):
    """A capture was registered as failed after its snapshot was frozen."""

    def __init__(self, case_id: str, message: str) -> None:
        super().__init__(message)
        self.case_id = case_id


def probe_media(path: str | Path, *, ffprobe_bin: str = "ffprobe") -> dict[str, Any]:
    """Read only the parameters necessary to protect the stream-copy contract."""
    try:
        result = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                (
                    "format=duration,start_time:"
                    "stream=codec_name,profile,width,height,r_frame_rate,avg_frame_rate,"
                    "has_b_frames,start_time,duration"
                ),
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        payload = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise RuntimeError("无法探测媒体参数") from exc
    streams = [stream for stream in payload.get("streams", []) if stream.get("codec_name")]
    if not streams:
        raise RuntimeError("媒体没有可用视频流")
    stream = streams[0]
    duration_s = _float_or_none(payload.get("format", {}).get("duration"))
    if duration_s is None:
        duration_s = _float_or_none(stream.get("duration"))
    if duration_s is None or duration_s <= 0:
        raise RuntimeError("媒体时长无效")
    return {
        "codec_name": str(stream.get("codec_name", "")).lower(),
        "profile": stream.get("profile"),
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": _frame_rate(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "has_b_frames": int(stream.get("has_b_frames") or 0),
        "duration_s": duration_s,
        "pts_start_s": _float_or_none(stream.get("start_time")),
    }


def validate_h264_media(media: dict[str, Any], config: LiveIngestConfig) -> str | None:
    if media.get("codec_name") != "h264":
        return "视频编码不是 H.264"
    if str(media.get("profile") or "") not in {"Main", "High"}:
        return "视频 H.264 Profile 必须是 Main 或 High"
    if media.get("width") != config.expected_width or media.get("height") != config.expected_height:
        return "视频分辨率与固定机位配置不一致"
    if abs(float(media.get("fps") or 0.0) - config.expected_fps) > 0.05:
        return "视频帧率与固定机位配置不一致"
    if int(media.get("has_b_frames") or 0) != 0:
        return "视频流包含 B 帧，不能作为 GOP 回溯缓冲"
    return None


class LiveSliceCoordinator:
    """Freeze three indexed GOP rings into one atomically registered review case."""

    def __init__(
        self,
        config: LiveIngestConfig,
        segment_index: SegmentIndex,
        case_store: LiveMultiviewCaseStore,
        *,
        ffmpeg_bin: str = "ffmpeg",
        ffprobe_bin: str = "ffprobe",
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.config = config
        self.segment_index = segment_index
        self.case_store = case_store
        self.ffmpeg_bin = ffmpeg_bin
        self.ffprobe_bin = ffprobe_bin
        self.clock = clock
        self._lock = threading.Lock()

    def trigger(self) -> MultiviewCase:
        """Synchronously produce a playable case; never publish a partial case."""
        with self._lock:
            trigger_time_s = self.clock()
            capture_end_s = self._capture_end(trigger_time_s)
            case_id = self._new_case_id(trigger_time_s)
            snapshots: dict[str, list[IndexedSegment]] = {}
            try:
                snapshots = self.segment_index.freeze_window(
                    (camera.camera_id for camera in self.config.cameras),
                    end_time_s=capture_end_s,
                    window_seconds=self.config.buffer_seconds,
                    max_gap_s=self.config.segment_seconds * 1.5,
                )
            except InsufficientLiveBuffer as exc:
                raise LiveCaptureUnavailable("三路缓冲尚不足以生成完整复核片段") from exc

            try:
                case, manifest_path = self._create_case(
                    case_id=case_id,
                    trigger_time_s=trigger_time_s,
                    capture_end_s=capture_end_s,
                    snapshots=snapshots,
                )
                return self.case_store.save(
                    case,
                    capture_state=CaptureState.READY,
                    manifest_path=manifest_path,
                    captured_at_s=trigger_time_s,
                )
            except Exception as exc:
                failed_case = self._failed_case(case_id, trigger_time_s)
                self.case_store.save(
                    failed_case,
                    capture_state=CaptureState.FAILED,
                    error=_safe_error(exc),
                    captured_at_s=trigger_time_s,
                )
                raise LiveCaptureFailed(case_id, "三路片段封装或验证失败") from exc
            finally:
                self.segment_index.release_window(snapshots)

    def _capture_end(self, now_s: float) -> float:
        ends: list[float] = []
        for camera in self.config.cameras:
            health = self.segment_index.camera_health(camera.camera_id)
            latest_end_s = self.segment_index.latest_segment_end(camera.camera_id)
            if (
                not health.online
                or latest_end_s is None
                or now_s - latest_end_s > self.config.max_segment_age_seconds
            ):
                raise LiveCaptureUnavailable("三路摄像机必须全部在线后才能进入复核")
            ends.append(latest_end_s)
        return min(ends)

    def _create_case(
        self,
        *,
        case_id: str,
        trigger_time_s: float,
        capture_end_s: float,
        snapshots: dict[str, list[IndexedSegment]],
    ) -> tuple[MultiviewCase, Path]:
        self.config.cases_root.mkdir(parents=True, exist_ok=True)
        temp_dir = self.config.cases_root / f".{case_id}.tmp"
        final_dir = self.config.cases_root / case_id
        if temp_dir.exists() or final_dir.exists():
            raise RuntimeError("capture case directory already exists")
        temp_dir.mkdir()
        try:
            outputs = self._remux_all(temp_dir, snapshots)
            reference_camera = next(
                (camera for camera in self.config.cameras if camera.role.value == "main"),
                self.config.cameras[0],
            )
            reference_start_s = snapshots[reference_camera.camera_id][0].start_time_s
            event_time_s = max(0.0, capture_end_s - reference_start_s)
            videos = []
            manifest_views = []
            for camera in self.config.cameras:
                output = outputs[camera.camera_id]
                view_start_s = snapshots[camera.camera_id][0].start_time_s
                view_event_time_s = max(0.0, capture_end_s - view_start_s)
                sync_offset_ms = round((view_event_time_s - event_time_s) * 1000)
                videos.append(
                    EvidenceView(
                        camera_id=camera.camera_id,
                        display_name=camera.display_name,
                        role=camera.role,
                        path=str(final_dir / output["filename"]),
                        sync_offset_ms=sync_offset_ms,
                        quality="H.264 GOP 缓冲",
                    )
                )
                manifest_views.append(
                    {
                        "camera_id": camera.camera_id,
                        "output_file": output["filename"],
                        "output_media": output["media"],
                        "slice_start_utc": _utc_iso(view_start_s),
                        "slice_end_utc": _utc_iso(snapshots[camera.camera_id][-1].end_time_s),
                        "event_time_s": round(view_event_time_s, 3),
                        "sync_offset_ms": sync_offset_ms,
                        "segments": [
                            {
                                "start_utc": _utc_iso(segment.start_time_s),
                                "end_utc": _utc_iso(segment.end_time_s),
                                "pts_start_s": segment.pts_start_s,
                                "pts_end_s": segment.pts_end_s,
                            }
                            for segment in snapshots[camera.camera_id]
                        ],
                    }
                )

            manifest = {
                "schema_version": 1,
                "case_id": case_id,
                "triggered_at_utc": _utc_iso(trigger_time_s),
                "capture_end_utc": _utc_iso(capture_end_s),
                "window_target_s": self.config.buffer_seconds,
                "reference_camera_id": reference_camera.camera_id,
                "event_time_s": round(event_time_s, 3),
                "views": manifest_views,
                "tools": {
                    "ffmpeg": self._tool_version(self.ffmpeg_bin),
                    "ffprobe": self._tool_version(self.ffprobe_bin),
                },
            }
            manifest_path = temp_dir / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temp_dir, final_dir)
            return (
                MultiviewCase(
                    case_id=case_id,
                    title="现场多视角复核",
                    match_name="固定机位实时采集",
                    match_clock=datetime.fromtimestamp(trigger_time_s, UTC).strftime("%H:%M:%S"),
                    description="裁判进入复核时冻结的三路 H.264 GOP 回溯片段。",
                    event_time_s=event_time_s,
                    risk_level=RiskLevel.MEDIUM,
                    zone="待人工确认",
                    videos=videos,
                    evidence_notes=[
                        "三路固定机位实时缓冲",
                        f"触发 UTC：{_utc_iso(trigger_time_s)}",
                    ],
                ),
                final_dir / "manifest.json",
            )
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise

    def _remux_all(
        self,
        temp_dir: Path,
        snapshots: dict[str, list[IndexedSegment]],
    ) -> dict[str, dict[str, Any]]:
        outputs: dict[str, dict[str, Any]] = {}
        for camera in self.config.cameras:
            camera_dir = temp_dir / camera.camera_id
            camera_dir.mkdir()
            staged_paths = []
            for index, segment in enumerate(snapshots[camera.camera_id]):
                staged = camera_dir / f"segment-{index:04d}.ts"
                shutil.copy2(segment.path, staged)
                staged_paths.append(staged)
            concat_path = camera_dir / "segments.txt"
            concat_path.write_text(
                "".join(f"file {_concat_quote(path)}\n" for path in staged_paths),
                encoding="utf-8",
            )
            filename = f"{camera.camera_id}.mp4"
            partial_path = temp_dir / f".{filename}.part"
            final_path = temp_dir / filename
            try:
                subprocess.run(
                    [
                        self.ffmpeg_bin,
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-fflags",
                        "+genpts",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        str(concat_path),
                        "-map",
                        "0:v:0",
                        "-an",
                        "-c:v",
                        "copy",
                        "-avoid_negative_ts",
                        "make_zero",
                        "-movflags",
                        "+faststart",
                        "-f",
                        "mp4",
                        str(partial_path),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=90,
                )
            except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError(f"{camera.camera_id} MP4 流复制失败") from exc
            media = probe_media(partial_path, ffprobe_bin=self.ffprobe_bin)
            parameter_error = validate_h264_media(media, self.config)
            if parameter_error:
                raise RuntimeError(f"{camera.camera_id} 输出校验失败：{parameter_error}")
            minimum_duration_s = max(
                self.config.segment_seconds,
                self.config.buffer_seconds - self.config.segment_seconds * 2,
            )
            if float(media["duration_s"]) < minimum_duration_s:
                raise RuntimeError(f"{camera.camera_id} 输出片段时长不足")
            os.replace(partial_path, final_path)
            shutil.rmtree(camera_dir)
            outputs[camera.camera_id] = {"filename": filename, "media": media}
        return outputs

    def _failed_case(self, case_id: str, trigger_time_s: float) -> MultiviewCase:
        return MultiviewCase(
            case_id=case_id,
            title="现场多视角复核捕获失败",
            match_name="固定机位实时采集",
            match_clock=datetime.fromtimestamp(trigger_time_s, UTC).strftime("%H:%M:%S"),
            description="三路片段未能完成封装；不会使用不完整证据。",
            event_time_s=0.0,
            risk_level=RiskLevel.MEDIUM,
            zone="待重新触发",
            videos=[
                EvidenceView(
                    camera_id=camera.camera_id,
                    display_name=camera.display_name,
                    role=camera.role,
                    quality="捕获失败",
                )
                for camera in self.config.cameras
            ],
            evidence_notes=["未提交任何不完整媒体。"],
        )

    @staticmethod
    def _new_case_id(trigger_time_s: float) -> str:
        stamp = datetime.fromtimestamp(trigger_time_s, UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"live-{stamp}-{uuid.uuid4().hex[:8]}"

    @staticmethod
    def _tool_version(binary: str) -> str | None:
        try:
            result = subprocess.run(
                [binary, "-version"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return None
        return result.stdout.splitlines()[0][:200] if result.stdout else None


def _concat_quote(path: Path) -> str:
    return "'" + str(path).replace("'", r"'\''") + "'"


def _frame_rate(raw: Any) -> float:
    if raw is None:
        return 0.0
    value = str(raw)
    if "/" not in value:
        return float(value)
    numerator, denominator = value.split("/", 1)
    denominator_value = float(denominator)
    return float(numerator) / denominator_value if denominator_value else 0.0


def _float_or_none(raw: Any) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _utc_iso(timestamp_s: float) -> str:
    return datetime.fromtimestamp(timestamp_s, UTC).isoformat().replace("+00:00", "Z")


def _safe_error(exc: Exception) -> str:
    return str(exc).replace("\n", " ")[:500] or exc.__class__.__name__
