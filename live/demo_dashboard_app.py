"""Local MP4 multiview demo dashboard (preprocess-only, no live YOLO).

Shared playback clock uses main camera as reference + per-camera offset_seconds.
2D pitch uses quiet interactive 4+4 calibration on the main camera (same UX as offside review).
"""

from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import cv2
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config.pitch import SoccerPitchConfiguration

logger = logging.getLogger("demo_dashboard")

IO_LOCK = threading.RLock()
REPO_ROOT = Path(__file__).resolve().parents[1]
FFMPEG_BIN = (
    os.environ.get("FFMPEG_BIN")
    or shutil.which("ffmpeg")
    or "/root/autodl-tmp/envs/sc-demo/bin/ffmpeg"
)
PITCH = SoccerPitchConfiguration()
# Runtime artifacts (calib / foul ROI). Override with DEMO_DASHBOARD_DIR.
DASHBOARD_DIR = Path(
    os.environ.get(
        "DEMO_DASHBOARD_DIR",
        str(REPO_ROOT / "demo_outputs" / "dashboard"),
    )
)


def _resolve_path(value: str | Path) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p.resolve()
    return (REPO_ROOT / p).resolve()


class CamSlot:
    def __init__(self, key: str, cfg: dict[str, Any]) -> None:
        self.key = key
        self.camera_id = str(cfg.get("camera_id", key))
        self.role = str(cfg.get("role", "secondary"))
        self.video = _resolve_path(cfg["video"])
        self.metadata = _resolve_path(cfg["metadata"])
        self.offset_seconds = float(cfg.get("offset_seconds", 0.0))
        self.frames: dict[int, dict[str, Any]] = {}
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.duration: float = 0.0
        self.status: str = "unloaded"
        self.error: Optional[str] = None
        self._lock = threading.RLock()

    def load(self) -> None:
        if not self.video.exists():
            self.status = "error"
            self.error = f"video missing: {self.video}"
            return
        if not self.metadata.exists():
            self.status = "error"
            self.error = f"metadata missing: {self.metadata}"
            return
        self.frames = {}
        with self.metadata.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                fi = int(row["frame_index"])
                # Normalize offside-style rows into dashboard fields.
                if "players" in row and "detections" not in row:
                    players = row.get("players") or []
                    row = {
                        **row,
                        "detections": len(players),
                        "track_ids": [int(p.get("track_id", -1)) for p in players],
                    }
                self.frames[fi] = row
        cap = cv2.VideoCapture(str(self.video))
        if not cap.isOpened():
            self.status = "error"
            self.error = f"cannot open {self.video}"
            return
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or len(self.frames) or 0)
        if self.frames:
            self.total_frames = max(self.total_frames, max(self.frames))
        self.duration = self.total_frames / self.fps if self.fps else 0.0
        cap.release()
        self.status = "loaded"
        self.error = None

    def frame_at_main_time(self, main_t: float) -> tuple[str, Optional[int], Optional[dict]]:
        local_t = main_t - self.offset_seconds
        if local_t < -1e-3:
            return "waiting", None, None
        if local_t > self.duration + 1e-3:
            return "ended", None, None
        idx = int(round(local_t * self.fps)) + 1
        idx = max(1, min(self.total_frames, idx))
        meta = self.frames.get(idx)
        return "ok", idx, meta

    def _ffmpeg_jpeg(self, frame_index: int) -> bytes:
        t = max(0.0, (frame_index - 1) / max(self.fps, 1e-6))
        cmd = [
            FFMPEG_BIN,
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{t:.6f}",
            "-i",
            str(self.video),
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-q:v",
            "5",
            "pipe:1",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=8)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"{self.camera_id} ffmpeg failed: {exc}") from exc
        if proc.returncode != 0 or not proc.stdout:
            err = (proc.stderr or b"").decode("utf-8", errors="ignore")[:200]
            raise HTTPException(404, f"{self.camera_id} frame {frame_index} unreadable: {err}")
        return proc.stdout

    def jpeg_at(self, frame_index: int) -> bytes:
        with self._lock:
            return self._ffmpeg_jpeg(frame_index)


class DashboardState:
    def __init__(self) -> None:
        self.config_path: Path | None = None
        self.cams: dict[str, CamSlot] = {}
        self.main_key: str = "main"
        self.layout_main: str = "main"
        self.layout_side1: str = "side1"
        self.layout_side2: str = "side2"
        self.events: list[dict[str, Any]] = []
        self.gpu_name: str = "unknown"
        self.cuda: bool = False
        self.model_name: str = "yolo11s.pt (preprocess only)"
        self.ready: bool = False
        self.message: str = "init"
        self.H: Optional[np.ndarray] = None
        self.H_status: str = "unavailable"
        self.proj_tracks: dict[int, dict[str, Any]] = {}
        self.proj_fps: float = 30.0
        self.proj_total: int = 0
        self.proj_label: str = "主机位固定标定投影"
        self.pitch_draw = {"out_w": 640, "out_h": 360, "pad": 28}
        self.virtual_cameras: list[str] = []
        self.calib_path: Path = DASHBOARD_DIR / "manual_homography.json"
        self.image_pts: list[list[float]] = []
        self.pitch_pts: list[list[float]] = []
        self.calib_pairs: list[dict[str, Any]] = []
        self.calib_confirmed: bool = False
        self.foul_roi_path: Path = DASHBOARD_DIR / "foul_roi.json"
        self.foul_roi: Optional[list[float]] = None  # [x1,y1,x2,y2] image coords
        self.foul_event_time: float = 4.5
        self.foul_involved_ids: list[int] = [1, 4]


STATE = DashboardState()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _setup_phase() -> str:
    if STATE.calib_confirmed and STATE.H is not None:
        return "done"
    if len(STATE.image_pts) < 4:
        return "image"
    if len(STATE.pitch_pts) < 4:
        return "pitch"
    return "confirm"


def recompute_H() -> None:
    if len(STATE.calib_pairs) < 4:
        STATE.H = None
        STATE.H_status = "unavailable"
        return
    src = np.array([p["image_xy"] for p in STATE.calib_pairs], dtype=np.float32)
    dst = np.array([p["pitch_xy"] for p in STATE.calib_pairs], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)
    STATE.H = H
    STATE.H_status = "fixed" if H is not None else "unavailable"


def _sync_pairs_from_buffers() -> None:
    n = min(len(STATE.image_pts), len(STATE.pitch_pts))
    STATE.calib_pairs = [
        {"image_xy": STATE.image_pts[i], "pitch_xy": STATE.pitch_pts[i], "index": i + 1}
        for i in range(n)
    ]
    if n >= 4:
        recompute_H()
    else:
        STATE.H = None
        STATE.H_status = "unavailable"


def _persist_calib() -> None:
    STATE.calib_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "image_pts": STATE.image_pts,
        "pitch_pts": STATE.pitch_pts,
        "pairs": STATE.calib_pairs,
        "confirmed": STATE.calib_confirmed,
        "homography_status": STATE.H_status,
        "updated_at": _utcnow(),
    }
    STATE.calib_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_persisted_calib(*, require_confirm: bool) -> None:
    STATE.image_pts, STATE.pitch_pts, STATE.calib_pairs = [], [], []
    STATE.calib_confirmed = False
    STATE.H, STATE.H_status = None, "unavailable"
    if not STATE.calib_path.exists():
        return
    try:
        data = json.loads(STATE.calib_path.read_text(encoding="utf-8"))
        confirmed = bool(data.get("confirmed"))
        STATE.image_pts = list(data.get("image_pts") or [])
        STATE.pitch_pts = list(data.get("pitch_pts") or [])
        STATE.calib_confirmed = confirmed
        _sync_pairs_from_buffers()
        # If require_confirm and not confirmed, keep points for operator to finish,
        # but do not treat as done.
        if require_confirm and not confirmed:
            STATE.calib_confirmed = False
    except Exception as exc:  # noqa: BLE001
        logger.warning("load dashboard calib failed: %s", exc)


def _project_players(row: dict[str, Any]) -> list[tuple[float, float, int]]:
    if STATE.H is None:
        return []
    pts, meta = [], []
    for p in row.get("players") or []:
        xyxy = p.get("xyxy")
        if not xyxy or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = xyxy
        pts.append([(x1 + x2) / 2.0, float(y2)])
        meta.append(int(p.get("track_id", -1)))
    if not pts:
        return []
    arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
    try:
        world = cv2.perspectiveTransform(arr, STATE.H).reshape(-1, 2)
    except cv2.error:
        return []
    return [(float(x), float(y), tid) for (x, y), tid in zip(world, meta)]


def draw_pitch_2d(
    players_xy: list[tuple[float, float, int]],
    calib_pitch_pts: Optional[list[tuple[float, float]]] = None,
    out_w: int = 640,
    out_h: int = 360,
    pad: int = 28,
) -> np.ndarray:
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = (34, 110, 45)
    scale_x = (out_w - 2 * pad) / float(PITCH.length)
    scale_y = (out_h - 2 * pad) / float(PITCH.width)

    def to_px(x: float, y: float) -> tuple[int, int]:
        return int(pad + x * scale_x), int(pad + y * scale_y)

    cv2.rectangle(canvas, to_px(0, 0), to_px(PITCH.length, PITCH.width), (230, 230, 230), 2)
    mid = PITCH.length / 2
    cv2.line(canvas, to_px(mid, 0), to_px(mid, PITCH.width), (230, 230, 230), 2)
    cx, cy = to_px(mid, PITCH.width / 2)
    r = int(PITCH.centre_circle_radius * scale_x)
    cv2.circle(canvas, (cx, cy), max(2, r), (230, 230, 230), 2)
    pb, pw = PITCH.penalty_box_length, PITCH.penalty_box_width
    y0, y1 = (PITCH.width - pw) / 2, (PITCH.width + pw) / 2
    cv2.rectangle(canvas, to_px(0, y0), to_px(pb, y1), (230, 230, 230), 2)
    cv2.rectangle(canvas, to_px(PITCH.length - pb, y0), to_px(PITCH.length, y1), (230, 230, 230), 2)

    if calib_pitch_pts and not STATE.calib_confirmed:
        for i, (x, y) in enumerate(calib_pitch_pts):
            px, py = to_px(x, y)
            cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)
            cv2.putText(canvas, str(i + 1), (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    for x, y, tid in players_xy:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if x < -200 or x > PITCH.length + 200 or y < -200 or y > PITCH.width + 200:
            continue
        px, py = to_px(float(x), float(y))
        cv2.circle(canvas, (px, py), 8, (0, 200, 255), -1)
        cv2.putText(canvas, f"ID{tid}", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    return canvas


def _overlay_image_calib_dots(jpeg_bytes: bytes) -> bytes:
    if STATE.calib_confirmed or not STATE.image_pts:
        return jpeg_bytes
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jpeg_bytes
    for i, xy in enumerate(STATE.image_pts):
        x, y = int(xy[0]), int(xy[1])
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(img, str(i + 1), (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return buf.tobytes() if ok else jpeg_bytes


def _load_foul_roi() -> None:
    STATE.foul_roi = None
    if not STATE.foul_roi_path.exists():
        return
    try:
        data = json.loads(STATE.foul_roi_path.read_text(encoding="utf-8"))
        xyxy = data.get("xyxy")
        if isinstance(xyxy, list) and len(xyxy) == 4:
            STATE.foul_roi = [float(x) for x in xyxy]
    except Exception as exc:  # noqa: BLE001
        logger.warning("load foul roi failed: %s", exc)


def _persist_foul_roi() -> None:
    STATE.foul_roi_path.parent.mkdir(parents=True, exist_ok=True)
    STATE.foul_roi_path.write_text(
        json.dumps(
            {
                "xyxy": STATE.foul_roi,
                "timestamp_seconds": STATE.foul_event_time,
                "updated_at": _utcnow(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _compute_foul_roi_from_tracks(pad: float = 16.0) -> Optional[list[float]]:
    """Union of person boxes near foul_event_time from main/proj tracks."""
    if not STATE.proj_tracks and STATE.main_key in STATE.cams:
        frames = STATE.cams[STATE.main_key].frames
    else:
        frames = STATE.proj_tracks
    if not frames:
        # also try main frames
        main = STATE.cams.get(STATE.main_key)
        frames = main.frames if main else {}
    if not frames:
        return None
    target = float(STATE.foul_event_time)
    best_fi, best_dt = None, 1e9
    for fi, row in frames.items():
        ts = row.get("timestamp_seconds")
        if ts is None:
            fps = STATE.proj_fps or 30.0
            ts = (int(fi) - 1) / max(fps, 1e-6)
        dt = abs(float(ts) - target)
        if dt < best_dt:
            best_dt, best_fi = dt, int(fi)
    if best_fi is None or best_dt > 0.25:
        return None
    row = frames.get(best_fi) or {}
    players = row.get("players") or []
    # Prefer involved IDs when present
    want = set(STATE.foul_involved_ids or [])
    chosen = [p for p in players if int(p.get("track_id", -1)) in want] if want else []
    if not chosen:
        chosen = list(players)
    boxes = []
    for p in chosen:
        xyxy = p.get("xyxy")
        if xyxy and len(xyxy) == 4:
            boxes.append([float(v) for v in xyxy])
    if not boxes:
        return None
    x1 = min(b[0] for b in boxes) - pad
    y1 = min(b[1] for b in boxes) - pad
    x2 = max(b[2] for b in boxes) + pad
    y2 = max(b[3] for b in boxes) + pad
    return [x1, y1, x2, y2]


def _seed_events() -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    def add(**kw: Any) -> None:
        events.append(
            {
                "id": uuid4().hex[:12],
                "source": kw.get("source", "system"),
                "algorithm_verified": bool(kw.get("algorithm_verified", False)),
                "camera_id": kw.get("camera_id"),
                "timestamp_seconds": float(kw.get("timestamp_seconds", 0.0)),
                "severity": kw.get("severity", "info"),
                "event_type": kw.get("event_type", "system"),
                "message": kw.get("message", ""),
                "suggested_card": kw.get("suggested_card"),
                "foul_type": kw.get("foul_type"),
                "track_ids": kw.get("track_ids"),
            }
        )

    add(
        source="system",
        algorithm_verified=False,
        event_type="system_status",
        message="Dashboard loaded preprocess videos (no live YOLO).",
        severity="info",
        timestamp_seconds=0.0,
        camera_id="CAM01",
    )
    add(
        source="system",
        algorithm_verified=False,
        event_type="model_status",
        message="Foul model not deployed — MVFoul weights/package missing；下列犯规条为人工辅助提示。",
        severity="warning",
        timestamp_seconds=0.0,
        camera_id=None,
    )
    ids = STATE.foul_involved_ids
    id_txt = "、".join(f"ID{i}" for i in ids) if ids else "相关球员"
    add(
        source="foul_assist",
        algorithm_verified=False,
        event_type="foul_assist",
        foul_type="holding",
        suggested_card="yellow",
        track_ids=list(ids),
        camera_id="CAM01",
        timestamp_seconds=STATE.foul_event_time,
        severity="review",
        message=(
            f"辅助提示：约 {STATE.foul_event_time:.1f}s，主机位检测框区域出现一次疑似犯规。"
            f"涉及球员 {id_txt}；动作研判为「拉人（holding）」。"
            "按一般比赛执法惯例，拉人不构成破坏明显进球机会时通常出示黄牌；"
            "若构成 DOGSO 则可升级红牌。本条建议：出示黄牌。"
            "播放经过该时刻时，将按检测框位置短暂红色高亮提示。"
        ),
    )
    return events


def bootstrap(config_path: Path) -> None:
    cfg = _load_yaml(config_path)
    STATE.config_path = config_path
    videos = cfg.get("videos") or {}
    STATE.cams = {}
    for key, vcfg in videos.items():
        slot = CamSlot(key, vcfg)
        slot.load()
        STATE.cams[key] = slot
        if slot.role == "main":
            STATE.main_key = key
    STATE.layout_main = STATE.main_key
    sides = [k for k, c in STATE.cams.items() if c.role != "main"]
    if len(sides) >= 1:
        STATE.layout_side1 = sides[0]
    if len(sides) >= 2:
        STATE.layout_side2 = sides[1]

    proj = cfg.get("projection") or {}
    STATE.proj_label = str(proj.get("label") or "主机位固定标定投影")
    STATE.proj_fps = float(proj.get("fps") or 30.0) or 30.0
    STATE.pitch_draw = {
        "out_w": int(proj.get("pitch_out_w") or 640),
        "out_h": int(proj.get("pitch_out_h") or 360),
        "pad": int(proj.get("pitch_pad") or 28),
    }
    STATE.calib_path = _resolve_path(
        proj.get("calib_store", str(DASHBOARD_DIR / "manual_homography.json"))
    )
    # Keep in-progress / confirmed calib across restarts (do not wipe).
    _load_persisted_calib(require_confirm=True)
    STATE.foul_roi_path = DASHBOARD_DIR / "foul_roi.json"

    tracks_path_raw = proj.get("tracks")
    STATE.proj_tracks = {}
    # Only load an explicit tracks file when configured (never silently use offside).
    if tracks_path_raw:
        tracks_path = _resolve_path(str(tracks_path_raw))
        if tracks_path.exists():
            with tracks_path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    STATE.proj_tracks[int(row["frame_index"])] = row
    # Prefer main camera frame cache when it already carries players.
    main = STATE.cams.get(STATE.main_key)
    if main and main.frames:
        for fi, row in main.frames.items():
            if row.get("players"):
                STATE.proj_tracks[fi] = row
    STATE.proj_total = max(STATE.proj_tracks) if STATE.proj_tracks else (main.total_frames if main else 0)
    if main and main.fps:
        STATE.proj_fps = float(main.fps)

    # Foul flash ROI = union of person detections near 4.5s (not hand-drawn).
    auto_roi = _compute_foul_roi_from_tracks()
    if auto_roi is not None:
        STATE.foul_roi = auto_roi
        _persist_foul_roi()
    else:
        _load_foul_roi()

    cams = cfg.get("virtual_cameras")
    STATE.virtual_cameras = (
        [str(x) for x in cams] if isinstance(cams, list) and cams else ["赛3", "赛4", "赛5", "赛6", "赛7", "赛8"]
    )

    STATE.cuda = bool(torch.cuda.is_available())
    if STATE.cuda:
        try:
            STATE.gpu_name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            STATE.gpu_name = "cuda"
    else:
        STATE.gpu_name = "cpu"
    STATE.model_name = str((cfg.get("system") or {}).get("model_name", STATE.model_name))
    STATE.events = _seed_events()
    STATE.ready = all(c.status == "loaded" for c in STATE.cams.values()) if STATE.cams else False
    STATE.message = "ready" if STATE.ready else "partial_or_error"


class SwapBody(BaseModel):
    side_key: str


class PointPayload(BaseModel):
    xy: list[float] = Field(default_factory=list)


class UndoPayload(BaseModel):
    target: str = "auto"  # image | pitch | auto


class FoulRoiPayload(BaseModel):
    xyxy: list[float] = Field(default_factory=list)


def create_app() -> FastAPI:
    app = FastAPI(title="Demo Multiview Dashboard", version="0.3.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "demo-dashboard",
            "ready": STATE.ready,
            "message": STATE.message,
            "homography_status": STATE.H_status,
            "setup_phase": _setup_phase(),
            "cameras": {
                k: {"status": c.status, "error": c.error, "frames": c.total_frames, "fps": c.fps}
                for k, c in STATE.cams.items()
            },
        }

    @app.get("/api/dashboard/status")
    def status() -> dict[str, Any]:
        return {
            "ready": STATE.ready,
            "gpu": STATE.gpu_name,
            "cuda": STATE.cuda,
            "model": STATE.model_name,
            "layout": {
                "main": STATE.layout_main,
                "side1": STATE.layout_side1,
                "side2": STATE.layout_side2,
            },
            "cameras": {
                k: {
                    "camera_id": c.camera_id,
                    "role": c.role,
                    "status": c.status,
                    "error": c.error,
                    "fps": c.fps,
                    "total_frames": c.total_frames,
                    "duration_seconds": c.duration,
                    "offset_seconds": c.offset_seconds,
                    "video": str(c.video),
                }
                for k, c in STATE.cams.items()
            },
            "projection_label": STATE.proj_label,
            "homography_status": STATE.H_status,
            "setup_phase": _setup_phase(),
            "calib_confirmed": STATE.calib_confirmed,
            "virtual_cameras": STATE.virtual_cameras,
            "pitch_layout": {
                **STATE.pitch_draw,
                "length_cm": float(PITCH.length),
                "width_cm": float(PITCH.width),
            },
            "foul_event_time": STATE.foul_event_time,
            "foul_roi": STATE.foul_roi,
            "foul_involved_ids": STATE.foul_involved_ids,
            "main_duration": STATE.cams[STATE.main_key].duration if STATE.main_key in STATE.cams else 0,
        }

    @app.post("/api/dashboard/swap")
    def swap(body: SwapBody) -> dict[str, Any]:
        if body.side_key not in STATE.cams:
            raise HTTPException(400, f"unknown camera key {body.side_key}")
        old_main = STATE.layout_main
        if body.side_key == STATE.layout_side1:
            STATE.layout_main, STATE.layout_side1 = STATE.layout_side1, old_main
        elif body.side_key == STATE.layout_side2:
            STATE.layout_main, STATE.layout_side2 = STATE.layout_side2, old_main
        else:
            STATE.layout_side1 = old_main
            STATE.layout_main = body.side_key
        return {
            "layout": {
                "main": STATE.layout_main,
                "side1": STATE.layout_side1,
                "side2": STATE.layout_side2,
            }
        }

    @app.post("/api/dashboard/calib/image")
    def calib_image(payload: PointPayload) -> dict[str, Any]:
        if STATE.calib_confirmed or len(STATE.image_pts) >= 4:
            return {"ok": False, "phase": _setup_phase()}
        if len(payload.xy) != 2:
            raise HTTPException(400, "xy=[x,y]")
        STATE.image_pts.append([float(payload.xy[0]), float(payload.xy[1])])
        _sync_pairs_from_buffers()
        _persist_calib()
        return {
            "ok": True,
            "phase": _setup_phase(),
            "image_pts": len(STATE.image_pts),
            "pitch_pts": len(STATE.pitch_pts),
        }

    @app.post("/api/dashboard/calib/pitch")
    def calib_pitch(payload: PointPayload) -> dict[str, Any]:
        if STATE.calib_confirmed or len(STATE.image_pts) < 4 or len(STATE.pitch_pts) >= 4:
            return {"ok": False, "phase": _setup_phase()}
        if len(payload.xy) != 2:
            raise HTTPException(400, "xy=[x,y]")
        STATE.pitch_pts.append([float(payload.xy[0]), float(payload.xy[1])])
        _sync_pairs_from_buffers()
        _persist_calib()
        return {
            "ok": True,
            "phase": _setup_phase(),
            "image_pts": len(STATE.image_pts),
            "pitch_pts": len(STATE.pitch_pts),
        }

    @app.post("/api/dashboard/calib/undo")
    def calib_undo(payload: UndoPayload) -> dict[str, Any]:
        """Undo last calib point. target=image|pitch|auto."""
        if STATE.calib_confirmed:
            return {"ok": False, "phase": "done", "message": "already confirmed"}
        target = str(payload.target or "auto")
        if target == "auto":
            if STATE.pitch_pts:
                target = "pitch"
            elif STATE.image_pts:
                target = "image"
            else:
                return {
                    "ok": False,
                    "phase": _setup_phase(),
                    "image_pts": 0,
                    "pitch_pts": 0,
                }
        if target == "pitch" and STATE.pitch_pts:
            STATE.pitch_pts.pop()
        elif target == "image" and STATE.image_pts:
            STATE.image_pts.pop()
            if len(STATE.pitch_pts) > len(STATE.image_pts):
                STATE.pitch_pts = STATE.pitch_pts[: len(STATE.image_pts)]
        else:
            return {
                "ok": False,
                "phase": _setup_phase(),
                "image_pts": len(STATE.image_pts),
                "pitch_pts": len(STATE.pitch_pts),
            }
        _sync_pairs_from_buffers()
        _persist_calib()
        return {
            "ok": True,
            "phase": _setup_phase(),
            "image_pts": len(STATE.image_pts),
            "pitch_pts": len(STATE.pitch_pts),
            "undid": target,
        }

    @app.post("/api/dashboard/calib/confirm")
    def calib_confirm() -> dict[str, Any]:
        if len(STATE.image_pts) < 4 or len(STATE.pitch_pts) < 4:
            return {"ok": False, "phase": _setup_phase()}
        STATE.calib_confirmed = True
        _sync_pairs_from_buffers()
        _persist_calib()
        return {"ok": True, "phase": "done", "homography_status": STATE.H_status}

    @app.post("/api/dashboard/calib/reset")
    def calib_reset() -> dict[str, Any]:
        STATE.image_pts, STATE.pitch_pts, STATE.calib_pairs = [], [], []
        STATE.calib_confirmed = False
        STATE.H, STATE.H_status = None, "unavailable"
        _persist_calib()
        return {"ok": True, "phase": "image"}

    @app.post("/api/dashboard/foul_roi")
    def foul_roi_set(payload: FoulRoiPayload) -> dict[str, Any]:
        if len(payload.xyxy) != 4:
            raise HTTPException(400, "xyxy=[x1,y1,x2,y2]")
        x1, y1, x2, y2 = [float(v) for v in payload.xyxy]
        STATE.foul_roi = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        _persist_foul_roi()
        return {"ok": True, "foul_roi": STATE.foul_roi, "timestamp_seconds": STATE.foul_event_time}

    @app.get("/api/dashboard/foul_roi")
    def foul_roi_get() -> dict[str, Any]:
        return {
            "foul_roi": STATE.foul_roi,
            "timestamp_seconds": STATE.foul_event_time,
            "involved_ids": STATE.foul_involved_ids,
        }

    @app.get("/api/dashboard/events")
    def events() -> dict[str, Any]:
        return {"events": STATE.events}

    @app.get("/api/dashboard/snapshot")
    def snapshot(t: float = 0.0) -> dict[str, Any]:
        if not STATE.cams:
            raise HTTPException(503, "no cameras configured")

        with IO_LOCK:
            views = {}
            for key, cam in STATE.cams.items():
                if cam.status != "loaded":
                    views[key] = {"status": cam.status, "error": cam.error}
                    continue
                st, idx, meta = cam.frame_at_main_time(t)
                item: dict[str, Any] = {
                    "status": st,
                    "camera_id": cam.camera_id,
                    "role": cam.role,
                    "offset_seconds": cam.offset_seconds,
                    "source_fps": cam.fps,
                }
                if st == "ok" and idx is not None:
                    item["frame_index"] = idx
                    item["timestamp_seconds"] = (idx - 1) / cam.fps
                    item["detections"] = (meta or {}).get("detections")
                    item["track_ids"] = (meta or {}).get("track_ids") or []
                    item["processing_fps"] = (meta or {}).get("processing_fps")
                    item["inference_ms"] = (meta or {}).get("inference_ms")
                    try:
                        jpg = cam.jpeg_at(idx)
                        if key == STATE.layout_main:
                            jpg = _overlay_image_calib_dots(jpg)
                        item["image_jpeg_base64"] = base64.b64encode(jpg).decode("ascii")
                    except HTTPException as exc:
                        item["status"] = "error"
                        item["error"] = str(exc.detail)
                views[key] = item

            pidx = int(round(t * STATE.proj_fps)) + 1
            if STATE.proj_total > 0:
                pidx = max(1, min(STATE.proj_total, pidx))
            prow = STATE.proj_tracks.get(pidx) or {"frame_index": pidx, "players": []}
            players_xy = _project_players(prow) if (STATE.H is not None and STATE.calib_confirmed) else []
            pitch = draw_pitch_2d(
                players_xy,
                calib_pitch_pts=[tuple(p) for p in STATE.pitch_pts],  # type: ignore[misc]
                out_w=STATE.pitch_draw["out_w"],
                out_h=STATE.pitch_draw["out_h"],
                pad=STATE.pitch_draw["pad"],
            )
            ok, buf = cv2.imencode(".jpg", pitch, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            pitch_b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else None

            geometry = {
                "label": STATE.proj_label,
                "homography_status": STATE.H_status if STATE.calib_confirmed else "pending_calib",
                "frame_index": pidx,
                "projected_objects": len(players_xy),
                "pitch_jpeg_base64": pitch_b64,
                "setup_phase": _setup_phase(),
                "calib_confirmed": STATE.calib_confirmed,
                "message": None if STATE.calib_confirmed and STATE.H is not None else "等待主机位标定",
            }
            return {
                "main_time_seconds": t,
                "layout": {
                    "main": STATE.layout_main,
                    "side1": STATE.layout_side1,
                    "side2": STATE.layout_side2,
                },
                "views": views,
                "geometry": geometry,
                "virtual_cameras": STATE.virtual_cameras,
                "foul_roi": STATE.foul_roi,
                "foul_event_time": STATE.foul_event_time,
                "pitch_layout": {
                    **STATE.pitch_draw,
                    "length_cm": float(PITCH.length),
                    "width_cm": float(PITCH.width),
                },
                "system": {
                    "gpu": STATE.gpu_name,
                    "cuda": STATE.cuda,
                    "model": STATE.model_name,
                    "main_camera_key": STATE.layout_main,
                },
            }

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return DASHBOARD_HTML

    return app


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>SC-live Multiview Demo Dashboard</title>
<style>
*{box-sizing:border-box}
body{margin:0;font-family:Segoe UI,Arial,sans-serif;background:#0e1116;color:#e8eaed}
header{padding:10px 14px;background:#161b22;border-bottom:1px solid #30363d;display:flex;gap:16px;flex-wrap:wrap;align-items:center}
.grid{display:grid;grid-template-columns:2fr 1fr 72px 1fr;grid-template-rows:auto auto;gap:8px;padding:8px}
.main{grid-column:1;grid-row:1/3}
.side1{grid-column:2;grid-row:1}
.side2{grid-column:4;grid-row:1}
.wheel{grid-column:3;grid-row:1/3;display:flex;flex-direction:column;align-items:center}
.geo{grid-column:2/5;grid-row:2}
.panel{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px}
.viewwrap{position:relative}
img.view{width:100%;background:#000;display:block;min-height:120px}
img.pitch{width:100%;max-height:320px;object-fit:contain;background:#0a2;display:block}
.muted{color:#9da3ae;font-size:12px}
.bar{display:flex;gap:6px;flex-wrap:wrap;align-items:center;padding:8px}
button{background:#238636;color:#fff;border:0;padding:6px 10px;border-radius:4px;cursor:pointer}
button.sec{background:#30363d}
input[type=range]{width:240px}
#events{max-height:220px;overflow:auto;font-size:13px}
.evt{padding:4px 0;border-bottom:1px solid #222;cursor:pointer}
.tag{display:inline-block;padding:1px 6px;border-radius:3px;background:#30363d;margin-right:4px;font-size:11px}
.warn{color:#d29922}
.cam-wheel{
  width:64px;height:100%;min-height:280px;max-height:420px;overflow-y:auto;
  scroll-snap-type:y mandatory;border:1px solid #30363d;border-radius:6px;background:#0d1117;
  scrollbar-width:thin;
}
.cam-item{
  scroll-snap-align:center;height:52px;display:flex;align-items:center;justify-content:center;
  font-size:15px;font-weight:600;color:#8b949e;border-bottom:1px solid #21262d;cursor:pointer;
  user-select:none;
}
.cam-item.active{background:#1f6feb33;color:#79c0ff;box-shadow:inset 3px 0 0 #1f6feb}
.cam-item:hover{color:#e6edf3}
.virt-preview{
  margin-top:8px;min-height:72px;background:#05080c;border:1px dashed #30363d;border-radius:4px;
  display:flex;align-items:center;justify-content:center;color:#6e7681;font-size:13px;text-align:center;padding:8px;
}
.wheel-label{font-size:11px;color:#8b949e;margin-bottom:4px}
/* Quiet finish control — visible enough to click, not loud */
#confirmGeo{
  display:none;position:absolute;right:8px;bottom:8px;
  padding:4px 10px;font-size:11px;
  opacity:0.35;background:#222c;color:#c9d1d9;
  border:1px solid #444;border-radius:4px;cursor:pointer;z-index:5;
}
#confirmGeo:hover{opacity:0.9;background:#333}
#foulFlash{
  display:none;position:absolute;pointer-events:none;z-index:4;
  background:rgba(220,40,40,0.45);border:2px solid rgba(255,80,80,0.9);
  box-sizing:border-box;
}
#foulDraft{
  display:none;position:absolute;pointer-events:none;z-index:3;
  border:2px dashed rgba(255,200,80,0.85);background:rgba(255,200,80,0.12);
}
</style>
</head>
<body>
<header>
  <strong>SC-live 多视角大屏</strong>
  <span id="sys" class="muted">loading…</span>
</header>
<div class="grid">
  <div class="panel main">
    <div class="muted" id="mainLabel">MAIN</div>
    <div class="viewwrap" id="mainWrap">
      <img id="mainImg" class="view"/>
      <div id="foulDraft"></div>
      <div id="foulFlash"></div>
    </div>
    <div id="mainMeta" class="muted"></div>
  </div>
  <div class="panel side1">
    <div class="muted">辅机位1 <button class="sec" onclick="swap('side1')">Swap</button></div>
    <img id="side1Img" class="view"/>
    <div id="side1Meta" class="muted"></div>
  </div>
  <div class="panel wheel">
    <div class="wheel-label">更多机位</div>
    <div id="camWheel" class="cam-wheel" title="演示机位列表（无视频源）"></div>
    <div id="virtPrev" class="virt-preview">滚轮选择机位<br/>当前无视频源</div>
  </div>
  <div class="panel side2">
    <div class="muted">辅机位2 <button class="sec" onclick="swap('side2')">Swap</button></div>
    <img id="side2Img" class="view"/>
    <div id="side2Meta" class="muted"></div>
  </div>
  <div class="panel geo">
    <div><b>二维球场投影</b> <span class="warn" id="geoLabel"></span></div>
    <div class="viewwrap">
      <img id="pitchImg" class="pitch" alt="pitch"/>
      <button id="confirmGeo" onclick="confirmGeo()" title="结束标定">确认</button>
    </div>
    <div id="geoBody" class="muted"></div>
  </div>
</div>
<div class="bar panel" style="margin:8px">
  <button onclick="toggle()">Play/Pause</button>
  <button class="sec" onclick="step(-1)">Prev</button>
  <button class="sec" onclick="step(1)">Next</button>
  <button class="sec" onclick="setRate(0.5)">0.5x</button>
  <button class="sec" onclick="setRate(1)">1x</button>
  <button class="sec" onclick="setRate(2)">2x</button>
  <input id="seek" type="range" min="0" max="1000" value="0" oninput="onSeek()"/>
  <span id="clock">t=0.000s</span>
</div>
<div class="panel" style="margin:8px">
  <b>Events</b>
  <div id="events"></div>
</div>
<script>
let playing=false, rate=1, t=0, duration=10, timer=null, layout={}, loading=false;
let virtualCams=[], virtIdx=0, phase='image', pitchLayout=null;
let foulRoi=null, foulTime=4.5, foulMarking=false, foulCorner=null;
let foulFlashUntil=0, foulFlashedForPass=false, lastT=0;
function buildWheel(){
  const box=document.getElementById('camWheel');
  box.innerHTML='';
  virtualCams.forEach((name,i)=>{
    const d=document.createElement('div');
    d.className='cam-item'+(i===virtIdx?' active':'');
    d.textContent=name;
    d.onclick=()=>selectVirt(i);
    box.appendChild(d);
  });
  updateVirtPrev();
}
function selectVirt(i){
  virtIdx=i;
  [...document.querySelectorAll('.cam-item')].forEach((el,j)=>{
    el.classList.toggle('active', j===virtIdx);
  });
  const el=document.querySelectorAll('.cam-item')[virtIdx];
  if(el) el.scrollIntoView({block:'nearest', behavior:'smooth'});
  updateVirtPrev();
}
function updateVirtPrev(){
  const name=virtualCams[virtIdx]||'—';
  document.getElementById('virtPrev').innerHTML=
    `<div><b>${name}</b><br/><span style="color:#d29922">无视频源</span></div>`;
}
function updateConfirmBtn(){
  const b=document.getElementById('confirmGeo');
  b.style.display = (phase==='confirm') ? 'block' : 'none';
}
function imgClickXY(ev,img){
  const rect=img.getBoundingClientRect();
  return [(ev.clientX-rect.left)*(img.naturalWidth/rect.width),
          (ev.clientY-rect.top)*(img.naturalHeight/rect.height)];
}
function pitchClickToCm(ev,img){
  const [x,y]=imgClickXY(ev,img);
  const pad=pitchLayout.pad, ow=pitchLayout.out_w, oh=pitchLayout.out_h;
  const sx=(ow-2*pad)/pitchLayout.length_cm, sy=(oh-2*pad)/pitchLayout.width_cm;
  return [(x-pad)/sx, (y-pad)/sy];
}
function mapImgBoxToOverlay(xyxy){
  const img=document.getElementById('mainImg');
  if(!img.naturalWidth) return null;
  const rect=img.getBoundingClientRect();
  const wrap=document.getElementById('mainWrap').getBoundingClientRect();
  const sx=rect.width/img.naturalWidth, sy=rect.height/img.naturalHeight;
  const left=rect.left-wrap.left + xyxy[0]*sx;
  const top=rect.top-wrap.top + xyxy[1]*sy;
  const w=(xyxy[2]-xyxy[0])*sx, h=(xyxy[3]-xyxy[1])*sy;
  return {left, top, width:w, height:h};
}
function placeOverlay(el, xyxy){
  const box=mapImgBoxToOverlay(xyxy);
  if(!box){ el.style.display='none'; return; }
  el.style.display='block';
  el.style.left=box.left+'px';
  el.style.top=box.top+'px';
  el.style.width=box.width+'px';
  el.style.height=box.height+'px';
}
function triggerFoulFlash(){
  if(!foulRoi) return;
  foulFlashUntil=performance.now()+500;
  placeOverlay(document.getElementById('foulFlash'), foulRoi);
}
function updateFoulFlash(){
  const el=document.getElementById('foulFlash');
  if(!foulRoi){ el.style.display='none'; return; }
  if(performance.now() < foulFlashUntil){
    placeOverlay(el, foulRoi);
  } else {
    el.style.display='none';
  }
  // auto flash once when playback crosses foulTime
  if(lastT < foulTime && t >= foulTime - 1e-6){
    if(!foulFlashedForPass){
      foulFlashedForPass=true;
      triggerFoulFlash();
    }
  }
  if(t < foulTime - 0.2) foulFlashedForPass=false;
  lastT=t;
}
async function init(){
  const s=await (await fetch('/api/dashboard/status')).json();
  duration=s.main_duration||10;
  document.getElementById('seek').max=Math.floor(duration*1000);
  document.getElementById('sys').textContent=
    `GPU=${s.gpu} CUDA=${s.cuda} model=${s.model} ready=${s.ready}`;
  layout=s.layout;
  phase=s.setup_phase||'image';
  pitchLayout=s.pitch_layout;
  foulRoi=s.foul_roi||null;
  foulTime=s.foul_event_time||4.5;
  foulMarking=false; foulCorner=null;
  virtualCams=s.virtual_cameras||['赛3','赛4','赛5','赛6'];
  buildWheel(); updateConfirmBtn();
  document.getElementById('camWheel').addEventListener('wheel', (ev)=>{
    ev.preventDefault();
    if(ev.deltaY>0) selectVirt(Math.min(virtualCams.length-1, virtIdx+1));
    else selectVirt(Math.max(0, virtIdx-1));
  }, {passive:false});
  const ev=await (await fetch('/api/dashboard/events')).json();
  window._allEvents = ev.events||[];
  renderEvents();
  await refresh();
  requestAnimationFrame(function tick(){ updateFoulFlash(); requestAnimationFrame(tick); });
}
function renderEvents(){
  const box=document.getElementById('events');
  box.innerHTML='';
  (window._allEvents||[]).forEach(e=>{
    const et=Number(e.timestamp_seconds||0);
    // Timed tips (foul assist etc.) only appear once playback reaches that moment.
    if(e.event_type==='foul_assist' && t + 1e-6 < et) return;
    const d=document.createElement('div');
    d.className='evt';
    const card=e.suggested_card?(` card=${e.suggested_card}`):'';
    d.innerHTML=`<span class="tag">${e.event_type}</span><span class="tag">${e.camera_id||'-'}</span>`+
      `<span class="tag">verified=${e.algorithm_verified}</span>${card?`<span class="tag">${card.trim()}</span>`:''}`+
      ` t=${et.toFixed(2)} — ${e.message}`;
    d.onclick=()=>{
      t=et;
      if(e.event_type==='foul_assist' && foulRoi) triggerFoulFlash();
      refresh();
    };
    box.appendChild(d);
  });
}
function setRate(r){ rate=r; if(playing){ stop(); play(); } }
function play(){
  playing=true;
  timer=setInterval(()=>{
    if(loading) return;
    t=Math.min(duration, t+0.10*rate);
    refresh();
  }, 120);
}
function stop(){ playing=false; if(timer) clearInterval(timer); timer=null; }
function toggle(){ if(playing) stop(); else play(); }
function step(d){ t=Math.max(0, Math.min(duration, t + d*(1/30))); refresh(); }
function onSeek(){ t=Number(document.getElementById('seek').value)/1000; refresh(); }
async function swap(which){
  const key = which==='side1'? layout.side1 : layout.side2;
  const r=await (await fetch('/api/dashboard/swap',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({side_key:key})})).json();
  layout=r.layout; refresh();
}
function show(imgId, metaId, view, label){
  const img=document.getElementById(imgId);
  const meta=document.getElementById(metaId);
  if(!view){ meta.textContent='missing'; return; }
  if(view.status==='ok' && view.image_jpeg_base64){
    img.src='data:image/jpeg;base64,'+view.image_jpeg_base64;
    meta.textContent=`${label} ${view.camera_id} frame=${view.frame_index} t=${(view.timestamp_seconds||0).toFixed(3)} det=${view.detections} tracks=${(view.track_ids||[]).length}`;
  } else {
    img.removeAttribute('src');
    meta.textContent=`${label} ${view.camera_id||''} status=${view.status} ${view.error||''}`;
  }
}
async function refresh(){
  if(loading) return;
  loading=true;
  try{
    document.getElementById('clock').textContent=`t=${t.toFixed(3)}s rate=${rate}x`;
    document.getElementById('seek').value=Math.floor(t*1000);
    const snap=await (await fetch(`/api/dashboard/snapshot?t=${t}`)).json();
    layout=snap.layout;
    pitchLayout=snap.pitch_layout||pitchLayout;
    if(snap.foul_roi) foulRoi=snap.foul_roi;
    if(snap.foul_event_time!=null) foulTime=snap.foul_event_time;
    const v=snap.views||{};
    show('mainImg','mainMeta', v[layout.main], 'MAIN');
    show('side1Img','side1Meta', v[layout.side1], '辅1');
    show('side2Img','side2Meta', v[layout.side2], '辅2');
    document.getElementById('mainLabel').textContent='主机位 ← '+layout.main;
    const g=snap.geometry||{};
    phase=g.setup_phase||phase;
    updateConfirmBtn();
    document.getElementById('geoLabel').textContent='';
    if(g.pitch_jpeg_base64){
      document.getElementById('pitchImg').src='data:image/jpeg;base64,'+g.pitch_jpeg_base64;
    }
    document.getElementById('geoBody').textContent='';
    document.getElementById('mainImg').style.cursor = (phase==='image') ? 'crosshair' : 'default';
    document.getElementById('pitchImg').style.cursor = (phase==='pitch') ? 'crosshair' : 'default';
    renderEvents();
    updateFoulFlash();
  } catch(e){
    console.error(e);
  } finally {
    loading=false;
  }
}
document.getElementById('mainImg').addEventListener('click', async (ev)=>{
  if(phase!=='image') return;
  const xy=imgClickXY(ev, ev.target);
  await fetch('/api/dashboard/calib/image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({xy})});
  await refresh();
});
document.getElementById('pitchImg').addEventListener('click', async (ev)=>{
  if(phase!=='pitch') return;
  const xy=pitchClickToCm(ev, ev.target);
  await fetch('/api/dashboard/calib/pitch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({xy})});
  await refresh();
});
async function undoCalib(target){
  if(phase==='done') return;
  await fetch('/api/dashboard/calib/undo',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({target:target||'auto'})});
  await refresh();
}
document.getElementById('pitchImg').addEventListener('contextmenu', async (ev)=>{
  ev.preventDefault();
  if(phase==='done') return;
  await undoCalib('pitch');
});
document.getElementById('mainImg').addEventListener('contextmenu', async (ev)=>{
  ev.preventDefault();
  if(phase==='done') return;
  await undoCalib('image');
});
async function confirmGeo(){
  const r=await (await fetch('/api/dashboard/calib/confirm',{method:'POST'})).json();
  phase=r.phase||'done'; updateConfirmBtn();
  await refresh();
}
document.addEventListener('keydown', (ev)=>{
  if(ev.key==='Enter' && phase==='confirm'){ confirmGeo(); }
});
init();
</script>
</body>
</html>
"""


def run_dashboard_server(
    *,
    config: str,
    host: str = "0.0.0.0",
    port: int = 6008,
) -> None:
    import uvicorn

    path = Path(config).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    bootstrap(path)
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/demo_dashboard.yaml")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=6008)
    args = p.parse_args()
    run_dashboard_server(config=args.config, host=args.host, port=args.port)
