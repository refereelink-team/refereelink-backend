"""Offside review UI — cache playback + quiet 4+4 point setup + 2D projection."""

from __future__ import annotations

import base64
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from app.config.pitch import SoccerPitchConfiguration

logger = logging.getLogger("offside_review")
REPO_ROOT = Path(__file__).resolve().parents[1]
PITCH = SoccerPitchConfiguration()


class ReviewState:
    def __init__(self) -> None:
        self.video_path: Path | None = None
        self.discovery_dir: Path | None = None
        self.config_path: Path = REPO_ROOT / "configs" / "offside_demo.yaml"
        self.review_dir: Path | None = None
        self.fps: float = 30.0
        self.total_frames: int = 0
        self.width: int = 1280
        self.height: int = 720
        self.frame_index: dict[int, dict[str, Any]] = {}
        self.lock = threading.RLock()
        self.ready: bool = False
        self.message: str = "initializing"
        self._cap: Optional[cv2.VideoCapture] = None
        self._last_idx: int = -1
        self.image_pts: list[list[float]] = []
        self.pitch_pts: list[list[float]] = []
        self.calib_pairs: list[dict[str, Any]] = []
        self.H: Optional[np.ndarray] = None
        self.H_inliers: int = 0
        self.H_status: str = "unavailable"
        self.calib_confirmed: bool = False
        self.pitch_draw = {"out_w": 640, "out_h": 420, "pad": 24}


STATE = ReviewState()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def _jump_marks(total: int) -> list[int]:
    if total <= 1:
        return [1]
    marks = [1]
    n = 100
    while n < total:
        marks.append(n)
        n += 100
    if marks[-1] != total:
        marks.append(total)
    return marks


def load_existing_frame_index(review_dir: Path, video_path: Path) -> dict[int, dict[str, Any]]:
    cache_path = review_dir / "frame_index.jsonl"
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing review cache {cache_path}")
    index: dict[int, dict[str, Any]] = {}
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            index[int(row["frame_index"])] = row
    return index


def recompute_H() -> None:
    if len(STATE.calib_pairs) < 4:
        STATE.H = None
        STATE.H_inliers = 0
        STATE.H_status = "unavailable"
        return
    src = np.array([p["image_xy"] for p in STATE.calib_pairs], dtype=np.float32)
    dst = np.array([p["pitch_xy"] for p in STATE.calib_pairs], dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        STATE.H = None
        STATE.H_inliers = 0
        STATE.H_status = "unavailable"
        return
    STATE.H = H
    STATE.H_inliers = len(STATE.calib_pairs)
    STATE.H_status = "ready"


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
        STATE.H_inliers = 0


def _setup_phase() -> str:
    if STATE.calib_confirmed and STATE.H is not None:
        return "done"
    if len(STATE.image_pts) < 4:
        return "image"
    if len(STATE.pitch_pts) < 4:
        return "pitch"
    return "confirm"


def _load_offside_context() -> dict[str, Any]:
    """Roles for offside overlay come only from yaml (explicit save), not sticky confirmation file."""
    cfg = _load_yaml(STATE.config_path)
    roles = cfg.get("manual_role_confirmations") or {}
    return {
        "attack_direction": cfg.get("attack_direction"),
        "attacker_ids": set(int(x) for x in (roles.get("attacker_track_ids") or [])),
        "defender_ids": set(int(x) for x in (roles.get("defender_track_ids") or [])),
        "passer_id": cfg.get("passer_track_id"),
        "second_last_override": roles.get("second_last_defender_track_id"),
        "review_confirmed": bool((cfg.get("review") or {}).get("confirmed")),
    }


def _clear_review_judgment() -> dict[str, Any]:
    cfg = _load_yaml(STATE.config_path)
    cfg["critical_frame"] = None
    cfg["critical_time_seconds"] = None
    cfg["critical_time_source"] = None
    cfg["attack_direction"] = None
    cfg["passer_track_id"] = None
    cfg["manual_role_confirmations"] = {
        "attacker_track_ids": [],
        "defender_track_ids": [],
        "second_last_defender_track_id": None,
    }
    ball = cfg.get("ball") or {}
    ball["source"] = "detector"
    ball["manual_point"] = None
    cfg["ball"] = ball
    cfg["review"] = {"confirmed": False, "confirmed_at": None, "notes": ""}
    _save_yaml(STATE.config_path, cfg)
    if STATE.review_dir is not None:
        path = STATE.review_dir / "CONFIRMATION_RESULT.json"
        if path.exists():
            path.unlink()
    return {"ok": True, "message": "已清除确认与越位判断"}


def _judge_offside(
    players_xy: list[tuple[float, float, int]],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Geometric assist: line // goal line at 2nd-last defender along attack axis."""
    result: dict[str, Any] = {
        "ready": False,
        "verdict": None,
        "line_x": None,
        "second_last_id": None,
        "offside_attacker_ids": [],
    }
    direction = ctx.get("attack_direction")
    defenders = ctx.get("defender_ids") or set()
    attackers = ctx.get("attacker_ids") or set()
    if direction not in {"left_to_right", "right_to_left"}:
        return result
    if len(defenders) < 2 or not attackers:
        return result
    if STATE.H is None:
        return result

    by_id = {tid: (x, y) for x, y, tid in players_xy}
    def_pos = [(tid, by_id[tid][0], by_id[tid][1]) for tid in defenders if tid in by_id]
    if len(def_pos) < 2:
        return result

    # Last defender = nearest to the goal being attacked.
    # Second-last = next. Offside line is parallel to that goal line (const X).
    if direction == "left_to_right":
        def_pos.sort(key=lambda t: t[1], reverse=True)  # max X first (right goal)
    else:
        def_pos.sort(key=lambda t: t[1])  # min X first (left goal)
    override = ctx.get("second_last_override")
    if override is not None and int(override) in by_id:
        second_last_id = int(override)
        line_x = float(by_id[second_last_id][0])
    else:
        second_last_id = int(def_pos[1][0])  # [0]=last, [1]=second-last
        line_x = float(def_pos[1][1])

    passer = ctx.get("passer_id")
    offside_ids: list[int] = []
    for tid in attackers:
        if passer is not None and int(tid) == int(passer):
            continue
        if tid not in by_id:
            continue
        ax = float(by_id[tid][0])
        if direction == "left_to_right":
            # nearer to right goal than the line
            if ax > line_x + 1.0:  # 1cm tolerance
                offside_ids.append(int(tid))
        else:
            if ax < line_x - 1.0:
                offside_ids.append(int(tid))

    verdict = "OFFSIDE" if offside_ids else "ONSIDE"
    result.update(
        {
            "ready": True,
            "verdict": verdict,
            "line_x": line_x,
            "second_last_id": second_last_id,
            "offside_attacker_ids": offside_ids,
            "attack_direction": direction,
        }
    )
    return result


def _draw_offside_line_on_pitch(
    canvas: np.ndarray,
    line_x: float,
    out_w: int,
    out_h: int,
) -> None:
    pad = STATE.pitch_draw["pad"]
    scale_x = (out_w - 2 * pad) / float(PITCH.length)
    scale_y = (out_h - 2 * pad) / float(PITCH.width)
    px = int(pad + line_x * scale_x)
    y0 = int(pad)
    y1 = int(pad + PITCH.width * scale_y)
    cv2.line(canvas, (px, y0), (px, y1), (0, 255, 255), 3, cv2.LINE_AA)


def _draw_offside_line_on_video(out: np.ndarray, line_x: float) -> None:
    """Map pitch-space goal-parallel line back to image via H^{-1}."""
    if STATE.H is None:
        return
    try:
        H_inv = np.linalg.inv(STATE.H)
    except np.linalg.LinAlgError:
        return
    ys = np.linspace(0, PITCH.width, 40)
    pts = np.array([[[line_x, float(y)] for y in ys]], dtype=np.float32)
    try:
        img_pts = cv2.perspectiveTransform(pts, H_inv).reshape(-1, 2)
    except cv2.error:
        return
    h, w = out.shape[:2]
    prev = None
    for x, y in img_pts:
        if not np.isfinite(x) or not np.isfinite(y):
            prev = None
            continue
        if x < -50 or y < -50 or x > w + 50 or y > h + 50:
            prev = None
            continue
        cur = (int(x), int(y))
        if prev is not None:
            cv2.line(out, prev, cur, (0, 255, 255), 3, cv2.LINE_AA)
        prev = cur


def draw_pitch_2d(
    players_xy: list[tuple[float, float, int]],
    ball_xy: Optional[tuple[float, float]] = None,
    calib_pitch_pts: Optional[list[tuple[float, float]]] = None,
    out_w: int = 640,
    out_h: int = 420,
    offside: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    canvas[:] = (34, 110, 45)
    pad = STATE.pitch_draw["pad"]
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

    if calib_pitch_pts and not (offside and offside.get("ready")):
        for i, (x, y) in enumerate(calib_pitch_pts):
            px, py = to_px(x, y)
            cv2.circle(canvas, (px, py), 5, (0, 255, 255), -1)
            cv2.putText(canvas, str(i + 1), (px + 6, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

    if offside and offside.get("ready") and offside.get("line_x") is not None:
        _draw_offside_line_on_pitch(canvas, float(offside["line_x"]), out_w, out_h)

    off_ids = set(offside.get("offside_attacker_ids") or []) if offside else set()
    for x, y, tid in players_xy:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        if x < -200 or x > PITCH.length + 200 or y < -200 or y > PITCH.width + 200:
            continue
        px, py = to_px(float(x), float(y))
        color = (0, 0, 255) if tid in off_ids else (0, 200, 255)
        cv2.circle(canvas, (px, py), 8, color, -1)
        cv2.putText(canvas, f"ID{tid}", (px + 6, py - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    if ball_xy is not None and np.isfinite(ball_xy[0]) and np.isfinite(ball_xy[1]):
        bx, by = to_px(float(ball_xy[0]), float(ball_xy[1]))
        cv2.circle(canvas, (bx, by), 7, (0, 0, 255), -1)

    if offside and offside.get("verdict") == "OFFSIDE":
        cv2.putText(
            canvas,
            "OFFSIDE",
            (out_w // 2 - 140, out_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (0, 0, 0),
            10,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "OFFSIDE",
            (out_w // 2 - 140, out_h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.2,
            (0, 0, 255),
            5,
            cv2.LINE_AA,
        )
    elif offside and offside.get("verdict") == "ONSIDE":
        cv2.putText(
            canvas,
            "ONSIDE",
            (out_w // 2 - 110, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return canvas


def _project_players(row: dict[str, Any]) -> tuple[list[tuple[float, float, int]], Optional[tuple[float, float]]]:
    players_xy: list[tuple[float, float, int]] = []
    ball_xy = None
    if STATE.H is None:
        return players_xy, ball_xy
    pts, meta = [], []
    for p in row.get("players") or []:
        x1, y1, x2, y2 = p["xyxy"]
        pts.append([(x1 + x2) / 2.0, float(y2)])
        meta.append(int(p.get("track_id", -1)))
    if pts:
        arr = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)
        try:
            world = cv2.perspectiveTransform(arr, STATE.H).reshape(-1, 2)
            for (x, y), tid in zip(world, meta):
                players_xy.append((float(x), float(y), tid))
        except cv2.error:
            pass
    ball = row.get("ball")
    if ball and ball.get("xyxy"):
        x1, y1, x2, y2 = ball["xyxy"]
        bx, by = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        try:
            w = cv2.perspectiveTransform(np.array([[[bx, by]]], dtype=np.float32), STATE.H).reshape(2)
            ball_xy = (float(w[0]), float(w[1]))
        except cv2.error:
            ball_xy = None
    return players_xy, ball_xy


def _draw_frame(
    frame: np.ndarray,
    row: dict[str, Any],
    fps: float,
    offside: Optional[dict[str, Any]] = None,
) -> np.ndarray:
    out = frame.copy()
    players = row.get("players") or []
    off_ids = set((offside or {}).get("offside_attacker_ids") or [])
    for p in players:
        x1, y1, x2, y2 = map(int, p["xyxy"])
        tid = int(p.get("track_id", -1))
        conf = float(p.get("confidence", 0.0))
        color = (0, 0, 255) if tid in off_ids else (0, 200, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"ID{tid}:{conf:.2f}"
        cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(out, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(out, (int((x1 + x2) / 2), y2), 4, (0, 255, 255), -1)
    ball = row.get("ball")
    if ball and ball.get("xyxy"):
        x1, y1, x2, y2 = map(int, ball["xyxy"])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
    if not STATE.calib_confirmed:
        for i, xy in enumerate(STATE.image_pts):
            x, y = map(int, xy)
            cv2.circle(out, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(out, str(i + 1), (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    if offside and offside.get("ready") and offside.get("line_x") is not None:
        _draw_offside_line_on_video(out, float(offside["line_x"]))
    if offside and offside.get("verdict") == "OFFSIDE":
        cv2.putText(out, "OFFSIDE", (out.shape[1] // 2 - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 10, cv2.LINE_AA)
        cv2.putText(out, "OFFSIDE", (out.shape[1] // 2 - 180, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5, cv2.LINE_AA)
    fi = int(row.get("frame_index", 0))
    for i, line in enumerate([f"Frame: {fi}", f"Time: {fi / max(fps, 1e-6):.3f}s", f"Tracks: {len(players)}"]):
        y = 22 + i * 22
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def read_video_frame(video_path: Path, zero_based: int) -> np.ndarray:
    with STATE.lock:
        if STATE._cap is None or not STATE._cap.isOpened():
            STATE._cap = cv2.VideoCapture(str(video_path))
            STATE._last_idx = -1
        cap = STATE._cap
        if zero_based != STATE._last_idx + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, zero_based)
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            STATE._cap = cv2.VideoCapture(str(video_path))
            frame = None
            for i in range(zero_based + 1):
                ok, fr = STATE._cap.read()
                if not ok:
                    break
                if i == zero_based:
                    frame = fr
            if frame is None:
                raise HTTPException(404, f"Frame {zero_based + 1} not readable")
            STATE._last_idx = zero_based
            return frame
        STATE._last_idx = zero_based
        return frame


class SavePayload(BaseModel):
    critical_frame: Optional[int] = None
    critical_time_seconds: Optional[float] = None
    attack_direction: Optional[str] = None
    passer_track_id: Optional[int] = None
    attacker_track_ids: list[int] = Field(default_factory=list)
    defender_track_ids: list[int] = Field(default_factory=list)
    second_last_defender_track_id: Optional[int] = None
    ball_source: str = "detector"
    ball_manual_point: Optional[list[float]] = None
    notes: str = ""
    confirmed: bool = False


class PointPayload(BaseModel):
    xy: list[float]
    frame_index: Optional[int] = None


def _persist_calib() -> None:
    if STATE.review_dir is None:
        return
    STATE.review_dir.mkdir(parents=True, exist_ok=True)
    (STATE.review_dir / "manual_homography.json").write_text(
        json.dumps(
            {
                "image_pts": STATE.image_pts,
                "pitch_pts": STATE.pitch_pts,
                "pairs": STATE.calib_pairs,
                "confirmed": STATE.calib_confirmed,
                "homography_status": STATE.H_status,
                "updated_at": _utcnow(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_persisted_calib() -> None:
    if STATE.review_dir is None:
        return
    path = STATE.review_dir / "manual_homography.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        STATE.image_pts = list(data.get("image_pts") or [])
        STATE.pitch_pts = list(data.get("pitch_pts") or [])
        STATE.calib_confirmed = bool(data.get("confirmed"))
        # Drop old landmark-only saves
        if not STATE.image_pts and data.get("pairs") and "landmark_id" in str(data.get("pairs")):
            STATE.image_pts, STATE.pitch_pts, STATE.calib_confirmed = [], [], False
        _sync_pairs_from_buffers()
    except Exception as exc:  # noqa: BLE001
        logger.warning("load calib failed: %s", exc)


def create_app() -> FastAPI:
    app = FastAPI(title="Offside Review UI", version="0.3.0")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "ok": True,
            "service": "offside-review",
            "ready": STATE.ready,
            "total_frames": STATE.total_frames,
            "fps": STATE.fps,
            "setup_phase": _setup_phase(),
        }

    @app.get("/api/review/status")
    def review_status() -> dict[str, Any]:
        cfg = _load_yaml(STATE.config_path)
        review = cfg.get("review") or {}
        roles = cfg.get("manual_role_confirmations") or {}
        ctx = _load_offside_context()
        return {
            "ready": STATE.ready,
            "total_frames": STATE.total_frames,
            "fps": STATE.fps,
            "jump_marks": _jump_marks(STATE.total_frames),
            "review_confirmed": bool(review.get("confirmed")),
            "critical_frame": cfg.get("critical_frame"),
            "attack_direction": ctx.get("attack_direction") or cfg.get("attack_direction"),
            "attacker_track_ids": sorted(ctx.get("attacker_ids") or []),
            "defender_track_ids": sorted(ctx.get("defender_ids") or []),
            "passer_track_id": ctx.get("passer_id") or cfg.get("passer_track_id"),
            "second_last_defender_track_id": roles.get("second_last_defender_track_id"),
            "setup_phase": _setup_phase(),
            "calib_confirmed": STATE.calib_confirmed,
            "image_pts": len(STATE.image_pts),
            "pitch_pts": len(STATE.pitch_pts),
            "homography_status": STATE.H_status,
            "pitch_layout": {**STATE.pitch_draw, "length_cm": float(PITCH.length), "width_cm": float(PITCH.width)},
        }

    @app.get("/api/review/calib")
    def calib_get() -> dict[str, Any]:
        return {
            "phase": _setup_phase(),
            "image_pts": len(STATE.image_pts),
            "pitch_pts": len(STATE.pitch_pts),
            "confirmed": STATE.calib_confirmed,
            "homography_status": STATE.H_status,
        }

    @app.post("/api/review/calib/image")
    def calib_image(payload: PointPayload) -> dict[str, Any]:
        if STATE.calib_confirmed or len(STATE.image_pts) >= 4:
            return {"ok": False, "phase": _setup_phase()}
        if len(payload.xy) != 2:
            raise HTTPException(400, "xy=[x,y]")
        STATE.image_pts.append([float(payload.xy[0]), float(payload.xy[1])])
        _sync_pairs_from_buffers()
        _persist_calib()
        return {"ok": True, "phase": _setup_phase(), "image_pts": len(STATE.image_pts), "pitch_pts": len(STATE.pitch_pts)}

    @app.post("/api/review/calib/pitch")
    def calib_pitch(payload: PointPayload) -> dict[str, Any]:
        if STATE.calib_confirmed or len(STATE.image_pts) < 4 or len(STATE.pitch_pts) >= 4:
            return {"ok": False, "phase": _setup_phase()}
        if len(payload.xy) != 2:
            raise HTTPException(400, "xy=[x,y] pitch cm")
        STATE.pitch_pts.append([float(payload.xy[0]), float(payload.xy[1])])
        _sync_pairs_from_buffers()
        _persist_calib()
        return {"ok": True, "phase": _setup_phase(), "image_pts": len(STATE.image_pts), "pitch_pts": len(STATE.pitch_pts)}

    @app.post("/api/review/calib/confirm")
    def calib_confirm() -> dict[str, Any]:
        if len(STATE.image_pts) < 4 or len(STATE.pitch_pts) < 4:
            return {"ok": False, "phase": _setup_phase()}
        STATE.calib_confirmed = True
        _sync_pairs_from_buffers()
        _persist_calib()
        return {"ok": True, "phase": "done", "homography_status": STATE.H_status}

    @app.post("/api/review/calib/reset")
    def calib_reset() -> dict[str, Any]:
        STATE.image_pts, STATE.pitch_pts, STATE.calib_pairs = [], [], []
        STATE.calib_confirmed = False
        STATE.H, STATE.H_status, STATE.H_inliers = None, "unavailable", 0
        _persist_calib()
        return {"ok": True, "phase": "image"}

    @app.get("/api/review/frame/{frame_index}")
    def review_frame(
        frame_index: int,
        attack_direction: Optional[str] = None,
        attacker_ids: Optional[str] = None,
        defender_ids: Optional[str] = None,
        passer_id: Optional[int] = None,
        second_last_id: Optional[int] = None,
    ) -> dict[str, Any]:
        if not STATE.ready or STATE.video_path is None:
            raise HTTPException(503, "not ready")
        if frame_index < 1 or frame_index > STATE.total_frames:
            raise HTTPException(400, "out of range")
        row = STATE.frame_index.get(frame_index) or {
            "frame_index": frame_index,
            "timestamp_seconds": (frame_index - 1) / STATE.fps,
            "players": [],
            "ball": None,
        }
        frame = read_video_frame(STATE.video_path, frame_index - 1)
        players_xy, ball_xy = _project_players(row)
        ctx = _load_offside_context()
        # Live UI overrides (before / without save)
        if attack_direction in {"left_to_right", "right_to_left"}:
            ctx["attack_direction"] = attack_direction
        if attacker_ids:
            ctx["attacker_ids"] = {int(x) for x in attacker_ids.split(",") if x.strip()}
        if defender_ids:
            ctx["defender_ids"] = {int(x) for x in defender_ids.split(",") if x.strip()}
        if passer_id is not None:
            ctx["passer_id"] = int(passer_id)
        if second_last_id is not None:
            ctx["second_last_override"] = int(second_last_id)
        offside = _judge_offside(players_xy, ctx)
        vis = _draw_frame(frame, row, STATE.fps, offside=offside)
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            raise HTTPException(500, "jpeg")
        pitch = draw_pitch_2d(
            players_xy if STATE.H is not None else [],
            ball_xy if STATE.H is not None else None,
            [tuple(p) for p in STATE.pitch_pts],
            out_w=STATE.pitch_draw["out_w"],
            out_h=STATE.pitch_draw["out_h"],
            offside=offside if offside.get("ready") else None,
        )
        ok2, buf2 = cv2.imencode(".jpg", pitch, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok2:
            raise HTTPException(500, "pitch jpeg")
        return {
            "frame_index": frame_index,
            "timestamp_seconds": row.get("timestamp_seconds"),
            "players": row.get("players") or [],
            "ball": row.get("ball"),
            "image_jpeg_base64": base64.b64encode(buf.tobytes()).decode("ascii"),
            "pitch_jpeg_base64": base64.b64encode(buf2.tobytes()).decode("ascii"),
            "setup_phase": _setup_phase(),
            "calib_confirmed": STATE.calib_confirmed,
            "homography_status": STATE.H_status,
            "offside": {
                "ready": bool(offside.get("ready")),
                "verdict": offside.get("verdict"),
                "line_x_cm": offside.get("line_x"),
                "second_last_defender_track_id": offside.get("second_last_id"),
                "offside_attacker_ids": offside.get("offside_attacker_ids") or [],
            },
            "pitch_layout": {**STATE.pitch_draw, "length_cm": float(PITCH.length), "width_cm": float(PITCH.width)},
        }

    @app.post("/api/review/save")
    def review_save(payload: SavePayload) -> dict[str, Any]:
        cfg = _load_yaml(STATE.config_path)
        if STATE.video_path is not None:
            cfg["video"] = str(STATE.video_path)
        missing = []
        if payload.critical_frame is None and payload.critical_time_seconds is None:
            missing.append("critical_frame")
        if payload.attack_direction not in {"left_to_right", "right_to_left"}:
            missing.append("attack_direction")
        if not payload.attacker_track_ids:
            missing.append("attacker_track_ids")
        if not payload.defender_track_ids:
            missing.append("defender_track_ids")
        if payload.confirmed and missing:
            return {
                "ok": False,
                "missing_fields": missing,
                "message": "缺少必要项",
                "result_panel": {"title": "确认未完成", "status": "incomplete", "missing_fields": missing},
            }
        if payload.critical_frame is not None:
            cfg["critical_frame"] = int(payload.critical_frame)
            cfg["critical_time_seconds"] = (int(payload.critical_frame) - 1) / max(STATE.fps, 1e-6)
            cfg["critical_time_source"] = "human_confirmed" if payload.confirmed else "human_draft"
        if payload.attack_direction in {"left_to_right", "right_to_left"}:
            cfg["attack_direction"] = payload.attack_direction
        cfg["passer_track_id"] = payload.passer_track_id
        cfg["manual_role_confirmations"] = {
            "attacker_track_ids": list(payload.attacker_track_ids),
            "defender_track_ids": list(payload.defender_track_ids),
            "second_last_defender_track_id": payload.second_last_defender_track_id,
        }
        ball = cfg.get("ball") or {}
        ball["source"] = payload.ball_source
        if payload.ball_source == "manual_click":
            ball["manual_point"] = payload.ball_manual_point
        cfg["ball"] = ball
        cfg["homography"] = {"status": STATE.H_status, "confirmed": STATE.calib_confirmed, "pairs": STATE.calib_pairs}
        cfg["review"] = {
            "confirmed": bool(payload.confirmed),
            "confirmed_at": _utcnow() if payload.confirmed else None,
            "notes": payload.notes,
        }
        _save_yaml(STATE.config_path, cfg)
        panel = {
            "title": "确认结果" if payload.confirmed else "草稿",
            "status": "confirmed" if payload.confirmed else "draft",
            "formal_verdict": None,
            "formal_verdict_note": "人工确认摘要，非最终 OFFSIDE/ONSIDE。",
            "critical_frame": cfg.get("critical_frame"),
            "attack_direction": cfg.get("attack_direction"),
            "passer_track_id": cfg.get("passer_track_id"),
            "attacker_track_ids": cfg["manual_role_confirmations"]["attacker_track_ids"],
            "defender_track_ids": cfg["manual_role_confirmations"]["defender_track_ids"],
            "ball": cfg.get("ball"),
        }
        if STATE.review_dir is not None:
            (STATE.review_dir / "CONFIRMATION_RESULT.json").write_text(
                json.dumps(panel, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        return {"ok": True, "result_panel": panel, "message": "已保存"}

    @app.get("/api/review/result")
    def review_result() -> dict[str, Any]:
        path = STATE.review_dir / "CONFIRMATION_RESULT.json" if STATE.review_dir else None
        if path and path.exists():
            return {"ok": True, "result_panel": json.loads(path.read_text(encoding="utf-8"))}
        return {"ok": False, "result_panel": {"title": "尚未确认", "status": "empty"}}

    @app.post("/api/review/clear")
    def review_clear() -> dict[str, Any]:
        return _clear_review_judgment()

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return REVIEW_HTML

    return app


REVIEW_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>半自动越位复核</title>
<style>
body{font-family:Arial,sans-serif;margin:0;background:#111;color:#eee}
.wrap{display:grid;grid-template-columns:2fr 1fr;gap:12px;padding:12px}
.views{display:grid;grid-template-columns:1fr 1fr;gap:8px}
img.view{width:100%;background:#000;border:1px solid #333}
.panel{background:#1b1b1b;padding:12px;border-radius:8px;max-height:92vh;overflow:auto}
button,select,input{margin:3px;padding:6px 8px}
.row{margin:6px 0}
.muted{color:#aaa;font-size:12px}
.tag{display:inline-block;padding:2px 6px;background:#333;border-radius:4px;margin:2px}
#resultBox{margin-top:10px;padding:10px;border:2px solid #2a5;border-radius:8px;background:#102318;display:none}
#resultBox.err{border-color:#a33;background:#2a1515}
#confirmGeo{display:none;position:absolute;right:10px;bottom:10px;padding:4px 10px;font-size:12px;opacity:.55;background:#333;color:#ddd;border:1px solid #555;border-radius:4px;cursor:pointer}
#confirmGeo:hover{opacity:1}
.viewwrap{position:relative}
</style>
</head>
<body>
<div class="wrap">
  <div>
    <div class="views">
      <div class="viewwrap"><img id="view" class="view" alt="video"/></div>
      <div class="viewwrap">
        <img id="pitch" class="view" alt="pitch"/>
        <button id="confirmGeo" onclick="confirmGeo()">确认</button>
      </div>
    </div>
    <div class="row" id="jumpRow"></div>
    <div class="row">
      <button onclick="step(-5)">-5</button>
      <button onclick="step(-1)">-1</button>
      <button onclick="togglePlay()">Play/Pause</button>
      <button onclick="step(1)">+1</button>
      <button onclick="step(5)">+5</button>
      <button onclick="setRate(0.25)">0.25x</button>
      <button onclick="setRate(0.5)">0.5x</button>
      <button onclick="setRate(1)">1x</button>
      <input id="jumpIn" type="number" min="1" style="width:90px"/>
      <button onclick="jump(Number(document.getElementById('jumpIn').value))">Go</button>
    </div>
    <div id="status" class="muted"></div>
    <div id="resultBox"></div>
  </div>
  <div class="panel">
    <h3>越位确认</h3>
    <div class="row">当前帧 <b id="fi">-</b> · <b id="tm">-</b>s</div>
    <div class="row">关键帧 <b id="critShow">未设置</b></div>
    <div class="row">
      进攻方向
      <select id="adir">
        <option value="">(未设置)</option>
        <option value="left_to_right">left_to_right</option>
        <option value="right_to_left">right_to_left</option>
      </select>
    </div>
    <div class="row"><button onclick="markCritical()">设为关键帧</button></div>
    <div class="row">Track ID
      <select id="selTrack"></select>
      <button onclick="setRole('attacker')">进攻</button>
      <button onclick="setRole('defender')">防守</button>
      <button onclick="setRole('ignore')">忽略</button>
    </div>
    <div class="row">Passer <input id="passer" type="number" style="width:80px"/></div>
    <div class="row">倒数第二防守 <input id="sld" type="number" style="width:80px"/></div>
    <div class="row">进攻 <span id="att" class="tag">[]</span></div>
    <div class="row">防守 <span id="def" class="tag">[]</span></div>
    <div class="row">球 <span id="ballInfo" class="muted">detector</span>
      <label class="muted"><input type="checkbox" id="manualBall"/> 点画面记球</label>
    </div>
    <div class="row"><textarea id="notes" rows="2" style="width:95%"></textarea></div>
    <div class="row">
      <button onclick="save(false)">保存草稿</button>
      <button onclick="save(true)" style="background:#2a5">确认并保存</button>
      <button onclick="clearJudgment()" style="background:#533">清除判断</button>
    </div>
    <div id="saveMsg" class="muted"></div>
  </div>
</div>
<script>
let frame=1,total=1,fps=30,playing=false,rate=1,timer=null,phase='image',layout=null;
let attackers=[],defenders=[],critical=null,ballSource='detector',ballPoint=null,players=[];
function showResult(panel,ok){
  const box=document.getElementById('resultBox');
  if(!panel || panel.status==='empty'){ box.style.display='none'; box.innerHTML=''; return; }
  box.style.display='block'; box.className=ok?'':'err';
  const lines=[panel.title||'', '状态:'+(panel.status||'-'), '关键帧:'+(panel.critical_frame??'-'),
    '方向:'+(panel.attack_direction??'-'), 'Passer:'+(panel.passer_track_id??'-'),
    '进攻:'+JSON.stringify(panel.attacker_track_ids||[]), '防守:'+JSON.stringify(panel.defender_track_ids||[]),
    panel.formal_verdict_note||'', (panel.missing_fields?('缺少:'+panel.missing_fields.join(',')):'')];
  box.innerHTML=lines.filter(Boolean).join('<br/>');
}
function showOffside(o){
  const box=document.getElementById('resultBox');
  if(!o||!o.ready){
    // keep confirmation text if any; strip only live OFFSIDE banner by reloading from empty overlay
    return;
  }
  box.style.display='block';
  box.className = (o.verdict==='OFFSIDE') ? 'err' : '';
  const extra = [
    '<b style="font-size:22px">'+(o.verdict||'')+'</b>',
    '越位线: 倒数第二防守 ID'+(o.second_last_defender_track_id??'-'),
    '越位进攻:'+JSON.stringify(o.offside_attacker_ids||[]),
  ];
  box.innerHTML = extra.join('<br/>');
}
function clearResultBox(){
  const box=document.getElementById('resultBox');
  box.style.display='none'; box.innerHTML=''; box.className='';
}
function updateCrit(){document.getElementById('critShow').textContent=critical==null?'未设置':String(critical);}
function updateConfirmBtn(){
  const b=document.getElementById('confirmGeo');
  b.style.display = (phase==='confirm') ? 'block' : 'none';
}
function imgClickXY(ev,img){
  const rect=img.getBoundingClientRect();
  return [(ev.clientX-rect.left)*(img.naturalWidth/rect.width), (ev.clientY-rect.top)*(img.naturalHeight/rect.height)];
}
function pitchClickToCm(ev,img){
  const [x,y]=imgClickXY(ev,img);
  const pad=layout.pad, ow=layout.out_w, oh=layout.out_h;
  const sx=(ow-2*pad)/layout.length_cm, sy=(oh-2*pad)/layout.width_cm;
  return [(x-pad)/sx, (y-pad)/sy];
}
function roleQuery(){
  const q=new URLSearchParams();
  const ad=document.getElementById('adir').value;
  if(ad) q.set('attack_direction', ad);
  if(attackers.length) q.set('attacker_ids', attackers.join(','));
  if(defenders.length) q.set('defender_ids', defenders.join(','));
  const p=Number(document.getElementById('passer').value); if(Number.isFinite(p)&&p) q.set('passer_id', String(p));
  const s=Number(document.getElementById('sld').value); if(Number.isFinite(s)&&s) q.set('second_last_id', String(s));
  return q.toString();
}
async function init(){
  const s=await (await fetch('/api/review/status')).json();
  total=s.total_frames; fps=s.fps; phase=s.setup_phase; layout=s.pitch_layout;
  document.getElementById('jumpRow').innerHTML=(s.jump_marks||[]).map(n=>`<button onclick="jump(${n})">${n}</button>`).join(' ');
  // Always start clean: do not auto-restore old roles / OFFSIDE on re-entry.
  attackers=[]; defenders=[]; critical=null; ballSource='detector'; ballPoint=null;
  document.getElementById('adir').value='';
  document.getElementById('passer').value='';
  document.getElementById('sld').value='';
  updateCrit(); updateConfirmBtn();
  document.getElementById('status').textContent='';
  clearResultBox();
  await load(1);
}
async function clearJudgment(){
  const r=await (await fetch('/api/review/clear',{method:'POST'})).json();
  attackers=[]; defenders=[]; critical=null; ballSource='detector'; ballPoint=null;
  document.getElementById('adir').value='';
  document.getElementById('passer').value='';
  document.getElementById('sld').value='';
  document.getElementById('notes').value='';
  document.getElementById('manualBall').checked=false;
  document.getElementById('ballInfo').textContent='detector';
  updateCrit();
  clearResultBox();
  document.getElementById('saveMsg').textContent=r.message||'已清除';
  await load(frame);
}
async function load(fi){
  frame=Math.max(1,Math.min(total,fi|0));
  const qs=roleQuery();
  const r=await (await fetch(`/api/review/frame/${frame}`+(qs?('?'+qs):''))).json();
  document.getElementById('view').src='data:image/jpeg;base64,'+r.image_jpeg_base64;
  document.getElementById('pitch').src='data:image/jpeg;base64,'+r.pitch_jpeg_base64;
  phase=r.setup_phase; layout=r.pitch_layout; updateConfirmBtn();
  document.getElementById('fi').textContent=r.frame_index;
  document.getElementById('tm').textContent=(r.timestamp_seconds||0).toFixed(3);
  players=r.players||[];
  const sel=document.getElementById('selTrack'); sel.innerHTML='';
  players.forEach(p=>{const o=document.createElement('option');o.value=p.track_id;o.textContent='ID'+p.track_id;sel.appendChild(o);});
  document.getElementById('att').textContent=JSON.stringify(attackers);
  document.getElementById('def').textContent=JSON.stringify(defenders);
  document.getElementById('view').style.cursor = (phase==='image') ? 'crosshair' : 'default';
  document.getElementById('pitch').style.cursor = (phase==='pitch') ? 'crosshair' : 'default';
  if(r.offside && r.offside.ready) showOffside(r.offside);
  else if(!qs) clearResultBox();
}
function step(d){load(frame+d);} function jump(fi){if(fi) load(fi);}
function setRate(r){rate=r; if(playing){stop();play();}}
function play(){playing=true; timer=setInterval(()=>{if(frame>=total){stop();return;} load(frame+1);},1000/(fps*rate));}
function stop(){playing=false; if(timer) clearInterval(timer); timer=null;}
function togglePlay(){if(playing) stop(); else play();}
function markCritical(){critical=frame; updateCrit();}
function setRole(role){
  const id=Number(document.getElementById('selTrack').value); if(!Number.isFinite(id)) return;
  attackers=attackers.filter(x=>x!==id); defenders=defenders.filter(x=>x!==id);
  if(role==='attacker') attackers.push(id); if(role==='defender') defenders.push(id);
  document.getElementById('att').textContent=JSON.stringify(attackers);
  document.getElementById('def').textContent=JSON.stringify(defenders);
  load(frame);
}
document.getElementById('adir').addEventListener('change', ()=>load(frame));
document.getElementById('passer').addEventListener('change', ()=>load(frame));
document.getElementById('sld').addEventListener('change', ()=>load(frame));
document.getElementById('view').addEventListener('click', async (ev)=>{
  if(document.getElementById('manualBall').checked){
    const xy=imgClickXY(ev,ev.target); ballSource='manual_click'; ballPoint=xy;
    document.getElementById('ballInfo').textContent='manual';
    return;
  }
  if(phase!=='image') return;
  const xy=imgClickXY(ev,ev.target);
  await fetch('/api/review/calib/image',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({xy,frame_index:frame})});
  await load(frame);
});
document.getElementById('pitch').addEventListener('click', async (ev)=>{
  if(phase!=='pitch') return;
  const xy=pitchClickToCm(ev,ev.target);
  await fetch('/api/review/calib/pitch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({xy,frame_index:frame})});
  await load(frame);
});
async function confirmGeo(){
  const r=await (await fetch('/api/review/calib/confirm',{method:'POST'})).json();
  phase=r.phase||'done'; updateConfirmBtn(); await load(frame);
}
async function save(confirmed){
  if(confirmed && critical==null){critical=frame; updateCrit();}
  const body={critical_frame:critical, attack_direction:document.getElementById('adir').value||null,
    passer_track_id:Number(document.getElementById('passer').value)||null,
    attacker_track_ids:attackers, defender_track_ids:defenders,
    second_last_defender_track_id:Number(document.getElementById('sld').value)||null,
    ball_source:ballSource, ball_manual_point:ballPoint,
    notes:document.getElementById('notes').value||'', confirmed:!!confirmed};
  const r=await (await fetch('/api/review/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  document.getElementById('saveMsg').textContent=r.message||'';
  showResult(r.result_panel, r.ok!==false && !(r.missing_fields&&r.missing_fields.length));
  await load(frame);
}
init();
</script>
</body>
</html>
"""


def run_review_server(
    *,
    input_path: str,
    discovery_dir: str,
    host: str = "0.0.0.0",
    port: int = 6006,
    device: str = "cuda",
    config_path: str | None = None,
) -> None:
    import uvicorn

    video = Path(input_path).resolve()
    discovery = Path(discovery_dir).resolve()
    # Default: sibling review/ next to discovery/; override with DEMO_OFFSIDE_REVIEW_DIR.
    import os

    review_dir = Path(
        os.environ.get(
            "DEMO_OFFSIDE_REVIEW_DIR",
            str(discovery.parent / "review"),
        )
    )
    review_dir.mkdir(parents=True, exist_ok=True)
    STATE.video_path = video
    STATE.discovery_dir = discovery
    STATE.review_dir = review_dir
    if config_path:
        STATE.config_path = Path(config_path).resolve()

    cap = cv2.VideoCapture(str(video))
    STATE.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    STATE.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    STATE.frame_index = load_existing_frame_index(review_dir, video)
    STATE.total_frames = max(STATE.total_frames, len(STATE.frame_index))
    _load_persisted_calib()
    # Prefer clean first-open experience unless already confirmed
    if not STATE.calib_confirmed:
        # keep partial progress if any; otherwise empty
        pass
    STATE.ready = True
    STATE.message = "ready"
    uvicorn.run(create_app(), host=host, port=port, log_level="info")
