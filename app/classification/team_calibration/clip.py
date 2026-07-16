"""Local-video clip processing for supervised team calibration.

The live pipeline remains available for normal match analysis.  Calibration
uses this small offline service so an operator can pause and seek a stable,
fully processed clip without restarting the live source.
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
import torch

from app.classification.team_calibration.session import CalibrationState, TeamCalibrationSession
from app.classification.team_calibration.types import CalibrationLabel
from app.classification.team_calibration.quality import CropQualityAssessor
from app.classification.team_calibration.roi import JerseyROIExtractor
from app.vision.core import VisionCore

logger = logging.getLogger(__name__)

MAX_CLIP_MS = 60_000
RECOMMENDED_CLIP_MS = 30_000


@dataclass(frozen=True)
class SourceInfo:
    duration_ms: int
    fps: float
    width: int
    height: int
    frame_count: int


@dataclass
class ClipJob:
    job_id: str
    clip_id: str
    source_path: str
    start_ms: int
    end_ms: int
    clip_path: str
    metadata_path: str
    status: str = "processing"
    error: Optional[str] = None


class CalibrationClipService:
    """Manage one local source preview and one offline calibration clip."""

    def __init__(
        self,
        session: TeamCalibrationSession,
        *,
        temp_root: Optional[str | Path] = None,
        vision_core_factory: Optional[Callable[..., VisionCore]] = None,
    ) -> None:
        self.session = session
        self._lock = threading.RLock()
        self._source_path: Optional[Path] = None
        self._source_info: Optional[SourceInfo] = None
        self._config: Any = None
        self._job: Optional[ClipJob] = None
        self._worker: Optional[threading.Thread] = None
        self._temp_root = Path(temp_root or (Path(tempfile.gettempdir()) / "sc-team-calibration"))
        self._temp_root.mkdir(parents=True, exist_ok=True)
        self._vision_core_factory = vision_core_factory or VisionCore
        self._roi_extractor = JerseyROIExtractor()
        self._quality_assessor = CropQualityAssessor()

    @property
    def source_path(self) -> Optional[Path]:
        with self._lock:
            return self._source_path

    @property
    def source_info(self) -> Optional[SourceInfo]:
        with self._lock:
            return self._source_info

    @property
    def job(self) -> Optional[ClipJob]:
        with self._lock:
            return self._job

    def preview(
        self,
        *,
        source_path: str,
        match_id: str,
        camera_id: str,
        device: str,
        bundle_path: Optional[str] = None,
        config: Any = None,
    ) -> dict[str, Any]:
        path = Path(source_path).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"local video source does not exist: {path}")
        info = inspect_video(path)
        with self._lock:
            if self.session.state != CalibrationState.IDLE:
                raise RuntimeError("reset the current calibration clip before loading another source")
            self._source_path = path
            self._source_info = info
            self._config = config
            self._job = None
            return self.session.begin_source_preview(
                match_id=match_id,
                camera_id=camera_id,
                source_url="/api/team-calibration/clip/source",
                bundle_path=bundle_path,
                device=device,
            )

    def start_clip(self, start_ms: int) -> dict[str, Any]:
        with self._lock:
            if self._source_info is None:
                raise RuntimeError("source preview is required before selecting a clip")
            if int(start_ms) < 0 or int(start_ms) >= self._source_info.duration_ms:
                raise ValueError("clip start time must be inside the source video")
            return self.session.begin_clip_selecting(int(start_ms))

    def finish_clip(self, end_ms: int) -> dict[str, Any]:
        with self._lock:
            if self._source_path is None or self._source_info is None:
                raise RuntimeError("source preview is required before processing a clip")
            if self._job is not None and self._worker is not None and self._worker.is_alive():
                raise RuntimeError("a calibration clip is already processing")
            start_ms = self.session.snapshot().get("clip_start_ms")
            if start_ms is None:
                raise RuntimeError("clip start time has not been selected")
            validate_clip_range(int(start_ms), int(end_ms), self._source_info.duration_ms)
            clip_id = uuid.uuid4().hex[:12]
            job_id = uuid.uuid4().hex[:12]
            directory = self._temp_root / clip_id
            directory.mkdir(parents=True, exist_ok=False)
            job = ClipJob(
                job_id=job_id,
                clip_id=clip_id,
                source_path=str(self._source_path),
                start_ms=int(start_ms),
                end_ms=int(end_ms),
                clip_path=str(directory / "review.mp4"),
                metadata_path=str(directory / "metadata.json"),
            )
            self._job = job
            duration_ms = int(end_ms) - int(start_ms)
            self.session.begin_processing(
                clip_id=clip_id,
                job_id=job_id,
                start_ms=int(start_ms),
                end_ms=int(end_ms),
                duration_ms=duration_ms,
                review_video_url="/api/team-calibration/clip/video",
                metadata_url="/api/team-calibration/clip/metadata",
            )
            self._worker = threading.Thread(
                target=self._process_job,
                args=(job,),
                daemon=True,
                name=f"calibration-clip-{clip_id}",
            )
            self._worker.start()
            return self.session.snapshot()

    def status(self) -> dict[str, Any]:
        return self.session.snapshot()

    def metadata(self) -> dict[str, Any]:
        with self._lock:
            if self._job is None:
                raise FileNotFoundError("no calibration clip has been processed")
            path = Path(self._job.metadata_path)
        if not path.is_file():
            raise FileNotFoundError("calibration clip metadata is not ready")
        return json.loads(path.read_text(encoding="utf-8"))

    def review_video_path(self) -> Path:
        with self._lock:
            if self._job is None:
                raise FileNotFoundError("no calibration clip has been processed")
            path = Path(self._job.clip_path)
        if not path.is_file():
            raise FileNotFoundError("calibration clip video is not ready")
        return path

    def label_track(self, track_id: int, label: CalibrationLabel | str) -> dict[str, Any]:
        with self._lock:
            if self._job is None:
                return self.session.label_track(track_id, label)
            metadata = self.metadata()
            job = self._job
        value = label if isinstance(label, CalibrationLabel) else CalibrationLabel(str(label))
        if value == CalibrationLabel.IGNORE:
            return self.session.label_track(track_id, value)

        track = next(
            (item for item in metadata.get("tracks", []) if int(item["track_id"]) == int(track_id)),
            None,
        )
        if track is None:
            raise ValueError(f"unknown clip track: {track_id}")
        observations = [
            {
                **player,
                "frame_index": frame["frame_index"],
            }
            for frame in metadata.get("frames", [])
            for player in frame.get("players", [])
            if int(player["track_id"]) == int(track_id)
            and bool(player.get("quality_accepted", False))
        ]
        observations.sort(key=lambda item: int(item["frame_index"]))
        max_samples = self.session.max_samples_per_track
        if len(observations) > max_samples:
            indices = np.linspace(0, len(observations) - 1, max_samples, dtype=int)
            observations = [observations[int(index)] for index in indices]

        samples: list[tuple[np.ndarray, float, int]] = []
        capture = cv2.VideoCapture(job.clip_path)
        try:
            for observation in observations:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(observation["frame_index"]))
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                roi = self._roi_extractor.extract(frame, observation["bbox"])
                quality = self._quality_assessor.assess(
                    roi,
                    detection_confidence=float(observation.get("confidence", 0.0)),
                )
                if quality.accepted:
                    samples.append((roi, quality.score, int(observation["frame_index"])))
        finally:
            capture.release()
        return self.session.label_track(track_id, value, samples=samples)

    def reset(self) -> dict[str, Any]:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                raise RuntimeError("cannot reset while a calibration clip is processing")
            job = self._job
            self._source_path = None
            self._source_info = None
            self._job = None
            self._worker = None
            if job is not None:
                shutil.rmtree(Path(job.clip_path).parent, ignore_errors=True)
            return self.session.reset()

    def _process_job(self, job: ClipJob) -> None:
        try:
            metadata = self._run_processing(job)
            Path(job.metadata_path).write_text(
                json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            tracks = metadata["tracks"]
            self.session.complete_processing(
                tracks=tracks,
                observed_frames=len(metadata["frames"]),
                last_frame_index=(metadata["frames"][-1]["frame_index"] if metadata["frames"] else None),
            )
            job.status = "completed"
        except Exception as exc:  # pragma: no cover - exercised by integration tests
            job.status = "failed"
            job.error = str(exc)
            logger.exception("Calibration clip processing failed")
            self.session.fail_processing(str(exc))

    def _run_processing(self, job: ClipJob) -> dict[str, Any]:
        source = cv2.VideoCapture(job.source_path)
        if not source.isOpened():
            raise ValueError(f"unable to open video source: {job.source_path}")
        fps = max(float(source.get(cv2.CAP_PROP_FPS) or 0.0), 1.0)
        width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        start_frame = max(0, int(round(job.start_ms * fps / 1000.0)))
        end_frame = max(start_frame, int(np.ceil(job.end_ms * fps / 1000.0)))
        expected_frames = max(1, end_frame - start_frame)
        source.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        writer = cv2.VideoWriter(
            job.clip_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            source.release()
            raise RuntimeError("unable to create review video")

        config = getattr(self, "_config", None)
        core = self._vision_core_factory(
            device=self.session.device,
            fps=fps,
            player_model_path=getattr(config, "player_model_path", "assets/weights/yolo11s.pt"),
            pitch_model_path=getattr(config, "pitch_model_path", "assets/weights/football-pitch-detection.pt"),
            camera_calibration_path=getattr(config, "camera_calibration_path", None),
            enable_undistortion=getattr(config, "enable_undistortion", True),
            calibration_alpha=getattr(config, "calibration_alpha", 0.0),
            pitch_detection_interval=getattr(config, "pitch_detection_interval", 5),
            imgsz=getattr(config, "imgsz", 640),
            enable_pitch=False,
            inference_backend=getattr(config, "inference_backend", "auto"),
        )
        core.load_models()
        frames: list[dict[str, Any]] = []
        track_summaries: dict[int, dict[str, Any]] = {}
        processed = 0
        try:
            while processed < expected_frames:
                ok, frame = source.read()
                if not ok or frame is None:
                    break
                vision = _process_core_frame(core, frame, processed)
                output = vision.undistorted_frame.copy()
                frame_players: list[dict[str, Any]] = []
                detections = vision.tracked_detections
                for index in range(len(detections)):
                    track_id = int(detections.tracker_id[index]) if detections.tracker_id is not None else index
                    bbox = [float(value) for value in detections.xyxy[index]]
                    confidence = float(detections.confidence[index]) if detections.confidence is not None else 0.0
                    roi = self._roi_extractor.extract(vision.undistorted_frame, bbox)
                    quality = self._quality_assessor.assess(roi, detection_confidence=confidence)
                    frame_players.append(
                        {
                            "track_id": track_id,
                            "bbox": bbox,
                            "confidence": confidence,
                            "quality_accepted": quality.accepted,
                            "quality_score": round(quality.score, 4),
                        }
                    )
                    summary = track_summaries.setdefault(
                        track_id,
                        {
                            "track_id": track_id,
                            "first_timestamp_ms": None,
                            "last_timestamp_ms": None,
                            "observation_count": 0,
                            "quality_observation_count": 0,
                            "representative_frame_index": processed,
                            "representative_timestamp_ms": 0,
                            "representative_bbox": bbox,
                            "representative_quality_score": 0.0,
                        },
                    )
                    timestamp_ms = int(round(processed * 1000.0 / fps))
                    summary["first_timestamp_ms"] = (
                        timestamp_ms if summary["first_timestamp_ms"] is None
                        else summary["first_timestamp_ms"]
                    )
                    summary["last_timestamp_ms"] = timestamp_ms
                    summary["observation_count"] += 1
                    if quality.accepted:
                        summary["quality_observation_count"] += 1
                    if quality.score >= float(summary["representative_quality_score"]):
                        summary["representative_frame_index"] = processed
                        summary["representative_timestamp_ms"] = timestamp_ms
                        summary["representative_bbox"] = bbox
                        summary["representative_quality_score"] = round(quality.score, 4)
                    _draw_overlay(output, bbox, track_id)
                writer.write(output)
                frames.append(
                    {
                        "frame_index": processed,
                        "timestamp_ms": int(round(processed * 1000.0 / fps)),
                        "players": frame_players,
                    }
                )
                processed += 1
                if processed == 1 or processed % max(1, int(fps)) == 0:
                    self.session.update_processing(
                        progress=min(processed / expected_frames, 0.99),
                        observed_frames=processed,
                    )
        finally:
            writer.release()
            source.release()
        if not frames:
            raise RuntimeError("selected clip contains no decodable frames")
        return {
            "schema_version": 1,
            "clip_id": job.clip_id,
            "fps": fps,
            "width": width,
            "height": height,
            "frame_count": len(frames),
            "duration_ms": int(round(len(frames) * 1000.0 / fps)),
            "frames": frames,
            "tracks": sorted(
                track_summaries.values(),
                key=lambda item: (-item["quality_observation_count"], -item["observation_count"], item["track_id"]),
            ),
        }


def inspect_video(path: str | Path) -> SourceInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"unable to open local video source: {path}")
    try:
        fps = max(float(capture.get(cv2.CAP_PROP_FPS) or 0.0), 1.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration_ms = int(round(frame_count * 1000.0 / fps)) if frame_count else 0
        return SourceInfo(duration_ms, fps, width, height, frame_count)
    finally:
        capture.release()


def validate_clip_range(start_ms: int, end_ms: int, source_duration_ms: int) -> None:
    start_ms = int(start_ms)
    end_ms = int(end_ms)
    if start_ms < 0 or end_ms < 0:
        raise ValueError("clip times must be non-negative")
    if end_ms <= start_ms:
        raise ValueError("clip end time must be after clip start time")
    if end_ms > int(source_duration_ms):
        raise ValueError("clip end time exceeds source duration")
    if end_ms - start_ms > MAX_CLIP_MS:
        raise ValueError(f"clip length cannot exceed {MAX_CLIP_MS} ms")


def _process_core_frame(core: VisionCore, frame: np.ndarray, frame_index: int):
    with torch.inference_mode():
        return core.process(frame, frame_index)


def _draw_overlay(frame: np.ndarray, bbox: list[float], track_id: int) -> None:
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    color = (0, 215, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    label = f"ID {track_id}"
    (text_width, text_height), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    )
    top = max(0, y1 - text_height - baseline - 5)
    cv2.rectangle(frame, (x1, top), (x1 + text_width + 10, y1), (10, 10, 10), -1)
    cv2.putText(
        frame,
        label,
        (x1 + 5, max(text_height + 2, y1 - baseline - 3)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
