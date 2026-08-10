#!/usr/bin/env python3
"""Extract leakage-safe video frames for pitch-registration annotation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from app.field_registration.annotations import ANNOTATION_FORMAT_VERSION, VALID_SPLITS
from app.field_registration.pitch_model import PitchDimensions, PitchModel


def _video_metadata(source: Path) -> tuple[float, int, tuple[int, int]]:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {source}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()
    if fps <= 0.0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError("video metadata is incomplete")
    return fps, frame_count, (width, height)


def select_frame_indices(
    fps: float,
    total_frames: int,
    requested_frames: int,
    start_sec: float,
    end_sec: float | None,
) -> np.ndarray:
    if requested_frames <= 0:
        raise ValueError("requested frame count must be positive")
    duration_sec = total_frames / fps
    end = duration_sec if end_sec is None else float(end_sec)
    if start_sec < 0.0 or end <= start_sec or end > duration_sec + 1.0 / fps:
        raise ValueError(
            f"invalid time range {start_sec:.3f}-{end:.3f}s for {duration_sec:.3f}s video"
        )
    first = min(total_frames - 1, int(round(start_sec * fps)))
    last = min(total_frames - 1, max(first, int(round(end * fps)) - 1))
    available = last - first + 1
    count = min(requested_frames, available)
    return np.unique(np.rint(np.linspace(first, last, count)).astype(np.int64))


def create_annotation_pack(
    source: str | Path,
    output_directory: str | Path,
    *,
    requested_frames: int = 30,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    split: str = "calibration",
    dimensions: PitchDimensions | None = None,
    overwrite: bool = False,
) -> Path:
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
    output_path = Path(output_directory).expanduser().resolve()
    manifest_path = output_path / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(f"annotation pack already exists: {manifest_path}")
    frames_path = output_path / "frames"
    frames_path.mkdir(parents=True, exist_ok=True)

    fps, total_frames, image_size = _video_metadata(source_path)
    indices = select_frame_indices(
        fps,
        total_frames,
        requested_frames,
        start_sec,
        end_sec,
    )
    capture = cv2.VideoCapture(str(source_path))
    frame_entries: list[dict[str, object]] = []
    try:
        for index in indices.tolist():
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError(f"failed to decode frame {index}")
            relative_path = Path("frames") / f"frame-{index:08d}.jpg"
            target = output_path / relative_path
            if not cv2.imwrite(str(target), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"failed to write {target}")
            frame_entries.append(
                {
                    "frame_index": index,
                    "timestamp_ms": round(index * 1000.0 / fps, 3),
                    "image_path": relative_path.as_posix(),
                    "tags": [],
                    "correspondences": [],
                    "line_polylines": [],
                    "contact_points": [],
                }
            )
    finally:
        capture.release()

    pitch_dimensions = dimensions or PitchDimensions()
    pitch_model = PitchModel(pitch_dimensions)
    manifest = {
        "version": ANNOTATION_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "split": split,
        "source": {
            "name": source_path.name,
            "path_at_creation": str(source_path),
            "fps": fps,
            "frame_count": total_frames,
            "duration_ms": round(total_frames * 1000.0 / fps, 3),
        },
        "image_size": list(image_size),
        "pitch_dimensions_m": {
            "length": pitch_dimensions.length_m,
            "width": pitch_dimensions.width_m,
            "penalty_area_depth": pitch_dimensions.penalty_area_depth_m,
            "penalty_area_width": pitch_dimensions.penalty_area_width_m,
            "goal_area_depth": pitch_dimensions.goal_area_depth_m,
            "goal_area_width": pitch_dimensions.goal_area_width_m,
            "centre_circle_radius": pitch_dimensions.centre_circle_radius_m,
            "penalty_spot_distance": pitch_dimensions.penalty_spot_distance_m,
        },
        "landmarks_m": {
            label: list(xy) for label, xy in pitch_model.landmarks.items()
        }
        | {
            label: list(xy) for label, xy in pitch_model.point_landmarks.items()
        },
        "semantic_elements_m": {
            label: points.round(6).tolist()
            for label, points in pitch_model.semantic_elements().items()
        },
        "annotation_policy": {
            "sequence_level_split": True,
            "minimum_correspondences_per_frame": 4,
            "recommended_correspondences_per_frame": 8,
            "contact_point_definition": "support centre between visible feet on pitch plane",
        },
        "frames": frame_entries,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    ui_source = Path(__file__).with_name("pitch_annotation_ui.html")
    if ui_source.is_file():
        shutil.copy2(ui_source, output_path / "index.html")
    readme = (
        "Pitch registration annotation pack\n\n"
        "Run this directory through a local HTTP server:\n"
        "  uv run python -m http.server 8765 --directory .\n\n"
        "Then open http://127.0.0.1:8765/index.html. Exported JSON must be "
        "validated before it is used for evaluation.\n"
    )
    (output_path / "README.txt").write_text(readme, encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--start-sec", type=float, default=0.0)
    parser.add_argument("--end-sec", type=float)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="calibration")
    parser.add_argument("--pitch-length-m", type=float, default=105.0)
    parser.add_argument("--pitch-width-m", type=float, default=68.0)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args()
    dimensions = PitchDimensions(
        length_m=arguments.pitch_length_m,
        width_m=arguments.pitch_width_m,
    )
    manifest = create_annotation_pack(
        arguments.source,
        arguments.output_directory,
        requested_frames=arguments.frames,
        start_sec=arguments.start_sec,
        end_sec=arguments.end_sec,
        split=arguments.split,
        dimensions=dimensions,
        overwrite=arguments.overwrite,
    )
    print(f"Created annotation pack: {manifest}")


if __name__ == "__main__":
    main()
