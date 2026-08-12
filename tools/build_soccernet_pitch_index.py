#!/usr/bin/env python3
"""Build a leakage-auditable index for local SoccerNet-Calibration data.

The command does not download or copy the research-only dataset. It records
paths relative to ``--root`` so a teammate can reproduce the same index after
placing SoccerNet under any local directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import cv2

from experiments.field_registration.soccernet import (
    SoccerNetFormatError,
    SoccerNetIndex,
    SoccerNetIndexFrame,
    parse_soccernet_annotation,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise SoccerNetFormatError(f"path is outside dataset root: {path}") from exc


def _find_image(annotation_path: Path, image_root: Path) -> Path:
    candidates: list[Path] = []
    relative_parent = annotation_path.parent
    for extension in IMAGE_EXTENSIONS:
        candidates.extend(
            (
                annotation_path.with_suffix(extension),
                image_root / relative_parent.name / f"{annotation_path.stem}{extension}",
                image_root / f"{annotation_path.stem}{extension}",
            )
        )
    existing = sorted({candidate.resolve() for candidate in candidates if candidate.is_file()})
    if len(existing) != 1:
        raise SoccerNetFormatError(
            f"expected exactly one image for {annotation_path}, found {len(existing)}; "
            "place matching images beside JSON files or under --image-root"
        )
    return existing[0]


def _frame_index(path: Path, fallback: int) -> int:
    matches = re.findall(r"\d+", path.stem)
    return int(matches[-1]) if matches else fallback


def build_index(
    root: str | Path,
    output: str | Path,
    *,
    split: str,
    annotation_glob: str = "**/*.json",
    image_root: str | Path | None = None,
) -> Path:
    root_path = Path(root).expanduser().resolve()
    image_root_path = (
        Path(image_root).expanduser().resolve()
        if image_root is not None
        else root_path
    )
    annotation_paths = sorted(root_path.glob(annotation_glob))
    output_path = Path(output).expanduser().resolve()
    annotation_paths = [
        path for path in annotation_paths if path.resolve() != output_path
    ]
    if not annotation_paths:
        raise FileNotFoundError(
            f"no annotation JSON files match {annotation_glob!r} under {root_path}"
        )
    frames: list[SoccerNetIndexFrame] = []
    for fallback, annotation_path in enumerate(annotation_paths):
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        image_path = _find_image(annotation_path, image_root_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise SoccerNetFormatError(f"cannot read image: {image_path}")
        height, width = image.shape[:2]
        parsed = parse_soccernet_annotation(payload, (width, height))
        labels = tuple(label for label, points in parsed.items() if points.shape[0] >= 2)
        if not labels:
            continue
        relative_annotation = _relative(annotation_path, root_path)
        relative_image = _relative(image_path, root_path)
        sequence_id = annotation_path.parent.relative_to(root_path).as_posix() or "."
        source_id = str(
            payload.get(
                "source_id",
                payload.get("image", Path(relative_image).as_posix()),
            )
        )
        frames.append(
            SoccerNetIndexFrame(
                source_id=source_id,
                sequence_id=sequence_id,
                frame_index=int(payload.get("frame_index", _frame_index(annotation_path, fallback))),
                image_path=relative_image,
                annotation_path=relative_annotation,
                image_size=(width, height),
                raw_labels_present=labels,
            )
        )
    if not frames:
        raise SoccerNetFormatError("no frame contains usable SoccerNet pitch labels")
    identifiers = [(frame.source_id, frame.frame_index) for frame in frames]
    if len(set(identifiers)) != len(identifiers):
        raise SoccerNetFormatError("duplicate (source_id, frame_index) identity")
    return SoccerNetIndex(split, str(root_path), tuple(frames)).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split", choices=("train", "validation", "test"), required=True
    )
    parser.add_argument("--annotation-glob", default="**/*.json")
    parser.add_argument("--image-root", type=Path)
    arguments = parser.parse_args()
    path = build_index(
        arguments.root,
        arguments.output,
        split=arguments.split,
        annotation_glob=arguments.annotation_glob,
        image_root=arguments.image_root,
    )
    print(path)


if __name__ == "__main__":
    main()
