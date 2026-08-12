#!/usr/bin/env python3
"""Build an index containing only accepted teacher-consensus frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.field_registration.teacher_cache import load_teacher_frame


def build_index(
    root: str | Path,
    output: str | Path,
    *,
    split: str,
    image_glob: str = "frames/**/*.*",
    cache_directory: str = "teacher-cache",
) -> Path:
    root_path = Path(root).expanduser().resolve()
    cache_root = root_path / cache_directory
    frames: list[dict[str, object]] = []
    for image_path in sorted(root_path.glob(image_glob)):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        relative_image = image_path.relative_to(root_path)
        cache_relative = relative_image
        if cache_relative.parts and cache_relative.parts[0] == "frames":
            cache_relative = Path(*cache_relative.parts[1:])
        candidate = cache_root / cache_relative.with_suffix(".json")
        if not candidate.is_file():
            continue
        teacher = load_teacher_frame(candidate)
        if teacher.metadata.get("teacher_consensus") is not True:
            continue
        frames.append(
            {
                "source_id": teacher.source_id,
                "frame_index": teacher.frame_index,
                "image_path": relative_image.as_posix(),
                "teacher_cache_path": candidate.relative_to(root_path).as_posix(),
            }
        )
    if not frames:
        raise ValueError("no accepted teacher-consensus frames were found")
    payload = {
        "format_version": 1,
        "split": split,
        "root": str(root_path),
        "frames": frames,
    }
    target = Path(output).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("distillation", "domain"), required=True)
    parser.add_argument("--image-glob", default="frames/**/*.*")
    parser.add_argument("--cache-directory", default="teacher-cache")
    arguments = parser.parse_args()
    print(
        build_index(
            arguments.root,
            arguments.output,
            split=arguments.split,
            image_glob=arguments.image_glob,
            cache_directory=arguments.cache_directory,
        )
    )


if __name__ == "__main__":
    main()
