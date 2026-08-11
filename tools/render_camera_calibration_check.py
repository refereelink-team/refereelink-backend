"""Render side-by-side source/rectified images for manual lens QA."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from app.field_registration.lens import LensCalibration, LensUndistorter

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def select_images(directory: Path, maximum: int) -> list[Path]:
    if maximum <= 0:
        raise ValueError("maximum image count must be positive")
    paths = sorted(
        path for path in directory.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"no images found in {directory}")
    if len(paths) <= maximum:
        return paths
    indices = np.unique(
        np.rint(np.linspace(0, len(paths) - 1, maximum)).astype(np.int64)
    )
    return [paths[int(index)] for index in indices]


def comparison_image(frame: np.ndarray, undistorter: LensUndistorter) -> np.ndarray:
    rectified = undistorter.rectify(frame)
    source = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for image, label in ((source, "SOURCE"), (rectified, "RECTIFIED")):
        cv2.rectangle(image, (0, 0), (190, 42), (0, 0, 0), thickness=-1)
        cv2.putText(
            image,
            label,
            (12, 30),
            font,
            0.8,
            (255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
    return np.concatenate((source, rectified), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--balance", type=float, default=0.0)
    arguments = parser.parse_args()

    calibration = LensCalibration.load(arguments.calibration)
    undistorter = LensUndistorter(calibration, balance=arguments.balance)
    paths = select_images(arguments.images, arguments.max_images)
    arguments.output.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"cannot read image: {path}")
        rendered = comparison_image(frame, undistorter)
        target = arguments.output / f"check-{index:02d}-{path.stem}.jpg"
        if not cv2.imwrite(str(target), rendered, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"failed to write {target}")
        print(f"Saved {target}")


if __name__ == "__main__":
    main()
