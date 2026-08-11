from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from app.field_registration.lens import LensCalibration, LensModel, LensUndistorter
from tools.render_camera_calibration_check import comparison_image, select_images


def test_select_images_is_bounded_and_spans_directory(tmp_path: Path) -> None:
    for index in range(10):
        cv2.imwrite(str(tmp_path / f"image-{index:02d}.jpg"), np.zeros((8, 8, 3)))

    selected = select_images(tmp_path, maximum=4)

    assert len(selected) == 4
    assert selected[0].name == "image-00.jpg"
    assert selected[-1].name == "image-09.jpg"


def test_select_images_rejects_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no images"):
        select_images(tmp_path, maximum=4)


def test_comparison_image_places_source_and_rectified_side_by_side() -> None:
    frame = np.full((48, 64, 3), 127, dtype=np.uint8)
    calibration = LensCalibration(
        lens_model=LensModel.PINHOLE,
        camera_matrix=np.array(
            [[60.0, 0.0, 32.0], [0.0, 60.0, 24.0], [0.0, 0.0, 1.0]]
        ),
        distortion_coefficients=np.zeros(5),
        image_size=(64, 48),
    )

    result = comparison_image(frame, LensUndistorter(calibration))

    assert result.shape == (48, 128, 3)
    assert result.dtype == np.uint8
