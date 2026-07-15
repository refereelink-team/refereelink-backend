import numpy as np
import pytest

from app.geometry.view import ViewTransformer


def test_view_transformer_rejects_mismatched_shapes() -> None:
    source = np.array([[0, 0], [1, 1], [2, 2]], dtype=np.float32)
    target = np.array([[0, 0], [1, 1]], dtype=np.float32)

    with pytest.raises(ValueError):
        ViewTransformer(source=source, target=target)


def test_view_transformer_returns_empty_points_unchanged() -> None:
    source = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    target = np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    transformer = ViewTransformer(source=source, target=target)

    transformed = transformer.transform_points(np.empty((0, 2), dtype=np.float32))

    assert transformed.shape == (0, 2)
