from __future__ import annotations

import numpy as np
import pytest

from app.vision.backends import CallableBackend, DetectorBackend, UltralyticsBackend


class _FakeUltralyticsModel:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, int, bool]] = []

    def __call__(self, frame: np.ndarray, *, imgsz: int, half: bool) -> dict[str, object]:
        self.calls.append((frame, imgsz, half))
        return {"boxes": np.array([[1, 2, 3, 4]], dtype=np.float32)}


class _QuantizeUltralyticsModel:
    def __init__(self) -> None:
        self.quantize: object = "not-called"

    def __call__(
        self,
        frame: np.ndarray,
        *,
        imgsz: int,
        verbose: bool,
        quantize: int | None = None,
    ) -> dict[str, object]:
        del frame, imgsz, verbose
        self.quantize = quantize
        return {"boxes": np.array([[1, 2, 3, 4]], dtype=np.float32)}


def test_callable_backend_forwards_the_shared_predict_arguments() -> None:
    frame = np.zeros((32, 48, 3), dtype=np.uint8)
    calls: list[tuple[np.ndarray, int, bool]] = []

    def fake_predictor(frame: np.ndarray, *, imgsz: int, half: bool) -> str:
        calls.append((frame, imgsz, half))
        return "fake-result"

    backend = CallableBackend(fake_predictor)

    assert isinstance(backend, DetectorBackend)
    assert backend.predict(frame, imgsz=320, half=True) == "fake-result"
    assert calls == [(frame, 320, True)]


def test_callable_backend_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="requires a callable"):
        CallableBackend(object())


@pytest.mark.parametrize(
    ("model_name", "expected_backend"),
    [("detector.pt", "pytorch"), ("detector.onnx", "onnx"), ("detector.engine", "tensorrt")],
)
def test_ultralytics_backend_infers_backend_from_suffix(
    tmp_path, model_name: str, expected_backend: str
) -> None:
    model_path = tmp_path / model_name
    model_path.touch()
    fake_model = _FakeUltralyticsModel()
    backend = UltralyticsBackend(model_path, model=fake_model)

    assert backend.backend == expected_backend
    assert isinstance(backend, DetectorBackend)


def test_ultralytics_backend_accepts_explicit_backend_for_custom_suffix(tmp_path) -> None:
    model_path = tmp_path / "detector.deploy"
    model_path.touch()

    backend = UltralyticsBackend(model_path, backend="trt", model=_FakeUltralyticsModel())

    assert backend.backend == "tensorrt"


def test_ultralytics_backend_uses_the_shared_predict_arguments(tmp_path) -> None:
    model_path = tmp_path / "detector.onnx"
    model_path.touch()
    fake_model = _FakeUltralyticsModel()
    frame = np.zeros((24, 24, 3), dtype=np.uint8)
    backend = UltralyticsBackend(model_path, model=fake_model)

    result = backend.predict(frame, imgsz=960, half=False)

    assert "boxes" in result
    assert fake_model.calls == [(frame, 960, False)]


@pytest.mark.parametrize(("half", "expected"), [(False, None), (True, 16)])
def test_ultralytics_backend_translates_precision_without_legacy_warning(
    tmp_path,
    half: bool,
    expected: int | None,
) -> None:
    model_path = tmp_path / "detector.pt"
    model_path.touch()
    fake_model = _QuantizeUltralyticsModel()
    backend = UltralyticsBackend(model_path, model=fake_model)

    backend.predict(np.zeros((24, 24, 3), dtype=np.uint8), half=half)

    assert fake_model.quantize == expected


def test_ultralytics_backend_reports_missing_model_clearly(tmp_path) -> None:
    missing_path = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="model file not found"):
        UltralyticsBackend(missing_path)


def test_ultralytics_backend_requires_supported_suffix_without_backend(tmp_path) -> None:
    model_path = tmp_path / "detector.bin"
    model_path.touch()

    with pytest.raises(ValueError, match="Supported model suffixes"):
        UltralyticsBackend(model_path, model=_FakeUltralyticsModel())


def test_ultralytics_backend_rejects_unknown_backend(tmp_path) -> None:
    model_path = tmp_path / "detector.pt"
    model_path.touch()

    with pytest.raises(ValueError, match="Supported backends"):
        UltralyticsBackend(model_path, backend="tensorflow", model=_FakeUltralyticsModel())
