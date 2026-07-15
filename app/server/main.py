from __future__ import annotations

import argparse
import logging
import threading
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from app.pipeline.buffer import PipelineMode
from app.pipeline.engine import InferencePipeline
from app.pipeline.source import create_video_source
from app.server.api.health import router as health_router
from app.server.api.status import router as status_router
from app.server.api.events import router as events_router
from app.server.api.pipeline import router as pipeline_router
from app.server.ws.state import router as ws_router
from app.services.publisher import WebSocketPublisher
from app.state.models import PipelineConfig, SourceStatus
from app.state.store import StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_store = StateStore()
_publisher = WebSocketPublisher(_store)
_pipeline: Optional[InferencePipeline] = None
_pipeline_lock = threading.Lock()
_device = "cpu"
_foul_checkpoint_path: Optional[str] = None


def _put_placeholder(text: str) -> None:
    """Write a black placeholder frame with text into the store so the
    MJPEG endpoint always has something to serve."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        frame, text, (40, 240),
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (180, 180, 180), 2, cv2.LINE_AA,
    )
    cv2.putText(
        frame, "Configure a video source and press START", (40, 280),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1, cv2.LINE_AA,
    )
    _store._latest_raw_frame = frame  # type: ignore[attr-defined]


def _generate_mjpeg() -> iter:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    while True:
        raw = getattr(_store, "_latest_raw_frame", None)
        if raw is None:
            _put_placeholder("NO VIDEO SOURCE")
            raw = getattr(_store, "_latest_raw_frame", None)
        if raw is None:
            time.sleep(0.05)
            continue
        _, jpeg = cv2.imencode(".jpg", raw, encode_params)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )
        time.sleep(1.0 / 30.0)


def create_pipeline(video_source: str, device: str = "cpu") -> InferencePipeline:
    """Factory used by both CLI startup and the REST API to build a
    pipeline bound to the shared store."""
    return InferencePipeline(
        source=create_video_source(video_source, store=_store),
        store=_store,
        device=device,
        mode=PipelineMode.REALTIME,
    )


def attach_and_start_pipeline(pipeline: InferencePipeline) -> None:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            try:
                _pipeline.stop()
            except Exception:
                pass
        _pipeline = pipeline
        app.state.pipeline = _pipeline
    _store.pipeline_running = True
    pipeline.start()


def stop_and_clear_pipeline() -> None:
    global _pipeline
    with _pipeline_lock:
        if _pipeline is not None:
            try:
                _pipeline.stop()
            except Exception:
                pass
            _pipeline = None
        app.state.pipeline = None
    _store._latest_raw_frame = None  # type: ignore[attr-defined]
    _store.source_status = SourceStatus.DISCONNECTED
    _store.pipeline_running = False
    _put_placeholder("PIPELINE STOPPED")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting...")
    _put_placeholder("SERVER READY")
    yield
    logger.info("Server shutting down...")
    stop_and_clear_pipeline()


app = FastAPI(title="Soccer Analysis Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.store = _store
app.state.publisher = _publisher
app.state.pipeline = None
app.state.create_pipeline = create_pipeline
app.state.attach_and_start_pipeline = attach_and_start_pipeline
app.state.stop_and_clear_pipeline = stop_and_clear_pipeline

app.include_router(health_router)
app.include_router(status_router)
app.include_router(events_router)
app.include_router(pipeline_router)
app.include_router(ws_router)


@app.get("/video/stream")
async def video_stream() -> StreamingResponse:
    return StreamingResponse(
        _generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def main() -> None:
    global _device

    parser = argparse.ArgumentParser(description="Soccer Analysis Server")
    parser.add_argument("--video_source", type=str, default=None,
                        help="Video file path or RTSP URL. If omitted, the "
                             "server starts idle and waits for a source "
                             "from the web UI.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference (cpu, cuda)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    _device = args.device

    if args.video_source:
        pipeline = create_pipeline(
            args.video_source,
            device=_device,
        )
        attach_and_start_pipeline(pipeline)
    else:
        logger.info("No --video_source provided; server starting idle. "
                    "Configure a source from the web dashboard.")

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()