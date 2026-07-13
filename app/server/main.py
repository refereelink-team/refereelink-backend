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
from app.state.store import StateStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_store = StateStore()
_publisher = WebSocketPublisher(_store)
_pipeline: Optional[InferencePipeline] = None


def _generate_mjpeg() -> iter:
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    no_frame_wait = 0
    while _store.pipeline_running or no_frame_wait < 100:
        raw = getattr(_store, '_latest_raw_frame', None)
        if raw is None:
            no_frame_wait += 1
            time.sleep(0.05)
            continue
        no_frame_wait = 0
        _, jpeg = cv2.imencode(".jpg", raw, encode_params)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )
        time.sleep(1.0 / 30.0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Server starting...")
    yield
    logger.info("Server shutting down...")
    if _pipeline is not None:
        _pipeline.stop()
    _store.pipeline_running = False


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
    parser = argparse.ArgumentParser(description="Soccer Analysis Server")
    parser.add_argument("--video_source", type=str, required=True,
                        help="Video file path or RTSP URL")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference (cpu, cuda)")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--foul_checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    global _pipeline
    _pipeline = InferencePipeline(
        source=create_video_source(args.video_source, store=_store),
        store=_store,
        device=args.device,
        mode=PipelineMode.REALTIME,
        foul_checkpoint_path=args.foul_checkpoint_path,
    )
    app.state.pipeline = _pipeline
    _pipeline.start()

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
