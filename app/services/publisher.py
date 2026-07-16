from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

from app.state.store import StateStore

logger = logging.getLogger(__name__)

PUSH_INTERVAL_SEC = 1.0 / 30.0
METRICS_INTERVAL_SEC = 1.0


class WebSocketPublisher:
    def __init__(self, store: StateStore) -> None:
        self._store = store
        self._clients: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            self._clients.add(websocket)
        logger.info("WebSocket client connected (total: %d)", len(self._clients))

    async def disconnect(self, websocket: WebSocket) -> None:
        async with self._lock:
            self._clients.discard(websocket)
        logger.info("WebSocket client disconnected (total: %d)", len(self._clients))

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            dead: list[WebSocket] = []
            for ws in self._clients:
                try:
                    await ws.send_json(message)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

    async def push_loop(self, stop_event: asyncio.Event) -> None:
        last_metrics_push = 0.0
        last_frame_id: int | None = None
        last_calibration_snapshot: dict[str, Any] | None = None
        while not stop_event.is_set():
            frame_state = self._store.latest_frame_state
            if frame_state is not None and frame_state.frame_id != last_frame_id:
                await self.broadcast(frame_state.model_dump())
                last_frame_id = frame_state.frame_id

            now = asyncio.get_event_loop().time()
            if now - last_metrics_push >= METRICS_INTERVAL_SEC:
                metrics = self._store.metrics
                await self.broadcast(metrics.model_dump())
                calibration = getattr(self._store, "team_calibration", None)
                if calibration is not None:
                    snapshot = calibration.snapshot()
                    if snapshot != last_calibration_snapshot:
                        await self.broadcast(snapshot)
                        last_calibration_snapshot = snapshot
                last_metrics_push = now

            await asyncio.sleep(PUSH_INTERVAL_SEC)

    async def handle_command_text(self, websocket: WebSocket, raw: str) -> None:
        try:
            data = json.loads(raw)
            command = data.get("command", "")
            if command == "start":
                calibration = getattr(self._store, "team_calibration", None)
                if (
                    self._store.config.require_team_calibration
                    and calibration is not None
                    and not calibration.can_run
                ):
                    await websocket.send_json(
                        {
                            "type": "ack",
                            "command": command,
                            "status": "error",
                            "code": "TEAM_CALIBRATION_REQUIRED",
                        }
                    )
                    return
                self._store.pipeline_running = True
            elif command == "stop":
                self._store.pipeline_running = False
            elif command == "update_config":
                params = data.get("params", {})
                self._store.update_config(params)
            await websocket.send_json({"type": "ack", "command": command, "status": "ok"})
        except Exception:
            pass
