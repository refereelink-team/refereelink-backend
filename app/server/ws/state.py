from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.publisher import WebSocketPublisher

router = APIRouter()
logger = logging.getLogger(__name__)


@router.websocket("/ws/state")
async def ws_state(websocket: WebSocket) -> None:
    publisher: WebSocketPublisher = websocket.app.state.publisher  # type: ignore[attr-defined]
    await publisher.connect(websocket)
    stop_event = asyncio.Event()
    push_task = asyncio.create_task(publisher.push_loop(stop_event))

    try:
        while True:
            try:
                raw = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=0.5,
                )
                await publisher.handle_command_text(websocket, raw)
            except asyncio.TimeoutError:
                pass
            except WebSocketDisconnect:
                break
    finally:
        stop_event.set()
        push_task.cancel()
        await publisher.disconnect(websocket)
