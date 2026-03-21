from __future__ import annotations

import csv
import os
import threading

from .store import GameStateManager


class AsyncPersistence(threading.Thread):
    """异步持久化线程。"""

    def __init__(
        self,
        game_state: GameStateManager,
        output_path: str,
        flush_interval: float = 0.5,
    ) -> None:
        super().__init__(daemon=True)
        self._game_state = game_state
        self._output_path = output_path
        self._flush_interval = flush_interval
        self._last_frame_id: int = -1
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        header = [
            "frame_id",
            "timestamp",
            "video_ts",
            "source",
            "player_id",
            "team",
            "pixel_x",
            "pixel_y",
            "field_x",
            "field_y",
            "speed",
            "confidence",
            "jersey_number",
        ]

        while not self._stop_event.is_set():
            try:
                frames = self._game_state.get_recent_frames(1000)
                new_frames = [f for f in frames if f.frame_id > self._last_frame_id]
                if new_frames:
                    new_frames.sort(key=lambda f: f.frame_id)
                    file_exists = os.path.exists(self._output_path)
                    with open(self._output_path, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(header)
                        for frame in new_frames:
                            for player in frame.players.values():
                                writer.writerow(
                                    [
                                        frame.frame_id,
                                        frame.timestamp,
                                        frame.video_ts if frame.video_ts is not None else "",
                                        frame.source,
                                        player.player_id,
                                        player.team.value,
                                        player.pixel_x,
                                        player.pixel_y,
                                        player.field_x,
                                        player.field_y,
                                        player.speed,
                                        player.confidence,
                                        player.jersey_number if player.jersey_number is not None else "",
                                    ]
                                )
                        self._last_frame_id = new_frames[-1].frame_id
            except Exception:
                pass
            self._stop_event.wait(self._flush_interval)
