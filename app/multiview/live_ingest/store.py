from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.multiview.models import CaptureState, MultiviewCase


class InsufficientLiveBuffer(RuntimeError):
    """The three-camera ring does not cover a safe review window."""


@dataclass(frozen=True)
class IndexedSegment:
    camera_id: str
    path: Path
    start_time_s: float
    end_time_s: float
    duration_s: float
    pts_start_s: float | None = None
    pts_end_s: float | None = None
    media: dict[str, Any] | None = None


@dataclass(frozen=True)
class CameraHealth:
    camera_id: str
    online: bool
    latest_segment_at_s: float | None
    reconnect_count: int
    last_error: str | None


class SegmentIndex:
    """Persistent ring metadata shared by ingest and the web API process."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._ensure_schema()

    def record_segment(self, segment: IndexedSegment) -> None:
        if segment.end_time_s <= segment.start_time_s:
            raise ValueError("segment end must be after its start")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO live_segments (
                    camera_id, path, start_time_s, end_time_s, duration_s,
                    pts_start_s, pts_end_s, media_json, ref_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(path) DO UPDATE SET
                    start_time_s = excluded.start_time_s,
                    end_time_s = excluded.end_time_s,
                    duration_s = excluded.duration_s,
                    pts_start_s = excluded.pts_start_s,
                    pts_end_s = excluded.pts_end_s,
                    media_json = excluded.media_json
                """,
                (
                    segment.camera_id,
                    str(segment.path.resolve()),
                    segment.start_time_s,
                    segment.end_time_s,
                    segment.duration_s,
                    segment.pts_start_s,
                    segment.pts_end_s,
                    json.dumps(segment.media or {}, ensure_ascii=False, separators=(",", ":")),
                ),
            )
            self._set_health(
                connection,
                camera_id=segment.camera_id,
                online=True,
                latest_segment_at_s=segment.end_time_s,
                last_error=None,
            )

    def set_camera_error(self, camera_id: str, message: str) -> None:
        with self._connect() as connection:
            self._set_health(
                connection,
                camera_id=camera_id,
                online=False,
                latest_segment_at_s=None,
                last_error=message[:500],
            )

    def increment_reconnect(self, camera_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO live_camera_health (
                    camera_id, online, latest_segment_at_s, reconnect_count, last_error
                ) VALUES (?, 0, NULL, 1, NULL)
                ON CONFLICT(camera_id) DO UPDATE SET
                    reconnect_count = live_camera_health.reconnect_count + 1
                """,
                (camera_id,),
            )

    def camera_health(self, camera_id: str) -> CameraHealth:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT camera_id, online, latest_segment_at_s, reconnect_count, last_error
                FROM live_camera_health WHERE camera_id = ?
                """,
                (camera_id,),
            ).fetchone()
        if row is None:
            return CameraHealth(camera_id, False, None, 0, None)
        return CameraHealth(
            camera_id=row["camera_id"],
            online=bool(row["online"]),
            latest_segment_at_s=row["latest_segment_at_s"],
            reconnect_count=int(row["reconnect_count"]),
            last_error=row["last_error"],
        )

    def latest_segment_end(self, camera_id: str) -> float | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT MAX(end_time_s) AS latest_end FROM live_segments WHERE camera_id = ?",
                (camera_id,),
            ).fetchone()
        return float(row["latest_end"]) if row is not None and row["latest_end"] is not None else None

    def buffer_seconds(self, camera_id: str, *, max_gap_s: float) -> float:
        segments = self._segments_for_camera(camera_id)
        if not segments:
            return 0.0
        latest = segments[-1]
        cursor = latest.end_time_s
        start = latest.start_time_s
        for segment in reversed(segments[:-1]):
            if segment.end_time_s < cursor - max_gap_s:
                break
            start = min(start, segment.start_time_s)
            cursor = min(cursor, segment.start_time_s)
        return max(0.0, latest.end_time_s - start)

    def freeze_window(
        self,
        camera_ids: Iterable[str],
        *,
        end_time_s: float,
        window_seconds: float,
        max_gap_s: float,
    ) -> dict[str, list[IndexedSegment]]:
        """Atomically pin contiguous, keyframe-aligned segments for all cameras."""
        snapshots: dict[str, list[IndexedSegment]] = {}
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                for camera_id in camera_ids:
                    segments = self._window_segments(
                        connection,
                        camera_id=camera_id,
                        end_time_s=end_time_s,
                        window_seconds=window_seconds,
                        max_gap_s=max_gap_s,
                    )
                    if not segments:
                        raise InsufficientLiveBuffer(f"{camera_id} 缓冲不足")
                    snapshots[camera_id] = segments
                for segment in (item for items in snapshots.values() for item in items):
                    connection.execute(
                        "UPDATE live_segments SET ref_count = ref_count + 1 WHERE path = ?",
                        (str(segment.path),),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise
        return snapshots

    def release_window(self, snapshots: dict[str, list[IndexedSegment]]) -> None:
        paths = [str(segment.path) for items in snapshots.values() for segment in items]
        if not paths:
            return
        with self._connect() as connection:
            connection.executemany(
                """
                UPDATE live_segments
                SET ref_count = CASE WHEN ref_count > 0 THEN ref_count - 1 ELSE 0 END
                WHERE path = ?
                """,
                ((path,) for path in paths),
            )

    def prune(self, *, older_than_s: float) -> list[Path]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT path FROM live_segments
                WHERE end_time_s < ? AND ref_count = 0
                """,
                (older_than_s,),
            ).fetchall()
            connection.executemany(
                "DELETE FROM live_segments WHERE path = ?",
                ((row["path"],) for row in rows),
            )
        return [Path(row["path"]) for row in rows]

    def _segments_for_camera(self, camera_id: str) -> list[IndexedSegment]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT camera_id, path, start_time_s, end_time_s, duration_s,
                       pts_start_s, pts_end_s, media_json
                FROM live_segments
                WHERE camera_id = ?
                ORDER BY start_time_s, end_time_s
                """,
                (camera_id,),
            ).fetchall()
        return [self._to_segment(row) for row in rows if Path(row["path"]).is_file()]

    def _window_segments(
        self,
        connection: sqlite3.Connection,
        *,
        camera_id: str,
        end_time_s: float,
        window_seconds: float,
        max_gap_s: float,
    ) -> list[IndexedSegment]:
        rows = connection.execute(
            """
            SELECT camera_id, path, start_time_s, end_time_s, duration_s,
                   pts_start_s, pts_end_s, media_json
            FROM live_segments
            WHERE camera_id = ?
            ORDER BY start_time_s DESC, end_time_s DESC
            """,
            (camera_id,),
        ).fetchall()
        cursor = end_time_s
        selected: list[IndexedSegment] = []
        for row in rows:
            segment = self._to_segment(row)
            if not segment.path.is_file():
                continue
            if segment.start_time_s > cursor + max_gap_s:
                continue
            if segment.end_time_s < cursor - max_gap_s:
                break
            selected.append(segment)
            cursor = min(cursor, segment.start_time_s)
            if end_time_s - cursor >= window_seconds:
                break
        if not selected or end_time_s - cursor < window_seconds:
            return []
        selected.reverse()
        return selected

    @staticmethod
    def _to_segment(row: sqlite3.Row) -> IndexedSegment:
        return IndexedSegment(
            camera_id=row["camera_id"],
            path=Path(row["path"]),
            start_time_s=float(row["start_time_s"]),
            end_time_s=float(row["end_time_s"]),
            duration_s=float(row["duration_s"]),
            pts_start_s=(float(row["pts_start_s"]) if row["pts_start_s"] is not None else None),
            pts_end_s=(float(row["pts_end_s"]) if row["pts_end_s"] is not None else None),
            media=json.loads(row["media_json"] or "{}"),
        )

    def _set_health(
        self,
        connection: sqlite3.Connection,
        *,
        camera_id: str,
        online: bool,
        latest_segment_at_s: float | None,
        last_error: str | None,
    ) -> None:
        connection.execute(
            """
            INSERT INTO live_camera_health (
                camera_id, online, latest_segment_at_s, reconnect_count, last_error
            ) VALUES (?, ?, ?, 0, ?)
            ON CONFLICT(camera_id) DO UPDATE SET
                online = excluded.online,
                latest_segment_at_s = COALESCE(
                    excluded.latest_segment_at_s, live_camera_health.latest_segment_at_s
                ),
                last_error = excluded.last_error
            """,
            (camera_id, int(online), latest_segment_at_s, last_error),
        )

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS live_segments (
                    camera_id TEXT NOT NULL,
                    path TEXT PRIMARY KEY,
                    start_time_s REAL NOT NULL,
                    end_time_s REAL NOT NULL,
                    duration_s REAL NOT NULL,
                    pts_start_s REAL,
                    pts_end_s REAL,
                    media_json TEXT NOT NULL,
                    ref_count INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS live_segments_camera_time
                    ON live_segments(camera_id, start_time_s, end_time_s);
                CREATE TABLE IF NOT EXISTS live_camera_health (
                    camera_id TEXT PRIMARY KEY,
                    online INTEGER NOT NULL DEFAULT 0,
                    latest_segment_at_s REAL,
                    reconnect_count INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT
                );
                """
            )


class LiveMultiviewCaseStore:
    """The mutable live-case source; demo cases remain untouched JSON."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self._ensure_schema()

    def save(
        self,
        case: MultiviewCase,
        *,
        capture_state: CaptureState,
        manifest_path: str | Path | None = None,
        error: str | None = None,
        captured_at_s: float | None = None,
    ) -> MultiviewCase:
        dynamic_case = case.model_copy(update={"capture_state": capture_state})
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO live_multiview_cases (
                    case_id, payload_json, capture_state, manifest_path, capture_error, captured_at_s
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    capture_state = excluded.capture_state,
                    manifest_path = excluded.manifest_path,
                    capture_error = excluded.capture_error,
                    captured_at_s = excluded.captured_at_s
                """,
                (
                    dynamic_case.case_id,
                    dynamic_case.model_dump_json(),
                    capture_state.value,
                    str(manifest_path) if manifest_path is not None else None,
                    error[:500] if error else None,
                    captured_at_s if captured_at_s is not None else time.time(),
                ),
            )
        return dynamic_case

    def list_cases(self) -> list[MultiviewCase]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json, capture_state FROM live_multiview_cases ORDER BY captured_at_s DESC"
            ).fetchall()
        return [self._case_from_row(row) for row in rows]

    def get_case(self, case_id: str) -> MultiviewCase | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json, capture_state
                FROM live_multiview_cases
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()
        return self._case_from_row(row) if row is not None else None

    def _case_from_row(self, row: sqlite3.Row) -> MultiviewCase:
        case = MultiviewCase.model_validate_json(row["payload_json"])
        return case.model_copy(update={"capture_state": CaptureState(row["capture_state"])})

    def _connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=10, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS live_multiview_cases (
                    case_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    capture_state TEXT NOT NULL,
                    manifest_path TEXT,
                    capture_error TEXT,
                    captured_at_s REAL NOT NULL
                )
                """
            )
