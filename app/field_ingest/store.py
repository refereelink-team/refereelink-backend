from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any
from uuid import UUID

from app.field_ingest.models import FieldSessionRegistration, utc_now


class FieldSessionStore:
    """Small durable index; video and telemetry remain files, never BLOBs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.database_path = self.root / "field_ingest.sqlite3"
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.database_path,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    device_id TEXT NOT NULL,
                    body_hash TEXT NOT NULL,
                    registration_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS live_epochs (
                    session_id TEXT NOT NULL,
                    stream_epoch INTEGER NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    token_hash TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    stopped_at TEXT,
                    PRIMARY KEY (session_id, stream_epoch),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    session_id TEXT NOT NULL,
                    artifact_id TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (session_id, artifact_id),
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                );
                """
            )

    @staticmethod
    def canonical_registration(registration: FieldSessionRegistration) -> tuple[str, str]:
        payload = registration.model_dump(mode="json", by_alias=True, exclude_none=True)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def register(self, registration: FieldSessionRegistration) -> tuple[str, dict[str, Any]]:
        encoded, body_hash = self.canonical_registration(registration)
        session_id = str(registration.session_id)
        now = utc_now()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row is not None:
                result = dict(row)
                return ("same" if result["body_hash"] == body_hash else "conflict", result)
            self._connection.execute(
                """
                INSERT INTO sessions
                    (session_id, device_id, body_hash, registration_json, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'registered', ?, ?)
                """,
                (
                    session_id,
                    str(registration.device_id),
                    body_hash,
                    encoded,
                    now,
                    now,
                ),
            )
        return "created", {
            "session_id": session_id,
            "device_id": str(registration.device_id),
            "status": "registered",
            "created_at": now,
            "updated_at": now,
        }

    def get_session(self, session_id: UUID | str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (str(session_id),)
            ).fetchone()
        return dict(row) if row is not None else None

    def next_epoch(self, session_id: UUID | str) -> int:
        with self._lock:
            row = self._connection.execute(
                "SELECT COALESCE(MAX(stream_epoch), 0) AS max_epoch FROM live_epochs WHERE session_id = ?",
                (str(session_id),),
            ).fetchone()
        return int(row["max_epoch"]) + 1

    def has_active_epoch(self) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM live_epochs WHERE status = 'active' LIMIT 1"
            ).fetchone()
        return row is not None

    def create_epoch(
        self,
        session_id: UUID | str,
        epoch: int,
        host: str,
        port: int,
        token_hash: str,
        profile: str,
        latency_ms: int,
    ) -> dict[str, Any]:
        now = utc_now()
        values = (
            str(session_id),
            epoch,
            host,
            port,
            token_hash,
            profile,
            latency_ms,
            "active",
            now,
        )
        with self._lock, self._connection:
            self._connection.execute(
                """
                INSERT INTO live_epochs
                    (session_id, stream_epoch, host, port, token_hash, profile, latency_ms, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            self._connection.execute(
                "UPDATE sessions SET status = 'live', updated_at = ? WHERE session_id = ?",
                (now, str(session_id)),
            )
        return {
            "session_id": str(session_id),
            "stream_epoch": epoch,
            "host": host,
            "port": port,
            "profile": profile,
            "latency_ms": latency_ms,
            "status": "active",
            "created_at": now,
        }

    def get_epoch(self, session_id: UUID | str, epoch: int) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM live_epochs WHERE session_id = ? AND stream_epoch = ?",
                (str(session_id), epoch),
            ).fetchone()
        return dict(row) if row is not None else None

    def release_epoch(self, session_id: UUID | str, epoch: int) -> bool:
        now = utc_now()
        with self._lock, self._connection:
            cursor = self._connection.execute(
                """
                UPDATE live_epochs SET status = 'released', stopped_at = ?
                WHERE session_id = ? AND stream_epoch = ? AND status = 'active'
                """,
                (now, str(session_id), epoch),
            )
            self._connection.execute(
                "UPDATE sessions SET status = 'registered', updated_at = ? WHERE session_id = ?",
                (now, str(session_id)),
            )
        return cursor.rowcount > 0

    def save_artifact(
        self,
        session_id: UUID | str,
        artifact_id: str,
        size_bytes: int,
        sha256: str,
        path: Path,
        status: str = "complete",
    ) -> tuple[str, dict[str, Any]]:
        now = utc_now()
        with self._lock, self._connection:
            row = self._connection.execute(
                "SELECT * FROM artifacts WHERE session_id = ? AND artifact_id = ?",
                (str(session_id), artifact_id),
            ).fetchone()
            if row is not None:
                existing = dict(row)
                return ("same" if existing["sha256"] == sha256 else "conflict", existing)
            self._connection.execute(
                """
                INSERT INTO artifacts
                    (session_id, artifact_id, size_bytes, sha256, path, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (str(session_id), artifact_id, size_bytes, sha256, str(path), status, now),
            )
        return "created", {
            "session_id": str(session_id),
            "artifact_id": artifact_id,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "path": str(path),
            "status": status,
            "created_at": now,
        }

    def list_artifacts(self, session_id: UUID | str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM artifacts WHERE session_id = ? ORDER BY artifact_id",
                (str(session_id),),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_artifact(self, session_id: UUID | str, artifact_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM artifacts WHERE session_id = ? AND artifact_id = ?",
                (str(session_id), artifact_id),
            ).fetchone()
        return dict(row) if row is not None else None

    def complete(self, session_id: UUID | str) -> None:
        now = utc_now()
        with self._lock, self._connection:
            self._connection.execute(
                "UPDATE sessions SET status = 'complete', updated_at = ? WHERE session_id = ?",
                (now, str(session_id)),
            )

    def close(self) -> None:
        with self._lock:
            self._connection.close()
