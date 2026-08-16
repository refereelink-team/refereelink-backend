from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from app.constants.paths import REPO_ROOT_DIR
from app.multiview.models import (
    FoulFacts,
    MultiviewDecision,
    ReviewRecord,
    ReviewState,
    RuleAssessment,
)

DEFAULT_REVIEW_DB_PATH = REPO_ROOT_DIR / "var" / "multiview" / "reviews.sqlite3"


class ReviewRevisionConflict(RuntimeError):
    def __init__(self, expected: int, current: int) -> None:
        super().__init__(f"review revision conflict: expected {expected}, current {current}")
        self.expected = expected
        self.current = current


class MultiviewReviewStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        configured = db_path or os.environ.get("SC_MULTIVIEW_REVIEW_DB")
        self.db_path = Path(configured).expanduser() if configured else DEFAULT_REVIEW_DB_PATH
        self._lock = threading.RLock()
        self._initialized = False

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _ensure_schema(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS analysis_runs (
                        analysis_id TEXT PRIMARY KEY,
                        case_id TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_analysis_case
                        ON analysis_runs(case_id, created_at DESC);

                    CREATE TABLE IF NOT EXISTS review_revisions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        case_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        analysis_id TEXT,
                        review_state TEXT NOT NULL,
                        facts_json TEXT NOT NULL,
                        assessment_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        UNIQUE(case_id, revision)
                    );
                    CREATE INDEX IF NOT EXISTS idx_review_case
                        ON review_revisions(case_id, revision DESC);
                    """
                )
            self._initialized = True

    def save_analysis(self, decision: MultiviewDecision) -> None:
        self._ensure_schema()
        payload = json.dumps(decision.model_dump(mode="json"), ensure_ascii=False)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO analysis_runs
                    (analysis_id, case_id, created_at, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (decision.analysis_id, decision.case_id, self._utc_now(), payload),
            )

    def get_analysis(self, analysis_id: str) -> MultiviewDecision | None:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload_json FROM analysis_runs WHERE analysis_id = ?",
                (analysis_id,),
            ).fetchone()
        return MultiviewDecision.model_validate_json(row["payload_json"]) if row else None

    def latest_analysis(self, case_id: str) -> MultiviewDecision | None:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM analysis_runs
                WHERE case_id = ? ORDER BY created_at DESC LIMIT 1
                """,
                (case_id,),
            ).fetchone()
        return MultiviewDecision.model_validate_json(row["payload_json"]) if row else None

    @staticmethod
    def _record_from_row(row: sqlite3.Row) -> ReviewRecord:
        created_at = str(row["created_at"])
        return ReviewRecord(
            case_id=str(row["case_id"]),
            analysis_id=row["analysis_id"],
            revision=int(row["revision"]),
            facts=FoulFacts.model_validate_json(row["facts_json"]),
            assessment=RuleAssessment.model_validate_json(row["assessment_json"]),
            review_state=ReviewState(str(row["review_state"])),
            created_at=created_at,
            updated_at=created_at,
        )

    def latest_review(self, case_id: str) -> ReviewRecord | None:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM review_revisions
                WHERE case_id = ? ORDER BY revision DESC LIMIT 1
                """,
                (case_id,),
            ).fetchone()
        return self._record_from_row(row) if row else None

    def get_review_revision(self, case_id: str, revision: int) -> ReviewRecord | None:
        self._ensure_schema()
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM review_revisions
                WHERE case_id = ? AND revision = ?
                """,
                (case_id, revision),
            ).fetchone()
        return self._record_from_row(row) if row else None

    def review_history(self, case_id: str) -> list[ReviewRecord]:
        self._ensure_schema()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM review_revisions
                WHERE case_id = ? ORDER BY revision DESC
                """,
                (case_id,),
            ).fetchall()
        return [self._record_from_row(row) for row in rows]

    def append_review(
        self,
        *,
        case_id: str,
        expected_revision: int,
        facts: FoulFacts,
        assessment: RuleAssessment,
        analysis_id: str | None,
        review_state: ReviewState,
    ) -> ReviewRecord:
        self._ensure_schema()
        created_at = self._utc_now()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) AS revision FROM review_revisions WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            current_revision = int(row["revision"])
            if current_revision != expected_revision:
                connection.rollback()
                raise ReviewRevisionConflict(expected_revision, current_revision)
            next_revision = current_revision + 1
            connection.execute(
                """
                INSERT INTO review_revisions (
                    case_id, revision, analysis_id, review_state,
                    facts_json, assessment_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    next_revision,
                    analysis_id,
                    review_state.value,
                    facts.model_dump_json(),
                    assessment.model_dump_json(),
                    created_at,
                ),
            )
            connection.commit()
        return ReviewRecord(
            case_id=case_id,
            analysis_id=analysis_id,
            revision=next_revision,
            facts=facts,
            assessment=assessment,
            review_state=review_state,
            created_at=created_at,
            updated_at=created_at,
        )


def new_analysis_id(case_id: str) -> str:
    return f"analysis-{case_id}-{uuid.uuid4().hex[:12]}"
