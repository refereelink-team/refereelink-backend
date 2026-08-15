from __future__ import annotations

import json
import os
from pathlib import Path

from app.constants.paths import REPO_ROOT_DIR
from app.multiview.models import MultiviewCase

DEFAULT_CASES_PATH = REPO_ROOT_DIR / "assets" / "multiview" / "cases.json"


class MultiviewCaseRepository:
    def __init__(self, cases_path: str | Path | None = None) -> None:
        configured = cases_path or os.environ.get("SC_MULTIVIEW_CASES_PATH")
        self.cases_path = Path(configured).expanduser() if configured else DEFAULT_CASES_PATH

    def list_cases(self) -> list[MultiviewCase]:
        if not self.cases_path.is_file():
            return []
        payload = json.loads(self.cases_path.read_text(encoding="utf-8"))
        raw_cases = payload if isinstance(payload, list) else payload.get("cases", [])
        return [MultiviewCase.model_validate(item) for item in raw_cases]

    def get_case(self, case_id: str) -> MultiviewCase | None:
        return next((case for case in self.list_cases() if case.case_id == case_id), None)

    @staticmethod
    def resolve_path(raw_path: str | None) -> Path | None:
        if not raw_path:
            return None
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT_DIR / path
        return path.resolve()

    def resolve_view_media(self, case_id: str, camera_id: str) -> tuple[Path, str] | None:
        case = self.get_case(case_id)
        if case is None:
            return None
        view = next((item for item in case.videos if item.camera_id == camera_id), None)
        if view is None:
            return None
        video_path = self.resolve_path(view.path)
        if video_path is not None and video_path.is_file() and video_path.stat().st_size > 1_000:
            return video_path, "video/mp4"
        preview_path = self.resolve_path(view.preview_path)
        if preview_path is not None and preview_path.is_file():
            suffix = preview_path.suffix.lower()
            media_type = "image/png" if suffix == ".png" else "image/jpeg"
            return preview_path, media_type
        return None

    def available_video_paths(self, case: MultiviewCase) -> list[Path]:
        paths: list[Path] = []
        for view in case.videos:
            path = self.resolve_path(view.path)
            if path is not None and path.is_file() and path.stat().st_size > 1_000:
                paths.append(path)
        return paths
