"""Versioned registry for non-runtime field-registration research assets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


class ResearchAssetError(ValueError):
    """Raised when a research asset registry is incomplete or unsafe."""


@dataclass(frozen=True)
class ResearchAsset:
    asset_id: str
    kind: str
    source_url: str
    version: str
    license: str
    data_use: str
    training_data: str
    allowed_in_runtime: bool
    notes: str = ""
    files: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ResearchAsset":
        required = (
            "id",
            "kind",
            "source_url",
            "version",
            "license",
            "data_use",
            "training_data",
            "allowed_in_runtime",
        )
        missing = [name for name in required if name not in payload]
        if missing:
            raise ResearchAssetError(f"asset is missing fields: {', '.join(missing)}")
        parsed = urlparse(str(payload["source_url"]))
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ResearchAssetError(f"asset {payload['id']!r} has an invalid source_url")
        files_payload = payload.get("files", [])
        files: list[tuple[str, str]] = []
        for item in files_payload:
            sha256 = str(item.get("sha256", "")).lower()
            if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
                raise ResearchAssetError(
                    f"asset {payload['id']!r} has an invalid SHA-256 entry"
                )
            files.append((str(item["path"]), sha256))
        return cls(
            asset_id=str(payload["id"]),
            kind=str(payload["kind"]),
            source_url=str(payload["source_url"]),
            version=str(payload["version"]),
            license=str(payload["license"]),
            data_use=str(payload["data_use"]),
            training_data=str(payload["training_data"]),
            allowed_in_runtime=bool(payload["allowed_in_runtime"]),
            notes=str(payload.get("notes", "")),
            files=tuple(files),
        )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_research_assets(path: str | Path) -> tuple[ResearchAsset, ...]:
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    if int(payload.get("format_version", 0)) != 1:
        raise ResearchAssetError("unsupported research asset registry format")
    assets = tuple(ResearchAsset.from_payload(item) for item in payload.get("assets", []))
    if not assets:
        raise ResearchAssetError("research asset registry is empty")
    identifiers = [asset.asset_id for asset in assets]
    if len(identifiers) != len(set(identifiers)):
        raise ResearchAssetError("research asset ids must be unique")
    return assets


def verify_registered_files(
    assets: Iterable[ResearchAsset],
    *,
    root: str | Path,
) -> tuple[str, ...]:
    root_path = Path(root)
    issues: list[str] = []
    for asset in assets:
        for relative_path, expected_hash in asset.files:
            candidate = root_path / relative_path
            if not candidate.is_file():
                issues.append(f"{asset.asset_id}: missing {relative_path}")
            elif sha256_file(candidate) != expected_hash:
                issues.append(f"{asset.asset_id}: SHA-256 mismatch for {relative_path}")
    return tuple(issues)
