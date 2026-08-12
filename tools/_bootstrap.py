"""Make direct ``python tools/<command>.py`` calls use this checkout."""

from __future__ import annotations

from pathlib import Path
import sys


def ensure_repository_root(script_file: str) -> Path:
    """Put the script's repository root before any editable global checkout."""

    root = Path(script_file).resolve().parents[1]
    root_text = str(root)
    sys.path[:] = [entry for entry in sys.path if entry != root_text]
    sys.path.insert(0, root_text)
    return root
