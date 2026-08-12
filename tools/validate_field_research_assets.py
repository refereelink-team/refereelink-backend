#!/usr/bin/env python3
"""Validate the field-registration research registry and optional local files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.field_registration.research_assets import (
    load_research_assets,
    verify_registered_files,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("research/field_registration/assets.json"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    arguments = parser.parse_args()
    assets = load_research_assets(arguments.registry)
    issues = verify_registered_files(assets, root=arguments.root)
    report = {
        "valid": not issues,
        "asset_count": len(assets),
        "runtime_allowed_count": sum(asset.allowed_in_runtime for asset in assets),
        "unverified_license_count": sum(asset.license == "UNVERIFIED" for asset in assets),
        "issues": list(issues),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
