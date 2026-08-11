#!/usr/bin/env python3
"""Validate a pitch-registration annotation pack before evaluation/training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.field_registration.annotations import validate_annotation_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="validate a newly generated pack before annotations have been added",
    )
    parser.add_argument(
        "--allow-no-contact-points",
        action="store_true",
        help="validate geometry-only camera-rig anchors without player contact points",
    )
    arguments = parser.parse_args()
    payload = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    report = validate_annotation_payload(
        payload,
        root=arguments.manifest.parent,
        require_annotated_frames=not arguments.allow_empty,
        require_contact_points=(
            not arguments.allow_empty and not arguments.allow_no_contact_points
        ),
    )
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if not report.valid:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
