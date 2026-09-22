from __future__ import annotations

import argparse
import logging

from app.multiview.live_ingest.config import LiveIngestConfigError, load_live_ingest_config
from app.multiview.live_ingest.supervisor import LiveIngestSupervisor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated three-camera RTSP/H.264 multiview ingest."
    )
    parser.add_argument(
        "--config",
        help="Protected JSON configuration file. Defaults to SC_MULTIVIEW_LIVE_CONFIG.",
    )
    args = parser.parse_args()
    try:
        config = load_live_ingest_config(args.config)
    except LiveIngestConfigError as exc:
        parser.error(str(exc))
    logging.info("Starting isolated live multiview ingest for %d cameras", len(config.cameras))
    LiveIngestSupervisor(config).run_forever()


if __name__ == "__main__":
    main()
