"""Entry point: download raw Quick, Draw! bitmap files.

Usage:
    python scripts/download_dataset.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG  # noqa: E402
from src.downloader import DatasetDownloader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        DatasetDownloader(CONFIG).run()
    except RuntimeError:
        logger.exception("Dataset download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
