"""Entry point: train the DoodleGenius CNN.

Usage:
    python scripts/train.py
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG  # noqa: E402
from src.trainer import Trainer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    Trainer(CONFIG).run()


if __name__ == "__main__":
    main()
