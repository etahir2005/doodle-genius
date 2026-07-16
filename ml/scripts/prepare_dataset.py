"""Entry point: transform raw bitmaps into a model-ready dataset.

Usage:
    python scripts/prepare_dataset.py
"""

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import CONFIG  # noqa: E402
from src.dataset import DatasetLoader, MissingCategoryFileError  # noqa: E402
from src.preprocessing import Preprocessor  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    CONFIG.data_processed_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_images, labels = DatasetLoader(CONFIG).load_all()
    except MissingCategoryFileError:
        logger.exception("Cannot prepare dataset — missing raw data")
        sys.exit(1)

    splits = Preprocessor(CONFIG).run(raw_images, labels)

    np.savez_compressed(CONFIG.processed_dataset_path, **splits)
    logger.info("Saved processed dataset to %s", CONFIG.processed_dataset_path)


if __name__ == "__main__":
    main()
