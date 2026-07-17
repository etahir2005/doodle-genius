"""Downloads raw Quick, Draw! bitmap files."""

import logging
from urllib.parse import quote

import requests

from src.config import Config
from src.dataset import DatasetLoader

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 60
CHUNK_SIZE_BYTES = 1024 * 1024


class DatasetDownloader:
    """
    Downloads the raw numpy_bitmap .npy file for every category in
    Config.categories from Google's public Quick, Draw! bucket.
    Idempotent — already-downloaded files are skipped.
    """

    def __init__(self, config: Config):
        self.config = config

    def download_category(self, category: str) -> None:
        """
        Download a single category's .npy file if not already present.

        Downloads to a .partial temp file first and renames on
        success, so an interrupted download never leaves a corrupt
        file at the final path that a later run would mistake for
        "already done".

        Raises:
            requests.RequestException: If the download fails.
        """
        destination = (
            self.config.data_raw_dir
            / f"{DatasetLoader.category_to_filename(category)}.npy"
        )
        if destination.exists():
            logger.info("Skipping '%s' — already downloaded", category)
            return

        url = f"{self.config.quickdraw_base_url}/{quote(category)}.npy"
        logger.info("Downloading '%s' from %s", category, url)

        response = requests.get(url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()

        temp_path = destination.with_suffix(".npy.partial")
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE_BYTES):
                f.write(chunk)
        temp_path.rename(destination)

        logger.info(
            "Saved '%s' (%.1f MB)", category, destination.stat().st_size / 1e6
        )

    def run(self) -> None:
        """
        Download every configured category.

        Raises:
            RuntimeError: If one or more categories failed to
                download, listing which ones — the caller decides
                whether/how to exit.
        """
        self.config.data_raw_dir.mkdir(parents=True, exist_ok=True)

        failed_categories = []
        for category in self.config.categories:
            try:
                self.download_category(category)
            except requests.RequestException:
                logger.exception("Failed to download category '%s'", category)
                failed_categories.append(category)

        if failed_categories:
            raise RuntimeError(f"Failed to download categories: {failed_categories}")

        logger.info(
            "All %d categories downloaded successfully", len(self.config.categories)
        )
