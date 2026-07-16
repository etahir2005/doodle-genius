"""Raw Quick, Draw! bitmap loading."""

import logging

import numpy as np

from src.config import Config

logger = logging.getLogger(__name__)


class MissingCategoryFileError(FileNotFoundError):
    """Raised when a category's raw .npy file has not been downloaded."""


class DatasetLoader:
    """
    Loads raw Quick, Draw! bitmap arrays from disk.

    Owns exactly one responsibility: reading downloaded .npy files
    and assembling them into raw (unnormalized, unsplit) NumPy
    arrays. Normalizing, reshaping, and splitting are Preprocessor's
    job, not this class's.
    """

    def __init__(self, config: Config):
        self.config = config

    @staticmethod
    def category_to_filename(category: str) -> str:
        """
        Convert a category name into a filesystem-safe filename stem.

        Multi-word categories (e.g. "ice cream") contain spaces,
        which are awkward in shell commands and inconsistent across
        platforms. This is the single source of truth for that
        mapping, shared with DatasetDownloader so the two never
        disagree on where a category's file lives.
        """
        return category.replace(" ", "_")

    def load_category(self, category: str) -> np.ndarray:
        """
        Load and subsample the raw bitmap array for a single category.

        Args:
            category: Category name matching a downloaded .npy file.

        Returns:
            Array of shape (n_samples, 784), dtype uint8, where
            n_samples <= config.samples_per_category.

        Raises:
            MissingCategoryFileError: If the .npy file is missing.
        """
        file_path = (
            self.config.data_raw_dir
            / f"{self.category_to_filename(category)}.npy"
        )
        if not file_path.exists():
            raise MissingCategoryFileError(
                f"Raw data file not found for category '{category}': {file_path}. "
                "Run scripts/download_dataset.py first."
            )

        data = np.load(file_path)

        rng = np.random.default_rng(self.config.random_seed)
        if len(data) > self.config.samples_per_category:
            indices = rng.choice(
                len(data), size=self.config.samples_per_category, replace=False
            )
            data = data[indices]

        logger.info("Loaded %d samples for category '%s'", len(data), category)
        return data

    def load_all(self) -> tuple:
        """
        Load raw bitmap arrays and integer labels for every configured category.

        Returns:
            Tuple of (images, labels):
                images: shape (total_samples, 784), dtype uint8.
                labels: shape (total_samples,), dtype int64 — each
                    value is the category's index in config.categories.

        Raises:
            MissingCategoryFileError: Propagated if any category file is missing.
        """
        all_images = []
        all_labels = []

        for label_index, category in enumerate(self.config.categories):
            images = self.load_category(category)
            all_images.append(images)
            all_labels.append(np.full(len(images), label_index, dtype=np.int64))

        return np.concatenate(all_images, axis=0), np.concatenate(all_labels, axis=0)
