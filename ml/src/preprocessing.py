"""Data preprocessing: normalization, reshaping, and dataset splitting."""

import logging

import numpy as np
from sklearn.model_selection import train_test_split

from src.config import Config

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    Transforms raw bitmap arrays into a model-ready, split dataset.

    Owns normalization/reshaping and stratified train/val/test
    splitting only — loading from disk is DatasetLoader's job,
    hyperparameters live in Config.
    """

    def __init__(self, config: Config):
        self.config = config

    def normalize(self, images: np.ndarray) -> np.ndarray:
        """
        Reshape flat uint8 bitmaps into normalized float32 tensors.

        Args:
            images: Array of shape (n_samples, 784), dtype uint8.

        Returns:
            Array of shape (n_samples, image_size, image_size, 1),
            dtype float32, scaled to [0, 1].
        """
        size = self.config.image_size
        reshaped = images.reshape(-1, size, size, 1)
        return reshaped.astype("float32") / 255.0

    def split(self, images: np.ndarray, labels: np.ndarray) -> dict:
        """
        Split images/labels into stratified train, validation, and test sets.

        Args:
            images: Normalized image array, shape (n_samples, H, W, 1).
            labels: Integer label array, shape (n_samples,).

        Returns:
            Dict with keys "X_train", "y_train", "X_val", "y_val",
            "X_test", "y_test".
        """
        val_split = self.config.validation_split
        test_split = self.config.test_split
        seed = self.config.random_seed

        X_train, X_temp, y_train, y_temp = train_test_split(
            images, labels,
            test_size=val_split + test_split,
            random_state=seed,
            stratify=labels,
        )

        relative_test_size = test_split / (val_split + test_split)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=relative_test_size,
            random_state=seed,
            stratify=y_temp,
        )

        logger.info(
            "Split dataset: train=%d, val=%d, test=%d",
            len(X_train), len(X_val), len(X_test),
        )

        return {
            "X_train": X_train, "y_train": y_train,
            "X_val": X_val, "y_val": y_val,
            "X_test": X_test, "y_test": y_test,
        }

    def run(self, raw_images: np.ndarray, labels: np.ndarray) -> dict:
        """Convenience method: normalize then split in one call."""
        return self.split(self.normalize(raw_images), labels)
