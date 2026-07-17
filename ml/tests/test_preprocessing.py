"""Unit tests for src.preprocessing.Preprocessor."""

import numpy as np

from src.config import Config
from src.preprocessing import Preprocessor


def _test_config() -> Config:
    return Config(categories=["a", "b", "c", "d"])


def test_normalize_scales_to_unit_range():
    preprocessor = Preprocessor(_test_config())
    raw = np.array([[0, 255] * 392], dtype=np.uint8)  # shape (1, 784)

    normalized = preprocessor.normalize(raw)

    assert normalized.shape == (1, 28, 28, 1)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_split_preserves_total_sample_count():
    preprocessor = Preprocessor(_test_config())
    images = np.random.rand(200, 28, 28, 1).astype("float32")
    labels = np.array([i % 4 for i in range(200)])

    splits = preprocessor.split(images, labels)

    total = len(splits["X_train"]) + len(splits["X_val"]) + len(splits["X_test"])
    assert total == 200


def test_split_stratifies_across_all_classes():
    preprocessor = Preprocessor(_test_config())
    images = np.random.rand(400, 28, 28, 1).astype("float32")
    labels = np.array([i % 4 for i in range(400)])

    splits = preprocessor.split(images, labels)

    assert set(np.unique(splits["y_train"])) == {0, 1, 2, 3}
    assert set(np.unique(splits["y_val"])) == {0, 1, 2, 3}
    assert set(np.unique(splits["y_test"])) == {0, 1, 2, 3}
