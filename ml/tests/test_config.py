"""Unit tests for src.config.Config."""

import pytest

from src.config import Config


def test_image_shape_derived_from_image_size():
    config = Config(image_size=28)

    assert config.image_shape == (28, 28, 1)


def test_num_classes_derived_from_categories():
    config = Config(categories=["a", "b", "c"])

    assert config.num_classes == 3


def test_derived_paths_are_relative_to_ml_root():
    config = Config()

    assert config.data_raw_dir == config.ml_root / "data" / "raw"
    assert config.trained_model_path == config.models_dir / "doodle_genius_cnn.keras"


def test_empty_categories_raises_value_error():
    with pytest.raises(ValueError):
        Config(categories=[])


def test_split_ratios_summing_to_one_or_more_raises_value_error():
    with pytest.raises(ValueError):
        Config(validation_split=0.6, test_split=0.5)
