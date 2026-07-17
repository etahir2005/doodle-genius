"""Unit tests for src.model.CNNModel."""

import numpy as np

from src.config import Config
from src.model import CNNModel


def _test_config() -> Config:
    return Config(categories=["a", "b", "c", "d"])


def test_cnn_model_output_shape_matches_category_count():
    cnn = CNNModel(_test_config())

    assert cnn.keras_model.output_shape == (None, 4)


def test_cnn_model_accepts_expected_input_shape():
    cnn = CNNModel(_test_config())

    assert cnn.keras_model.input_shape == (None, 28, 28, 1)


def test_cnn_model_predictions_are_a_valid_probability_distribution():
    cnn = CNNModel(_test_config())
    dummy_input = np.zeros((2, 28, 28, 1), dtype="float32")

    predictions = cnn.keras_model.predict(dummy_input, verbose=0)

    assert predictions.shape == (2, 4)
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=1e-5)
