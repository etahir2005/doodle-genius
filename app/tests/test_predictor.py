"""Integration test for Predictor -- skipped if the trained model
isn't present, since it's a real 5MB artifact rather than something to
fake."""

import numpy as np
import pytest

from game.predictor import ML_CONFIG, Predictor

MODEL_EXISTS = ML_CONFIG.trained_model_path.exists()


@pytest.mark.skipif(not MODEL_EXISTS, reason="trained model artifact not present")
def test_predict_topk_returns_k_sorted_predictions():
    predictor = Predictor()
    dummy_input = np.random.rand(1, 28, 28, 1).astype(np.float32)

    results = predictor.predict_topk(dummy_input, k=3)

    assert len(results) == 3
    categories, confidences = zip(*results)
    assert all(c in predictor.categories for c in categories)
    assert list(confidences) == sorted(confidences, reverse=True)
