"""Loads the trained CNN once and runs inference for the game."""

import sys
from pathlib import Path

import numpy as np

# ml/ is a sibling project with its own package (src.config.CONFIG,
# src.model.CNNModel) built for the training pipeline. Importing it
# here instead of hardcoding a second copy of the category list means
# the label order the app displays can never drift out of sync with
# what the model was trained on -- ml/ stays the single source of
# truth for both the weights and the label mapping. This resolves
# correctly regardless of the app's working directory, same pattern
# ml/scripts/train.py already uses for its own imports. It only works
# cleanly because this package is named "game", not "src" -- otherwise
# inserting ml/ onto sys.path would create two same-named "src"
# packages and Python would silently pick whichever came first.
_ML_ROOT = Path(__file__).resolve().parents[2] / "ml"
if str(_ML_ROOT) not in sys.path:
    sys.path.insert(0, str(_ML_ROOT))

from src.config import CONFIG as ML_CONFIG  # noqa: E402
from src.model import CNNModel  # noqa: E402


class Predictor:
    """Wraps the trained Keras model for single-image inference."""

    def __init__(self, ml_config=None):
        self.ml_config = ml_config or ML_CONFIG
        self.categories = self.ml_config.categories
        self.model = CNNModel.load(self.ml_config.trained_model_path)

    def predict_topk(self, model_input: np.ndarray, k: int = 3) -> list:
        """
        Args:
            model_input: shape (1, 28, 28, 1), float32, values in
                [0, 1] -- already preprocessed by CanvasProcessor.

        Returns:
            List of (category, confidence) tuples, length k, sorted
            descending by confidence.
        """
        probs = self.model.predict(model_input, verbose=0)[0]
        top_indices = np.argsort(probs)[::-1][:k]
        return [(self.categories[i], float(probs[i])) for i in top_indices]
