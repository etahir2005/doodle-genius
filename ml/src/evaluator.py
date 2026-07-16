"""Evaluation orchestration for the DoodleGenius CNN."""

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from src.config import Config
from src.model import CNNModel

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Orchestrates evaluation of a trained model against the held-out
    test set: accuracy/loss, per-class precision/recall/F1, and a
    confusion matrix.
    """

    def __init__(self, config: Config):
        self.config = config

    def load_model_and_test_set(self) -> tuple:
        """
        Load the trained model and test split from disk.

        Returns:
            Tuple of (keras_model, X_test, y_test).

        Raises:
            FileNotFoundError: If the model or processed dataset doesn't exist yet.
        """
        model_path = self.config.trained_model_path
        dataset_path = self.config.processed_dataset_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run scripts/train.py first."
            )
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at {dataset_path}. "
                "Run scripts/prepare_dataset.py first."
            )

        model = CNNModel.load(model_path)
        data = np.load(dataset_path)
        return model, data["X_test"], data["y_test"]

    def plot_confusion_matrix(self, matrix: np.ndarray) -> None:
        """Save a heatmap of the confusion matrix to disk."""
        categories = self.config.categories
        fig, ax = plt.subplots(figsize=(11, 11))
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(range(len(categories)))
        ax.set_yticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=90)
        ax.set_yticklabels(categories)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("DoodleGenius Confusion Matrix")
        fig.colorbar(im)
        fig.tight_layout()
        fig.savefig(self.config.confusion_matrix_plot_path)
        logger.info(
            "Saved confusion matrix plot to %s", self.config.confusion_matrix_plot_path
        )

    def run(self) -> dict:
        """
        Execute the full evaluation and persist the report.

        Returns:
            The evaluation report dict that was also written to disk.
        """
        model, X_test, y_test = self.load_model_and_test_set()

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        logger.info("Test accuracy: %.4f | Test loss: %.4f", test_accuracy, test_loss)

        predictions = np.argmax(model.predict(X_test, verbose=0), axis=1)
        report = classification_report(
            y_test, predictions, target_names=self.config.categories, output_dict=True
        )
        matrix = confusion_matrix(y_test, predictions)
        self.plot_confusion_matrix(matrix)

        result = {
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "per_class": report,
        }
        # default=float coerces numpy float64/int64 (returned by sklearn's
        # classification_report) into JSON-serializable native floats.
        self.config.evaluation_report_path.write_text(
            json.dumps(result, indent=2, default=float)
        )
        logger.info("Saved evaluation report to %s", self.config.evaluation_report_path)

        return result
