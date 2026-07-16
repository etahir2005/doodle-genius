"""Training orchestration for the DoodleGenius CNN."""

import logging

import matplotlib.pyplot as plt
import numpy as np

from src.config import Config
from src.model import CNNModel

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates a single training run: loads the processed dataset,
    builds the model, fits it, and persists training artifacts (best
    checkpoint via callback, accuracy/loss plot).
    """

    def __init__(self, config: Config):
        self.config = config

    def load_processed_dataset(self) -> dict:
        """
        Load the train/val splits produced by prepare_dataset.py.

        Returns:
            Dict with "X_train", "y_train", "X_val", "y_val" arrays.

        Raises:
            FileNotFoundError: If the processed dataset doesn't exist yet.
        """
        path = self.config.processed_dataset_path
        if not path.exists():
            raise FileNotFoundError(
                f"Processed dataset not found at {path}. "
                "Run scripts/prepare_dataset.py first."
            )
        data = np.load(path)
        return {key: data[key] for key in data.files}

    def plot_history(self, history) -> None:
        """Save accuracy/loss curves for the training run to disk."""
        fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(12, 4))

        ax_acc.plot(history.history["accuracy"], label="train")
        ax_acc.plot(history.history["val_accuracy"], label="validation")
        ax_acc.set_title("Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.legend()

        ax_loss.plot(history.history["loss"], label="train")
        ax_loss.plot(history.history["val_loss"], label="validation")
        ax_loss.set_title("Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.legend()

        fig.tight_layout()
        fig.savefig(self.config.training_history_plot_path)
        logger.info(
            "Saved training history plot to %s", self.config.training_history_plot_path
        )

    def run(self) -> CNNModel:
        """
        Execute the full training run.

        Returns:
            The CNNModel wrapper; its keras_model holds the best
            weights restored by early stopping.
        """
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

        data = self.load_processed_dataset()
        cnn = CNNModel(self.config)
        cnn.keras_model.summary(print_fn=logger.info)

        history = cnn.keras_model.fit(
            data["X_train"], data["y_train"],
            validation_data=(data["X_val"], data["y_val"]),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            callbacks=cnn.get_callbacks(),
            verbose=2,
        )

        self.plot_history(history)
        logger.info("Training complete. Best model saved via ModelCheckpoint.")
        return cnn
