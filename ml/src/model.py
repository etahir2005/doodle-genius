"""CNN architecture for DoodleGenius."""

import tensorflow as tf
from tensorflow.keras import layers, models

from src.config import Config


class CNNModel:
    """
    Wraps the DoodleGenius CNN architecture, compilation, and the
    Keras training callbacks it needs.

    Architecture: three convolutional blocks with increasing filter
    depth, each followed by max pooling, then a dense classification
    head with dropout for regularization. Sized for 28x28 grayscale
    doodle bitmaps across 20-30 categories — deliberately unchanged
    since compute (Kaggle GPU) was never the constraint on model size.
    """

    def __init__(self, config: Config):
        self.config = config
        self.keras_model = self._build()

    def _build(self) -> tf.keras.Model:
        model = models.Sequential([
            layers.Input(shape=self.config.image_shape),
            layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(self.config.num_classes, activation="softmax"),
        ], name="doodle_genius_cnn")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def get_callbacks(self) -> list:
        """
        Build the standard callback set used for every training run.

        Returns:
            List of Keras callbacks: early stopping on validation
            loss, checkpointing the best model, and learning-rate
            reduction on plateau.
        """
        return [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=self.config.early_stopping_patience,
                restore_best_weights=True,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(self.config.trained_model_path),
                monitor="val_accuracy",
                save_best_only=True,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-6,
            ),
        ]

    @staticmethod
    def load(model_path) -> tf.keras.Model:
        """Load a previously trained model from disk."""
        return tf.keras.models.load_model(model_path)
