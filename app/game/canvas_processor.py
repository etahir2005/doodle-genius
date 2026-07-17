"""Converts a raw drawable-canvas frame into model-ready input."""

import numpy as np
from PIL import Image


class CanvasProcessor:
    """
    Mirrors ml/src/preprocessing.py's normalization, but for one live
    canvas frame instead of a batch of dataset images. The model can
    only be trusted on inputs matching the distribution it was trained
    on, so every step here reproduces what a Quick, Draw! bitmap looks
    like: 28x28, grayscale, black background with white strokes,
    scaled to [0, 1].
    """

    def __init__(self, image_size: int = 28, ink_pixel_threshold: int = 15):
        self.image_size = image_size
        self.ink_pixel_threshold = ink_pixel_threshold

    def has_ink(self, rgba: np.ndarray) -> bool:
        """
        True once enough has been drawn to bother running inference.

        The canvas starts as a blank black frame; running the model on
        that would just return noisy near-uniform probabilities across
        all 30 classes, so the app skips inference until there's a
        meaningful amount of bright (stroke) pixels.
        """
        if rgba is None:
            return False
        brightness = rgba[:, :, :3].mean(axis=2)
        return int((brightness > 40).sum()) > self.ink_pixel_threshold

    def to_model_input(self, rgba: np.ndarray) -> np.ndarray:
        """
        Args:
            rgba: canvas frame, shape (H, W, 4), uint8 -- image_data
                from streamlit-drawable-canvas, with the canvas
                configured for a black background and white strokes
                (see streamlit_app.py), so no inversion step is needed.

        Returns:
            Array of shape (1, image_size, image_size, 1), float32,
            scaled to [0, 1].
        """
        image = Image.fromarray(rgba[:, :, :3].astype(np.uint8)).convert("L")
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        return array.reshape(1, self.image_size, self.image_size, 1)
