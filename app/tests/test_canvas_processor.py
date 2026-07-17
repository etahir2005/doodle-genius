"""Unit tests for CanvasProcessor -- synthetic RGBA arrays only, no
real canvas or model needed."""

import numpy as np
import pytest

from game.canvas_processor import CanvasProcessor


@pytest.fixture
def processor():
    return CanvasProcessor(image_size=28, ink_pixel_threshold=15)


def blank_canvas(size=280):
    return np.zeros((size, size, 4), dtype=np.uint8)


def drawn_canvas(size=280):
    canvas = np.zeros((size, size, 4), dtype=np.uint8)
    canvas[100:150, 100:150, :3] = 255
    canvas[100:150, 100:150, 3] = 255
    return canvas


def test_has_ink_false_on_blank_canvas(processor):
    assert processor.has_ink(blank_canvas()) is False


def test_has_ink_false_on_none(processor):
    assert processor.has_ink(None) is False


def test_has_ink_true_once_drawn(processor):
    assert processor.has_ink(drawn_canvas()) is True


def test_to_model_input_shape(processor):
    result = processor.to_model_input(drawn_canvas())
    assert result.shape == (1, 28, 28, 1)


def test_to_model_input_value_range(processor):
    result = processor.to_model_input(drawn_canvas())
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_to_model_input_dtype(processor):
    result = processor.to_model_input(drawn_canvas())
    assert result.dtype == np.float32
