"""Game-specific configuration for the Day 2 Streamlit app."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    """
    Constants that only matter to gameplay -- separate from
    ml/src/config.py's Config, which governs the training pipeline.
    Keeping these apart means tweaking round length or scoring never
    risks touching anything that affects the model itself.
    """

    round_seconds: int = 20
    top_k: int = 3
    refresh_interval_ms: int = 500
    min_points: int = 10
    max_points: int = 100
    canvas_size: int = 280
    stroke_width: int = 14
