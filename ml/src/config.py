"""Centralized configuration for the DoodleGenius ML pipeline.

Config is a frozen dataclass rather than module-level constants so it
can be constructed with overrides — every other stateful component
(DatasetLoader, DatasetDownloader, Preprocessor, CNNModel, Trainer,
Evaluator) takes a Config instance in its constructor instead of
importing values directly, which also makes each of them trivially
testable with a small, fast Config (e.g. 4 categories instead of 30).
"""

from dataclasses import dataclass, field
from pathlib import Path


def _default_categories() -> list:
    # 30 categories chosen for visual distinctiveness and broad
    # coverage across everyday object types. A category's index
    # position doubles as its integer label throughout training and
    # inference, and will be mirrored in the Streamlit app on Day 2.
    return [
        "airplane", "apple", "banana", "bicycle", "bird",
        "book", "car", "cat", "chair", "clock",
        "cloud", "cup", "dog", "door", "fish",
        "flower", "guitar", "hat", "house", "ice cream",
        "key", "ladder", "moon", "mountain", "pizza",
        "shoe", "star", "sun", "tree", "umbrella",
    ]


@dataclass(frozen=True)
class Config:
    """Single source of truth for pipeline paths, categories, and hyperparameters."""

    ml_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent
    )
    quickdraw_base_url: str = (
        "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"
    )
    categories: list = field(default_factory=_default_categories)

    image_size: int = 28
    # Cap is a network/disk consideration (download size) now that
    # training runs on a Kaggle GPU, not a compute one.
    samples_per_category: int = 12000
    validation_split: float = 0.1
    test_split: float = 0.1
    random_seed: int = 42

    # BATCH_SIZE raised for GPU parallelism (Kaggle T4); EPOCHS is a
    # ceiling only — early_stopping_patience governs actual duration.
    batch_size: int = 256
    epochs: int = 30
    learning_rate: float = 1e-3
    early_stopping_patience: int = 4

    def __post_init__(self) -> None:
        """
        Validate configuration values that would otherwise fail
        confusingly downstream.
        """
        if not self.categories:
            raise ValueError("Config.categories must not be empty")
        if self.validation_split + self.test_split >= 1.0:
            raise ValueError("validation_split + test_split must be < 1.0")

    @property
    def image_shape(self) -> tuple:
        return (self.image_size, self.image_size, 1)

    @property
    def num_classes(self) -> int:
        return len(self.categories)

    @property
    def data_raw_dir(self) -> Path:
        return self.ml_root / "data" / "raw"

    @property
    def data_processed_dir(self) -> Path:
        return self.ml_root / "data" / "processed"

    @property
    def models_dir(self) -> Path:
        return self.ml_root / "models"

    @property
    def processed_dataset_path(self) -> Path:
        return self.data_processed_dir / "dataset.npz"

    @property
    def trained_model_path(self) -> Path:
        return self.models_dir / "doodle_genius_cnn.keras"

    @property
    def training_history_plot_path(self) -> Path:
        return self.models_dir / "training_history.png"

    @property
    def confusion_matrix_plot_path(self) -> Path:
        return self.models_dir / "confusion_matrix.png"

    @property
    def evaluation_report_path(self) -> Path:
        return self.models_dir / "evaluation_report.json"


CONFIG = Config()
