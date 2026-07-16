# DoodleGenius — ML Training Pipeline

Trains a CNN to classify 28x28 grayscale doodle bitmaps from a
30-category subset of Google's Quick, Draw! dataset.

## Categories (30)
airplane, apple, banana, bicycle, bird, book, car, cat, chair, clock,
cloud, cup, dog, door, fish, flower, guitar, hat, house, ice cream,
key, ladder, moon, mountain, pizza, shoe, star, sun, tree, umbrella

This list must stay in sync with `CATEGORIES` in `src/config.py` —
that file is the source of truth; this is just for readability.

## Model Details
- **Architecture**: 3 convolutional blocks (32 → 64 → 64 filters, 3x3
  kernels, ReLU, max pooling) → flatten → dense(128, ReLU) →
  dropout(0.3) → dense(30, softmax)
- **Input**: 28x28 grayscale bitmap, normalized to [0, 1]
- **Loss**: sparse categorical crossentropy
- **Optimizer**: Adam, learning rate 1e-3
- **Batch size**: 256 (tuned for GPU throughput on Kaggle T4)
- **Epochs**: up to 30, with early stopping (patience 4 on
  validation loss) and learning-rate reduction on plateau
- **Data split**: 80/10/10 train/validation/test, stratified by category

## Data Preparation
- **Source**: raw `.npy` bitmap files downloaded per category from
  Google's public Quick, Draw! bucket (no auth required)
- **Sampling**: capped at 12,000 drawings per category (raw category
  files contain 100k-300k+; the cap keeps download size and dataset
  size manageable). Subsampling uses a fixed random seed (42) for
  reproducibility — the same 12,000 drawings are selected every run.
- **Normalization**: flat 784-value uint8 arrays reshaped to
  (28, 28, 1) and scaled from [0, 255] to [0, 1] float32
- **Splitting**: stratified 80/10/10 train/validation/test split
  (`sklearn.train_test_split`, stratified by category, same fixed
  seed) — stratification guarantees every category is proportionally
  represented in all three splits, which matters since per-class
  accuracy is a key evaluation metric
- **Output**: saved as a single compressed `.npz`
  (`data/processed/dataset.npz`) so training doesn't repeat this work

## Recommended: Train on Kaggle (GPU)
Clone this branch into a Kaggle Notebook with GPU + internet enabled,
run the four pipeline scripts there, then download `models/` back
into this folder afterward. See the project's Kaggle notebook for
the exact cell-by-cell walkthrough.

## Local setup (CPU fallback)
```bash
cd ml
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Pipeline (same commands locally or on Kaggle)
1. `python scripts/download_dataset.py` — downloads 30 categories,
   several GB total; safe to re-run, skips completed files.
2. `python scripts/prepare_dataset.py` — normalizes and splits into
   train/val/test.
3. `python scripts/train.py` — trains the CNN with early stopping.
4. `python scripts/evaluate.py` — reports test accuracy, per-class
   precision/recall/F1, and a confusion matrix.

## Architecture
All logic lives in `src/` as small, testable classes; `scripts/`
contains only thin entry points that wire those classes together:

- `Config` — single source of truth for paths, categories, hyperparameters
- `DatasetDownloader` — downloads raw Quick, Draw! `.npy` files
- `DatasetLoader` — reads and subsamples raw `.npy` files
- `Preprocessor` — normalizes, reshapes, splits into train/val/test
- `CNNModel` — architecture, compilation, training callbacks
- `Trainer` — orchestrates a training run and saves artifacts
- `Evaluator` — orchestrates evaluation and saves the report

## Configuration
All categories, dataset sizing, and hyperparameters live in
`src/config.py`'s `Config` dataclass. Nothing else needs to change to
adjust them.

## Tests
```bash
pytest tests/
```

## Outputs
- `models/doodle_genius_cnn.keras` — trained model
- `models/training_history.png` — accuracy/loss curves
- `models/confusion_matrix.png` — per-class confusion matrix
- `models/evaluation_report.json` — test accuracy + per-class precision/recall/F1