# DoodleGenius

A real-time drawing game where a custom-trained convolutional neural
network guesses what you're sketching as you draw — inspired by
Google's Quick, Draw! Unlike projects that call an existing model or
external API, DoodleGenius trains, evaluates, exports, and deploys
its own CNN from scratch.

## Status
Day 2 of 2 — ML training pipeline and Streamlit game UI both complete.

## How It Works
1. **Train** (Python / TensorFlow, on a Kaggle GPU notebook) — a CNN
   is trained on a curated 30-category subset of Google's Quick,
   Draw! dataset.
2. **Play** (Day 2, Streamlit) — the trained model is loaded
   in-process and runs inference server-side as the user draws on a
   canvas, with a timer, scoring, and round history.

## Tech Stack
- Python, TensorFlow/Keras — model training (Kaggle GPU notebook)
- Google Quick, Draw! dataset (`numpy_bitmap` format)
- Streamlit — game UI (Day 2)

## Branching
`main` (stable) ← `development` (integration) ← `feature/*` (work branches).

## Project Structure
doodle-genius/
├── ml/       # Python training pipeline (Day 1)
├── app/      # Streamlit game UI (Day 2)
└── .github/workflows/

## Day 1: ML Training Pipeline
See [`ml/README.md`](ml/README.md) for setup and usage.

## Day 2: Game UI
See [`app/README.md`](app/README.md) for setup, usage, and gameplay rules.

## License
MIT — see [LICENSE](LICENSE).