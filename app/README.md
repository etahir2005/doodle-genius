# DoodleGenius — Game UI

Streamlit app that loads the CNN trained in `ml/` and plays a
real-time drawing-guessing game against it.

## Setup
```bash
cd app
python -m venv venv
source venv/bin/activate   # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run
```bash
streamlit run streamlit_app.py
```

## Gameplay
- A random word is picked from the 30 categories `ml/src/config.py` trains on.
- Each round lasts `AppConfig.round_seconds` (20s by default) -- draw the
  word before time runs out.
- The model guesses live as you draw; if its top guess matches the word
  before time's up, the round is scored a win.
- Scoring: `AppConfig.min_points`-`AppConfig.max_points` (10-100),
  scaled linearly by how much round time was left when the correct
  guess landed -- faster, clearer drawings score more. A wrong guess
  or a timeout scores 0.
- Modeled on Google's Quick, Draw!: the model's job is to recognize
  the sketch, the player's job is to draw clearly and quickly enough
  that it can -- a correct AI guess is a player win, not a loss.
- Full rules and a worked scoring example are shown in-app under
  "How to play".

## How it works
- `game/predictor.py` loads the trained model from `ml/models/` and
  reuses `ml/src`'s `Config`/`CNNModel` directly, so the category
  labels can never drift out of sync with what the model was trained
  on.
- `game/canvas_processor.py` converts the live drawing canvas into the
  same 28x28 grayscale, black-background/white-stroke format the
  model was trained on.
- `game/game_state.py` manages round lifecycle, scoring, and history
  through an injected state mapping (`st.session_state` in
  production), so it's fully unit-testable without a Streamlit
  runtime.
- The round timer and live predictions update via
  `st.fragment(run_every=...)`, not a full-page refresh.
- The canvas widget is `streamlit-drawable-canvas-fix`, a maintained
  fork used because the original `streamlit-drawable-canvas` package
  is archived and incompatible with current Streamlit.

## Tests
```bash
pytest tests/
```