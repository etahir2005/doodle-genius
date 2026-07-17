"""
DoodleGenius -- Day 2 game UI.

Run with: streamlit run app/streamlit_app.py

Kept thin on purpose -- same convention as ml/scripts/: all real logic
lives in app/game/ as small classes, this file just wires them
together and renders the page.
"""

import sys
from pathlib import Path

import streamlit as st
from streamlit_drawable_canvas import st_canvas

sys.path.insert(0, str(Path(__file__).resolve().parent))

from game.canvas_processor import CanvasProcessor  # noqa: E402
from game.config import AppConfig  # noqa: E402
from game.game_state import GameState  # noqa: E402
from game.predictor import Predictor  # noqa: E402

st.set_page_config(
    page_title="DoodleGenius", page_icon="\u270f\ufe0f", layout="centered"
)


@st.cache_resource
def load_predictor() -> Predictor:
    """
    Load the trained model exactly once per server process, not once
    per rerun. Streamlit reruns this whole script on every stroke and
    every fragment tick -- without this cache, a 5MB Keras model would
    get reloaded from disk dozens of times a minute. st.cache_resource
    is Streamlit's decorator specifically for objects that shouldn't
    be rebuilt on rerun (models, DB connections); the other decorator,
    st.cache_data, is for serializable return values like dataframes
    and isn't appropriate for a live model object.
    """
    return Predictor()


config = AppConfig()
predictor = load_predictor()
processor = CanvasProcessor(image_size=predictor.ml_config.image_size)
game = GameState(categories=predictor.categories, config=config, state=st.session_state)


@st.fragment(run_every=config.refresh_interval_ms / 1000)
def render_active_round() -> None:
    """
    A fragment reruns on its own timer (run_every), independent of the
    rest of the script -- so the countdown and live predictions keep
    updating even if the user pauses mid-drawing. Everything outside
    this function (title, score metric, round history) is untouched by
    those ticks; only this fragment's output redraws, which is cheaper
    than a full-page refresh and needs no third-party autorefresh
    package (that package is unmaintained and has reported breakage on
    current Streamlit -- st.fragment(run_every=...) is Streamlit's own
    built-in, actively maintained equivalent).
    """
    remaining = game.remaining()
    # st.header, not st.subheader -- this is the first thing a viewer
    # should read, so it gets the largest text on the page short of
    # the app title itself.
    st.header(f"\u270f\ufe0f Draw: {st.session_state.target}")
    st.progress(remaining / config.round_seconds)
    st.write(f"{remaining:.1f}s left")

    # Key includes the round count so each new round gets a fresh,
    # empty canvas widget -- reusing a static key would carry the
    # previous round's drawing over into the next one.
    canvas_key = f"canvas_{len(st.session_state.history)}"
    # Wrapped in a bordered container so the canvas reads as a
    # deliberate drawing area instead of a floating black square with
    # no visual separation from the page background.
    with st.container(border=True):
        canvas_result = st_canvas(
            stroke_width=config.stroke_width,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=config.canvas_size,
            width=config.canvas_size,
            drawing_mode="freedraw",
            update_streamlit=True,
            key=canvas_key,
        )

    predictions = []
    if (
        canvas_result.image_data is not None
        and processor.has_ink(canvas_result.image_data)
    ):
        model_input = processor.to_model_input(canvas_result.image_data)
        predictions = predictor.predict_topk(model_input, k=config.top_k)

    if predictions:
        st.write("Model's guesses:")
        for category, confidence in predictions:
            st.write(f"**{category}** -- {confidence:.0%}")

    top_guess = predictions[0][0] if predictions else None
    correct = top_guess == st.session_state.target

    # Round outcomes are handed off to the static result screen below
    # (rendered outside this fragment) instead of showing a message
    # here and immediately rerunning -- that message would disappear
    # almost instantly, before there's any real chance to read it.
    if correct:
        game.finish_round(guessed=top_guess, correct=True, predictions=predictions)
        st.rerun()
    elif game.is_time_up():
        game.finish_round(guessed=top_guess, correct=False, predictions=predictions)
        st.rerun()


st.title("DoodleGenius")
st.caption("Draw the prompt before time runs out -- the model guesses live.")

with st.expander("How to play"):
    # Computed from config rather than hardcoded, so the example stays
    # accurate even if round length or point range ever change.
    half_time_example = round(config.round_seconds / 2, 1)
    half_time_points = round(
        config.max_points * (half_time_example / config.round_seconds)
    )
    st.markdown(
        f"""
A random word is picked from {len(predictor.categories)} categories.
Draw it before the timer runs out -- the model guesses live as you sketch.

**Scoring:** if the model's top guess matches the word before time's up,
you score {config.min_points}-{config.max_points} points, scaled by how
much time you had left -- faster, clearer drawings score more. A wrong
guess or a timeout scores 0 points.

*Example:* the round is {config.round_seconds}s long.
- Guess correctly right away, with nearly the full {config.round_seconds}s
  left, and you'd score close to the max ({config.max_points}).
- Guess correctly with {half_time_example}s left (halfway through) and
  you'd score about {half_time_points} points.
- Guess correctly right at the buzzer and you'd still get the minimum
  ({config.min_points}) -- correct always beats incorrect, no matter
  how slow.
- Guess wrong, or run out of time, and you score 0 -- speed only
  matters once you're actually right.

**Why a correct AI guess is a win:** this game is modeled on Google's
Quick, Draw! -- the model's job is to recognize your sketch, and your
job is to draw clearly and quickly enough that it can. A correct guess
means your drawing worked, not that the model beat you.
"""
    )

st.metric("Score", st.session_state.score)

if st.session_state.round_active:
    render_active_round()
elif st.session_state.round_result is not None:
    # Static screen shown after a round ends -- stays until the player
    # clicks "Next round", so the outcome and the model's guesses are
    # actually readable instead of flashing by.
    result = st.session_state.round_result
    if result.correct:
        st.success(f"Correct! It was **{result.target}**. +{result.points} points")
        if game.should_celebrate():
            st.balloons()
    else:
        st.error(f"Time's up! It was **{result.target}**.")

    if result.predictions:
        st.write("The model's guesses were:")
        for category, confidence in result.predictions:
            st.write(f"**{category}** -- {confidence:.0%}")

    if st.button("Next round", type="primary"):
        game.dismiss_result()
        st.rerun()
else:
    if st.button("Start round", type="primary"):
        game.start_new_round()
        st.rerun()

if st.session_state.history:
    st.divider()
    st.subheader("Round history")
    st.table(
        [
            {
                "Round": r.round_number,
                "Target": r.target,
                "Guessed": r.guessed,
                "Correct": "Yes" if r.correct else "No",
                "Time (s)": r.time_taken,
                "Points": r.points,
            }
            for r in st.session_state.history
        ]
    )
