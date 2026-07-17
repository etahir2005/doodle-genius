"""State wrapper managing one player's round-by-round game."""

import random
import time
from dataclasses import dataclass, field
from typing import MutableMapping

from .config import AppConfig


@dataclass
class RoundResult:
    round_number: int
    target: str
    guessed: str
    correct: bool
    time_taken: float
    points: int
    predictions: list = field(default_factory=list)


class GameState:
    """
    Wraps a mutable state mapping (st.session_state in production, a
    plain dict in tests) instead of importing st.session_state
    directly -- same dependency-injection pattern as ml/src/config.py's
    Config being passed into every pipeline class rather than imported
    globally. It's what makes this class fully unit-testable with no
    Streamlit runtime at all. Streamlit reruns the whole script
    top-to-bottom on every interaction, so anything held in a normal
    local variable resets on every rerun -- state has to live in
    something external to survive between reruns, which is what this
    injected mapping is for.
    """

    def __init__(self, categories: list, config: AppConfig, state: MutableMapping):
        self.categories = categories
        self.config = config
        self.state = state
        self.state.setdefault("score", 0)
        self.state.setdefault("history", [])
        self.state.setdefault("round_active", False)
        self.state.setdefault("round_result", None)
        self.state.setdefault("celebrated", False)
        self.state.setdefault("target", None)
        self.state.setdefault("round_start", None)
        self.state.setdefault("last_target", None)

    def start_new_round(self) -> None:
        # Exclude the immediately previous target so two rounds in a
        # row never ask for the same doodle.
        choices = [c for c in self.categories if c != self.state["last_target"]]
        target = random.choice(choices)
        self.state["target"] = target
        self.state["last_target"] = target
        self.state["round_start"] = time.time()
        self.state["round_active"] = True
        self.state["round_result"] = None

    def elapsed(self) -> float:
        return time.time() - self.state["round_start"]

    def remaining(self) -> float:
        return max(0.0, self.config.round_seconds - self.elapsed())

    def is_time_up(self) -> bool:
        return self.remaining() <= 0

    def finish_round(
        self, guessed: str, correct: bool, predictions: list = None
    ) -> None:
        points = 0
        if correct:
            # Faster correct guesses score more -- scaled linearly by
            # fraction of round time left, floored at min_points so
            # every correct guess is still worth something.
            fraction_left = self.remaining() / self.config.round_seconds
            points = max(
                self.config.min_points,
                round(self.config.max_points * fraction_left),
            )
            self.state["score"] += points

        result = RoundResult(
            round_number=len(self.state["history"]) + 1,
            target=self.state["target"],
            guessed=guessed or "(none)",
            correct=correct,
            time_taken=round(self.elapsed(), 1),
            points=points,
            predictions=predictions or [],
        )
        self.state["history"].append(result)
        self.state["round_result"] = result
        self.state["round_active"] = False
        self.state["celebrated"] = False

    def should_celebrate(self) -> bool:
        """
        True exactly once per correct round result. Streamlit reruns
        the whole script on every interaction -- including the click
        on "Next round" itself, one rerun before dismiss_result()
        actually clears round_result -- so without this guard,
        st.balloons() would fire a second time on that transitional
        rerun instead of exactly once when the round was actually won.
        """
        result = self.state.get("round_result")
        if result is not None and result.correct and not self.state["celebrated"]:
            self.state["celebrated"] = True
            return True
        return False

    def dismiss_result(self) -> None:
        """
        Clears the just-finished round's result so the app returns to
        the start screen. Kept as a separate step from finish_round
        (rather than clearing automatically) specifically so the
        result -- and the model's guesses that led to it -- stays on
        screen until the player is actually ready to move on, instead
        of vanishing the instant the round ends.
        """
        self.state["round_result"] = None
