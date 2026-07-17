"""Unit tests for GameState -- uses a plain dict instead of a real
st.session_state, since GameState takes any MutableMapping."""

import time

import pytest

from game.config import AppConfig
from game.game_state import GameState


@pytest.fixture
def config():
    return AppConfig(round_seconds=10, min_points=10, max_points=100)


@pytest.fixture
def game(config):
    return GameState(categories=["cat", "dog", "sun"], config=config, state={})


def test_initializes_defaults(game):
    assert game.state["score"] == 0
    assert game.state["history"] == []
    assert game.state["round_active"] is False


def test_start_new_round_sets_target_and_activates(game):
    game.start_new_round()
    assert game.state["target"] in ["cat", "dog", "sun"]
    assert game.state["round_active"] is True
    assert game.state["round_start"] is not None


def test_start_new_round_avoids_immediate_repeat(config):
    game = GameState(categories=["cat", "dog"], config=config, state={})
    game.start_new_round()
    first = game.state["target"]
    for _ in range(10):
        game.start_new_round()
        assert game.state["target"] != first
        first = game.state["target"]


def test_remaining_counts_down(game):
    game.start_new_round()
    remaining = game.remaining()
    assert 0 < remaining <= game.config.round_seconds


def test_is_time_up_false_immediately_after_start(game):
    game.start_new_round()
    assert game.is_time_up() is False


def test_is_time_up_true_after_round_length_elapses(game):
    game.start_new_round()
    game.state["round_start"] = time.time() - (game.config.round_seconds + 1)
    assert game.is_time_up() is True


def test_finish_round_correct_awards_points_and_updates_history(game):
    game.start_new_round()
    target = game.state["target"]
    game.finish_round(guessed=target, correct=True)

    assert game.state["round_active"] is False
    assert game.state["score"] > 0
    assert len(game.state["history"]) == 1
    result = game.state["history"][0]
    assert result.correct is True
    assert result.target == target
    assert result.points >= game.config.min_points


def test_finish_round_incorrect_awards_no_points(game):
    game.start_new_round()
    game.finish_round(guessed="wrong", correct=False)

    assert game.state["score"] == 0
    assert game.state["history"][0].correct is False
    assert game.state["history"][0].points == 0


def test_finish_round_faster_guess_scores_more(config):
    game_fast = GameState(categories=["cat"], config=config, state={})
    game_fast.start_new_round()
    game_fast.finish_round(guessed="cat", correct=True)
    fast_points = game_fast.state["history"][0].points

    game_slow = GameState(categories=["cat"], config=config, state={})
    game_slow.start_new_round()
    game_slow.state["round_start"] = time.time() - (config.round_seconds - 1)
    game_slow.finish_round(guessed="cat", correct=True)
    slow_points = game_slow.state["history"][0].points

    assert fast_points > slow_points
