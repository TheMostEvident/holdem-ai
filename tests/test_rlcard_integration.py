"""Runs only where RLCard is installed; verifies a real hand end-to-end."""

import pytest

pytest.importorskip('rlcard')

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine


def test_real_rlcard_random_hand():
    engine = HandEngine(starting_stack=200, seed=42)
    result = engine.play_hand((RandomAgent(1), RandomAgent(2)))
    assert abs(sum(result.payoffs_chips)) <= 1e-6
    assert len(result.action_history) > 0
    assert all(len(result.payoffs_bb) == 2 for _ in range(1))
