"""Offline tests for the public adapter API with a stand-in RLCard Env."""
from types import SimpleNamespace

import pytest

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine


class FakeEnv:
    def __init__(self):
        self.game = SimpleNamespace(big_blind=2, dealer_id=0)
        self.actions = []
        self.counter = 0

    def reset(self):
        self.counter = 0
        self.actions = []
        return {'legal_actions': {1: None, 2: None}}, 0

    def is_over(self):
        return self.counter == 2

    def step(self, action):
        self.actions.append(action)
        self.counter += 1
        if self.counter == 1:
            return {'legal_actions': {0: None, 1: None}}, 1
        return {'legal_actions': {}}, 0

    def get_payoffs(self):
        return (4, -4)


@pytest.fixture
def engine(monkeypatch):
    from holdem_ai.environment import corrected_game
    monkeypatch.setattr(corrected_game, 'make_corrected_rlcard_env', lambda **kwargs: FakeEnv())
    return HandEngine(starting_stack=200, seed=123)


def test_hand_engine_play_hand_is_zero_sum(engine):
    result = engine.play_hand((RandomAgent(1), RandomAgent(2)))
    assert result.payoffs_chips == (4.0, -4.0)
    assert result.payoffs_bb == (2.0, -2.0)
    assert len(result.action_history) == 2
    assert [actor for actor, _ in result.action_history] == [0, 1]


def test_reject_illegal_action(engine):
    engine.reset()
    with pytest.raises(ValueError, match='Illegal action'):
        engine.step(999)


def test_random_agent_never_leaves_legal_actions():
    agent = RandomAgent(seed=7)
    state = {'legal_actions': {0: None, 3: None, 4: None}}
    for _ in range(100):
        assert agent.select_action(state) in state['legal_actions']


def test_cannot_read_result_before_hand_finished(engine):
    engine.reset()
    with pytest.raises(RuntimeError, match='not over'):
        engine.get_result()


def test_button_rotates_between_hands(engine):
    engine.reset()
    assert engine.env.game.dealer_id == 0
    engine.reset()
    assert engine.env.game.dealer_id == 1
    engine.reset()
    assert engine.env.game.dealer_id == 0
