import random

import pytest

from holdem_agent_phase5 import (
    InformationState, RandomLegalPolicy, Terminal, Turn, run_self_play, snapshot,
)


def state(seat, step, num_actions=5):
    return Turn(InformationState(seat, (float(step),), tuple(i in (1, 3) for i in range(num_actions))))


class FakeMultiSeat:
    def __init__(self, order=(0, 1, 0, 2, 0, 3, 4, 5), n=6):
        self.order = tuple(order)
        self.n = n
        self.cursor = 0
        self.actions = []

    def reset(self, seed=None):
        self.cursor = 0
        self.actions = []
        return state(self.order[0], 0)

    def step(self, action_index):
        assert action_index in (1, 3)
        self.actions.append(action_index)
        self.cursor += 1
        if self.cursor >= len(self.order):
            return Terminal(tuple(float(i - (self.n - 1) / 2) for i in range(self.n)))
        return state(self.order[self.cursor], self.cursor)


def test_selfplay_6_max_trajectories_link_next_same_seat():
    env = FakeMultiSeat()
    policies = {i: RandomLegalPolicy() for i in range(6)}
    result = run_self_play(env, policies, seed=9)
    assert result.num_decisions == 8
    assert len(result.transitions) == 8
    zero = result.for_seat(0)
    assert len(zero) == 3
    assert [transition.done for transition in zero] == [False, False, True]
    assert [transition.reward for transition in zero] == [0, 0, -2.5]
    assert [t.next_observation.features for t in zero[:-1]] == [(2.0,), (4.0,)]
    assert result.for_seat(1)[0].done is True
    assert result.for_seat(1)[0].reward == -1.5
    assert set(env.actions).issubset({1, 3})


def test_deterministic_selfplay_actions_seed():
    policies = {i: RandomLegalPolicy() for i in range(6)}
    a = run_self_play(FakeMultiSeat(), policies, seed=77)
    b = run_self_play(FakeMultiSeat(), policies, seed=77)
    assert [t.action_index for t in a.transitions] == [t.action_index for t in b.transitions]


def test_selfplay_missing_policy():
    with pytest.raises(KeyError):
        run_self_play(FakeMultiSeat(order=(0, 1)), {0: RandomLegalPolicy()}, seed=1)


def test_selfplay_guard_against_infinite_episode():
    with pytest.raises(RuntimeError, match="max_decisions"):
        run_self_play(FakeMultiSeat(), {i: RandomLegalPolicy() for i in range(6)}, max_decisions=3)


class IllegalPolicy:
    def act(self, observation, rng):
        return 2


def test_selfplay_invalid_policy_action_raises():
    with pytest.raises(ValueError, match="illegal"):
        run_self_play(FakeMultiSeat(), {0: IllegalPolicy()})


class SeededChance:
    def __init__(self):
        self.rng = random.Random(12)
        self.first = None

    def reset(self, seed=None):
        self.rng.seed(seed)
        self.first = self.rng.randrange(10000)
        return state(0, self.first)

    def step(self, action_index):
        return Terminal((1.0, -1.0))


def test_snapshot_deep_clone_preserves_rng_and_isolates_mutations():
    original = SeededChance()
    original.reset(23)
    copy = snapshot(original)
    assert copy is not original
    assert copy.rng.random() == original.rng.random()
    copy.first = -1
    assert original.first != -1


def test_returned_transition_has_frozen_observation():
    result = run_self_play(FakeMultiSeat(order=(0, 1), n=2), {0: RandomLegalPolicy(), 1: RandomLegalPolicy()})
    with pytest.raises(Exception):
        result.transitions[0].observation.player_id = 1
