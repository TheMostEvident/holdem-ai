import json
import math
import random

import pytest

from holdem_agent_phase5 import (
    CFRTablePolicy,
    EpsilonGreedyPolicy,
    InformationState,
    RandomLegalPolicy,
    masked_distribution,
)


def obs(*, seat=0, mask=(True, False, True), hand=("SA", "SK"), history=("0:call",)):
    return InformationState(
        player_id=seat,
        features=(0.0, 1.0),
        legal_action_mask=mask,
        private_cards=hand,
        public_cards=("H2", "D3", "C4"),
        stage="flop",
        public_state=(("pot", "100"),),
        history=history,
    )


def test_infoset_key_stable_and_private_distinguished():
    first = obs()
    assert first.infoset_key() == obs().infoset_key()
    assert first.infoset_key() != obs(hand=("SA", "SQ")).infoset_key()
    assert first.infoset_key() != obs(history=("0:raise",)).infoset_key()
    decoded = json.loads(first.infoset_key())
    assert decoded["player"] == 0
    assert decoded["private"] == ["SA", "SK"]


def test_legal_mask_rejects_illegal_and_type():
    x = obs()
    assert x.legal_actions == (0, 2)
    for bad in (-1, 1, 3):
        with pytest.raises(ValueError):
            x.assert_legal(bad)
    with pytest.raises(TypeError):
        x.assert_legal(True)
    x.assert_legal(2)


@pytest.mark.parametrize("mask", [(), (False, False)])
def test_nonterminal_must_have_legal_action(mask):
    with pytest.raises(ValueError):
        obs(mask=mask)


def test_nan_feature_rejected():
    with pytest.raises(ValueError):
        InformationState(0, (math.nan,), (True,))


def test_masked_distribution_invalid_never_selected():
    p = masked_distribution((1.0, 100.0, 3.0), (True, False, True))
    assert p == pytest.approx((0.25, 0.0, 0.75))
    assert sum(p) == pytest.approx(1)


def test_masked_distribution_uniform_fallback():
    assert masked_distribution((0, 0, 0), (True, False, True)) == pytest.approx((0.5, 0, 0.5))


@pytest.mark.parametrize("values", [(0, -1, 0), (0, float("inf"), 0), (0, math.nan, 0)])
def test_masked_distribution_rejects_invalid_weights_even_if_masked(values):
    with pytest.raises(ValueError):
        masked_distribution(values, (True, False, True))


def test_masked_distribution_rejects_length():
    with pytest.raises(ValueError):
        masked_distribution((1, 2), (True, False, True))


def test_random_policy_deterministic_and_masked():
    r1, r2 = random.Random(4), random.Random(4)
    x = obs()
    policy = RandomLegalPolicy()
    a = [policy.act(x, r1) for _ in range(100)]
    b = [policy.act(x, r2) for _ in range(100)]
    assert a == b
    assert set(a) == {0, 2}


def test_cfr_policy_normalizes_mask_and_uses_fallback():
    x = obs()
    selected = CFRTablePolicy({x.infoset_key(): (0, 1, 1)})
    assert [selected.act(x, random.Random(i)) for i in range(30)] == [2] * 30
    absent = CFRTablePolicy({})
    assert {absent.act(x, random.Random(i)) for i in range(30)} == {0, 2}


def test_epsilon_greedy_does_not_select_illegal_even_if_high_q():
    pol = EpsilonGreedyPolicy(lambda _: [0.1, 10_000, 0.2], epsilon=0)
    assert pol.act(obs(), random.Random(1)) == 2
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(lambda _: [0.0], epsilon=-0.1)
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(lambda _: [0.0], epsilon=1.1)


def test_epsilon_greedy_rejects_bad_q_dimension():
    with pytest.raises(ValueError):
        EpsilonGreedyPolicy(lambda _: [1], epsilon=0).act(obs(), random.Random(1))


def test_equal_q_tie_breaker_is_lowest_legal_action():
    assert EpsilonGreedyPolicy(lambda _: [5, 99, 5], epsilon=0).act(obs(), random.Random(9)) == 0
