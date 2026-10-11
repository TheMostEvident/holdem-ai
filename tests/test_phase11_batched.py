from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from holdem_agent_phase8 import NetworkConfig, TransformerActor
from holdem_agent_phase9 import PPOConfig, collect_self_play
from holdem_agent_phase11 import collect_batched_self_play, evaluate_balanced


@pytest.fixture
def actor():
    torch.manual_seed(100)
    return TransformerActor(NetworkConfig(embed_dim=32, attention_heads=2,
                                         transformer_layers=1, ff_dim=64,
                                         hidden_dim=48)).eval()


def test_batched_rows_are_legal_and_seat_specific(actor):
    cfg = PPOConfig(hands_per_update=8, seed=21)
    rows = collect_batched_self_play(actor, cfg, env_batch_size=4)
    assert rows.hands == 8 and len(rows.hand_rewards) == 8
    assert len(rows.actions) >= 8
    for r, a, seat in zip(rows.actors, rows.actions, rows.acting_seats):
        assert r.legal_action_mask[a]
        assert r.features[0] == seat
    assert all(abs(sum(x)) < 1e-9 for x in rows.hand_rewards)


def test_batch_one_matches_legacy_sequential(actor):
    cfg = PPOConfig(hands_per_update=3, seed=78)
    rows = collect_batched_self_play(actor, cfg, env_batch_size=1)
    old = collect_self_play(actor, cfg)
    assert rows.hand_rewards == old.hand_rewards
    assert rows.actions == old.actions
    assert rows.acting_seats == old.acting_seats
    assert rows.returns_bb == old.returns_bb
    assert rows.actors == old.actors
    assert rows.critics == old.critics


def test_batched_reproducible_and_changes_iteration(actor):
    cfg = PPOConfig(hands_per_update=5, seed=14)
    a = collect_batched_self_play(actor, cfg, env_batch_size=5)
    b = collect_batched_self_play(actor, cfg, env_batch_size=2)
    c = collect_batched_self_play(actor, cfg, iteration=1, env_batch_size=2)
    assert a.actions == b.actions
    assert a.hand_rewards == b.hand_rewards
    assert c.hand_rewards != a.hand_rewards or c.actions != a.actions


@pytest.mark.parametrize('size', [0, -1, 2.3, True])
def test_reject_bad_batch_size(actor, size):
    with pytest.raises(ValueError):
        collect_batched_self_play(actor, PPOConfig(hands_per_update=2),
                                  env_batch_size=size)


def test_reject_negative_iteration(actor):
    with pytest.raises(ValueError):
        collect_batched_self_play(actor, PPOConfig(hands_per_update=2), iteration=-1)


def test_balanced_eval_against_random_and_frozen_anchor(actor):
    other = TransformerActor(actor.config).eval()
    other.load_state_dict(actor.state_dict())
    for opponent in (None, other):
        stats = evaluate_balanced(actor, opponent=opponent, hands=8, seed=12)
        assert stats['hands'] == 8
        assert stats['seat0_hands'] == stats['neural_button_hands'] == 4
        assert stats['ci95_lower_bb100'] <= stats['mean_bb100'] <= stats['ci95_upper_bb100']
    with pytest.raises(ValueError):
        evaluate_balanced(actor, opponent=actor, hands=8)
    with pytest.raises(ValueError):
        evaluate_balanced(actor, hands=5)
