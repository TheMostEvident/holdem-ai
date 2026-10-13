"""Regression tests: hand-weighted payoff tails and preflop termination types."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from holdem_ai.evaluation import terminal_reward_audit as audit
from holdem_ai.training.ppo import TrajectoryRows


def _actor(stage: int):
    f = [0.] * 20
    f[2] = float(stage)
    f[1] = 1.
    return SimpleNamespace(
        features=tuple(f),
        legal_action_mask=(True,) * 7,
    )


def _rows():
    # 1. Preflop fold
    # 2. Preflop explicit all-in then a call -> full-board automatic showdown
    # 3. Preflop call then flop fold
    # 4. Preflop/flop/turn/river decisions -> river nonfold showdown
    acts = (0, 6, 2, 2, 0, 2, 1, 1, 2)
    stages = (0, 0, 0, 0, 1, 0, 1, 2, 3)
    return TrajectoryRows(
        actors=tuple(_actor(s) for s in stages),
        critics=(None,) * len(acts),
        actions=acts,
        acting_seats=(0, 0, 1, 0, 1, 0, 1, 0, 1),
        returns_bb=(-0.5, -200., 200., -8., 8., 150., -150., 150., -150.),
        hands=4,
        hand_rewards=((-0.5, 0.5), (-200., 200.), (-8., 8.), (150., -150.)),
        terminal_actions=(0, 2, 0, 2),
        decisions_per_hand=(1, 2, 2, 4),
    )


def test_preflop_exit_and_payoff_tail_exact_counts():
    report = audit.summarize_terminal_distribution(_rows())
    assert report["hands"] == 4
    assert report["decisions"] == 9
    assert report["last_decision_street_counts"] == {
        "preflop": 2, "flop": 1, "turn": 0, "river": 1,
    }
    assert report["preflop"]["last_decision_hand_count"] == 2
    assert report["preflop"]["fold_ended_count"] == 1
    assert report["preflop"]["nonfold_showdown_count"] == 1
    assert report["preflop"]["explicit_all_in_hand_count"] == 1
    assert report["preflop"]["explicit_all_in_and_nonfold_last_preflop"] == 1
    assert report["last_decision_street_x_termination"]["flop:fold"] == 1
    assert report["decision_coverage"] == {
        "flop_hands": 2, "turn_hands": 1, "river_hands": 1,
    }
    dist = report["seat0_hand_weighted_reward"]
    assert dist["count"] == 4
    assert dist["near_full_stack_fraction"] == pytest.approx(0.25)
    assert dist["abs_ge_190bb_fraction"] == pytest.approx(0.25)
    assert dist["signed_full_stack"]["win"] == 0
    assert dist["signed_full_stack"]["loss"] == 1
    assert report["full_stack_settlements"]["hand_count"] == 1
    assert report["payoff_by_exit_group"]["preflop:showdown"]["near_full_stack_fraction"] == 1.0
    assert report["payoff_by_exit_group"]["preflop:fold"]["near_full_stack_fraction"] == 0.0
    assert report["preflop"]["last_action_counts"]["call"] == 1
    assert report["preflop"]["last_action_counts"]["fold"] == 1
    assert report["actor_decision_weighted_return"]["count"] == 9


def test_missing_boundaries_fail_without_guessing_terminations():
    rows = _rows()
    rows = TrajectoryRows(
        rows.actors, rows.critics, rows.actions, rows.acting_seats,
        rows.returns_bb, rows.hands, rows.hand_rewards)
    with pytest.raises(ValueError, match="hand boundaries"):
        audit.summarize_terminal_distribution(rows)


def test_last_decision_not_matching_terminal_action_rejected():
    rows = _rows()
    rows = TrajectoryRows(
        rows.actors, rows.critics, rows.actions, rows.acting_seats,
        rows.returns_bb, rows.hands, rows.hand_rewards,
        terminal_actions=(0, 6, 0, 2),
        decisions_per_hand=(1, 2, 2, 4))
    with pytest.raises(ValueError, match="terminal action does not match"):
        audit.summarize_terminal_distribution(rows)


def test_nonzero_sum_terminal_reward_rejected():
    rows = _rows()
    bad_rewards = list(rows.hand_rewards)
    bad_rewards[0] = (-0.5, 1.0)
    rows = TrajectoryRows(
        rows.actors, rows.critics, rows.actions, rows.acting_seats,
        rows.returns_bb, rows.hands, tuple(bad_rewards),
        rows.terminal_actions, rows.decisions_per_hand)
    with pytest.raises(ValueError, match="non-zero-sum"):
        audit.summarize_terminal_distribution(rows)


def test_two_checkpoint_sampling_protocol(monkeypatch):
    seen = []
    before = object()
    after = object()
    models = {"best.pt": before, "latest.pt": after}

    def fake_load(path, *, device):
        return models[str(path)]

    def fake_collect(actor, cfg, *, iteration, device, env_batch_size):
        seen.append((actor, cfg.seed, cfg.hands_per_update, env_batch_size))
        return _rows()

    monkeypatch.setattr(audit, "load_actor", fake_load)
    monkeypatch.setattr(audit, "collect_batched_self_play", fake_collect)
    result = audit.run_terminal_reward_audit(
        checkpoints={"best": "best.pt", "latest": "latest.pt"},
        device="cpu", hands=4, seed=580000000, env_batch_size=2)
    assert seen == [(before, 580000000, 4, 2), (after, 580000000, 4, 2)]
    assert result["models"]["latest"]["preflop"]["nonfold_showdown_count"] == 1


def test_invalid_config_and_empty_rows_are_rejected():
    with pytest.raises(ValueError, match="stack_bb"):
        audit.summarize_terminal_distribution(_rows(), stack_bb=-1)
    with pytest.raises(ValueError, match="env_batch_size"):
        audit.run_terminal_reward_audit(
            checkpoints={"best": "best.pt"}, hands=4, env_batch_size=0)
