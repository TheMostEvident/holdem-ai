"""Phase12 robust evaluation: same-deck, opposite-seat, unit tests."""
from __future__ import annotations

from dataclasses import replace
import random

import pytest
import torch

from holdem_agent_phase8 import NetworkConfig, TransformerActor
from holdem_agent_phase12 import (
    DealGroup, EvaluationConfig, EvaluationResult,
    action_counts, compare_results, evaluate_seed_blocks,
    model_fingerprint, summarize_block_scores,
)


@pytest.fixture(scope='module')
def net():
    torch.manual_seed(17)
    cfg = NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                        ff_dim=64, hidden_dim=48, max_tokens=256)
    return TransformerActor(cfg)


def clone(model):
    copy = TransformerActor(model.config)
    copy.load_state_dict(model.state_dict())
    return copy


@pytest.mark.parametrize('bad', [0, 1, -1, True, 2.3])
def test_invalid_group_counts(bad):
    with pytest.raises(ValueError):
        EvaluationConfig(groups=bad)


def test_invalid_eval_config():
    defaults = EvaluationConfig()
    for changes in ({'seed': -4}, {'seed': True}, {'mode': 'broken'},
                    {'big_blind': 0}, {'max_decisions': 0}, {'device': 'gpu'}):
        with pytest.raises(ValueError):
            replace(defaults, **changes)


def test_statistics_use_independent_groups_not_four_hands():
    data = summarize_block_scores((1.0, -1.0, 2.0, -2.0))
    assert data['mean_bb100'] == 0.0
    assert data['standard_error_bb100'] > 0
    with pytest.raises(ValueError):
        summarize_block_scores((1.0,))
    with pytest.raises(ValueError):
        summarize_block_scores((1.0, float('nan')))


def test_four_layouts_and_identical_actor_zero_net(net):
    # Identical actors necessarily produce the same game action sequence in
    # a fixed physical layout; reversing policy assignment reverses reward.
    report = evaluate_seed_blocks(net, opponent=clone(net),
                                  config=EvaluationConfig(groups=3, seed=877, mode='greedy'))
    assert report.summary()['hands'] == 12
    assert report.summary()['candidate_button_hands'] == 6
    assert report.summary()['candidate_seat0_hands'] == 6
    assert report.summary()['mean_bb100'] == 0
    assert report.summary()['ci95_lower_bb100'] == 0
    assert report.summary()['ci95_upper_bb100'] == 0
    for block in report.blocks:
        assert block.payoffs_bb[0] == -block.payoffs_bb[1]
        assert block.payoffs_bb[2] == -block.payoffs_bb[3]


def test_sampled_self_match_same_model_cancels(net):
    report = evaluate_seed_blocks(net, opponent=clone(net), config=EvaluationConfig(
        groups=2, seed=1501, mode='sampled'))
    assert report.means_bb == (0.0, 0.0)


def test_random_reproducible_and_action_histograms(net):
    cfg = EvaluationConfig(groups=3, seed=101)
    first = evaluate_seed_blocks(net, config=cfg)
    second = evaluate_seed_blocks(net, config=cfg)
    assert first.blocks == second.blocks
    assert first.summary() == second.summary()
    assert len(first.summary()['actor_actions']) == 7
    assert sum(first.summary()['actor_actions']) > 0
    assert sum(first.summary()['opponent_actions']) > 0
    assert len({block.seed for block in first.blocks}) == 3


def test_sampling_changes_policy_interaction_at_least_statistic_contract(net):
    result = evaluate_seed_blocks(net, config=EvaluationConfig(groups=2, seed=44, mode='sampled'))
    assert result.summary()['mode'] == 'sampled'
    assert result.summary()['hands'] == 8
    assert isinstance(result.means_bb, tuple)


def test_action_count_validation():
    assert action_counts((0, 0, 6)) == [2, 0, 0, 0, 0, 0, 1]
    with pytest.raises(ValueError):
        action_counts((7,))


def test_compare_paired_results_zero_against_itself(net):
    cfg = EvaluationConfig(groups=3, seed=404)
    a = evaluate_seed_blocks(net, config=cfg)
    b = evaluate_seed_blocks(net, config=cfg)
    diff = compare_results(a, b)
    assert diff['delta_mean_bb100'] == 0
    assert diff['delta_se_bb100'] == 0
    assert diff['delta_ci95_lower_bb100'] == 0
    assert diff['hands_per_agent'] == 12


def test_compare_rejects_inconsistent_opponent_and_seed(net):
    cfg = EvaluationConfig(groups=2, seed=901)
    original = evaluate_seed_blocks(net, config=cfg)
    alt = evaluate_seed_blocks(net, config=replace(cfg, seed=902))
    with pytest.raises(ValueError):
        compare_results(original, alt)
    fake = replace(original, opponent_id='different_policy')
    with pytest.raises(ValueError):
        compare_results(original, fake)


def test_actor_fingerprint_changes_with_weights(net):
    other = clone(net)
    assert model_fingerprint(net) == model_fingerprint(other)
    with torch.no_grad():
        next(other.parameters()).add_(0.01)
    assert model_fingerprint(net) != model_fingerprint(other)


def test_reject_alias_actor_and_keep_training_flags(net):
    with pytest.raises(ValueError):
        evaluate_seed_blocks(net, opponent=net, config=EvaluationConfig(groups=2))
    opponent = clone(net)
    net.train(True)
    opponent.train(True)
    evaluate_seed_blocks(net, opponent=opponent, config=EvaluationConfig(groups=2))
    assert net.training and opponent.training


def test_dropout_forbidden():
    model = TransformerActor(NetworkConfig(embed_dim=16, attention_heads=2,
                                           transformer_layers=1, ff_dim=32,
                                           hidden_dim=16, dropout=0.1))
    with pytest.raises(ValueError):
        evaluate_seed_blocks(model, config=EvaluationConfig(groups=2))
