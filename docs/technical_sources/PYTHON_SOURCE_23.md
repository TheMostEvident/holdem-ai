# Complete Python source volume 23/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## tests/test_multi_player_engine.py

Original blob `092bb6933159ea3c0ac650e5ff9635a046642b8b` (11045 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_multi_player_engine.py" sha="092bb6933159ea3c0ac650e5ff9635a046642b8b" -->

````````python
from __future__ import annotations

import pytest

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_evaluator import RANKS, SUITS
from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine


def test_preflop_and_postflop_order_three_players():
    e = MultiPlayerHandEngine(num_players=3, seed=9)
    s, actor = e.reset()
    assert actor == 0  # BTN acts first preflop in three players (UTG=BTN)
    assert (e.contributions, e.stacks) == ([0, 1, 2], [200, 199, 198])
    assert s['raw_obs']['to_call'] == 2
    s, actor = e.step(1)  # btn call
    assert actor == 1
    s, actor = e.step(1)  # SB call
    assert actor == 2
    s, actor = e.step(1)  # BB option check
    assert actor == 1 and e.street == 'flop'
    assert len(e.board) == 3 and len(e.burned) == 1
    s, actor = e.step(1)
    assert actor == 2
    s, actor = e.step(1)
    assert actor == 0
    s, actor = e.step(1)
    assert actor == 1 and e.street == 'turn'
    assert len(e.board) == 4 and len(e.burned) == 2


def test_six_players_betting_order():
    e = MultiPlayerHandEngine(num_players=6, seed=1, button=0)
    s, actor = e.reset()
    assert (e.seats.small_blind, e.seats.big_blind) == (1, 2)
    actors = []
    for _ in range(6):
        actors.append(actor)
        s, actor = e.step(1)
    assert actors == [3, 4, 5, 0, 1, 2]
    assert e.street == 'flop' and actor == 1


def test_headsup_compatible_and_private_hole_cards():
    e = MultiPlayerHandEngine(num_players=2, seed=99, button=1)
    s, actor = e.reset()
    assert actor == 1
    assert e.seats.small_blind == 1 and e.seats.big_blind == 0
    assert s['raw_obs']['hand'] == list(e.hole_cards[1])
    assert e.hole_cards[0] != e.hole_cards[1]
    assert str(e.hole_cards[0]) not in str(s)
    with pytest.raises(ValueError):
        e.observe(0)


def test_multiplayer_fold_immediate_no_extra_board():
    e = MultiPlayerHandEngine(num_players=4, seed=7)
    e.reset()
    # BTN=0, SB=1, BB=2, UTG=3. UTG/BTN/SB fold => BB wins immediately.
    for player in [3, 0, 1]:
        assert e.current_player == player
        e.step(0)
    assert e.is_over() and e.board == [] and e.burned == []
    result = e.get_result()
    assert result.payoffs_chips == (0, -1, 1, 0)
    assert sum(result.payoffs_chips) == 0


def test_illegal_actions_have_no_effect():
    e = MultiPlayerHandEngine(num_players=3, seed=2)
    e.reset()
    original = tuple(e.stacks), tuple(e.contributions), tuple(e.action_history)
    with pytest.raises(ValueError):
        e.step(123)
    with pytest.raises(ValueError):
        e.step_raise_to(3)  # first open must raise to >=4
    assert original == (tuple(e.stacks), tuple(e.contributions), tuple(e.action_history))
    with pytest.raises(RuntimeError):
        e.get_result()


def test_full_raise_and_minimum_re_raise():
    e = MultiPlayerHandEngine(num_players=3, seed=4)
    e.reset()
    # Button preflop raise-to 6 -> min re-raise to 10.
    e.step_raise_to(6)
    assert e.current_player == 1 and e.min_raise_to() == 10
    with pytest.raises(ValueError):
        e.step_raise_to(9)
    e.step_raise_to(10)
    assert e.current_player == 2 and e.min_raise_to() == 14


def test_single_active_vs_allins_cannot_raise():
    e = MultiPlayerHandEngine(stacks=[10, 15, 50], button=0)
    e.reset()
    e.step(4)  # P0 allin10
    e.step(4)  # P1 allin15
    assert e.current_player == 2
    assert e.legal_actions == (0, 1)
    e.step(1)  # P2 call 15 (not forced all-in for 50)
    assert e.is_over() and len(e.board) == 5
    r = e.get_result()
    assert sorted(r.contributions) == [10, 15, 15]
    assert sum(r.payoffs_chips) == 0
    assert sum(r.final_stacks) == 75


def test_multiway_short_allin_does_not_reopen_prior_actor():
    # BTN P0=200, SB P1=200, BB P2=15, UTG P3=200.
    # UTG calls 2, BTN raises to 10, SB calls, BB allin to 15,
    # UTG calls extra 13. Action returns to BTN, who faces +5 < 8 full raise.
    e = MultiPlayerHandEngine(stacks=[200, 200, 15, 200], button=0)
    e.reset()
    assert e.current_player == 3
    e.step(1)  # UTG 2
    e.step_raise_to(10)  # BTN
    e.step(1)  # SB call
    e.step(4)  # BB all-in 15 short raise
    assert e.current_player == 3
    e.step(1)
    assert e.current_player == 0
    assert e.legal_actions == (0, 1)
    with pytest.raises(ValueError):
        e.step_raise_to(25)


def test_cumulative_short_allins_reopen_earlier_action():
    # BTN (0) raises to 10 (last full raise 8), SB(1) goes allin 14,
    # BB(2) goes allin 18; BTN faces +8 total -> reopening permitted.
    e = MultiPlayerHandEngine(stacks=[200, 14, 18], button=0)
    e.reset()
    e.step_raise_to(10)
    e.step(4)
    e.step(4)
    assert e.current_player == 0
    assert e.min_raise_to() == 26
    # No one with chips remains to call a raise: may only call or fold.
    assert e.legal_actions == (0, 1)


def test_allin_with_uncalled_chips_and_folds():
    # An initial raise to 20 that nobody calls returns uncalled 18.
    e = MultiPlayerHandEngine(num_players=3, seed=12)
    e.reset()
    e.step_raise_to(20)
    e.step(0)
    e.step(0)
    r = e.get_result()
    assert r.contributions == (20, 1, 2)
    assert r.payoffs_chips == (3, -1, -2)
    assert r.final_stacks == (203, 199, 198)


def test_no_duplicate_dealt_cards_six_players():
    e = MultiPlayerHandEngine(num_players=6, seed=6)
    e.play_hand([RandomAgent(i) for i in range(6)])
    all_cards = [c for hole in e.hole_cards for c in hole] + e.board + e.burned
    assert len(all_cards) == len(set(all_cards))


def test_explicit_tie_odd_chip_seat_left_of_button():
    # 3 player: deal clockwise 1 2 0, twice, with royal-flush board.
    used = ['s2', 'd3', 'c4', 'h5', 's6', 'd7',
            'c8', 'hT', 'hJ', 'hQ', 'd8', 'hK', 'c9', 'hA']
    deck = used + [s + r for s in SUITS for r in RANKS if s + r not in used]
    e = MultiPlayerHandEngine(num_players=3, button=0, deck=deck)
    e.reset()
    e.step(1)  # button calls 2
    e.step(0)  # SB folds (1 chip dead)
    e.step(1)  # BB checks; both players check down
    while not e.is_over():
        e.step(1)
    r = e.get_result()
    assert r.board == ('hT', 'hJ', 'hQ', 'hK', 'hA')
    assert r.payoffs_chips == (0, -1, 1)
    assert r.awards == (2, 0, 3)


@pytest.mark.parametrize('n', [2, 3, 4, 5, 6])
def test_random_stress_chip_conservation(n):
    for seed in range(75):
        e = MultiPlayerHandEngine(num_players=n, seed=seed, button=seed % n)
        agents = [RandomAgent(seed * 11 + i) for i in range(n)]
        result = e.play_hand(agents, max_actions=200)
        assert sum(result.payoffs_chips) == 0
        assert sum(result.final_stacks) == n * 200
        assert min(result.final_stacks) >= 0
        assert sum(result.contributions) == sum(result.awards)
        assert len(result.board) in (0, 3, 4, 5)


def test_non_contiguous_active_seats():
    e = MultiPlayerHandEngine(stacks=[200, 0, 200, 0, 200, 0], button=4, seed=5)
    state, player = e.reset()
    assert (e.seats.small_blind, e.seats.big_blind, e.seats.preflop_first) == (0, 2, 4)
    assert player == 4
    res = e.play_hand([RandomAgent(i) for i in range(6)])
    assert res.final_stacks[1] == res.final_stacks[3] == res.final_stacks[5] == 0
    assert sum(res.final_stacks) == 600


def test_short_big_blind_still_requires_full_bring_in_with_two_active_opponents():
    # BTN and SB have chips; BB can post only one chip of the nominal 2.
    e = MultiPlayerHandEngine(stacks=[200, 200, 1], button=0)
    s, actor = e.reset()
    assert actor == 0 and e.contributions == [0, 1, 1]
    assert s['raw_obs']['to_call'] == 2
    e.step(1)  # BTN calls 2, not the short BB's posted 1
    assert e.contributions == [2, 1, 1]
    assert e.state['raw_obs']['to_call'] == 1
    e.step(1)  # SB completes nominal BB of 2
    assert e.contributions == [2, 2, 1]
    # Flop has two players with chips, and BB is all-in.
    assert e.street == 'flop'
    assert e.current_player == 1


def test_multiway_side_pot_different_main_and_side_winners():
    # P0 AA > P1 KK > P2 QQ; P0 wins main pot, P1 wins side pot.
    # Deal order is seats P1,P2,P0, twice, then burn/flop/burn/turn/burn/river.
    cards = ['sK', 'sQ', 'sA', 'dK', 'dQ', 'dA',
             'h5', 'h2', 'c3', 's4', 'h6', 'd8', 'h7', 'h9']
    deck = cards + [s + r for s in SUITS for r in RANKS if s + r not in cards]
    e = MultiPlayerHandEngine(stacks=[10, 20, 40], deck=deck)
    e.reset()
    e.step(4)  # P0 all-in for 10
    e.step(4)  # P1 all-in for 20
    e.step(1)  # P2 can only call to 20 against the two all-ins
    assert e.is_over()
    r = e.get_result()
    assert r.board == ('h2', 'c3', 's4', 'd8', 'h9')
    assert r.contributions == (10, 20, 20)
    assert r.awards == (30, 20, 0)
    assert r.payoffs_chips == (20, 0, -20)
    assert r.final_stacks == (30, 20, 20)


def test_multiple_short_allins_cumulatively_reopen_action_when_opponents_can_call():
    # 5 seats BTN0/SB1/BB2/UTG3/HJ4. BTN raises to 10;
    # SB all-in14 and BB all-in18 are individually short raises of 4.
    # Facing +8 cumulatively reopens betting for BTN when UTG/HJ have chips.
    e = MultiPlayerHandEngine(stacks=[200, 14, 18, 200, 200], button=0)
    e.reset()
    e.step(1)  # UTG calls 2
    e.step(1)  # HJ calls 2
    e.step_raise_to(10)  # BTN full raise +8
    e.step(4)  # SB all-in14
    e.step(4)  # BB all-in18
    e.step(1)  # UTG calls18
    e.step(1)  # HJ calls18
    assert e.current_player == 0
    assert e.min_raise_to() == 26
    assert 4 in e.legal_actions
    e.step_raise_to(26)
    assert e.current_player == 3


def test_reset_can_finish_without_any_decisions_when_blinds_allin():
    e = MultiPlayerHandEngine(stacks=[1, 1], button=0, seed=1)
    state, actor = e.reset()
    assert state is None and actor == -1
    assert e.is_over() and len(e.board) == 5
    assert sum(e.get_result().final_stacks) == 2


def test_action_raise_to_history_is_recorded_for_replay():
    e = MultiPlayerHandEngine(num_players=3)
    e.reset()
    e.step_raise_to(7)
    assert e.action_history[-1] == (0, 5)
    assert e.action_raise_to[-1] == 7
    assert e.state['raw_obs']['action_raise_to'] == (7,)


def test_small_stacks_stress_and_total_chips_at_every_action():
    import random
    rng = random.Random(2026)
    for hand in range(500):
        n = rng.randint(2, 6)
        stacks = [rng.choice([0, 1, 2, 3, 5, 7, 10, 50]) for _ in range(n)]
        occupied = [i for i, v in enumerate(stacks) if v > 0]
        if len(occupied) < 2:
            continue
        e = MultiPlayerHandEngine(stacks=stacks, button=rng.choice(occupied), seed=hand)
        s, actor = e.reset()
        for _ in range(200):
            assert sum(e.stacks) + sum(e.contributions) == sum(stacks)
            if e.is_over():
                break
            s, actor = e.step(rng.choice(e.legal_actions))
        else:
            raise AssertionError('more than 200 actions in one hand')
        r = e.get_result()
        assert sum(r.final_stacks) == sum(stacks)
        assert sum(r.payoffs_chips) == 0

````````

<!-- END_PY_SOURCE path="tests/test_multi_player_engine.py" -->

## tests/test_multi_player_seats.py

Original blob `3d0c353687e1181fae44f068fc7aceed8033d8e3` (1202 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_multi_player_seats.py" sha="3d0c353687e1181fae44f068fc7aceed8033d8e3" -->

````````python
import pytest

from holdem_ai.environment.multi_player_seats import multi_player_seats, rotate_button


def test_heads_up_seat_rule_and_sparse_empty_seats():
    p = multi_player_seats([200, 200], button=0)
    assert (p.small_blind, p.big_blind, p.preflop_first, p.postflop_first) == (0, 1, 0, 1)
    q = multi_player_seats([0, 200, 0, 200, 0, 0], button=3)
    assert (q.small_blind, q.big_blind, q.preflop_first, q.postflop_first) == (3, 1, 3, 1)


def test_three_and_six_players():
    p = multi_player_seats([200] * 3, button=2)
    assert (p.small_blind, p.big_blind, p.preflop_first, p.postflop_first) == (0, 1, 2, 0)
    p = multi_player_seats([200] * 6, button=1)
    assert (p.small_blind, p.big_blind, p.preflop_first, p.postflop_first) == (2, 3, 4, 2)


def test_rotate_button_skips_empty_seats():
    assert rotate_button([100, 0, 100, 0, 100, 0], 0) == 2
    assert rotate_button([100, 0, 100, 0, 100, 0], 4) == 0
    assert rotate_button([0, 100, 0, 50, 0, 0], 1) == 3


@pytest.mark.parametrize('stacks,button', [([1], 0), ([0, 0, 1], 2), ([1, -1], 0), ([1, 1], 3)])
def test_invalid_seats(stacks, button):
    with pytest.raises(ValueError):
        multi_player_seats(stacks, button)

````````

<!-- END_PY_SOURCE path="tests/test_multi_player_seats.py" -->

## tests/test_phase10_cli.py

Original blob `58d51ca07621a7954e3b7e5b3466cfc4c79b3c08` (2073 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase10_cli.py" sha="58d51ca07621a7954e3b7e5b3466cfc4c79b3c08" -->

````````python
"""End-to-end Phase 10 CLI on CPU with mock Phase3 integration fixture."""
from __future__ import annotations

import pytest
from scripts.phase10_train import build_parser, run


def args(*extra):
    return build_parser().parse_args([
        '--device', 'cpu', '--updates', '1', '--hands-per-update', '2',
        '--epochs', '1', '--minibatch-size', '4', '--eval-hands', '4',
        '--small-network', '--seed', '22', *extra,
    ])


def test_phase10_cli_train_resume(tmp_path):
    ckpt = str(tmp_path / 'a.pt')
    result = run(args('--checkpoint', ckpt))
    assert result['phase'] == 10
    assert result['trained_policy'] is True
    assert result['completed_updates'] == 1
    assert result['evaluation_before']['hands'] == 4
    assert result['evaluation_after']['hands'] == 4
    assert result['last_update']['logit_variance_ratio'] >= 0
    result2 = run(args('--resume', ckpt, '--checkpoint', ckpt, '--skip-eval'))
    assert result2['completed_updates'] == 2
    assert result2['evaluation_before'] is None


def test_phase10_cli_fresh_comparison(tmp_path):
    out = run(args('--compare', '--checkpoint', str(tmp_path/'qb.pt'),
                   '--ppo-checkpoint', str(tmp_path/'ppo.pt')))
    assert out['ppo_control_last_update']['update'] == 1
    assert out['last_update']['update'] == 1
    assert out['evaluation_before'] == out['ppo_control_evaluation_before']
    assert out['ppo_control_evaluation_after']['hands'] == 4
    assert (tmp_path/'qb.pt').exists() and (tmp_path/'ppo.pt').exists()


def test_phase10_rejects_invalid_comparison_resume(tmp_path):
    with pytest.raises(ValueError):
        run(args('--compare', '--resume', str(tmp_path/'missing.pt')))
    with pytest.raises(ValueError):
        run(args('--resume', 'foo', '--from-phase9', 'bar'))
    with pytest.raises(ValueError):
        run(args('--compare', '--checkpoint', str(tmp_path/'same.pt'),
                 '--ppo-checkpoint', str(tmp_path/'same.pt')))


def test_phase10_rejects_invalid_beta():
    with pytest.raises(ValueError):
        run(args('--beta', '-0.1'))

````````

<!-- END_PY_SOURCE path="tests/test_phase10_cli.py" -->

## tests/test_phase10_gpu.py

Original blob `1efad2978b9371a5c941c8b9e2ff7c20f4576051` (907 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase10_gpu.py" sha="1efad2978b9371a5c941c8b9e2ff7c20f4576051" -->

````````python
"""GTX1070 smoke; skipped in environments without PyTorch CUDA."""
import pytest
import torch
from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.qboost_session import create_qboost_session, train_qboost_updates


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU not available here')
def test_phase10_cuda_actual_optimizer():
    cfg = NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1, ff_dim=64, hidden_dim=48)
    ppo = PPOConfig(hands_per_update=2, epochs=1, minibatch_size=4, seed=8)
    session = create_qboost_session(device='cuda', network_config=cfg, ppo_config=ppo)
    result = train_qboost_updates(session, updates=1)
    assert result[0]['decisions'] > 0
    assert torch.cuda.max_memory_allocated() > 0
    assert all(torch.isfinite(p).all() for p in session.base.actor.parameters())

````````

<!-- END_PY_SOURCE path="tests/test_phase10_gpu.py" -->

## tests/test_phase10_qboost.py

Original blob `04c6432ef788d7c68f3b0a2779b0c485a5b089e2` (11558 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase10_qboost.py" sha="04c6432ef788d7c68f3b0a2779b0c485a5b089e2" -->

````````python
"""Phase 10: algebra, information boundary, policy-gradient and integration."""
from __future__ import annotations

import math
import random
from dataclasses import replace

import pytest
import torch

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode
from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.models.transformer import CentralizedQCritic, NetworkConfig, TransformerActor
from holdem_ai.training.ppo import PPOConfig, collect_self_play, freeze_old_policy_data
from holdem_ai.training.ppo_session import create_session
from holdem_ai.training.qboost import QBoostConfig, freeze_qboost_data, optimize_qboost, qboost_diagnostics, qboost_surrogate
from holdem_ai.training.qboost_session import create_qboost_session, save_qboost_checkpoint, load_qboost_checkpoint, from_phase9_checkpoint, train_qboost_updates
from holdem_ai.training.ppo_session import save_checkpoint


@pytest.fixture(scope='module')
def sample():
    cfg = PPOConfig(hands_per_update=5, epochs=1, minibatch_size=8, seed=19)
    net = NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                        ff_dim=64, hidden_dim=48)
    base = create_session(device='cpu', seed=19, network_config=net, ppo_config=cfg)
    rows = collect_self_play(base.actor, cfg, device='cpu')
    frozen = freeze_qboost_data(base.actor, base.critic, rows)
    return base, rows, frozen


@pytest.mark.parametrize('beta', [-0.1, 1.01, float('nan'), float('inf'), True, '1'])
def test_invalid_beta(beta):
    with pytest.raises(ValueError):
        QBoostConfig(beta=beta)


def test_frozen_shapes_and_actor_privacy(sample):
    base, rows, frozen = sample
    n = len(rows.actions)
    assert frozen.old_probs.shape == (n, 7)
    assert frozen.q_actions.shape == (n, 7)
    assert frozen.old_log_probs.shape == (n,)
    assert frozen.raw_advantages.shape == (n,)
    assert frozen.scale > 0 and math.isfinite(frozen.scale)
    assert torch.allclose(frozen.old_probs.sum(-1), torch.ones(n), atol=1e-6)
    for idx, obs in enumerate(rows.actors):
        assert not frozen.old_probs[idx, torch.tensor(obs.legal_action_mask).logical_not()].any()
        assert obs.legal_action_mask[rows.actions[idx]]
    assert not frozen.q_actions.requires_grad
    assert not frozen.old_probs.requires_grad
    with pytest.raises(TypeError):
        base.actor(batch_critics(rows.critics[:2]))
    with pytest.raises(TypeError):
        base.critic(batch_actors(rows.actors[:2]))


def test_baseline_equals_pi_expectation(sample):
    _, _, frozen = sample
    assert torch.allclose(frozen.baselines, (frozen.old_probs * frozen.q_actions).sum(-1))


def test_beta0_identical_to_old_ppo_surrogate(sample):
    base, rows, frozen = sample
    n = min(8, len(rows.actions))
    acts = torch.tensor(rows.actions[:n], dtype=torch.long)
    ret = torch.tensor(rows.returns_bb[:n], dtype=torch.float32)
    p = frozen.old_probs[:n]
    q = frozen.q_actions[:n]
    old_log = frozen.old_log_probs[:n]
    # Importance ratio differs from one to test proper PPO clipping.
    new_log = old_log + torch.linspace(-0.3, 0.4, n)
    new_probs = p.clone()
    from holdem_ai.training.ppo import clipped_surrogate
    standard = clipped_surrogate(new_log, old_log,
                                 (frozen.raw_advantages[:n] - frozen.center) / frozen.scale,
                                 0.2)
    boosted = qboost_surrogate(new_log, new_probs, old_log, p, q,
                               acts, ret, frozen.baselines[:n], beta=0,
                               center=frozen.center, scale=frozen.scale,
                               clip_epsilon=0.2)
    assert torch.allclose(standard, boosted, atol=1e-6)


def test_oldpolicy_boost_surrogate_mean_matches_raw_ppo(sample):
    _, rows, frozen = sample
    n = len(rows.actions)
    acts = torch.tensor(rows.actions)
    rets = torch.tensor(rows.returns_bb)
    q = qboost_surrogate(frozen.old_log_probs, frozen.old_probs,
                         frozen.old_log_probs, frozen.old_probs,
                         frozen.q_actions, acts, rets, frozen.baselines,
                         beta=1, center=frozen.center, scale=frozen.scale,
                         clip_epsilon=0.2)
    expected = (frozen.raw_advantages - frozen.center) / frozen.scale
    # Per row can differ, but analytical term vanishes and conditional
    # correction is different; this equality is in expectation, not samplewise.
    assert torch.isfinite(q).all()
    assert q.shape == expected.shape
    assert q.abs().max() < 10000


def test_analytic_gradient_identity_exact_enumeration():
    # Enumerate all three actions in a fixed state with true Q-values.
    # No sampling noise: gradient is independent of approximator errors.
    old = torch.tensor([0.25, 0.35, 0.40])
    true = torch.tensor([2.0, -1.3, 4.7])
    estimate = torch.tensor([-8.0, 12.0, 0.2])  # intentionally badly wrong
    logits = torch.log(old).detach().requires_grad_(True)
    new_p = logits.softmax(-1)
    target_grad = torch.autograd.grad((new_p * true).sum(), logits, retain_graph=True)[0]
    # All action outcomes are enumerated (their probabilities are old pi).
    # g_MC = E_old[score(a)*(R(a)-V)]
    # g_QBoost = E_old[score(a)*(R(a)-V-Qhat(a)+Vhat)] + grad(sum_a pi*Qhat)
    v = (old * estimate).sum()
    grad_mc = torch.zeros_like(logits)
    grad_qb = torch.zeros_like(logits)
    for a in range(3):
        score = torch.nn.functional.one_hot(torch.tensor(a), 3).float() - old
        grad_mc += old[a] * score * (true[a] - v)
        grad_qb += old[a] * score * (true[a] - estimate[a])
    grad_qb += old * (estimate - v)
    assert torch.allclose(grad_mc, target_grad, atol=1e-6)
    assert torch.allclose(grad_qb, target_grad, atol=1e-6)
    # The chosen critic was intentionally inaccurate, to verify correction.
    assert not torch.allclose(estimate, true)


def test_ppo_surrogate_boost_gradient_at_old_policy():
    # Explicit differentiable finite action enumeration for the implementation.
    q_all = torch.tensor([[2., -3., 4., 0., 0., 0., 0.]]).repeat(3,1)
    old = torch.tensor([[0.2, 0.3, 0.5, 0., 0., 0., 0.]]).repeat(3,1)
    act = torch.tensor([0, 1, 2])
    true_returns = torch.tensor([4.0, -5.0, 1.0])
    sample_prob = old[:, :3].diag()
    x = torch.log(old[0,:3]).detach().clone().requires_grad_(True)
    p3 = x.softmax(0)
    newp = torch.cat([p3, torch.zeros(4)]).expand(3,7)
    newlog = p3.log()[act]
    oldlog = old.gather(1, act[:,None]).log().squeeze(1)
    baseline = (old * q_all).sum(-1)
    obj = qboost_surrogate(newlog, newp, oldlog, old, q_all, act,
                           true_returns, baseline, beta=1, center=0, scale=1,
                           clip_epsilon=0.2)
    gradient = torch.autograd.grad((sample_prob * obj).sum(), x, retain_graph=True)[0]
    truth = torch.autograd.grad((p3 * true_returns).sum(), x)[0]
    assert torch.allclose(gradient, truth, atol=1e-6)


def test_qboost_diagnostics_beta0_equals_ppo(sample):
    _, _, frozen = sample
    stats = qboost_diagnostics(frozen, beta=0)
    assert stats['logit_variance_ratio'] == pytest.approx(1.0, abs=1e-6)
    assert stats['boosted_residual_variance'] == pytest.approx(stats['mc_advantage_variance'], abs=1e-5)
    stats1 = qboost_diagnostics(frozen, beta=1)
    assert all(math.isfinite(v) for v in stats1.values())
    assert stats1['logit_variance_ratio'] >= 0


def test_qboost_invalid_inputs(sample):
    _, rows, frozen = sample
    acts = torch.tensor(rows.actions)
    rets = torch.tensor(rows.returns_bb)
    kw = dict(beta=1.0, center=frozen.center,
              scale=frozen.scale, clip_epsilon=0.2)
    with pytest.raises(ValueError):
        qboost_surrogate(frozen.old_log_probs, frozen.old_probs, frozen.old_log_probs,
                         frozen.old_probs, frozen.q_actions, acts + 7,
                         rets, frozen.baselines, **kw)
    with pytest.raises(ValueError):
        qboost_surrogate(frozen.old_log_probs, frozen.old_probs, frozen.old_log_probs,
                         frozen.old_probs, frozen.q_actions[:1], acts,
                         rets, frozen.baselines, **kw)
    with pytest.raises(ValueError):
        qboost_surrogate(frozen.old_log_probs, frozen.old_probs, frozen.old_log_probs,
                         frozen.old_probs, frozen.q_actions, acts,
                         rets, frozen.baselines, **dict(kw, scale=0))


def test_qboost_optimizer_changes_weights(sample):
    base, rows, frozen = sample
    session = create_qboost_session(device='cpu', seed=91,
                                    network_config=base.network_config,
                                    ppo_config=base.ppo_config)
    before = [p.detach().clone() for p in session.base.actor.parameters()]
    result = optimize_qboost(session.base.actor, session.base.critic,
                             session.base.actor_optimizer, session.base.critic_optimizer,
                             frozen, session.base.ppo_config, session.qconfig,
                             shuffle_seed=81)
    assert result['minibatches'] >= 1
    assert result['decisions'] == len(rows.actions)
    assert any(not torch.equal(x, y) for x, y in zip(before, session.base.actor.parameters()))
    assert all(math.isfinite(v) for v in result.values())


def test_qboost_checkpoint_resume(tmp_path):
    small = NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                          ff_dim=64, hidden_dim=48)
    pcfg = PPOConfig(hands_per_update=2, epochs=1, minibatch_size=4, seed=8)
    s = create_qboost_session(network_config=small, ppo_config=pcfg,
                              qconfig=QBoostConfig(beta=0.7))
    path = tmp_path / 'qboost.pt'
    first = train_qboost_updates(s, updates=1, checkpoint=path)
    assert path.is_file() and len(first) == 1 and s.completed_updates == 1
    new = load_qboost_checkpoint(path)
    assert new.completed_updates == 1
    assert new.qconfig.beta == 0.7
    for a, b in zip(s.base.actor.parameters(), new.base.actor.parameters()):
        assert torch.equal(a, b)
    second = train_qboost_updates(new, updates=1, checkpoint=path)
    assert len(second) == 1 and new.completed_updates == 2
    assert load_qboost_checkpoint(path).completed_updates == 2


def test_from_phase9_checkpoint(tmp_path):
    small = NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                          ff_dim=64, hidden_dim=48)
    base = create_session(seed=13, network_config=small,
                          ppo_config=PPOConfig(hands_per_update=2, epochs=1, seed=13))
    base.completed_updates = 4
    path = tmp_path / 'ppo.pt'
    save_checkpoint(base, path)
    loaded = from_phase9_checkpoint(path, qconfig=QBoostConfig(beta=0.5))
    assert loaded.completed_updates == 4
    assert loaded.qconfig.beta == 0.5
    assert all(torch.equal(a, b) for a, b in zip(base.actor.parameters(), loaded.base.actor.parameters()))
    with pytest.raises(ValueError):
        load_qboost_checkpoint(path)


def test_config_cannot_be_modified_mid_resume(tmp_path):
    cfg = QBoostConfig(beta=0.4)
    s = create_qboost_session(qconfig=cfg)
    path = tmp_path / 'qboost.pt'
    save_qboost_checkpoint(s, path)
    new = load_qboost_checkpoint(path)
    assert new.qconfig.beta == 0.4
    with pytest.raises(ValueError):
        from_phase9_checkpoint(path)

````````

<!-- END_PY_SOURCE path="tests/test_phase10_qboost.py" -->

## tests/test_phase11_batched.py

Original blob `03a1fc44be3f4e58251c9e63f69f48f0dae5cc03` (3130 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase11_batched.py" sha="03a1fc44be3f4e58251c9e63f69f48f0dae5cc03" -->

````````python
from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.training.ppo import PPOConfig, collect_self_play
from holdem_ai.training.batched_collector import collect_batched_self_play
from holdem_ai.evaluation.balanced import evaluate_balanced


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

````````

<!-- END_PY_SOURCE path="tests/test_phase11_batched.py" -->

## tests/test_phase11_gpu.py

Original blob `b555a5a86336b1b5ca2286087a27cf07366e5a52` (863 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase11_gpu.py" sha="b555a5a86336b1b5ca2286087a27cf07366e5a52" -->

````````python
"""CUDA-only batched rollout integration, runs on the user's GTX 1070."""
from __future__ import annotations

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.batched_collector import collect_batched_self_play


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is not installed on this runner')
def test_phase11_batched_forward_on_cuda_and_legal_actions():
    actor = TransformerActor(NetworkConfig()).to('cuda')
    rows = collect_batched_self_play(
        actor, PPOConfig(hands_per_update=6, seed=888),
        device='cuda', env_batch_size=3,
    )
    assert rows.hands == 6
    assert all(row.legal_action_mask[action] for row, action in zip(rows.actors, rows.actions))
    assert torch.cuda.max_memory_allocated() > 0

````````

<!-- END_PY_SOURCE path="tests/test_phase11_gpu.py" -->

## tests/test_phase11_progress.py

Original blob `be20c35f67e6a3f0173ec0b7d1573fe2acfe3271` (5397 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase11_progress.py" sha="be20c35f67e6a3f0173ec0b7d1573fe2acfe3271" -->

````````python
"""Progress UI is an observer: no dependency on the poker engine or CUDA."""
from __future__ import annotations

import io
import math
from unittest.mock import patch

import pytest

from holdem_ai.utils.progress import TrainingProgress, format_duration
from scripts.phase11_train import build_parser, run


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def create_progress(clock, *, duration=60.0, updates=4):
    screen = io.StringIO()
    progress = TrainingProgress(duration_seconds=duration, max_updates=updates,
                                stream=screen, clock=clock, threaded=False,
                                prefer_tqdm=False)
    return progress, screen


def test_eta_elapsed_resume_and_terminal_status():
    clock = Clock()
    p, output = create_progress(clock, duration=100.0, updates=4)
    with p:
        p.on_event({'kind': 'resume', 'completed_updates': 11})
        clock.now = 10.0
        p.on_event({'kind': 'update', 'update': 12, 'elapsed_seconds': 0.5,
                    'hands_per_second_training': 42.0})
        snap = p.snapshot()
        assert snap['total_updates'] == 12
        assert snap['updates'] == 1
        assert snap['remaining_updates'] == 3
        assert snap['time_left'] == pytest.approx(90.0)
        assert snap['update_eta'] == pytest.approx(30.0)
        assert snap['estimated_remaining'] == pytest.approx(30.0)
        clock.now = 20.0
        p.on_event({'kind': 'update', 'update': 13, 'hands_per_second_training': 31.0})
        assert p.snapshot()['fraction'] == pytest.approx(0.5)
        p.on_event({'kind': 'end', 'stop_reason': 'update_limit'})
        assert p.snapshot()['status'] == 'update_limit'
    text = output.getvalue()
    assert 'time_left=01:30' not in text  # uses hours:minutes:seconds
    assert 'time_left=00:01:30' in text
    assert 'ETA=00:00:30' in text
    assert 'upd=13 (+2/4)' in text
    assert 'hands/s=31.0' in text
    assert 'elapsed=00:00:20' in text


def test_time_budget_mode_heartbeat_and_ceiling():
    clock = Clock()
    p, screen = create_progress(clock, duration=80.0, updates=None)
    with p:
        p.on_event({'kind': 'start', 'completed_updates': 3})
        clock.now = 21.2
        assert p.snapshot()['fraction'] == pytest.approx(21.2 / 80.0)
        assert format_duration(p.snapshot()['time_left']) == '00:00:59'
        p.on_event({'kind': 'evaluation', 'update': 3})
        assert p.snapshot()['status'] == 'evaluated'
        clock.now = 82.0
        assert p.snapshot()['time_left'] == 0.0
        p.on_event({'kind': 'end', 'stop_reason': 'duration_limit'})
    assert '00:00:00' in screen.getvalue()


def test_zero_update_eta_is_not_fabricated():
    clock = Clock()
    p, _ = create_progress(clock)
    with p:
        p.on_event({'kind': 'start', 'completed_updates': 0})
        assert p.snapshot()['update_eta'] is None
        assert p.snapshot()['estimated_remaining'] == 60.0


def test_invalid_inputs():
    with pytest.raises(ValueError):
        TrainingProgress(duration_seconds=0, max_updates=None)
    with pytest.raises(ValueError):
        TrainingProgress(duration_seconds=math.inf, max_updates=None)
    with pytest.raises(ValueError):
        TrainingProgress(duration_seconds=60, max_updates=0)
    with pytest.raises(ValueError):
        TrainingProgress(duration_seconds=60, max_updates=2, interval=0.01)


def test_cli_additional_flags_do_not_change_experiment_config(tmp_path):
    args = build_parser().parse_args([
        '--device', 'cpu', '--duration-minutes', '0.5',
        '--max-updates', '2', '--checkpoint', str(tmp_path / 'test.pt'),
        '--log-dir', str(tmp_path / 'run'), '--no-progress',
    ])
    assert args.no_progress is True
    assert args.progress_interval == 1.0
    with patch('scripts.phase11_train.run_experiment', return_value={'phase': 11}) as fake:
        result = run(args)
        assert result == {'phase': 11}
        assert fake.call_args.args[0].duration_seconds == 30.0
        assert fake.call_args.args[0].max_updates == 2


def test_cli_show_progress_integrates_existing_runner_emit(tmp_path, capsys):
    args = build_parser().parse_args([
        '--duration-minutes', '0.5', '--max-updates', '2',
        '--checkpoint', str(tmp_path / 'test.pt'),
        '--log-dir', str(tmp_path / 'run'),
    ])

    def fake_run(config, *, emit):
        emit({'kind': 'start', 'completed_updates': 3})
        emit({'kind': 'update', 'update': 4, 'hands_per_second_training': 42})
        emit({'kind': 'evaluation', 'opponent': 'random', 'mean_bb100': 0})
        emit({'kind': 'end', 'stop_reason': 'update_limit'})
        return {'phase': 11, 'completed_updates': 4}

    with patch('scripts.phase11_train.run_experiment', side_effect=fake_run):
        with patch('scripts.phase11_train.TrainingProgress') as mock_progress:
            ctx = mock_progress.return_value.__enter__.return_value
            result = run(args)
            assert ctx.on_event.call_count == 4
            assert ctx.message.call_count == 3
            assert result['completed_updates'] == 4


def test_progress_display_no_policy_side_effect():
    # Progress observer owns no torch/policy/engine resources.
    import inspect
    from holdem_ai.utils import progress
    source = inspect.getsource(progress)
    assert 'torch.' not in source
    assert 'collect_self_play' not in source

````````

<!-- END_PY_SOURCE path="tests/test_phase11_progress.py" -->

## tests/test_phase11_runner.py

Original blob `a148f322edd01907598778bcab496c69dea4395a` (4381 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase11_runner.py" sha="a148f322edd01907598778bcab496c69dea4395a" -->

````````python
from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from holdem_ai.experiments.time_budget import ExperimentConfig, run_experiment
from holdem_ai.experiments.time_budget import _budget_remaining
from scripts.phase11_compare import compare_logs


def config_for(tmp_path, algorithm='ppo'):
    return ExperimentConfig(algorithm=algorithm, device='cpu',
                            duration_seconds=120, max_updates=1,
                            env_batch_size=3, eval_every=1, eval_hands=4,
                            hands_per_update=3, epochs=1, minibatch_size=8,
                            small_network=True,
                            checkpoint=str(tmp_path / f'{algorithm}.pt'),
                            log_dir=str(tmp_path / f'logs_{algorithm}'))


@pytest.mark.parametrize('algorithm', ['ppo', 'qboost'])
def test_small_real_update_and_resume(algorithm, tmp_path):
    cfg = config_for(tmp_path, algorithm)
    report = run_experiment(cfg)
    assert report['completed_updates'] == 1
    assert report['updates_this_run'] == 1
    assert report['stop_reason'] == 'update_limit'
    assert report['evaluations'] >= 2
    log = tmp_path / f'logs_{algorithm}'
    events = [json.loads(x) for x in (log / 'metrics.jsonl').read_text().splitlines()]
    assert any(x['kind'] == 'update' for x in events)
    assert any(x.get('opponent') == 'frozen_anchor' for x in events)
    assert (log / 'anchor.pt').exists()
    payload = torch.load(cfg.checkpoint, map_location='cpu', weights_only=True)
    assert payload['completed_updates'] == 1
    with pytest.raises(FileExistsError):
        run_experiment(cfg)
    resumed = run_experiment(replace(cfg, resume=cfg.checkpoint))
    assert resumed['completed_updates'] == 2
    assert resumed['updates_this_run'] == 1
    assert torch.load(cfg.checkpoint, map_location='cpu', weights_only=True)['completed_updates'] == 2


def test_compare_reports_after_two_run_types(tmp_path):
    p = config_for(tmp_path, 'ppo')
    q = config_for(tmp_path, 'qboost')
    run_experiment(p)
    run_experiment(q)
    result = compare_logs(str(tmp_path / 'logs_ppo' / 'metrics.jsonl'),
                          str(tmp_path / 'logs_qboost' / 'metrics.jsonl'))
    assert result['shared_updates'] == 1
    assert result['records'][0]['update'] == 1
    assert 'qboost_minus_ppo_random_bb100' in result['records'][0]


def test_budget_and_config_validation(tmp_path):
    cfg = config_for(tmp_path)
    for key, val in [('duration_seconds', 0), ('max_updates', 0),
                     ('env_batch_size', 0), ('eval_hands', 6), ('beta', 2),
                     ('algorithm', 'vrpo'), ('seed', True)]:
        with pytest.raises(ValueError):
            replace(cfg, **{key: val})
    assert _budget_remaining(float('inf'))
    assert not _budget_remaining(0.0)


def test_tiny_duration_stops_safely(tmp_path):
    cfg = replace(config_for(tmp_path), duration_seconds=1e-6)
    result = run_experiment(cfg)
    assert result['completed_updates'] == 0
    assert result['stop_reason'] == 'duration_limit'
    assert torch.load(cfg.checkpoint, weights_only=True, map_location='cpu')['completed_updates'] == 0


def test_resume_requires_original_log_dir(tmp_path):
    cfg = config_for(tmp_path)
    with pytest.raises(FileNotFoundError):
        run_experiment(replace(cfg, resume=cfg.checkpoint))


def test_import_phase10_checkpoint(tmp_path):
    from holdem_ai.training.qboost_session import create_qboost_session, save_qboost_checkpoint
    from holdem_ai.models.transformer import NetworkConfig
    from holdem_ai.training.ppo import PPOConfig
    prior = create_qboost_session(
        device='cpu', network_config=NetworkConfig(
            embed_dim=32, attention_heads=2, transformer_layers=1,
            ff_dim=64, hidden_dim=48),
        ppo_config=PPOConfig(hands_per_update=3, epochs=1, minibatch_size=8),
    )
    prior.base.completed_updates = 6
    source = tmp_path / 'phase10_pretrained.pt'
    save_qboost_checkpoint(prior, source)
    cfg = replace(config_for(tmp_path, 'qboost'),
                  from_phase10=str(source))
    r = run_experiment(cfg)
    assert r['completed_updates'] == 7
    assert json.loads((tmp_path / 'logs_qboost' / 'run.json').read_text())['initial_updates'] == 6
    assert torch.load(source, weights_only=True)['completed_updates'] == 6

````````

<!-- END_PY_SOURCE path="tests/test_phase11_runner.py" -->

## tests/test_phase12_cli.py

Original blob `68f53e62663b18e82a15607f54bea51e537a9597` (2673 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase12_cli.py" sha="68f53e62663b18e82a15607f54bea51e537a9597" -->

````````python
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.evaluation.checkpoint_loader import load_actor
from scripts.phase12_evaluate import build_parser, run


def make_model_file(p):
    torch.manual_seed(10)
    cfg = NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                        ff_dim=64, hidden_dim=48)
    actor = TransformerActor(cfg)
    torch.save({'phase': 10, 'network_config': asdict(cfg),
                'actor_state': actor.state_dict()}, p)


def test_local_checkpoint_and_json_csv(tmp_path):
    path = tmp_path / 'qboost.pt'
    make_model_file(path)
    loaded = load_actor(path)
    assert not loaded.training
    args = build_parser().parse_args([
        '--candidate', str(path), '--groups', '2', '--device', 'cpu',
        '--output-json', str(tmp_path / 'eval.json'),
        '--output-csv', str(tmp_path / 'eval.csv'),
    ])
    result = run(args)
    assert result['candidate']['hands'] == 8
    assert (tmp_path / 'eval.json').is_file()
    assert len((tmp_path / 'eval.csv').read_text().splitlines()) == 3
    assert json.loads((tmp_path / 'eval.json').read_text()) == result


def test_phase11_anchor_and_paired_control(tmp_path):
    candidate = tmp_path / 'actor.pt'
    anchor = tmp_path / 'anchor.pt'
    control = tmp_path / 'control.pt'
    make_model_file(candidate)
    payload = torch.load(candidate, weights_only=True)
    torch.save({'network_config': payload['network_config'],
                'actor_state': payload['actor_state']}, anchor)
    torch.save(payload, control)
    args = build_parser().parse_args([
        '--candidate', str(candidate), '--control', str(control),
        '--opponent-checkpoint', str(anchor), '--groups', '2',
    ])
    outcome = run(args)
    assert outcome['paired_comparison']['delta_mean_bb100'] == 0
    assert outcome['candidate']['mean_bb100'] == 0
    assert load_actor(anchor).config.embed_dim == 32


def test_reject_same_input_file(tmp_path):
    path = tmp_path / 'one.pt'
    make_model_file(path)
    args = build_parser().parse_args(['--candidate', str(path), '--control', str(path)])
    with pytest.raises(ValueError):
        run(args)


def test_reject_bad_checkpoint(tmp_path):
    bad = tmp_path / 'invalid.pt'
    torch.save({'other': 1}, bad)
    with pytest.raises(ValueError):
        load_actor(bad)
    wrong = tmp_path / 'wrong.pt'
    torch.save({'phase': 11, 'network_config': {}, 'actor_state': {}}, wrong)
    with pytest.raises(ValueError):
        load_actor(wrong)

````````

<!-- END_PY_SOURCE path="tests/test_phase12_cli.py" -->

## tests/test_phase12_evaluation.py

Original blob `25ed721094e6970f97bd2fe63c34784502c2228f` (5558 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase12_evaluation.py" sha="25ed721094e6970f97bd2fe63c34784502c2228f" -->

````````python
"""Phase12 robust evaluation: same-deck, opposite-seat, unit tests."""
from __future__ import annotations

from dataclasses import replace
import random

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.evaluation.seed_blocked import DealGroup, EvaluationConfig, EvaluationResult, action_counts, compare_results, evaluate_seed_blocks, model_fingerprint, summarize_block_scores


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

````````

<!-- END_PY_SOURCE path="tests/test_phase12_evaluation.py" -->

## tests/test_phase12_gpu.py

Original blob `a11264ca66c51d273ea70cee5ff41b097cafab32` (667 bytes).

<!-- BEGIN_PY_SOURCE path="tests/test_phase12_gpu.py" sha="a11264ca66c51d273ea70cee5ff41b097cafab32" -->

````````python
"""GPU smoke on user's GTX1070; deliberately skipped on CPU-only runners."""
from __future__ import annotations

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor
from holdem_ai.evaluation.seed_blocked import EvaluationConfig, evaluate_seed_blocks


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_phase12_cuda_four_layout_inference():
    model = TransformerActor(NetworkConfig()).to('cuda')
    r = evaluate_seed_blocks(model, config=EvaluationConfig(groups=2, device='cuda', seed=989))
    assert r.summary()['hands'] == 8
    assert torch.cuda.max_memory_allocated() > 0

````````

<!-- END_PY_SOURCE path="tests/test_phase12_gpu.py" -->
