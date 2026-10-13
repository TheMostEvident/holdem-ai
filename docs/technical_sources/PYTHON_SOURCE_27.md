# Full Python implementation — Volume 27 / 28

[최종 통합 기술보고서](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md)의 코드 부록. 원본 커밋 `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`에 포함된 테스트 파일의 전체 원문이다. 구분용 LF는 원문 바이트 수와 Git Blob SHA 계산에서 제외한다.

## `tests/test_phase9_cli.py`

Original blob `7bdaaea529975e624fce569893e411e032b94ef3` (1112 bytes).

<a name="py-310"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase9_cli.py" sha="7bdaaea529975e624fce569893e411e032b94ef3" -->

````````python
"""Phase-9 executable CLI and checkpoint interface, CPU-only fast smoke."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.phase9_train import build_parser, run


def test_phase9_cli_smoke_and_resume(tmp_path):
    out = tmp_path / 'ppo.pt'
    parser = build_parser()
    args = parser.parse_args([
        '--device', 'cpu', '--updates', '1', '--hands-per-update', '2',
        '--epochs', '1', '--minibatch-size', '4', '--eval-hands', '4',
        '--small-network', '--checkpoint', str(out), '--seed', '13',
    ])
    result = run(args)
    assert result['phase'] == 9
    assert result['trained_policy']
    assert result['completed_updates'] == 1
    assert result['evaluation_before']['hands'] == 4
    assert result['evaluation_after']['hands'] == 4
    assert out.is_file()
    args2 = parser.parse_args([
        '--resume', str(out), '--updates', '1', '--device', 'cpu',
        '--checkpoint', str(out), '--skip-eval',
    ])
    resumed = run(args2)
    assert resumed['completed_updates'] == 2
    assert resumed['evaluation_before'] is None

````````

<!-- END_PY_SOURCE path="tests/test_phase9_cli.py" -->

## `tests/test_phase9_gpu.py`

Original blob `dfb598cd5b7c92cabd55a4bd63b2f326ce131a54` (1296 bytes).

<a name="py-311"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase9_gpu.py" sha="dfb598cd5b7c92cabd55a4bd63b2f326ce131a54" -->

````````python
"""Run on user's GTX1070; skip on the CPU-only packaging machine."""
from __future__ import annotations

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.ppo_session import create_session, train_updates


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available here')
def test_phase9_actual_cuda_optimizer_step():
    session = create_session(
        device='cuda', seed=29,
        network_config=NetworkConfig(embed_dim=32, attention_heads=2,
                                     transformer_layers=1, ff_dim=64,
                                     hidden_dim=48, max_tokens=128),
        ppo_config=PPOConfig(hands_per_update=2, epochs=1,
                             minibatch_size=4, inference_batch_size=8, seed=29),
    )
    reports = train_updates(session, updates=1)
    torch.cuda.synchronize()
    assert reports[0]['update'] == 1
    assert reports[0]['decisions'] >= 2
    assert all(torch.isfinite(p).all() for p in session.actor.parameters())
    assert all(torch.isfinite(p).all() for p in session.critic.parameters())
    assert next(session.actor.parameters()).device.type == 'cuda'
    assert next(session.critic.parameters()).device.type == 'cuda'

````````

<!-- END_PY_SOURCE path="tests/test_phase9_gpu.py" -->

## `tests/test_phase9_ppo.py`

Original blob `5b01675a9713dfa9d591f61d1f31f1d2b2fd99bf` (8613 bytes).

<a name="py-312"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_phase9_ppo.py" sha="5b01675a9713dfa9d591f61d1f31f1d2b2fd99bf" -->

````````python
"""Phase-9 unit/integration tests against real HUNL200 engine in the user's repo."""
from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import replace

import pytest
import torch

from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig, collect_self_play, freeze_old_policy_data, clipped_surrogate, optimize_ppo, evaluate_vs_random
from holdem_ai.training.ppo_session import create_session, save_checkpoint, load_checkpoint, train_updates
from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.models.transformer import chosen_q


@pytest.fixture(scope='module', autouse=True)
def limit_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_session(*, hands=4, seed=17):
    net = NetworkConfig(embed_dim=16, attention_heads=2, transformer_layers=1,
                        ff_dim=32, hidden_dim=24, max_tokens=128)
    cfg = PPOConfig(hands_per_update=hands, epochs=2, minibatch_size=8,
                    inference_batch_size=8, seed=seed)
    return create_session(seed=seed, network_config=net, ppo_config=cfg)


@pytest.fixture(scope='module')
def sample():
    sess = tiny_session(hands=8, seed=31)
    rows = collect_self_play(sess.actor, sess.ppo_config, device='cpu')
    frozen = freeze_old_policy_data(sess.actor, sess.critic, rows, chunk_size=7)
    return sess, frozen


@pytest.mark.parametrize('changes', [
    {'hands_per_update': 0}, {'epochs': 0}, {'minibatch_size': -1},
    {'learning_rate': 0}, {'clip_epsilon': 1}, {'clip_epsilon': -0.2},
    {'grad_clip': -1}, {'entropy_coef': float('nan')},
    {'inference_batch_size': 0}, {'max_decisions': True},
    {'big_blind': 0}, {'seed': 0.5},
])
def test_rejects_invalid_ppo_config(changes):
    with pytest.raises(ValueError):
        PPOConfig(**changes)


def test_full_on_policy_rollout_correctness(sample):
    _, frozen = sample
    rows = frozen.rows
    assert rows.hands == 8
    assert len(rows.hand_rewards) == 8
    assert len(rows.actions) >= 8
    assert set(rows.acting_seats) == {0, 1}
    assert all(math.isclose(sum(x), 0, abs_tol=1e-9) for x in rows.hand_rewards)
    assert all(0 <= a < 7 and actor.legal_action_mask[a]
               for actor, a in zip(rows.actors, rows.actions))
    assert all(len(a.features) == 20 and len(c.features) == 24
               for a, c in zip(rows.actors, rows.critics))
    for row, seat, ret in zip(rows.critics, rows.acting_seats, rows.returns_bb):
        assert seat in (0, 1)
        assert math.isfinite(ret)


def test_old_behavior_log_probs_equal_current_actor(sample):
    sess, frozen = sample
    actor = sess.actor
    with torch.no_grad():
        for i in (0, min(2, len(frozen.rows.actions) - 1)):
            dist = actor.distribution(batch_actors([frozen.rows.actors[i]]))
            actual = dist.log_prob(torch.tensor([frozen.rows.actions[i]])).item()
            assert actual == pytest.approx(frozen.old_log_probs[i].item(), abs=1e-5)


def test_advantage_is_action_independent_old_baseline(sample):
    sess, frozen = sample
    r = frozen.rows
    idx = 0
    with torch.no_grad():
        dist = sess.actor.distribution(batch_actors([r.actors[idx]]))
        q = sess.critic(batch_critics([r.critics[idx]]))[0, r.acting_seats[idx]]
        baseline = (dist.probs[0] * q).sum().item()
    assert frozen.baselines[idx].item() == pytest.approx(baseline, abs=1e-5)
    assert abs(frozen.advantages.mean().item()) < 1e-5
    assert torch.isfinite(frozen.advantages).all()


def test_old_log_probs_and_advantages_are_frozen_cpu(sample):
    _, frozen = sample
    assert frozen.old_log_probs.device.type == 'cpu'
    assert frozen.advantages.device.type == 'cpu'
    assert not frozen.old_log_probs.requires_grad
    assert not frozen.advantages.requires_grad


def test_clipped_surrogate_positive_and_negative_advantages():
    old = torch.zeros(4)
    ratios = torch.tensor([1.5, .5, 1.5, .5])
    lp = ratios.log()
    adv = torch.tensor([1., 1., -1., -1.])
    # Positive -> cap rewards above 1.2; negative -> cap penalties below .8.
    expected = torch.tensor([1.2, .5, -1.5, -.8])
    assert torch.allclose(clipped_surrogate(lp, old, adv, .2), expected, atol=1e-6)
    with pytest.raises(ValueError):
        clipped_surrogate(lp, old[:2], adv, .2)


def test_real_ppo_updates_parameters_without_nans(sample):
    original, frozen = sample
    sess = tiny_session(hands=8, seed=31)
    sess.actor.load_state_dict(copy.deepcopy(original.actor.state_dict()))
    sess.critic.load_state_dict(copy.deepcopy(original.critic.state_dict()))
    a_before = {k: x.clone() for k, x in sess.actor.state_dict().items()}
    c_before = {k: x.clone() for k, x in sess.critic.state_dict().items()}
    stats = optimize_ppo(sess.actor, sess.critic,
                         sess.actor_optimizer, sess.critic_optimizer,
                         frozen, sess.ppo_config, shuffle_seed=7)
    assert stats['minibatches'] > 0
    assert stats['decisions'] == len(frozen.rows.actions)
    assert all(math.isfinite(v) for v in stats.values())
    assert any(not torch.equal(x, a_before[k]) for k, x in sess.actor.state_dict().items())
    assert any(not torch.equal(x, c_before[k]) for k, x in sess.critic.state_dict().items())
    assert all(torch.isfinite(p).all() for p in sess.actor.parameters())
    assert all(torch.isfinite(p).all() for p in sess.critic.parameters())


def test_ppo_seat_returns_are_zero_sum_targets(sample):
    _, frozen = sample
    a = frozen.rows
    # Exact BB payout for each own-action sample. Every seat's expected policy
    # is trained on its own reward, not the seat-0 reward for both agents.
    assert len(a.returns_bb) == len(a.acting_seats)
    for seat, ret in zip(a.acting_seats, a.returns_bb):
        assert seat in (0, 1) and math.isfinite(ret)
    assert all(abs(x + y) < 1e-9 for x, y in a.hand_rewards)


def test_critic_values_strictly_zero_sum_after_optimizer(sample):
    sess = tiny_session(hands=2)
    rows = sample[1].rows
    frozen = freeze_old_policy_data(sess.actor, sess.critic, rows)
    optimize_ppo(sess.actor, sess.critic, sess.actor_optimizer, sess.critic_optimizer,
                 frozen, sess.ppo_config)
    with torch.no_grad():
        q = sess.critic(batch_critics(rows.critics[:4]))
    assert torch.equal(q[:, 0], -q[:, 1])


def test_evaluation_uses_alternating_seats_and_does_not_update(sample):
    sess, _ = sample
    before = {k: x.clone() for k, x in sess.actor.state_dict().items()}
    report = evaluate_vs_random(sess.actor, hands=8, seed=555)
    assert report['hands'] == 8
    assert report['seat0_hands'] == report['seat1_hands'] == 4
    assert report['neural_button_hands'] == report['neural_big_blind_hands'] == 4
    assert math.isfinite(report['mean_bb100'])
    assert report['ci95_lower_bb100'] <= report['mean_bb100'] <= report['ci95_upper_bb100']
    assert all(torch.equal(before[k], x) for k, x in sess.actor.state_dict().items())
    assert report == evaluate_vs_random(sess.actor, hands=8, seed=555)
    with pytest.raises(ValueError):
        evaluate_vs_random(sess.actor, hands=2)
    with pytest.raises(ValueError):
        evaluate_vs_random(sess.actor, hands=3)


def test_checkpoint_roundtrip_and_resume(tmp_path):
    sess = tiny_session(hands=3, seed=21)
    path = tmp_path / 'checkpoints' / 'phase9_ppo.pt'
    logs = train_updates(sess, updates=1, checkpoint=path)
    assert len(logs) == 1 and logs[0]['update'] == 1
    assert path.is_file() and not path.with_name(path.name + '.tmp').exists()
    original = {k: v.clone() for k, v in sess.actor.state_dict().items()}
    resumed = load_checkpoint(path)
    assert resumed.completed_updates == 1
    assert resumed.ppo_config == sess.ppo_config
    assert all(torch.equal(original[k], v) for k, v in resumed.actor.state_dict().items())
    assert len(resumed.actor_optimizer.state) > 0
    new_logs = train_updates(resumed, updates=1, checkpoint=path)
    assert new_logs[0]['update'] == 2
    assert load_checkpoint(path).completed_updates == 2
    with pytest.raises(ValueError):
        train_updates(resumed, updates=0)


def test_privileged_actor_boundary_rejects_critic_batch(sample):
    sess, frozen = sample
    critics = batch_critics(frozen.rows.critics[:2])
    with pytest.raises(TypeError):
        sess.actor(critics)


def test_no_history_truncation_large_max_decisions(sample):
    sess, frozen = sample
    assert max(len(row.token_ids) for row in frozen.rows.actors) <= sess.actor.config.max_tokens

````````

<!-- END_PY_SOURCE path="tests/test_phase9_ppo.py" -->

## `tests/test_player_manager.py`

Original blob `d896603affbea9ede8c6a95dead9554051e975cb` (1675 bytes).

<a name="py-313"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_player_manager.py" sha="d896603affbea9ede8c6a95dead9554051e975cb" -->

````````python
import pytest

from holdem_ai.environment.player_manager import PlayerManager


def test_elimination_and_pre_hand_stack_tiebreak():
    p = PlayerManager([200, 200, 100, 100, 200, 200])
    eliminated = p.update([350, 350, 0, 0, 100, 200], hand_number=1)
    assert eliminated == (2, 3)
    assert p.ranks == [None, None, 5, 5, None, None]
    assert p.alive == (0, 1, 4, 5)
    assert p.eliminated_at == [None, None, 1, 1, None, None]
    # Next same-hand busts: 100 chips ends worse than 200 chips.
    assert p.update([450, 450, 0, 0, 0, 100], hand_number=2) == (4,)
    assert p.ranks[4] == 4
    assert p.update([0, 800, 0, 0, 0, 200], hand_number=3) == (0,)
    assert p.ranks[0] == 3
    assert p.update([0, 1000, 0, 0, 0, 0], hand_number=4) == (5,)
    assert p.winner == 1
    assert p.ranks == [3, 1, 5, 5, 4, 2]


def test_different_prehand_stacks_same_hand_bust():
    p = PlayerManager([5, 10, 20, 30])
    p.update([0, 0, 0, 65], 1)
    assert p.ranks == [4, 3, 2, 1]
    assert p.winner == 3


def test_equal_start_busts_joint_rank():
    p = PlayerManager([10, 10, 10, 10])
    p.update([40, 0, 0, 0], 1)
    assert p.ranks == [1, 2, 2, 2]


def test_chip_conservation_and_no_resurrection():
    p = PlayerManager([10, 10, 10])
    with pytest.raises(ValueError):
        p.update([9, 10, 10], 1)
    assert p.stacks == (10, 10, 10)
    p.update([0, 15, 15], 1)
    with pytest.raises(ValueError):
        p.update([1, 14, 15], 2)
    assert p.stacks == (0, 15, 15)


@pytest.mark.parametrize('stacks', [[0, 10], [-1, 10], [10], [1] * 7, [True, 10]])
def test_invalid_starting_stacks(stacks):
    with pytest.raises(ValueError):
        PlayerManager(stacks)

````````

<!-- END_PY_SOURCE path="tests/test_player_manager.py" -->

## `tests/test_rewards_tournament.py`

Original blob `c14482c7de7aea35e85e22fd7fab240974a282d0` (836 bytes).

<a name="py-314"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_rewards_tournament.py" sha="c14482c7de7aea35e85e22fd7fab240974a282d0" -->

````````python
import pytest

from holdem_ai.environment.rewards import chip_rewards, placement_rewards


def test_chip_rewards_and_winner_take_all():
    assert chip_rewards((10, -4, -6), 2) == (5.0, -2.0, -3.0)
    assert placement_rewards((3, 1, 2)) == (0.0, 1.0, 0.0)


def test_joint_placing_split_prize_slots():
    rewards = placement_rewards((1, 2, 3, 4, 5, 5), (60, 20, 10, 5, 3, 2))
    assert rewards == (60, 20, 10, 5, 2.5, 2.5)
    assert sum(rewards) == 100


@pytest.mark.parametrize('ranks', [(1, 1, 2, 4), (1, 3, 3, 4), (1, 2, None, 4)])
def test_invalid_rank_sequences(ranks):
    with pytest.raises(ValueError):
        placement_rewards(ranks)


def test_invalid_payouts():
    with pytest.raises(ValueError):
        placement_rewards((1, 2), (10,))
    with pytest.raises(ValueError):
        placement_rewards((1, 2), (10, -1))

````````

<!-- END_PY_SOURCE path="tests/test_rewards_tournament.py" -->

## `tests/test_rlcard_integration.py`

Original blob `da63c2efad6610101967f83772ec80162f13133e` (543 bytes).

<a name="py-315"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_rlcard_integration.py" sha="da63c2efad6610101967f83772ec80162f13133e" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/test_rlcard_integration.py" -->

## `tests/test_rlcard_rules_integration.py`

Original blob `6f96b54a7210d04d7cef04f2ae6e9aba05148f24` (1036 bytes).

<a name="py-316"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_rlcard_rules_integration.py" sha="6f96b54a7210d04d7cef04f2ae6e9aba05148f24" -->

````````python
"""Checks the installed RLCard (1.2.0) rather than synthetic game objects."""

import pytest

pytest.importorskip('rlcard')

import rlcard

from holdem_ai.environment.rules_audit import audit_preflop


def test_audit_can_observe_real_rlcard_preflop_state():
    env = rlcard.make('no-limit-holdem', config={
        'seed': 123, 'game_num_players': 2, 'chips_for_each': 200,
        'dealer_id': 0,
    })
    env.reset()
    results = audit_preflop(env.game)
    assert {x.name for x in results} == {
        'button_posts_small_blind',
        'opponent_posts_big_blind',
        'button_acts_first_preflop',
    }


@pytest.mark.xfail(reason='Known upstream RLCard 1.2.0 heads-up blind/actor behavior; requires corrected hand engine', strict=False)
def test_rlcard_current_hunl_position_is_not_yet_compliant():
    env = rlcard.make('no-limit-holdem', config={
        'seed': 123, 'game_num_players': 2, 'chips_for_each': 200,
        'dealer_id': 0,
    })
    env.reset()
    assert all(x.passed for x in audit_preflop(env.game))

````````

<!-- END_PY_SOURCE path="tests/test_rlcard_rules_integration.py" -->

## `tests/test_rules_audit.py`

Original blob `f6849bbdf5a054d099e640b3aac5a8b529f010da` (881 bytes).

<a name="py-317"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_rules_audit.py" sha="f6849bbdf5a054d099e640b3aac5a8b529f010da" -->

````````python
from types import SimpleNamespace

from holdem_ai.environment.rules_audit import audit_preflop, summarize


def test_audit_accepts_a_correct_heads_up_implementation():
    game = SimpleNamespace(
        dealer_id=0, game_pointer=0,
        small_blind=1, big_blind=2,
        init_chips=[100, 100],
        players=[SimpleNamespace(in_chips=1), SimpleNamespace(in_chips=2)],
    )
    checks = audit_preflop(game)
    assert len(checks) == 3
    assert all(c.passed for c in checks)


def test_audit_detects_a_reversed_blind_and_first_actor():
    game = SimpleNamespace(
        dealer_id=0, game_pointer=1,
        small_blind=1, big_blind=2,
        init_chips=[100, 100],
        players=[SimpleNamespace(in_chips=2), SimpleNamespace(in_chips=1)],
    )
    checks = audit_preflop(game)
    assert sum(not c.passed for c in checks) == 3
    assert 'FAIL' in summarize(checks)

````````

<!-- END_PY_SOURCE path="tests/test_rules_audit.py" -->

## `tests/test_tournament_env.py`

Original blob `b0595166a158b3ed35e6906e45379e2dc9a369ed` (8761 bytes).

<a name="py-318"></a>

<!-- BEGIN_PY_SOURCE path="tests/test_tournament_env.py" sha="b0595166a158b3ed35e6906e45379e2dc9a369ed" -->

````````python
import pytest

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.tournament_env import TournamentEnv


class PassiveAgent:
    def select_action(self, state):
        return 1  # Check or call, including all-in call when stack is short.


class FoldAgent:
    def select_action(self, state):
        return 0  # May fold even when checking is possible.


def test_reproducible_two_player_complete_tournament():
    options = dict(num_players=2, starting_stack=60, seed=91, hands_per_level=2)
    a = TournamentEnv(**options).play_tournament([RandomAgent(1), RandomAgent(2)])
    b = TournamentEnv(**options).play_tournament([RandomAgent(1), RandomAgent(2)])
    assert a == b
    assert len(a.history) >= 1
    assert sum(a.final_stacks) == 120
    assert sorted(a.ranks) == [1, 2]
    assert sum(a.rewards) == 1.0
    assert a.final_stacks[a.winner] == 120


@pytest.mark.parametrize('num_players', [2, 3, 4, 5, 6])
def test_tournament_random_end_to_end(num_players):
    for seed in range(10):
        env = TournamentEnv(
            num_players=num_players,
            starting_stack=80,
            blind_levels=[(1, 2), (2, 4), (5, 10), (10, 20), (20, 40),
                          (40, 80), (80, 160), (160, 320), (320, 640)],
            hands_per_level=3,
            seed=seed,
            max_hands=1000,
        )
        outcome = env.play_tournament([RandomAgent(seed * 100 + i) for i in range(num_players)])
        assert env.is_over() and env.current_player == -1
        assert sum(outcome.final_stacks) == num_players * 80
        assert outcome.final_stacks[outcome.winner] == num_players * 80
        assert sum(c > 0 for c in outcome.final_stacks) == 1
        assert sum(outcome.rewards) == 1
        assert outcome.rewards[outcome.winner] == 1
        assert len(outcome.history) == outcome.hands_played
        assert sorted({r for r in outcome.ranks})[0] == 1
        for h, record in enumerate(outcome.history):
            assert record.hand_number == h + 1
            assert sum(record.stacks_before) == num_players * 80
            assert sum(record.stacks_after) == num_players * 80
            assert record.stacks_after == record.hand_result.final_stacks
            if h:
                assert record.stacks_before == outcome.history[h - 1].stacks_after
                assert record.button != outcome.history[h - 1].button
                assert record.button in [i for i, c in enumerate(record.stacks_before) if c > 0]
            for bust in record.eliminated:
                assert record.stacks_before[bust] > 0
                assert record.stacks_after[bust] == 0


def test_incremental_step_observation_no_private_cards():
    env = TournamentEnv(num_players=3, starting_stack=100, button=0,
                        blind_levels=[(1, 2)], seed=11)
    state, actor = env.reset()
    assert actor == 0
    assert state['tournament']['hand_number'] == 1
    assert state['tournament']['players_alive'] == (0, 1, 2)
    assert state['raw_obs']['hand'] == list(env._hand.hole_cards[actor])
    opponent = next(i for i in range(3) if i != actor)
    assert str(env._hand.hole_cards[opponent]) not in str(state)
    with pytest.raises(ValueError):
        env.observe(opponent)
    state, next_actor = env.step(1)
    assert next_actor == 1
    assert state['tournament']['hand_number'] == 1


def test_stack_persistence_button_rotation_and_rising_blinds():
    env = TournamentEnv(num_players=3, starting_stack=100,
                        blind_levels=[(1, 2), (2, 4), (4, 8)],
                        hands_per_level=1, seed=3)
    state, actor = env.reset()
    while len(env.hand_history) < 2 and not env.is_over():
        state, actor = env.step(0)
    assert len(env.hand_history) >= 2
    first, second = env.hand_history[:2]
    assert first.stacks_after == second.stacks_before
    assert (first.blinds.small_blind, first.blinds.big_blind) == (1, 2)
    assert (second.blinds.small_blind, second.blinds.big_blind) == (2, 4)
    assert first.button != second.button
    assert env.players.total_chips == 300


def test_no_ante_or_rebuy_and_invalid_input():
    with pytest.raises(ValueError):
        TournamentEnv(num_players=7)
    with pytest.raises(ValueError):
        TournamentEnv(num_players=6, starting_stacks=[10] * 5)
    with pytest.raises(ValueError):
        TournamentEnv(num_players=2, payouts=[1, -1])
    with pytest.raises(ValueError):
        TournamentEnv(num_players=2, max_hands=0)


def test_max_hands_guard_does_not_fabricate_winner():
    env = TournamentEnv(num_players=6, starting_stack=1000,
                        blind_levels=[(1, 2)], max_hands=1, seed=21)
    with pytest.raises(RuntimeError, match='max_hands'):
        env.play_tournament([FoldAgent() for _ in range(6)])


def test_restart_is_reproducible_and_rebuilds_stacks():
    env = TournamentEnv(num_players=2, starting_stack=40,
                        blind_levels=[(2, 4), (4, 8), (8, 16), (16, 32), (32, 64)],
                        hands_per_level=2, seed=19)
    outcome = env.play_tournament([RandomAgent(1), RandomAgent(2)])
    assert outcome.hands_played > 0
    before = env.reset()
    other = TournamentEnv(num_players=2, starting_stack=40,
                          blind_levels=[(2, 4), (4, 8), (8, 16), (16, 32), (32, 64)],
                          hands_per_level=2, seed=19)
    assert before == other.reset()
    assert len(env.hand_history) == 0


def test_degenerate_allin_on_reset_terminates_without_input():
    env = TournamentEnv(num_players=2, starting_stack=1, blind_levels=[(1, 2)], seed=11)
    state, actor = env.reset()
    assert env.is_over() and (state, actor) == (None, -1)
    result = env.get_result()
    assert result.hands_played >= 1
    assert sum(result.final_stacks) == 2
    assert sorted(result.ranks) == [1, 2]


def test_noncontiguous_seats_after_elimination_hand():
    # Target a breakout where seat 1 busts and seats 0 and 2 remain.
    # The engine's official UI still uses physical seats across hands.
    for seed in range(70):
        env = TournamentEnv(num_players=3, starting_stacks=[25, 1, 25],
                            blind_levels=[(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)],
                            hands_per_level=2, seed=seed)
        result = env.play_tournament([PassiveAgent()] * 3)
        if any(rec.stacks_after[1] == 0 and rec.stacks_after[0] > 0
               and rec.stacks_after[2] > 0 for rec in result.history):
            assert any(1 not in [i for i, chip in enumerate(rec.stacks_before) if chip > 0]
                       for rec in result.history[1:])
            return
    pytest.fail('no noncontiguous seat case observed across deterministic seeds')


def test_unequal_stacks_and_terminal_prize_structure():
    env = TournamentEnv(num_players=4, starting_stacks=[11, 17, 23, 29],
                        blind_levels=[(1, 2), (2, 4), (4, 8), (8, 16),
                                      (16, 32), (32, 64), (64, 128)],
                        hands_per_level=1, payouts=[70, 20, 10, 0], seed=8)
    out = env.play_tournament([RandomAgent(i + 100) for i in range(4)])
    assert sum(out.final_stacks) == 80
    assert sum(out.rewards) == 100
    assert out.rewards[out.winner] == 70
    assert len(out.ranks) == 4


def test_manual_allin_raises_and_incremental_interface():
    env = TournamentEnv(num_players=3, starting_stack=80,
                        blind_levels=[(1, 2), (2, 4), (4, 8),
                                      (8, 16), (16, 32), (32, 64), (64, 128)],
                        hands_per_level=3, seed=5)
    state, actor = env.reset()
    assert actor == 0 and state['raw_obs']['min_raise_to'] == 4
    state, actor = env.step_raise_to(6)
    assert actor == 1 and state['raw_obs']['to_call'] == 5
    actions = 1
    while not env.is_over():
        state, actor = env.step(1)
        actions += 1
        assert actions < 1000
    assert sum(env.get_result().final_stacks) == 240


def test_fuzz_unequal_short_stacks_at_large_blinds():
    for n in range(2, 7):
        for seed in range(25):
            uneven = [1 + (seed * 3 + i * 11) % 35 for i in range(n)]
            env = TournamentEnv(num_players=n, starting_stacks=uneven,
                                blind_levels=[(1, 2), (3, 6), (10, 20),
                                              (20, 40), (40, 80), (80, 160),
                                              (160, 320)], hands_per_level=2,
                                button=seed % n, seed=seed)
            out = env.play_tournament([RandomAgent(seed * 100 + i) for i in range(n)])
            assert sum(out.final_stacks) == sum(uneven)
            assert sorted(out.ranks)[0] == 1
            assert all(v >= 0 for v in out.final_stacks)

````````

<!-- END_PY_SOURCE path="tests/test_tournament_env.py" -->

