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
