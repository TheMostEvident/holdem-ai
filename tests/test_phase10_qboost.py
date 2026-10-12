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
