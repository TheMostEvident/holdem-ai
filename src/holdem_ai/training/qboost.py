"""Phase 10: doubly-corrected Q-control-variate PPO research baseline.

At theta_old, E[score*(G - V - beta*(Q(a)-V))]
+ beta * sum_a grad(pi(a))*Q(a) equals the Monte Carlo policy gradient.
PPO clipping subsequently introduces ordinary on-policy surrogate bias.
This is a VRPO-inspired *Lite* experiment, NOT a faithful paper reimplementation.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.models.transformer import CentralizedQCritic, TransformerActor, chosen_q
from holdem_ai.training.ppo import FrozenPPOData, PPOConfig, TrajectoryRows, clipped_surrogate


@dataclass(frozen=True)
class QBoostConfig:
    beta: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.beta, bool) or not isinstance(self.beta, (float, int)) or not math.isfinite(self.beta) or not 0.0 <= self.beta <= 1.0:
            raise ValueError('beta must be a finite number in [0,1]')


@dataclass(frozen=True)
class FrozenQBoostData:
    rows: TrajectoryRows
    old_log_probs: torch.Tensor    # CPU [N]
    old_probs: torch.Tensor        # CPU [N,7]
    q_actions: torch.Tensor        # CPU [N,7], acting seat, no gradients
    baselines: torch.Tensor        # CPU [N]
    raw_advantages: torch.Tensor   # CPU [N] = MC - V
    center: float                 # same MC-advantage normalizer for both terms
    scale: float


def freeze_qboost_data(actor: TransformerActor, critic: CentralizedQCritic,
                       rows: TrajectoryRows, *, device: str = 'cpu',
                       chunk_size: int = 64) -> FrozenQBoostData:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError('chunk_size must be positive integer')
    actor.eval()
    critic.eval()
    probs, qs, logs = [], [], []
    with torch.no_grad():
        for start in range(0, len(rows.actions), chunk_size):
            stop = min(start + chunk_size, len(rows.actions))
            ab = batch_actors(rows.actors[start:stop], device=device)
            cb = batch_critics(rows.critics[start:stop], device=device)
            d = actor.distribution(ab)
            seats = torch.tensor(rows.acting_seats[start:stop], device=device, dtype=torch.long)
            acts = torch.tensor(rows.actions[start:stop], device=device, dtype=torch.long)
            q = critic(cb)
            q_seat = q[torch.arange(stop - start, device=device), seats]
            if not bool(torch.isfinite(q_seat).all().item()):
                raise FloatingPointError('nonfinite old action values')
            probs.append(d.probs.detach().cpu())
            logs.append(d.log_prob(acts).detach().cpu())
            qs.append(q_seat.detach().cpu())
    old_p, old_log, q_all = torch.cat(probs), torch.cat(logs), torch.cat(qs)
    baselines = (old_p * q_all).sum(-1)
    raw = torch.tensor(rows.returns_bb, dtype=torch.float32) - baselines
    center = float(raw.mean().item())
    scale = max(float(raw.std(unbiased=False).item()), 1e-6)
    if not all(bool(torch.isfinite(z).all().item()) for z in (old_p, old_log, q_all, baselines, raw)):
        raise FloatingPointError('nonfinite frozen QBoost information')
    return FrozenQBoostData(rows, old_log, old_p, q_all, baselines, raw, center, scale)


def qboost_surrogate(new_log: torch.Tensor, new_probs: torch.Tensor,
                     old_log: torch.Tensor, old_probs: torch.Tensor,
                     q_values: torch.Tensor, actions: torch.Tensor,
                     returns: torch.Tensor, baseline: torch.Tensor,
                     *, beta: float, center: float, scale: float,
                     clip_epsilon: float) -> torch.Tensor:
    """Per-row actor objective to MAXIMIZE, sharing one centering/scaling.

    MC normalized A = (G-V-center)/scale.
    Residual normalized A = (G-V-beta*(Q(a)-V)-center)/scale.
    Analytic = beta * sum_a (pi_new(a)-pi_old(a))*Q(a)/scale.
    beta=0 agrees exactly with Phase9 clipped PPO on identically frozen data.
    """
    n = new_log.numel()
    if not isinstance(beta, (int, float)) or isinstance(beta, bool) or not 0 <= beta <= 1:
        raise ValueError('beta must be in [0,1]')
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(center):
        raise ValueError('invalid advantage normalization')
    if not 0 < clip_epsilon < 1:
        raise ValueError('clip epsilon must be in (0,1)')
    if not (new_log.shape == old_log.shape == actions.shape == returns.shape == baseline.shape == (n,)):
        raise ValueError('scalar row tensors must align')
    if (new_probs.shape != (n, 7) or old_probs.shape != (n, 7) or q_values.shape != (n, 7)):
        raise ValueError('policy and Q must be [N,7]')
    if actions.dtype != torch.long or not bool(((actions >= 0) & (actions < 7)).all().item()):
        raise ValueError('invalid actions')
    if not all(bool(torch.isfinite(t).all().item()) for t in
               (new_log, old_log, new_probs, old_probs, q_values, returns, baseline)):
        raise FloatingPointError('nonfinite surrogate inputs')
    observed_q = q_values.gather(1, actions[:, None]).squeeze(1)
    residual = (returns - baseline - beta * (observed_q - baseline) - center) / scale
    mc_clip = clipped_surrogate(new_log, old_log, residual, clip_epsilon)
    analytic = beta * ((new_probs - old_probs) * q_values).sum(-1) / scale
    return mc_clip + analytic


def qboost_diagnostics(frozen: FrozenQBoostData, beta: float = 1.0) -> dict[str, float]:
    """Variance of empirical logit-gradient proxies, NOT parameter-gradient variance.

    They compare within the same observed sample, using frozen old policy and Q.
    Improvement (ratio <1) is not guaranteed; Q error can make it worse.
    """
    a = torch.tensor(frozen.rows.actions, dtype=torch.long)
    ret = torch.tensor(frozen.rows.returns_bb, dtype=torch.float32)
    qa = frozen.q_actions.gather(1, a[:, None]).squeeze(1)
    score = F.one_hot(a, num_classes=7).float() - frozen.old_probs
    old = frozen.raw_advantages - frozen.center
    residual = old - beta * (qa - frozen.baselines)
    p = frozen.old_probs
    correction = beta * p * (frozen.q_actions - frozen.baselines[:, None])
    g_old = score * (old / frozen.scale)[:, None]
    g_new = score * (residual / frozen.scale)[:, None] + correction / frozen.scale
    def variance(x: torch.Tensor) -> float:
        return float(x.var(unbiased=False).item())
    def logit_variance(x: torch.Tensor) -> float:
        return float(x.var(dim=0, unbiased=False).sum().item())
    old_v, new_v = logit_variance(g_old), logit_variance(g_new)
    return {
        'mc_advantage_variance': round(variance(frozen.raw_advantages), 6),
        'boosted_residual_variance': round(variance(residual), 6),
        'logit_gradient_proxy_variance_ppo': round(old_v, 6),
        'logit_gradient_proxy_variance_qboost': round(new_v, 6),
        'logit_variance_ratio': round(new_v / max(old_v, 1e-12), 6),
    }


def optimize_qboost(actor: TransformerActor, critic: CentralizedQCritic,
                    actor_optimizer: torch.optim.Optimizer,
                    critic_optimizer: torch.optim.Optimizer,
                    frozen: FrozenQBoostData, config: PPOConfig,
                    qconfig: QBoostConfig, *, device: str = 'cpu',
                    shuffle_seed: int = 0) -> dict[str, float | int]:
    actor.train()
    critic.train()
    n = len(frozen.rows.actions)
    rng = random.Random(shuffle_seed)
    keys = ('actor_loss', 'critic_loss', 'entropy', 'approx_kl', 'clip_fraction',
            'grad_norm_actor', 'grad_norm_critic', 'analytical_boost_mean')
    totals = dict.fromkeys(keys, 0.0)
    count = 0
    for _ in range(config.epochs):
        order = list(range(n))
        rng.shuffle(order)
        for offset in range(0, n, config.minibatch_size):
            ind = order[offset:offset + config.minibatch_size]
            ab = batch_actors([frozen.rows.actors[i] for i in ind], device=device)
            cb = batch_critics([frozen.rows.critics[i] for i in ind], device=device)
            acts = torch.tensor([frozen.rows.actions[i] for i in ind], dtype=torch.long, device=device)
            seats = torch.tensor([frozen.rows.acting_seats[i] for i in ind], dtype=torch.long, device=device)
            rewards = torch.tensor([frozen.rows.returns_bb[i] for i in ind], dtype=torch.float32, device=device)
            oldlog = frozen.old_log_probs[ind].to(device)
            oldp = frozen.old_probs[ind].to(device)
            q_old = frozen.q_actions[ind].to(device)
            v_old = frozen.baselines[ind].to(device)
            distribution = actor.distribution(ab)
            log_new = distribution.log_prob(acts)
            surrogate = qboost_surrogate(
                log_new, distribution.probs, oldlog, oldp, q_old, acts,
                rewards, v_old, beta=qconfig.beta, center=frozen.center,
                scale=frozen.scale, clip_epsilon=config.clip_epsilon)
            a_loss = -surrogate.mean()
            entropy = distribution.entropy().mean()
            policy_loss = a_loss - config.entropy_coef * entropy
            q_prediction = chosen_q(critic(cb), seats, acts)
            c_loss = F.smooth_l1_loss(q_prediction, rewards)
            actor_optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            (config.value_coef * c_loss).backward()
            an = nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip)
            cn = nn.utils.clip_grad_norm_(critic.parameters(), config.grad_clip)
            if not all(math.isfinite(float(x)) for x in (policy_loss.item(), c_loss.item(), an.item(), cn.item())):
                raise FloatingPointError('nonfinite Phase10 loss/grad')
            actor_optimizer.step()
            critic_optimizer.step()
            with torch.no_grad():
                ratio = (log_new - oldlog).exp()
                log_ratio = log_new - oldlog
                residual = (rewards - v_old - qconfig.beta *
                            (q_old.gather(1, acts[:, None]).squeeze(1) - v_old) - frozen.center) / frozen.scale
                clip_frac = ((ratio - 1).abs() > config.clip_epsilon).float().mean()
                boost_mean = qconfig.beta * ((distribution.probs - oldp) * q_old).sum(-1).mean() / frozen.scale
                stats = {'actor_loss': a_loss.item(), 'critic_loss': c_loss.item(),
                         'entropy': entropy.item(), 'approx_kl': (ratio - 1 - log_ratio).mean().item(),
                         'clip_fraction': clip_frac.item(), 'grad_norm_actor': an.item(),
                         'grad_norm_critic': cn.item(), 'analytical_boost_mean': boost_mean.item()}
            for k in keys:
                totals[k] += stats[k]
            count += 1
    return {**{k: round(v / count, 6) for k, v in totals.items()},
            **qboost_diagnostics(frozen, beta=qconfig.beta),
            'minibatches': count, 'decisions': n,
            'terminal_abs_mean_bb': round(sum(abs(x) for x in frozen.rows.returns_bb)/n, 5)}
