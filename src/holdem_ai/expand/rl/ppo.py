"""Clipped PPO on frozen collected 2-6 player trajectories."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch.distributions import Categorical

from .returns import PreparedSample
from .model import ActorCritic
from holdem_ai.expand.information.schema import ACTION_NAMES


@dataclass(frozen=True)
class PPOConfig:
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 0.0003
    epochs: int = 2
    minibatch_size: int = 64
    max_grad_norm: float = 1.0
    target_kl: float = 0.20

    def __post_init__(self):
        if not (0 < self.clip_epsilon < 1 and self.value_coef >= 0 and self.entropy_coef >= 0
                and self.learning_rate > 0 and self.epochs > 0 and self.minibatch_size > 0
                and self.max_grad_norm > 0 and self.target_kl > 0):
            raise ValueError('invalid PPO hyperparameters')


def optimize_ppo(model: ActorCritic, optimizer: torch.optim.Optimizer,
                 samples: tuple[PreparedSample, ...], config: PPOConfig) -> dict:
    if not samples:
        raise ValueError('cannot optimize empty batch')
    model.train()
    device = next(model.parameters()).device
    advantages = torch.tensor([s.advantage for s in samples], dtype=torch.float32,device=device)
    if len(samples) > 1:
        advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
    returns = torch.tensor([s.return_bb for s in samples], dtype=torch.float32,device=device)
    old_lp = torch.tensor([s.transition.action_logprob for s in samples],dtype=torch.float32,device=device)
    actions = torch.tensor([ACTION_NAMES.index(s.transition.action) for s in samples],dtype=torch.int64,device=device)
    stats=[]
    early_stop=False
    for epoch in range(config.epochs):
        for indices in torch.randperm(len(samples),device='cpu').split(config.minibatch_size):
            indices = indices.to(device)
            obs = [samples[int(i)].transition.encoded for i in indices.cpu().tolist()]
            logits, v = model(obs)
            dist=Categorical(logits=logits)
            new_lp=dist.log_prob(actions[indices])
            log_ratio=new_lp-old_lp[indices]
            ratio=log_ratio.exp()
            adv=advantages[indices]
            loss_pi=-torch.minimum(ratio*adv, ratio.clamp(1-config.clip_epsilon,1+config.clip_epsilon)*adv).mean()
            loss_v=torch.square(v-returns[indices]).mean()
            entropy=dist.entropy().mean()
            loss=loss_pi+config.value_coef*loss_v-config.entropy_coef*entropy
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite PPO loss: checkpoint should not be saved')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError('nonfinite PPO gradient: optimizer step aborted')
            optimizer.step()
            # Not an unbiased KL estimator, but a common PPO proxy.
            approx_kl=float(((ratio-1)-log_ratio).mean().detach().item())
            stats.append((float(loss_pi.detach()), float(loss_v.detach()),float(entropy.detach()),approx_kl))
            if approx_kl > config.target_kl:
                early_stop=True
                break
        if early_stop:
            break
    def avg(i): return sum(x[i] for x in stats)/len(stats)
    return {'actor_loss':round(avg(0),6), 'value_loss':round(avg(1),6),
            'entropy':round(avg(2),6),'approx_kl':round(avg(3),6),
            'optimizer_steps':len(stats), 'early_kl_stop':early_stop,
            'samples':len(samples)}
