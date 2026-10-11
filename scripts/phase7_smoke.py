"""Run from repo root: uv run --extra gpu python -m scripts.phase7_smoke --hands 100 --device cuda

No policy optimization occurs in Phase 7: only HUNL trajectories -> tensors.
"""
from __future__ import annotations

import argparse
import json
import time

from holdem_agent_phase5.policy import RandomLegalPolicy
from holdem_agent_phase6 import HUNL200Adapter, collect_global_episode
from holdem_agent_phase7 import (
    batch_actors, batch_critics, encode_actor, encode_critic, encode_action,
    masked_categorical,
)


def run(hands: int = 100, device: str = 'cpu', seed: int = 42) -> dict:
    if hands < 1:
        raise ValueError('hands must be positive')
    import torch

    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available: check torch installation and NVIDIA driver')
    policy = {0: RandomLegalPolicy(), 1: RandomLegalPolicy()}
    actor_samples, critic_samples = [], []
    decisions = 0
    start = time.perf_counter()
    for i in range(hands):
        env = HUNL200Adapter(button=i % 2, expose_training_state=True)
        episode = collect_global_episode(env, policy, seed=seed + i, privileged=True)
        if abs(sum(episode.terminal_rewards)) > 1e-9:
            raise AssertionError('HUNL200 zero-sum reward violation')
        for step in episode.decisions:
            assert step.critic_state is not None and step.next_critic_state is not None
            actor = encode_actor(step.observation)
            critic = encode_critic(step.critic_state)
            encode_critic(step.next_critic_state)  # includes terminal states
            encode_action(step.action_index, actor.legal_action_mask)
            actor_samples.append(actor)
            critic_samples.append(critic)
            decisions += 1
    max_length = max(len(s.token_ids) for s in actor_samples)
    # Small synthetic head is only an integration/gradient test, NOT a trained Agent.
    torch.manual_seed(seed)
    embedding = torch.nn.Embedding(73, 64, padding_idx=0).to(device)
    head = torch.nn.Linear(64 + 20, 7).to(device)
    gradient_batches = 0
    for i in range(0, decisions, 64):
        ab = batch_actors(actor_samples[i:i + 64], device=device)
        cb = batch_critics(critic_samples[i:i + 64], device=device)
        representations = embedding(ab.token_ids)
        weights = ab.attention_mask.unsqueeze(-1)
        pooled = (representations * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1)
        logits = head(torch.cat((pooled, ab.features), dim=-1))
        distribution = masked_categorical(logits, ab.legal_action_mask)
        actions = distribution.sample()
        assert bool(ab.legal_action_mask.gather(1, actions.unsqueeze(1)).all().item())
        if gradient_batches == 0:
            (-distribution.log_prob(actions).mean()).backward()
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all())
                       for p in (*embedding.parameters(), *head.parameters()))
        gradient_batches += 1
        assert cb.features.shape[0] == ab.features.shape[0]
    if device.startswith('cuda'):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return {
        'phase': 7, 'hands': hands, 'decisions': decisions,
        'hands_per_second_including_encoding': round(hands / elapsed, 3),
        'elapsed_seconds': round(elapsed, 4),
        'max_actor_tokens': max_length, 'batch_rows': min(64, decisions),
        'actor_feature_dim': 20, 'critic_feature_dim': 24,
        'action_count': 7, 'device': str(device), 'torch': torch.__version__,
        'synthetic_head_backward': True, 'trained_policy': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Phase7 actual HUNL -> tensor smoke test')
    parser.add_argument('--hands', type=int, default=100)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    opts = parser.parse_args()
    print(json.dumps(run(opts.hands, opts.device, opts.seed), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
