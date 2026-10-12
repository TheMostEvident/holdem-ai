"""Phase-8 integration smoke: real Phase-6 rollout -> Phase-7 encoders -> trainable nets.

Run from repository root:
    uv run --extra dev --extra gpu python -m scripts.phase8_smoke --hands 10 --device cuda

THIS IS NOT A VRPO TRAINING RUN. No optimizer updates are performed. The
losses below only verify that finite gradients can flow on real observations.
"""
from __future__ import annotations

import argparse
import json
import random
import time

import torch

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.training.episode_runner import run_self_play
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode
from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.models.transformer import CentralizedQCritic, NetworkConfig, TransformerActor, chosen_q
from holdem_ai.agents.neural_policy import NeuralActorPolicy


def run(*, hands: int = 10, device: str = 'cpu', batch_size: int = 16, seed: int = 42) -> dict:
    if type(hands) is not int or hands < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('hands and batch_size must be positive integers')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. Verify GPU, driver and torch installation')
    dev = torch.device(device)
    torch.manual_seed(seed)
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(dev)

    policy = {0: RandomLegalPolicy(), 1: RandomLegalPolicy()}
    actor_samples = []
    critic_samples = []
    actions = []
    seats = []
    final_rewards = []
    begin = time.perf_counter()
    max_tokens = 0
    for i in range(hands):
        env = HUNL200Adapter(button=i % 2, expose_training_state=True)
        episode = collect_global_episode(env, policy, seed=seed + i, privileged=True)
        if abs(sum(episode.terminal_rewards)) > 1e-9:
            raise AssertionError('HUNL200 zero-sum reward violated')
        for dec in episode.decisions:
            assert dec.critic_state is not None
            a = encode_actor(dec.observation)
            c = encode_critic(dec.critic_state)
            actor_samples.append(a)
            critic_samples.append(c)
            actions.append(dec.action_index)
            seats.append(dec.player_id)
            final_rewards.append(episode.terminal_rewards[dec.player_id])
            max_tokens = max(max_tokens, len(a.token_ids))
    if not actions:
        raise RuntimeError('no decisions collected')
    cfg = NetworkConfig()  # 2-layer, 64-dim, 2 heads, max_tokens=256
    actor = TransformerActor(cfg).to(dev)
    critic = CentralizedQCritic(cfg).to(dev)
    actor.train()
    critic.train()
    n = min(batch_size, len(actions))
    ab = batch_actors(actor_samples[:n], device=dev)
    cb = batch_critics(critic_samples[:n], device=dev)
    chosen = torch.tensor(actions[:n], dtype=torch.long, device=dev)
    acting_seats = torch.tensor(seats[:n], dtype=torch.long, device=dev)
    reward_targets = torch.tensor(final_rewards[:n], dtype=torch.float32, device=dev)
    if not bool(ab.legal_action_mask.gather(1, chosen[:, None]).all()):
        raise AssertionError('illegal sampled action')

    # Demonstration gradients only; these are NOT PPO or VRPO objectives.
    log_probs = actor.distribution(ab).log_prob(chosen)
    actor_demo_loss = -log_probs.mean()
    q = critic(cb)
    critic_demo_loss = ((chosen_q(q, acting_seats, chosen) - reward_targets) ** 2).mean()
    actor_demo_loss.backward()
    critic_demo_loss.backward()
    for name, model in [('actor', actor), ('critic', critic)]:
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        if not grads or any(g is None or not bool(torch.isfinite(g).all()) for g in grads):
            raise AssertionError(f'{name}: missing or nonfinite gradient')
    if not bool(torch.allclose(q.sum(dim=1), torch.zeros_like(q[:, 0]), atol=1e-7)):
        raise AssertionError('zero-sum critic output violated')

    # Exercise the complete inference contract with an UNTRAINED policy.
    neural = NeuralActorPolicy(actor, device=str(dev), greedy=False)
    probe = run_self_play(
        HUNL200Adapter(button=0),
        {0: neural, 1: RandomLegalPolicy()},
        seed=seed + hands,
        max_decisions=200,
    )
    if abs(sum(probe.rewards)) > 1e-9:
        raise AssertionError('neural-policy HUNL zero-sum reward violated')
    if dev.type == 'cuda':
        torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - begin
    return {
        'phase': 8,
        'device': str(dev),
        'torch': torch.__version__,
        'hands_collected': hands,
        'decisions_collected': len(actions),
        'max_actor_tokens': max_tokens,
        'gradient_batch': n,
        'actor_parameters': sum(p.numel() for p in actor.parameters()),
        'critic_parameters': sum(p.numel() for p in critic.parameters()),
        'actor_logits_shape': list(actor(ab).shape),
        'critic_q_shape': list(q.shape),
        'actor_demo_loss': round(actor_demo_loss.item(), 5),
        'critic_demo_loss': round(critic_demo_loss.item(), 5),
        'backward_pass': True,
        'neural_policy_played_hand': True,
        'neural_policy_decisions': probe.num_decisions,
        'elapsed_seconds': round(elapsed, 4),
        'cuda_peak_allocated_mb': round(torch.cuda.max_memory_allocated(dev) / 2 ** 20, 2)
        if dev.type == 'cuda' else None,
        'trained_policy': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='VRPO-Lite Phase-8 trainable network integration smoke')
    parser.add_argument('--hands', type=int, default=10)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(hands=args.hands, device=args.device,
                         batch_size=args.batch_size, seed=args.seed),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
