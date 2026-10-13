# Python source code: volume 5 of 28

Exact full .py sources from baseline commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. Integral appendix of [final technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md).

## scripts/phase6_benchmark.py

Original Git blob SHA: `4a633fdc219c78a1ad10ba667c4d81d06e499a0a` (3422 bytes).

<a id="py-049"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase6_benchmark.py" sha="4a633fdc219c78a1ad10ba667c4d81d06e499a0a" -->

````````python
"""CPU-only Phase 6 HUNL200 benchmark; no PyTorch/CUDA required.

Run at repository root:
    uv run python scripts/phase6_benchmark.py --hands 200 --workers 1,2,4
The default 100 warmup/test hands per worker mode is not a five-hour train.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import time

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.training.episode_runner import run_self_play
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter


def _run_one(hand_index: int, seed: int, max_decisions: int) -> tuple[float, int]:
    env = HUNL200Adapter(button=hand_index % 2)
    ep = run_self_play(
        env, {0: RandomLegalPolicy(), 1: RandomLegalPolicy()},
        seed=seed + hand_index, max_decisions=max_decisions,
    )
    if abs(sum(ep.rewards)) > 1e-9 or not ep.transitions:
        raise AssertionError("bad zero-sum rewards or missing decisions")
    if any(t.action_index not in t.observation.legal_actions for t in ep.transitions):
        raise AssertionError("illegal action in trajectory")
    return ep.rewards[0], ep.num_decisions


def run_benchmark(n_hands: int, n_workers: int, *, seed: int = 42,
                  max_decisions: int = 200) -> dict[str, float | int]:
    if n_hands < 1 or n_workers < 1 or max_decisions < 1:
        raise ValueError("hands, workers and max_decisions must be positive")
    begin = time.perf_counter()
    args = ((i, seed, max_decisions) for i in range(n_hands))
    if n_workers == 1:
        samples = [_run_one(*arg) for arg in args]
    else:
        # spawn-compatible Windows worker function; no GPU context created.
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_run_one, *arg) for arg in args]
            samples = [f.result() for f in futures]
    elapsed = time.perf_counter() - begin
    return {
        "hands": n_hands, "workers": n_workers, "elapsed_seconds": round(elapsed, 4),
        "hands_per_second": round(n_hands / elapsed, 3),
        "decisions": sum(x[1] for x in samples),
        "decisions_per_second": round(sum(x[1] for x in samples) / elapsed, 3),
        "seat0_mean_bb100": round(100.0 * sum(x[0] for x in samples) / n_hands, 4),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="HUNL200 Phase 6 CPU rollout speed test")
    p.add_argument("--hands", type=int, default=100)
    p.add_argument("--workers", default="1,2,4", help="comma separated, e.g. 1,2,4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-decisions", type=int, default=200)
    p.add_argument("--json-out", default=None, help="optional JSON report filepath")
    args = p.parse_args()
    modes = [int(x) for x in args.workers.split(",")]
    # Warm up imports + one hand; excluded from timed benchmark.
    _run_one(-2, args.seed, args.max_decisions)
    reports = [run_benchmark(args.hands, w, seed=args.seed, max_decisions=args.max_decisions) for w in modes]
    for report in reports:
        print(json.dumps(report, ensure_ascii=False))
    if args.json_out:
        from pathlib import Path
        output = Path(args.json_out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {output}")


if __name__ == "__main__":
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase6_benchmark.py" -->

## scripts/phase6_hardware_probe.py

Original Git blob SHA: `d8ba8655d715d9799e1e22d390ecff9fb2ae95d7` (1664 bytes).

<a id="py-050"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase6_hardware_probe.py" sha="d8ba8655d715d9799e1e22d390ecff9fb2ae95d7" -->

````````python
"""Safe, tiny PyTorch CUDA compute probe for Pascal-class GPUs (optional)."""
from __future__ import annotations

import json
import os
import platform
import time


def main() -> None:
    report = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "logical_cpus": os.cpu_count(),
    }
    try:
        import torch
    except ImportError:
        report["torch"] = "not installed (Phase 6 engine benchmark still works)"
        print(json.dumps(report, indent=2))
        return
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        report["gpu"] = torch.cuda.get_device_name(device)
        report["compute_capability"] = torch.cuda.get_device_capability(device)
        report["vram_gb"] = round(torch.cuda.get_device_properties(device).total_memory / (1024 ** 3), 2)
        try:
            torch.manual_seed(1)
            a = torch.randn((128, 128), device=device, requires_grad=True)
            start = time.perf_counter()
            loss = (a @ a).square().mean()
            loss.backward()
            torch.cuda.synchronize(device)
            report["forward_backward_pass"] = bool(torch.isfinite(loss).item() and torch.isfinite(a.grad).all().item())
            report["tiny_step_seconds"] = round(time.perf_counter() - start, 5)
        except Exception as exc:
            report["forward_backward_pass"] = False
            report["error"] = f"{type(exc).__name__}: {exc}"
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase6_hardware_probe.py" -->

## scripts/phase7_smoke.py

Original Git blob SHA: `ca4efa3f92b1d8e8994b9007561576728f56198a` (4254 bytes).

<a id="py-051"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase7_smoke.py" sha="ca4efa3f92b1d8e8994b9007561576728f56198a" -->

````````python
"""Run from repo root: uv run --extra gpu python -m scripts.phase7_smoke --hands 100 --device cuda

No policy optimization occurs in Phase 7: only HUNL trajectories -> tensors.
"""
from __future__ import annotations

import argparse
import json
import time

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode
from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.encoding.actions import encode_action, masked_categorical


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

````````

<!-- END_PY_SOURCE path="scripts/phase7_smoke.py" -->

## scripts/phase8_smoke.py

Original Git blob SHA: `10814a36ab96ef0295124bdd7a2dd4bad46b6bf0` (6235 bytes).

<a id="py-052"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase8_smoke.py" sha="10814a36ab96ef0295124bdd7a2dd4bad46b6bf0" -->

````````python
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

````````

<!-- END_PY_SOURCE path="scripts/phase8_smoke.py" -->

## scripts/phase9_train.py

Original Git blob SHA: `70ae3ada85f7700f92ca680093c0ac1f46c08daa` (4240 bytes).

<a id="py-053"></a>

<!-- BEGIN_PY_SOURCE path="scripts/phase9_train.py" sha="70ae3ada85f7700f92ca680093c0ac1f46c08daa" -->

````````python
"""CLI: Phase-9 actual PPO updates, checkpoint/resume, random baseline evaluation.

From repo root:
  uv run --extra gpu python -m scripts.phase9_train --device cuda --updates 2 --hands-per-update 16 --eval-hands 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig, evaluate_vs_random
from holdem_ai.training.ppo_session import create_session, load_checkpoint, train_updates


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Phase 9 PPO baseline on HUNL200; NOT VRPO')
    p.add_argument('--device', default='cpu', choices=('cpu', 'cuda'))
    p.add_argument('--updates', type=int, default=2, help='additional optimizer updates')
    p.add_argument('--hands-per-update', type=int, default=16)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--minibatch-size', type=int, default=32)
    p.add_argument('--eval-hands', type=int, default=20, help='even hand count >=2; same seed pre/post')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint', default='checkpoints/phase9_ppo.pt')
    p.add_argument('--resume', default=None, help='checkpoint path; ignores fresh-run training hyperparameters')
    p.add_argument('--small-network', action='store_true', help='1-layer tiny test network instead of GTX1070 default')
    p.add_argument('--skip-eval', action='store_true')
    return p


def run(args: argparse.Namespace) -> dict:
    begin = time.perf_counter()
    if args.resume:
        session = load_checkpoint(args.resume, device=args.device)
    else:
        net = (NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1,
                             ff_dim=64, hidden_dim=48, max_tokens=256)
               if args.small_network else NetworkConfig())
        cfg = PPOConfig(hands_per_update=args.hands_per_update,
                        epochs=args.epochs, minibatch_size=args.minibatch_size,
                        seed=args.seed)
        session = create_session(device=args.device, seed=args.seed,
                                 network_config=net, ppo_config=cfg)
    eval_seed = 900000 + session.ppo_config.seed * 10
    before = None if args.skip_eval else evaluate_vs_random(
        session.actor, hands=args.eval_hands, device=session.device,
        seed=eval_seed, big_blind=session.ppo_config.big_blind)
    if before is not None:
        print(json.dumps({'evaluation': 'before', **before}, ensure_ascii=False), flush=True)
    checkpoint_path = (args.resume if args.resume and args.checkpoint == 'checkpoints/phase9_ppo.pt'
                       else args.checkpoint)
    update_reports = train_updates(session, updates=args.updates,
                                   checkpoint=checkpoint_path)
    for record in update_reports:
        print(json.dumps({'kind': 'update', **record}, ensure_ascii=False), flush=True)
    after = None if args.skip_eval else evaluate_vs_random(
        session.actor, hands=args.eval_hands, device=session.device,
        seed=eval_seed, big_blind=session.ppo_config.big_blind)
    if after is not None:
        print(json.dumps({'evaluation': 'after', **after}, ensure_ascii=False), flush=True)
    return {
        'phase': 9, 'algorithm': 'PPO_baseline_MC_returns',
        'trained_policy': True, 'device': session.device, 'torch': torch.__version__,
        'completed_updates': session.completed_updates,
        'checkpoint': str(Path(checkpoint_path)),
        'actor_parameters': sum(p.numel() for p in session.actor.parameters()),
        'critic_parameters': sum(p.numel() for p in session.critic.parameters()),
        'last_update': update_reports[-1],
        'evaluation_before': before, 'evaluation_after': after,
        'elapsed_seconds': round(time.perf_counter() - begin, 3),
        'cuda_peak_allocated_mb': round(torch.cuda.max_memory_allocated() / 2**20, 2)
        if session.device == 'cuda' else None,
    }


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(run(args), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/phase9_train.py" -->

## scripts/run_tournament.py

Original Git blob SHA: `a62f710111e25e43ab7f3f0a94587bf1ca132222` (1534 bytes).

<a id="py-054"></a>

<!-- BEGIN_PY_SOURCE path="scripts/run_tournament.py" sha="a62f710111e25e43ab7f3f0a94587bf1ca132222" -->

````````python
"""Run one 2-6 seat tournament (random-agent baseline) and show eliminations."""
from __future__ import annotations

import argparse

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.tournament_env import TournamentEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', type=int, default=6)
    parser.add_argument('--stack', type=int, default=1000)
    parser.add_argument('--hands-per-level', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    env = TournamentEnv(num_players=args.players, starting_stack=args.stack,
                        hands_per_level=args.hands_per_level, seed=args.seed)
    agents = [RandomAgent(args.seed + i) for i in range(args.players)]
    result = env.play_tournament(agents)
    print(f'Tournament complete: champion=P{result.winner}, hands={result.hands_played}')
    for record in result.history:
        if args.verbose or record.eliminated:
            print(f'Hand {record.hand_number:3d} | BTN=P{record.button} '
                  f'blinds={record.blinds.small_blind}/{record.blinds.big_blind} '
                  f'| stacks={record.stacks_after} '
                  f'| busted={list(record.eliminated)}')
    print(f'Final ranks (seat order): {result.ranks}')
    print(f'Final stacks: {result.final_stacks}')
    print(f'Win rewards: {result.rewards}')


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/run_tournament.py" -->

## scripts/tournament_env1_smoke.py

Original Git blob SHA: `2db589c928eb440d5ccf8bc0a046f4b4895ba027` (1891 bytes).

<a id="py-059"></a>

<!-- BEGIN_PY_SOURCE path="scripts/tournament_env1_smoke.py" sha="2db589c928eb440d5ccf8bc0a046f4b4895ba027" -->

````````python
"""API smoke: uv run --extra dev --extra gui python -m scripts.tournament_env1_smoke"""
from __future__ import annotations

from fastapi.testclient import TestClient

from backend.main import create_app


def main() -> None:
    with TestClient(create_app()) as client:
        response = client.post("/api/v1/tournament/sessions", json={
            "seed": 19, "starting_stack": 100,
            "blind_levels": [[5, 10], [10, 20], [20, 40], [40, 80], [80, 160]],
            "hands_per_level": 2,
        })
        response.raise_for_status()
        payload = response.json()
        sid = payload["session_id"]
        total = payload["state"]["total_chips"]
        actions = 0
        while not payload["state"]["tournament_over"]:
            state = payload["state"]
            assert state["your_turn"] and state["to_act"] == 0
            assert sum(s["stack"] for s in state["seats"]) + state["pot"] == total
            assert len(state["seats"]) == 6
            response = client.post(f"/api/v1/tournament/sessions/{sid}/actions", json={
                "expected_revision": state["revision"], "action_index": 1,
            })
            response.raise_for_status()
            payload = response.json()
            actions += 1
            if actions > 1000:
                raise RuntimeError("smoke exceeded 1000 human decisions")
        final = payload["state"]
        assert final["winner_seat"] in range(6)
        assert sum(s["stack"] for s in final["seats"]) == total
        history = client.get(f"/api/v1/tournament/sessions/{sid}/history").json()
        assert history["total"] == final["hands_completed"]
        print({"phase": "Tournament ENV 1.1", "human_actions": actions,
               "hands": final["hands_completed"], "winner_seat": final["winner_seat"],
               "chips": total, "result": "passed"})


if __name__ == "__main__":
    main()

````````

<!-- END_PY_SOURCE path="scripts/tournament_env1_smoke.py" -->

## scripts/tournament_env13_smoke.py

Original Git blob SHA: `0533fa41e8d9080b170dd65184e03604875b75fa` (1499 bytes).

<a id="py-056"></a>

<!-- BEGIN_PY_SOURCE path="scripts/tournament_env13_smoke.py" sha="0533fa41e8d9080b170dd65184e03604875b75fa" -->

````````python
"""Reproducible six-seat Tournament ENV 1.3 bot lineup smoke, no HTTP server."""
from __future__ import annotations

import json
from holdem_ai.tournament_service.improved_bot import PROFILES
from holdem_ai.tournament_service.session import TournamentGameSession


def main() -> None:
    game = TournamentGameSession(
        seed=42, starting_stack=100, hands_per_level=2,
        blind_levels=[(5, 10), (10, 20), (20, 40), (40, 80), (80, 160)],
        lineup="improved_4",
    )
    actions = 0
    while True:
        view = game.view()
        assert [x['role'] for x in view['seats']] == [
            'human', 'tournament_agent_placeholder', *(['improved_bot'] * 4),
        ]
        assert [x['policy_name'] for x in view['seats'][2:]] == list(PROFILES)
        assert sum(x['stack'] for x in view['seats']) + view['pot'] == 600
        if view['tournament_over']:
            break
        assert view['your_turn']
        game.play_action(1, expected_revision=view['revision'])
        actions += 1
        if actions > 1500:
            raise AssertionError('tournament exceeded smoke action budget')
    print(json.dumps(dict(phase='Tournament ENV 1.3', result='passed',
                          human_actions=actions, hands=view['hands_completed'],
                          winner_seat=view['winner_seat'],
                          lineup=view['lineup'], total_chips=view['total_chips']),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/tournament_env13_smoke.py" -->

## scripts/tournament_env131_diagnose.py

Original Git blob SHA: `91f1f830fd0974bb595626d8ce9e3f8779ab2d0f` (1557 bytes).

<a id="py-055"></a>

<!-- BEGIN_PY_SOURCE path="scripts/tournament_env131_diagnose.py" sha="91f1f830fd0974bb595626d8ce9e3f8779ab2d0f" -->

````````python
"""Show conditional preflop heuristic raise mix (NOT actual in-game raise frequency).

Execute from project root:
  uv run --extra gui python -m scripts.tournament_env131_diagnose
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

from holdem_ai.tournament_service.improved_bot import ImprovedTournamentBot, PROFILES


def main() -> None:
    deck = [f'{s}{r}' for s in 'shdc' for r in '23456789TJQKA']
    print('Preflop heuristic check: 1,326 possible 2-card combos')
    print('Assumption: 6 active seats, call 10, pot 45, all raises legal, stack 20,000')
    print('These rates are *conditional policy choices*, NOT measured game win/raise rates.')
    for profile in PROFILES:
        bot = ImprovedTournamentBot(profile, seed=2026)
        counts: Counter[int] = Counter()
        for hole in combinations(deck, 2):
            obs = dict(legal_actions={a: None for a in range(5)}, raw_obs={
                'hand': list(hole), 'public_cards': [], 'player_id': 2,
                'to_call': 10, 'pot': 45, 'stacks': [20000]*6, 'status':['active']*6,
            })
            counts[bot.select_action(obs)] += 1
        pct = 100 * (counts[2]+counts[3])/1326
        print(f'{profile:<21} Fold={counts[0]:4} Check/Call={counts[1]:4} '
              f'HalfPot={counts[2]:4} Pot={counts[3]:4} AllIn={counts[4]:4}  '
              f'RaiseShare={pct:5.1f}%')
    print('Seat 2 remains an UNTRAINED Tournament Agent placeholder using a heuristic fallback.')


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/tournament_env131_diagnose.py" -->

## scripts/tournament_env14_smoke.py

Original Git blob SHA: `8a98b6230d1144ebdbe108b80fba5fc71c76b87f` (1760 bytes).

<a id="py-057"></a>

<!-- BEGIN_PY_SOURCE path="scripts/tournament_env14_smoke.py" sha="8a98b6230d1144ebdbe108b80fba5fc71c76b87f" -->

````````python
"""Real trusted Expand Phase3 checkpoint smoke for Tournament ENV 1.4.

Requires HOLDEM_AI_TOURNAMENT_CHECKPOINT environment variable pointing to
an EXISTING compatible 6-seat Expand E03 checkpoint. Never mutates weights.
"""
from __future__ import annotations

import json

from holdem_ai.tournament_service.session import TournamentGameSession
from holdem_ai.tournament_service.trained_actor import status


def main() -> None:
    if not status()['file_found']:
        raise SystemExit('Set HOLDEM_AI_TOURNAMENT_CHECKPOINT to runs/expand_phase03_pilot/best.pt before smoke')
    game = TournamentGameSession(seed=42, starting_stack=500, hands_per_level=5,
                                 lineup='trained_1_improved_4')
    moves=0
    while moves<50 and not game.view()['tournament_over']:
        view=game.view()
        assert view['seats'][1]['role']=='tournament_agent'
        assert sum(s['stack'] for s in view['seats'])+view['pot']==view['total_chips']
        if not view['your_turn']:
            raise AssertionError('session must auto-run all trained/improved bots until human turn')
        legal=[int(a['index']) for a in view['actions']]
        action=1 if 1 in legal else legal[0]
        game.play_action(action,expected_revision=view['revision'])
        moves+=1
    view=game.view()
    print(json.dumps({'phase':'Tournament ENV 1.4','checkpoint':status()['checkpoint_name'],
                      'agent':view['seats'][1]['policy_name'],'human_actions':moves,
                      'hands_completed':view['hands_completed'],
                      'chip_conservation':sum(s['stack'] for s in view['seats'])+view['pot']==view['total_chips'],
                      'result':'passed'},indent=2))


if __name__ == '__main__':
    main()

````````

<!-- END_PY_SOURCE path="scripts/tournament_env14_smoke.py" -->

## scripts/tournament_env17_release_audit.py

Original Git blob SHA: `c96316da5e205ceea6a6f6685d6d6d4a49ee8b2f` (5498 bytes).

<a id="py-058"></a>

<!-- BEGIN_PY_SOURCE path="scripts/tournament_env17_release_audit.py" sha="c96316da5e205ceea6a6f6685d6d6d4a49ee8b2f" -->

````````python
"""Offline Git safety audit for Tournament ENV 1.7. Read-only; no git mutations.

Usage (from repo root):
    uv run python -m scripts.tournament_env17_release_audit
    uv run python -m scripts.tournament_env17_release_audit --strict

The checker does not inspect secrets inside arbitrary source text. Manual review
of staged diffs is still required before committing and pushing.
"""
from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import subprocess
import sys


DANGEROUS_DIRS = (
    '.local/', 'runs/', 'checkpoints/', '.venv/', 'venv/',
    'frontend/node_modules/', 'frontend/dist/', 'frontend/coverage/',
    'data/checkpoints/', 'data/hand_histories/',
)
DANGEROUS_EXT = (
    '.pt', '.pth', '.ckpt', '.safetensors', '.sqlite', '.sqlite3',
    '.sqlite3-wal', '.sqlite3-shm', '.jsonl', '.ndjson', '.parquet',
)
SAMPLE_PRIVATE_PATHS = (
    '.local/tournament_history.sqlite3',
    'runs/expand_phase03_pilot/best.pt',
    'checkpoints/private.pt',
    'frontend/node_modules/example.js',
)


def run_git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ['git', '-C', str(root), *args], capture_output=True, check=check,
    )


def unsafe_reason(path: str) -> str | None:
    clean = path.replace('\\', '/')
    parts = PurePosixPath(clean).parts
    name = parts[-1].lower() if parts else ''
    if name == '.gitkeep' and clean in ('data/hand_histories/.gitkeep', 'data/checkpoints/.gitkeep'):
        return None
    if clean.startswith(DANGEROUS_DIRS) or any(p in ('__pycache__', '.pytest_cache') for p in parts):
        return 'generated/private directory'
    if name == '.env' or name.startswith('.env.') and name != '.env.example':
        return 'environment secrets'
    if name.endswith(DANGEROUS_EXT):
        return 'model/data/binary artifact'
    if name.endswith(('.sqlite3-journal', '.db', '.db-wal', '.db-shm')):
        return 'database artifact'
    return None


def git_paths(root: Path, *args: str) -> list[str]:
    out = run_git(root, *args).stdout
    return [x.decode('utf-8', errors='replace') for x in out.split(b'\0') if x]


def check_release(root: Path, *, strict: bool = False) -> tuple[int, list[str]]:
    log: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    try:
        actual = Path(run_git(root, 'rev-parse', '--show-toplevel').stdout.decode().strip()).resolve()
    except (OSError, subprocess.CalledProcessError):
        return 2, ['FAIL: not in a Git repository']
    if actual != root.resolve():
        return 2, [f'FAIL: root mismatch: expected {root}, Git uses {actual}']
    if not (root / 'frontend' / 'package.json').is_file() or not (root / 'backend' / 'main.py').is_file():
        return 2, ['FAIL: not the holdem-ai project root']
    branch = run_git(root, 'branch', '--show-current').stdout.decode('utf-8', errors='replace').strip()
    log.append(f'Git root: {root}')
    log.append(f'Current branch: {branch or "detached HEAD"}')
    staged = git_paths(root, 'diff', '--cached', '--name-only', '-z', '--diff-filter=ACMR')
    tracked = git_paths(root, 'ls-files', '-z')
    risky_staged = [(p, unsafe_reason(p)) for p in staged if unsafe_reason(p)]
    risky_tracked = [(p, unsafe_reason(p)) for p in tracked if unsafe_reason(p)]
    for p, why in risky_staged:
        errors.append(f'BLOCK: staged private/generated file: {p} ({why})')
    for p, why in risky_tracked:
        errors.append(f'BLOCK: private/generated file is tracked: {p} ({why})')
    for relative in SAMPLE_PRIVATE_PATHS:
        ignored = run_git(root, 'check-ignore', '-q', '--no-index', '--', relative, check=False)
        if ignored.returncode != 0:
            warnings.append(f'IGNORE MISSING: {relative}')
    if staged:
        diff = run_git(root, 'diff', '--cached', '--check', check=False)
        if diff.returncode != 0:
            errors.append('BLOCK: git diff --cached --check failed (whitespace/conflict markers)')
    elif strict:
        errors.append('BLOCK: no files staged; run the curated stage script first')
    else:
        warnings.append('No staged changes yet (normal before staging).')
    remote = run_git(root, 'remote', 'get-url', 'origin', check=False)
    if remote.returncode == 0:
        log.append('Origin: ' + remote.stdout.decode('utf-8', errors='replace').strip())
    else:
        warnings.append('No origin remote configured.')
    log.append(f'Staged files: {len(staged)}; tracked files: {len(tracked)}')
    log.extend('WARNING: ' + line for line in warnings)
    log.extend(errors)
    # In non-strict mode, report missing ignore rules as warnings so the user
    # can inspect the repository without changing it. In strict mode, block push.
    if strict and any(w.startswith('IGNORE MISSING') for w in warnings):
        log.append('BLOCK: update .gitignore before pushing')
        errors.append('missing ignore patterns')
    return (2 if errors else 0), log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--strict', action='store_true', help='fail if no staged files or unsafe ignore patterns')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    code, lines = check_release(root, strict=args.strict)
    print('\n'.join(lines))
    print('AUDIT:', 'FAILED' if code else 'PASSED (review staged diff manually)')
    return code


if __name__ == '__main__':
    sys.exit(main())

````````

<!-- END_PY_SOURCE path="scripts/tournament_env17_release_audit.py" -->

## src/holdem_ai/__init__.py

Original Git blob SHA: `203de1be16785e861c512ce99819dc0396384229` (65 bytes).

<a id="py-060"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/__init__.py" sha="203de1be16785e861c512ce99819dc0396384229" -->

````````python
"""No-Limit Texas Hold'em AI platform."""

__version__ = "0.1.0"

````````

<!-- END_PY_SOURCE path="src/holdem_ai/__init__.py" -->
