"""Measure inference batching gains without PPO optimizer steps."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace

import torch

from holdem_agent_phase8 import NetworkConfig, TransformerActor
from holdem_agent_phase9 import PPOConfig
from holdem_agent_phase11 import collect_batched_self_play


def run(*, device='cpu', hands=100, batch_sizes=(1, 4, 8, 16), seed=42,
        small_network=False) -> list[dict]:
    if type(hands) is not int or hands < 1:
        raise ValueError('hands must be a positive integer')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.manual_seed(seed)
    net = (NetworkConfig(embed_dim=32, transformer_layers=1, attention_heads=2,
                         ff_dim=64, hidden_dim=48)
           if small_network else NetworkConfig())
    actor = TransformerActor(net).to(device).eval()
    cfg = PPOConfig(hands_per_update=hands, seed=seed)
    warm = replace(cfg, hands_per_update=min(hands, 4))
    collect_batched_self_play(actor, warm, device=device, env_batch_size=1)
    reports = []
    for n in batch_sizes:
        if type(n) is not int or n < 1:
            raise ValueError('batch size must be positive')
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rows = collect_batched_self_play(actor, cfg, iteration=0,
                                         device=device, env_batch_size=n)
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        reports.append({
            'phase': 11, 'device': device, 'hands': rows.hands,
            'decisions': len(rows.actions), 'env_batch_size': n,
            'elapsed_seconds': round(elapsed, 4),
            'hands_per_second': round(rows.hands / elapsed, 3),
            'decisions_per_second': round(len(rows.actions) / elapsed, 3),
            'peak_cuda_allocated_mb': (round(torch.cuda.max_memory_allocated()/2**20, 3)
                                       if device == 'cuda' else None),
        })
    return reports


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--hands', type=int, default=100)
    p.add_argument('--batch-sizes', default='1,4,8,16')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--small-network', action='store_true')
    args = p.parse_args()
    for item in run(device=args.device, hands=args.hands,
                    batch_sizes=tuple(int(x) for x in args.batch_sizes.split(',')),
                    seed=args.seed, small_network=args.small_network):
        print(json.dumps(item, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
