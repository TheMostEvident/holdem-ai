"""CUDA-only batched rollout integration, runs on the user's GTX 1070."""
from __future__ import annotations

import pytest
import torch

from holdem_agent_phase8 import NetworkConfig, TransformerActor
from holdem_agent_phase9 import PPOConfig
from holdem_agent_phase11 import collect_batched_self_play


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
