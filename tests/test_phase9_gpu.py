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
