"""GTX1070 smoke; skipped in environments without PyTorch CUDA."""
import pytest
import torch
from holdem_agent_phase8 import NetworkConfig
from holdem_agent_phase9 import PPOConfig
from holdem_agent_phase10 import create_qboost_session, train_qboost_updates


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA GPU not available here')
def test_phase10_cuda_actual_optimizer():
    cfg = NetworkConfig(embed_dim=32, attention_heads=2, transformer_layers=1, ff_dim=64, hidden_dim=48)
    ppo = PPOConfig(hands_per_update=2, epochs=1, minibatch_size=4, seed=8)
    session = create_qboost_session(device='cuda', network_config=cfg, ppo_config=ppo)
    result = train_qboost_updates(session, updates=1)
    assert result[0]['decisions'] > 0
    assert torch.cuda.max_memory_allocated() > 0
    assert all(torch.isfinite(p).all() for p in session.base.actor.parameters())
