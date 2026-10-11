"""Runs a minimal real Phase-7 batch on CUDA when available (e.g. GTX 1070)."""
from __future__ import annotations

import pytest
import torch

from holdem_agent_phase6 import HUNL200Adapter
from holdem_agent_phase7 import batch_actors, batch_critics, encode_actor, encode_critic
from holdem_agent_phase8 import CentralizedQCritic, NetworkConfig, TransformerActor


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA runtime not installed on this machine')
def test_phase8_cuda_forward_backward():
    env = HUNL200Adapter(expose_training_state=True)
    observation = env.reset(seed=41).observation
    privileged = env.training_state()
    ab = batch_actors([encode_actor(observation)], device='cuda')
    cb = batch_critics([encode_critic(privileged)], device='cuda')
    cfg = NetworkConfig(embed_dim=32, transformer_layers=1, ff_dim=64, hidden_dim=64)
    actor, critic = TransformerActor(cfg).cuda(), CentralizedQCritic(cfg).cuda()
    p, q = actor.distribution(ab), critic(cb)
    action = torch.tensor([observation.legal_actions[0]], dtype=torch.long, device='cuda')
    loss = -p.log_prob(action).mean() + q[:, 0, 0].square().mean()
    loss.backward()
    torch.cuda.synchronize()
    assert bool(torch.isfinite(loss).item())
    assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in actor.parameters())
    assert all(x.grad is not None and bool(torch.isfinite(x.grad).all()) for x in critic.parameters())
