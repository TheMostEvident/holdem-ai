"""GPU smoke on user's GTX1070; deliberately skipped on CPU-only runners."""
from __future__ import annotations

import pytest
import torch

from holdem_agent_phase8 import NetworkConfig, TransformerActor
from holdem_agent_phase12 import EvaluationConfig, evaluate_seed_blocks


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_phase12_cuda_four_layout_inference():
    model = TransformerActor(NetworkConfig()).to('cuda')
    r = evaluate_seed_blocks(model, config=EvaluationConfig(groups=2, device='cuda', seed=989))
    assert r.summary()['hands'] == 8
    assert torch.cuda.max_memory_allocated() > 0
