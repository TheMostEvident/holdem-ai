"""Real CUDA invocation: only runs on CUDA-enabled hosts."""
from __future__ import annotations

import pytest
import torch

from holdem_ai.experiments.longrun import LongRunConfig, run_longrun
from holdem_ai.models.transformer import NetworkConfig
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.training.ppo_session import create_session, save_checkpoint


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA not available')
def test_cuda_one_update(tmp_path):
    source=tmp_path/'source.pt'
    session=create_session(device='cuda',seed=2,
        network_config=NetworkConfig(embed_dim=16,attention_heads=2,transformer_layers=1,
                                     ff_dim=32,hidden_dim=24,max_tokens=256),
        ppo_config=PPOConfig(hands_per_update=4,epochs=1,minibatch_size=4))
    save_checkpoint(session,source)
    output=run_longrun(LongRunConfig(root=str(tmp_path/'gpu'),source_checkpoint=str(source),
                                   algorithm='ppo',device='cuda',duration_minutes=1,
                                   max_updates=1,progress=False,hands_per_update=4,
                                   env_batch_size=2,minibatch_size=4,epochs=1,
                                   eval_groups=2,final_groups=2))
    assert output['completed_updates']==1
    assert output['cuda_peak_allocated_mb'] is not None
