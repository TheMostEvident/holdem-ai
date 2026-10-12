"""A tiny live CUDA end-to-end check for the user's Windows GPU."""
import pytest
import torch

from holdem_ai.experiments.controlled_study import ComparisonConfig, run_study
from holdem_ai.training.ppo_session import create_session, save_checkpoint
from holdem_ai.training.ppo import PPOConfig
from holdem_ai.models.transformer import NetworkConfig


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
def test_phase14_cuda_smoke(tmp_path):
    source=tmp_path/'source.pt'
    save_checkpoint(create_session(device='cpu',seed=77,
        network_config=NetworkConfig(embed_dim=16,attention_heads=2,
            transformer_layers=1, ff_dim=32,hidden_dim=24),
        ppo_config=PPOConfig(hands_per_update=2,epochs=1,minibatch_size=4)), source)
    result=run_study(ComparisonConfig(source_checkpoint=str(source),
        root=str(tmp_path/'study'),seeds=(1,),updates=1,hands_per_update=4,
        env_batch_size=2,minibatch_size=4,epochs=1,eval_every=1,
        validation_groups=2,test_groups=2,device='cuda',
        arm_duration_minutes=2,progress=False))
    assert result['status']=='complete'
