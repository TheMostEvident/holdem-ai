"""Actual CUDA run, skipped in environments without a CUDA GPU."""
from __future__ import annotations

import pytest
import torch

from holdem_ai.experiments.batch_scaling import BatchExperimentConfig, run_batch_experiment


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_phase13_cuda_optimization(tmp_path):
    config = BatchExperimentConfig(
        algorithm='qboost', device='cuda', duration_seconds=60,
        hands_per_update=8, env_batch_size=4, minibatch_size=8,
        epochs=1, max_updates=1, skip_eval=True,
        small_network=True,
        checkpoint=str(tmp_path/'model.pt'), log_dir=str(tmp_path/'log'))
    record = run_batch_experiment(config)
    assert record['completed_updates'] == 1
    assert record['cuda_peak_allocated_mb'] > 0
