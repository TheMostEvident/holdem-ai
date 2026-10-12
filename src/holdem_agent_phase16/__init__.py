"""Phase16: standalone opponent pool + observational strategy diagnostics.

This module imports neither Phase15 training code nor Phase15 checkpoints.
"""
from .opponents import BUILTIN_OPPONENTS, OpponentSpec, build_builtin
from .diagnostics import ActionProfile, SPR_BUCKETS, spr_bucket
from .evaluation import PoolEvaluationConfig, evaluate_opponent_pool

__all__ = ['BUILTIN_OPPONENTS', 'OpponentSpec', 'build_builtin',
           'ActionProfile', 'SPR_BUCKETS', 'spr_bucket',
           'PoolEvaluationConfig', 'evaluate_opponent_pool']
