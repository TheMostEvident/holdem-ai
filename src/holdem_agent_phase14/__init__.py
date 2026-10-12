"""Phase 14: preregistered multi-seed PPO vs QBoost comparison."""
from .study import ComparisonConfig, run_study
from .statistics import seed_level_summary

__all__ = ['ComparisonConfig', 'run_study', 'seed_level_summary']
