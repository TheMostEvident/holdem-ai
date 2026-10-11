"""Phase 9: PPO Monte-Carlo baseline (not full VRPO)."""
from .ppo import (PPOConfig, TrajectoryRows, FrozenPPOData,
                  collect_self_play, freeze_old_policy_data,
                  clipped_surrogate, optimize_ppo, evaluate_vs_random)
from .trainer import (TrainingSession, create_session, save_checkpoint,
                      load_checkpoint, train_updates)

__all__ = [
    'PPOConfig', 'TrajectoryRows', 'FrozenPPOData', 'collect_self_play',
    'freeze_old_policy_data', 'clipped_surrogate', 'optimize_ppo',
    'evaluate_vs_random', 'TrainingSession', 'create_session',
    'save_checkpoint', 'load_checkpoint', 'train_updates',
]
