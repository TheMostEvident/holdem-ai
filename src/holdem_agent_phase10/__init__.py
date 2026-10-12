"""Q-corrected control variate for an experimental VRPO-Lite PPO baseline."""
from .boost import (QBoostConfig, FrozenQBoostData, freeze_qboost_data,
                    qboost_surrogate, qboost_diagnostics, optimize_qboost)
from .trainer import (QBoostSession, create_qboost_session, from_phase9_checkpoint,
                      save_qboost_checkpoint, load_qboost_checkpoint, train_qboost_updates)

__all__ = ['QBoostConfig', 'FrozenQBoostData', 'freeze_qboost_data',
           'qboost_surrogate', 'qboost_diagnostics', 'optimize_qboost',
           'QBoostSession', 'create_qboost_session', 'from_phase9_checkpoint',
           'save_qboost_checkpoint', 'load_qboost_checkpoint', 'train_qboost_updates']
