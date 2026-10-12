"""Expand Phase 3: multiagent PPO on a public-information six-seat game."""
from .model import ActorCritic, ValueV2
from .collector import MultiAgentCollector, RolloutConfig, RolloutResult
from .returns import MultiAgentTransition, PreparedSample, prepare_gae
from .population import OpponentMix, scripted_action
from .ppo import PPOConfig, optimize_ppo

__all__ = ["ActorCritic", "ValueV2", "MultiAgentCollector", "RolloutConfig", "RolloutResult",
           "MultiAgentTransition", "PreparedSample", "prepare_gae", "OpponentMix", "scripted_action",
           "PPOConfig", "optimize_ppo"]
from .trainer import TrainConfig, build_model, train, evaluate_checkpoint, load_trained_actor
__all__.extend(['TrainConfig','build_model','train','evaluate_checkpoint','load_trained_actor'])
