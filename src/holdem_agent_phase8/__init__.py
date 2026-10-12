"""Phase 8: GTX 1070-friendly trainable VRPO-Lite Actor and Q-Critic."""
from .network import CausalTokenEncoder, CentralizedQCritic, NetworkConfig, TransformerActor, chosen_q
from .policy import NeuralActorPolicy

__all__ = [
    'CausalTokenEncoder', 'CentralizedQCritic', 'NetworkConfig',
    'TransformerActor', 'chosen_q', 'NeuralActorPolicy',
]
