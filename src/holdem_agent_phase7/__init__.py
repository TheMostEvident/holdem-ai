"""Phase 7 HUNL200 actor/critic input encoders and 7-head action codec."""
from .actions import encode_action, masked_categorical, masked_logits
from .encoding import (
    ACTOR_FEATURE_DIM, CRITIC_FEATURE_DIM, NUM_ACTIONS,
    EncodedActor, EncodedCritic, encode_actor, encode_critic,
)
from .tensors import ActorBatch, CriticBatch, batch_actors, batch_critics
from .vocabulary import VOCAB_SIZE

__all__ = [
    'ACTOR_FEATURE_DIM', 'CRITIC_FEATURE_DIM', 'NUM_ACTIONS', 'VOCAB_SIZE',
    'EncodedActor', 'EncodedCritic', 'encode_actor', 'encode_critic',
    'encode_action', 'masked_logits', 'masked_categorical',
    'ActorBatch', 'CriticBatch', 'batch_actors', 'batch_critics',
]
