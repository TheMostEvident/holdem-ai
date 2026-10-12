"""Expand Phase 2 acting-player information, action translation and encoding."""
from .schema import (PublicSeatFeatures, PublicAction, LegalActionSpec,
                     InformationState, TrainingOnlyCriticState, ACTION_NAMES)
from .builder import make_information_state
from .actions import legal_action_spec, to_table_action
from .encoder import (EncodedObservation, encode_observation, SEAT_DIM,
                      GLOBAL_DIM, HISTORY_DIM, MAX_HISTORY)
from .adapter import Phase2TableAdapter

__all__ = [
    'PublicSeatFeatures', 'PublicAction', 'LegalActionSpec', 'InformationState',
    'TrainingOnlyCriticState', 'ACTION_NAMES', 'make_information_state',
    'legal_action_spec', 'to_table_action', 'EncodedObservation',
    'encode_observation', 'SEAT_DIM', 'GLOBAL_DIM', 'HISTORY_DIM',
    'MAX_HISTORY', 'Phase2TableAdapter', 'PolicyV2',
]


def __getattr__(name: str):
    if name == 'PolicyV2':
        from .policy_v2 import PolicyV2
        return PolicyV2
    raise AttributeError(name)
