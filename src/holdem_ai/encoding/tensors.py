"""Optional PyTorch adapters; importable without torch until batching is requested."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from holdem_ai.encoding.observation import EncodedActor, EncodedCritic, ACTOR_FEATURE_DIM, CRITIC_FEATURE_DIM
from holdem_ai.encoding.vocabulary import PAD


@dataclass(frozen=True, slots=True)
class ActorBatch:
    token_ids: object
    token_values: object
    attention_mask: object
    features: object
    legal_action_mask: object


@dataclass(frozen=True, slots=True)
class CriticBatch:
    token_ids: object
    token_values: object
    attention_mask: object
    features: object


def _common(rows, device):
    import torch

    if not rows:
        raise ValueError('cannot batch empty sequences')
    length = max(len(row.token_ids) for row in rows)
    token_ids = torch.full((len(rows), length), PAD, dtype=torch.long, device=device)
    values = torch.zeros((len(rows), length), dtype=torch.float32, device=device)
    attention = torch.zeros((len(rows), length), dtype=torch.bool, device=device)
    for row_index, row in enumerate(rows):
        k = len(row.token_ids)
        token_ids[row_index, :k] = torch.tensor(row.token_ids, dtype=torch.long, device=device)
        values[row_index, :k] = torch.tensor(row.token_values, dtype=torch.float32, device=device)
        attention[row_index, :k] = True
    return token_ids, values, attention


def batch_actors(rows: Sequence[EncodedActor], *, device='cpu') -> ActorBatch:
    import torch

    ids, values, attention = _common(rows, device)
    features = torch.tensor([r.features for r in rows], dtype=torch.float32, device=device)
    legal = torch.tensor([r.legal_action_mask for r in rows], dtype=torch.bool, device=device)
    if features.shape[1] != ACTOR_FEATURE_DIM:
        raise AssertionError('bad actor dimensions')
    return ActorBatch(ids, values, attention, features, legal)


def batch_critics(rows: Sequence[EncodedCritic], *, device='cpu') -> CriticBatch:
    import torch

    ids, values, attention = _common(rows, device)
    features = torch.tensor([r.features for r in rows], dtype=torch.float32, device=device)
    if features.shape[1] != CRITIC_FEATURE_DIM:
        raise AssertionError('bad critic dimensions')
    return CriticBatch(ids, values, attention, features)
