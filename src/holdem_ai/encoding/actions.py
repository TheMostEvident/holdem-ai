"""Phase-5 compatible fixed 7-head action encoding and masked distributions.

This is VRPO-Lite v1. It is NOT the paper's base-16 multistep action codec.
"""
from __future__ import annotations

from holdem_ai.environment.bet_abstraction import BET_ACTION_LABELS

NUM_ACTIONS = len(BET_ACTION_LABELS)


def encode_action(action_index: int, legal_mask: tuple[bool, ...]) -> tuple[float, ...]:
    if len(legal_mask) != NUM_ACTIONS or not any(legal_mask) or any(type(x) is not bool for x in legal_mask):
        raise ValueError('expected nonempty 7-action bool mask')
    if type(action_index) is not int or not 0 <= action_index < NUM_ACTIONS or not legal_mask[action_index]:
        raise ValueError('action is illegal or out of range')
    return tuple(float(index == action_index) for index in range(NUM_ACTIONS))


def masked_logits(logits, legal_mask):
    """Torch [B,7] logits and bool mask; illegal gradients and probs vanish."""
    import torch

    if logits.ndim != 2 or logits.shape[1] != NUM_ACTIONS:
        raise ValueError('logits must be [B,7]')
    if legal_mask.shape != logits.shape or legal_mask.dtype != torch.bool:
        raise ValueError('mask must be bool [B,7]')
    if not bool(legal_mask.any(dim=-1).all().item()):
        raise ValueError('each observation needs a legal action')
    if not bool(torch.isfinite(logits).all().item()):
        raise ValueError('nonfinite logits')
    return logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)


def masked_categorical(logits, legal_mask):
    import torch

    return torch.distributions.Categorical(logits=masked_logits(logits, legal_mask))
