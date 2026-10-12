"""VRPO-Lite v1 trainable networks for the existing Phase-7 batch contracts.

This is a lightweight research baseline, not a reproduction of the full
2026 VRPO paper. Actor inputs NEVER contain privileged CriticState.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from holdem_ai.encoding.actions import masked_categorical, masked_logits
from holdem_ai.encoding.observation import ACTOR_FEATURE_DIM, CRITIC_FEATURE_DIM, NUM_ACTIONS
from holdem_ai.encoding.tensors import ActorBatch, CriticBatch
from holdem_ai.encoding.vocabulary import PAD, VOCAB_SIZE


@dataclass(frozen=True)
class NetworkConfig:
    embed_dim: int = 64
    attention_heads: int = 2
    transformer_layers: int = 2
    ff_dim: int = 128
    hidden_dim: int = 128
    max_tokens: int = 256
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for key in ('embed_dim', 'attention_heads', 'transformer_layers', 'ff_dim', 'hidden_dim', 'max_tokens'):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f'{key} must be a positive integer')
        if self.embed_dim % self.attention_heads:
            raise ValueError('embed_dim must be divisible by attention_heads')
        if not isinstance(self.dropout, (int, float)) or not 0 <= self.dropout < 1:
            raise ValueError('dropout must be in [0, 1)')


class CausalTokenEncoder(nn.Module):
    """Use the *last valid* token after a causal Transformer.

    The attention mask is True for real tokens and False for right padding.
    Token amounts enter as a scalar embedding, separate from token identity.
    No history is silently truncated; oversize batches raise ValueError.
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__()
        self.config = config
        dim = config.embed_dim
        self.token_embed = nn.Embedding(VOCAB_SIZE, dim, padding_idx=PAD)
        self.position_embed = nn.Embedding(config.max_tokens, dim)
        self.value_embed = nn.Linear(1, dim, bias=False)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=config.attention_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=config.transformer_layers, enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, token_ids: Tensor, token_values: Tensor, attention_mask: Tensor) -> Tensor:
        if token_ids.ndim != 2 or token_ids.dtype != torch.long:
            raise ValueError('token_ids must be int64 [batch, time]')
        bsz, steps = token_ids.shape
        if bsz < 1 or steps < 1 or steps > self.config.max_tokens:
            raise ValueError('invalid time dimension or max_tokens exceeded')
        if token_values.shape != token_ids.shape or token_values.dtype != torch.float32:
            raise ValueError('token_values must be float32 and match token_ids')
        if attention_mask.shape != token_ids.shape or attention_mask.dtype != torch.bool:
            raise ValueError('attention_mask must be bool and match token_ids')
        if not bool(attention_mask[:, 0].all().item()):
            raise ValueError('every sample needs at least the first valid token')
        if not bool((attention_mask[:, 1:] <= attention_mask[:, :-1]).all().item()):
            raise ValueError('only contiguous right padding is supported')
        if not bool(((token_ids != PAD) == attention_mask).all().item()):
            raise ValueError('token ids must be PAD exactly at padded positions')
        if not bool(torch.isfinite(token_values).all().item()):
            raise ValueError('nonfinite token values')
        positions = torch.arange(steps, device=token_ids.device)
        x = (self.token_embed(token_ids)
             + self.position_embed(positions)[None, :, :]
             + self.value_embed(torch.asinh(token_values).unsqueeze(-1)))
        # In a PyTorch bool attention mask, True means blocked.
        causal = torch.ones((steps, steps), device=token_ids.device, dtype=torch.bool).triu(1)
        x = self.transformer(x, mask=causal, src_key_padding_mask=~attention_mask)
        last = attention_mask.long().sum(dim=1) - 1
        return self.output_norm(x[torch.arange(bsz, device=x.device), last])


class TransformerActor(nn.Module):
    """Decentralized policy: ActorBatch -> masked action logits [B,7]."""

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.tokens = CausalTokenEncoder(self.config)
        self.policy_head = nn.Sequential(
            nn.Linear(self.config.embed_dim + ACTOR_FEATURE_DIM, self.config.hidden_dim),
            nn.GELU(), nn.Linear(self.config.hidden_dim, NUM_ACTIONS),
        )

    def forward(self, batch: ActorBatch) -> Tensor:
        if not isinstance(batch, ActorBatch):
            raise TypeError('Actor requires ActorBatch only (no privileged CriticBatch)')
        if batch.features.ndim != 2 or batch.features.shape[-1] != ACTOR_FEATURE_DIM:
            raise ValueError('Actor features must be [B,20]')
        if not bool(torch.isfinite(batch.features).all().item()):
            raise ValueError('nonfinite Actor features')
        state = self.tokens(batch.token_ids, batch.token_values, batch.attention_mask)
        logits = self.policy_head(torch.cat((state, torch.asinh(batch.features)), dim=-1))
        return masked_logits(logits, batch.legal_action_mask)

    def distribution(self, batch: ActorBatch) -> torch.distributions.Categorical:
        return masked_categorical(self.forward(batch), batch.legal_action_mask)


class CentralizedQCritic(nn.Module):
    """Privileged 2-player zero-sum Q-values in big-blind reward units.

    Output shape: [batch, seat=2, action=7].  Both players' returns
    are opposite for each full-game state/action.  At terminal states,
    bootstrap Q(s_terminal, a) is identically zero for both players.

    The Q head does not select poker actions; the Actor always determines
    legal-action probabilities using player-visible InformationState.
    """

    def __init__(self, config: NetworkConfig | None = None) -> None:
        super().__init__()
        self.config = config or NetworkConfig()
        self.tokens = CausalTokenEncoder(self.config)
        self.q_head = nn.Sequential(
            nn.Linear(self.config.embed_dim + CRITIC_FEATURE_DIM, self.config.hidden_dim),
            nn.GELU(), nn.Linear(self.config.hidden_dim, NUM_ACTIONS),
        )

    def forward(self, batch: CriticBatch) -> Tensor:
        if not isinstance(batch, CriticBatch):
            raise TypeError('Critic requires training-only CriticBatch')
        if batch.features.ndim != 2 or batch.features.shape[-1] != CRITIC_FEATURE_DIM:
            raise ValueError('Critic features must be [B,24]')
        if not bool(torch.isfinite(batch.features).all().item()):
            raise ValueError('nonfinite Critic features')
        state = self.tokens(batch.token_ids, batch.token_values, batch.attention_mask)
        q0 = self.q_head(torch.cat((state, torch.asinh(batch.features)), dim=-1))
        # The final Phase-7 Critic feature is the terminal-state indicator.
        # Terminal states have no legal actions, so Q is used only as V=0.
        q0 = q0.masked_fill(batch.features[:, -1:].to(dtype=torch.bool), 0.0)
        return torch.stack((q0, -q0), dim=1)


def chosen_q(q_values: Tensor, acting_seats: Tensor, actions: Tensor) -> Tensor:
    """Gather player-conditioned Q for actual actions. All tensors batched."""
    if q_values.ndim != 3 or q_values.shape[1:] != (2, NUM_ACTIONS):
        raise ValueError('q_values must be [B,2,7]')
    if acting_seats.shape != (q_values.shape[0],) or acting_seats.dtype != torch.long:
        raise ValueError('acting_seats must be int64 [B]')
    if actions.shape != acting_seats.shape or actions.dtype != torch.long:
        raise ValueError('actions must be int64 [B]')
    if not bool(((acting_seats >= 0) & (acting_seats < 2)).all().item()):
        raise ValueError('invalid player index')
    if not bool(((actions >= 0) & (actions < NUM_ACTIONS)).all().item()):
        raise ValueError('invalid action index')
    rows = torch.arange(q_values.shape[0], device=q_values.device)
    return q_values[rows, acting_seats, actions]
