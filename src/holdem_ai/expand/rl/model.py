"""Expand 3: strictly actor-visible shared multi-seat Actor + Value networks."""
from __future__ import annotations

import torch
from torch import nn

from holdem_ai.expand.information.encoder import EncodedObservation, GLOBAL_DIM, SEAT_DIM, HISTORY_DIM
from holdem_ai.expand.information.policy_v2 import PolicyV2


class ValueV2(nn.Module):
    """Unprivileged value baseline: receives only EncodedObservation."""
    def __init__(self, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(GLOBAL_DIM + SEAT_DIM + HISTORY_DIM, width),
            nn.Tanh(), nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))

    def forward(self, observations: list[EncodedObservation]) -> torch.Tensor:
        if not observations or any(not isinstance(o, EncodedObservation) for o in observations):
            raise TypeError('ValueV2 requires list[EncodedObservation]')
        device = next(self.parameters()).device
        g = torch.tensor([o.global_features for o in observations], dtype=torch.float32, device=device)
        seats = torch.tensor([o.seat_features for o in observations], dtype=torch.float32, device=device)
        sm = torch.tensor([o.seat_mask for o in observations], dtype=torch.float32, device=device)
        histories = torch.tensor([o.history_features for o in observations], dtype=torch.float32, device=device)
        hm = torch.tensor([o.history_mask for o in observations], dtype=torch.float32, device=device)
        seat_pool = (seats * sm.unsqueeze(-1)).sum(1) / sm.sum(1, keepdim=True).clamp(min=1)
        history_pool = (histories * hm.unsqueeze(-1)).sum(1) / hm.sum(1, keepdim=True).clamp(min=1)
        return self.net(torch.cat((g, seat_pool, history_pool), dim=-1)).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, *, width: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        self.width, self.heads, self.layers = width, heads, layers
        self.actor = PolicyV2(width=width, heads=heads, layers=layers)
        self.value = ValueV2(width=width)

    def forward(self, observations: list[EncodedObservation]):
        return self.actor(observations), self.value(observations)

    @torch.no_grad()
    def predict(self, observation: EncodedObservation):
        self.eval()
        logits, values = self([observation])
        return logits[0].softmax(-1).detach().cpu().tolist(), float(values[0].item())
