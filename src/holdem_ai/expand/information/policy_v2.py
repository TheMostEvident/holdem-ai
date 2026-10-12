"""Small, trainable multi-seat neural policy. No training in Expand Phase 2.

Actor input is *only* EncodedObservation; illegal logits are always masked.
History is encoded by GRU over the most recent 128 public betting actions.
"""
from __future__ import annotations

import torch
from torch import nn

from .encoder import (EncodedObservation, GLOBAL_DIM, HISTORY_DIM,
                      SEAT_DIM, MAX_HISTORY)
from .schema import ACTION_NAMES


class PolicyV2(nn.Module):
    def __init__(self, *, width: int = 64, heads: int = 4, layers: int = 2):
        super().__init__()
        if type(width) is not int or width < 16 or width % heads:
            raise ValueError('width must be >=16 and divisible by heads')
        self.width = width
        self.seat_project = nn.Sequential(nn.Linear(SEAT_DIM, width), nn.LayerNorm(width), nn.GELU())
        layer = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dim_feedforward=width*2,
                                           dropout=0.0, batch_first=True)
        self.seat_attention = nn.TransformerEncoder(layer, num_layers=layers,
                                                    enable_nested_tensor=False)
        self.history_project = nn.Sequential(nn.Linear(HISTORY_DIM, width), nn.GELU())
        self.history_rnn = nn.GRU(width, width, num_layers=1, batch_first=True)
        self.policy_head = nn.Sequential(nn.Linear(width*2+GLOBAL_DIM, width), nn.GELU(),
                                         nn.Linear(width, len(ACTION_NAMES)))

    def forward(self, observations: list[EncodedObservation]):
        if not observations or any(not isinstance(x, EncodedObservation) for x in observations):
            raise TypeError('PolicyV2 accepts nonempty list[EncodedObservation] only')
        device = next(self.parameters()).device
        seats = torch.tensor([o.seat_features for o in observations], device=device, dtype=torch.float32)
        seat_valid = torch.tensor([o.seat_mask for o in observations], device=device, dtype=torch.bool)
        globals_ = torch.tensor([o.global_features for o in observations], device=device, dtype=torch.float32)
        hist = torch.tensor([o.history_features for o in observations], device=device, dtype=torch.float32)
        history_valid = torch.tensor([o.history_mask for o in observations], device=device, dtype=torch.bool)
        action_mask = torch.tensor([o.legal_mask for o in observations], device=device, dtype=torch.bool)
        if torch.any(~seat_valid.any(dim=1)) or torch.any(~action_mask.any(dim=1)):
            raise ValueError('at least one occupied seat and legal action required')
        encoded = self.seat_attention(self.seat_project(seats), src_key_padding_mask=~seat_valid)
        pooled = (encoded*seat_valid.unsqueeze(-1)).sum(1)/seat_valid.sum(1, keepdim=True)
        # Use pack_padded_sequence to exclude padding from the recurrent state.
        lengths = history_valid.sum(1)
        packed = nn.utils.rnn.pack_padded_sequence(self.history_project(hist),
                    lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
        _, last = self.history_rnn(packed)
        h_summary = last[-1] * (lengths>0).unsqueeze(-1)
        logits = self.policy_head(torch.cat((pooled, h_summary, globals_), dim=-1))
        masked_logits = logits.masked_fill(~action_mask, torch.finfo(logits.dtype).min)
        return masked_logits

    @torch.no_grad()
    def action_probabilities(self, observation: EncodedObservation) -> tuple[float, ...]:
        self.eval()
        probabilities = self.forward([observation]).softmax(dim=-1)[0]
        return tuple(probabilities.cpu().tolist())

    @torch.no_grad()
    def select_action(self, observation: EncodedObservation, *, rng_seed: int, greedy: bool = False) -> str:
        self.eval()
        if type(rng_seed) is not int or rng_seed < 0:
            raise ValueError('nonnegative rng_seed required')
        logits = self.forward([observation])[0]
        if greedy:
            selected = logits.argmax().item()
        else:
            gen = torch.Generator(device='cpu')
            gen.manual_seed(rng_seed)
            # CPU generator works regardless of policy CUDA placement.
            probs = logits.softmax(-1).detach().cpu()
            selected = torch.multinomial(probs, 1, generator=gen).item()
        return ACTION_NAMES[selected]
