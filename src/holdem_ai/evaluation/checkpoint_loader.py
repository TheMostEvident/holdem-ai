"""Read trusted Phase-9/10/11 checkpoint or Phase-11 anchor, actor ONLY."""
from __future__ import annotations

from pathlib import Path
import torch

from holdem_ai.models.transformer import NetworkConfig, TransformerActor


def load_actor(path: str | Path, *, device: str = 'cpu') -> TransformerActor:
    # Load only from a trusted local training run; torch weights_only=True.
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or 'network_config' not in payload or 'actor_state' not in payload:
        raise ValueError('not a Phase9/10/11 checkpoint or Phase11 anchor.pt')
    if payload.get('phase') not in (None, 9, 10):
        raise ValueError('unrecognized training checkpoint phase')
    actor = TransformerActor(NetworkConfig(**payload['network_config']))
    actor.load_state_dict(payload['actor_state'], strict=True)
    return actor.to(device).eval().requires_grad_(False)
