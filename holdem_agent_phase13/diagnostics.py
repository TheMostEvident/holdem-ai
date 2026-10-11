"""Action-rate and legal-opportunity diagnostics on an on-policy trajectory.

Rows contain only actor-visible encodings, actual action, and a training-only
critic row; never derive actor statistics from privileged critic features.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch

from holdem_agent_phase7 import batch_actors
from holdem_agent_phase9.ppo import TrajectoryRows

STREETS = ('preflop', 'flop', 'turn', 'river')
ACTION_LABELS = ('fold', 'check', 'call', 'raise_half_pot',
                 'raise_pot', 'raise_double_pot', 'all_in')


def action_diagnostics(rows: TrajectoryRows, actor, *, device: str = 'cpu',
                       chunk_size: int = 64) -> dict:
    """Report actual frequencies and *conditional-on-legal* choice frequencies.

    A choice probability of zero for Check cannot be interpreted without
    knowing if Check was ever legal. Legal opportunities are tracked per action.
    """
    if type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    counts = [0] * 7
    opportunities = [0] * 7
    by_street = {name: {'actions': [0]*7, 'legal_opportunities': [0]*7,
                        'decisions': 0} for name in STREETS}
    by_position = {name: {'actions': [0]*7, 'legal_opportunities': [0]*7,
                          'decisions': 0} for name in ('button', 'big_blind')}
    for obs, action in zip(rows.actors, rows.actions):
        street_index = int(obs.features[2])
        if street_index not in range(4) or float(street_index) != obs.features[2]:
            raise ValueError('invalid stage in actor features')
        street = STREETS[street_index]
        position = 'button' if obs.features[1] == 1.0 else 'big_blind'
        if obs.features[1] not in (0.0, 1.0):
            raise ValueError('invalid button indicator')
        counts[action] += 1
        for key in ('actions',):
            by_street[street][key][action] += 1
            by_position[position][key][action] += 1
        by_street[street]['decisions'] += 1
        by_position[position]['decisions'] += 1
        for i, legal in enumerate(obs.legal_action_mask):
            if legal:
                opportunities[i] += 1
                by_street[street]['legal_opportunities'][i] += 1
                by_position[position]['legal_opportunities'][i] += 1
    total = len(rows.actions)
    old_mode = actor.training
    entropy_total = 0.0
    entropy_norm_total = 0.0
    try:
        actor.eval()
        with torch.inference_mode():
            for start in range(0, total, chunk_size):
                batch = batch_actors(rows.actors[start:start + chunk_size], device=device)
                dist = actor.distribution(batch)
                h = dist.entropy()
                nlegal = batch.legal_action_mask.sum(dim=1)
                # Normalized entropy = H/log(num legal); 1-legal => 0.
                h_norm = torch.where(nlegal > 1, h / nlegal.clamp_min(2).log(),
                                     torch.zeros_like(h))
                entropy_total += float(h.sum().item())
                entropy_norm_total += float(h_norm.sum().item())
    finally:
        actor.train(old_mode)
    if not all(math.isfinite(x) for x in (entropy_total, entropy_norm_total)):
        raise FloatingPointError('nonfinite actor entropy')
    return {
        'decisions': total,
        'actions': {name: counts[i] for i, name in enumerate(ACTION_LABELS)},
        'legal_opportunities': {name: opportunities[i] for i, name in enumerate(ACTION_LABELS)},
        'conditional_action_rate': {
            name: (round(counts[i] / opportunities[i], 6)
                   if opportunities[i] else None)
            for i, name in enumerate(ACTION_LABELS)
        },
        'mean_policy_entropy': round(entropy_total / total, 6),
        'mean_normalized_entropy': round(entropy_norm_total / total, 6),
        'by_street': by_street,
        'by_position': by_position,
    }
