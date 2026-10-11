"""Actor-observation-only behavioral diagnostics for evaluated HUNL200 hands."""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import torch

from holdem_agent_phase5.schema import InformationState
from holdem_agent_phase7 import batch_actors, encode_actor
from holdem_agent_phase13.diagnostics import ACTION_LABELS, STREETS

SPR_BUCKETS = ('spr_lt_1', 'spr_1_to_4', 'spr_4_to_10', 'spr_ge_10')
POSITIONS = ('button', 'big_blind')


def spr_bucket(observation: InformationState) -> str:
    """Effective stack-to-pot ratio using ONLY actor-visible public features.

    At an action decision, effective stack = min(hero stack, opponent stack).
    Feature indices are supplied by the Phase6 HUNL200 adapter.
    """
    if len(observation.features) < 8:
        raise ValueError('HUNL200 observation requires at least 8 features')
    pot = float(observation.features[3])
    hero = float(observation.features[6])
    villain = float(observation.features[7])
    if not all(math.isfinite(x) and x >= 0 for x in (pot, hero, villain)):
        raise ValueError('invalid observable pot or stacks')
    ratio = min(hero, villain) / max(pot, 1e-9)
    if ratio < 1:
        return SPR_BUCKETS[0]
    if ratio < 4:
        return SPR_BUCKETS[1]
    if ratio < 10:
        return SPR_BUCKETS[2]
    return SPR_BUCKETS[3]


def _empty_bucket() -> dict:
    return {'decisions': 0, 'actions': [0] * 7, 'legal_opportunities': [0] * 7}


def _summary_bucket(bucket: dict) -> dict:
    count, legal = bucket['actions'], bucket['legal_opportunities']
    return {'decisions': bucket['decisions'],
            'actions': dict(zip(ACTION_LABELS, count)),
            'legal_opportunities': dict(zip(ACTION_LABELS, legal)),
            'conditional_action_rate': {
                label: round(count[i] / legal[i], 6) if legal[i] else None
                for i, label in enumerate(ACTION_LABELS)
            }}


@dataclass
class ActionProfile:
    """Only public/own-player observations. No CriticState or engine access."""
    samples: list[InformationState] = field(default_factory=list, repr=False)
    actions: list[int] = field(default_factory=list, repr=False)

    def record(self, obs: InformationState, action: int) -> None:
        obs.assert_legal(action)
        if len(obs.legal_action_mask) != len(ACTION_LABELS):
            raise ValueError('wrong HUNL200 action mask size')
        if obs.stage not in STREETS or obs.features[1] not in (0.0, 1.0):
            raise ValueError('unrecognized HUNL200 street or button feature')
        self.samples.append(obs)
        self.actions.append(action)

    def summary(self, actor=None, *, device: str = 'cpu', big_blind: int = 100,
                entropy_batch_size: int = 64) -> dict:
        if type(entropy_batch_size) is not int or entropy_batch_size < 1:
            raise ValueError('entropy_batch_size must be positive int')
        total = _empty_bucket()
        by_street = {name: _empty_bucket() for name in STREETS}
        by_position = {name: _empty_bucket() for name in POSITIONS}
        by_spr = {name: _empty_bucket() for name in SPR_BUCKETS}
        for obs, action in zip(self.samples, self.actions):
            pos = 'button' if obs.features[1] == 1.0 else 'big_blind'
            bucket_name = spr_bucket(obs)
            for bucket in (total, by_street[obs.stage], by_position[pos], by_spr[bucket_name]):
                bucket['decisions'] += 1
                bucket['actions'][action] += 1
                for i, valid in enumerate(obs.legal_action_mask):
                    if valid:
                        bucket['legal_opportunities'][i] += 1
        out = {**_summary_bucket(total),
               'by_street': {k: _summary_bucket(v) for k, v in by_street.items()},
               'by_position': {k: _summary_bucket(v) for k, v in by_position.items()},
               'by_spr': {k: _summary_bucket(v) for k, v in by_spr.items()},
               'mean_policy_entropy': None,
               'mean_normalized_entropy': None}
        if actor is not None and self.samples:
            old_mode = actor.training
            entropy = normalized = 0.0
            try:
                actor.eval()
                with torch.inference_mode():
                    for start in range(0, len(self.samples), entropy_batch_size):
                        part = self.samples[start:start + entropy_batch_size]
                        enc = [encode_actor(obs, big_blind=big_blind,
                                            max_tokens=actor.config.max_tokens) for obs in part]
                        batch = batch_actors(enc, device=device)
                        distribution = actor.distribution(batch)
                        ent = distribution.entropy()
                        num_legal = batch.legal_action_mask.sum(dim=-1)
                        nent = torch.where(num_legal > 1,
                                           ent / num_legal.clamp_min(2).log(),
                                           torch.zeros_like(ent))
                        entropy += float(ent.sum().item())
                        normalized += float(nent.sum().item())
            finally:
                actor.train(old_mode)
            if not all(math.isfinite(x) for x in (entropy, normalized)):
                raise FloatingPointError('non-finite policy entropy')
            out['mean_policy_entropy'] = round(entropy / len(self.samples), 6)
            out['mean_normalized_entropy'] = round(normalized / len(self.samples), 6)
        return out
