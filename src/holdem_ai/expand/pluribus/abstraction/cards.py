"""Reproducible (deliberately *untrained*) toy 6-max hold'em abstraction.

Preflop hero cards are EXACT. Postflop hash buckets are an infrastructure
prototype, NOT Pluribus' equity-based EMD/learned card buckets. The public
board and betting history are always exact and opponent private cards excluded.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json


@dataclass(frozen=True)
class AbstractionConfig:
    postflop_buckets: int = 16
    allow_half_pot: bool = False
    schema: str = 'e04-abstraction-v1-hash-prototype'

    def __post_init__(self):
        if type(self.postflop_buckets) is not int or not 2 <= self.postflop_buckets <= 4096:
            raise ValueError('postflop_buckets must be 2..4096')
        if type(self.allow_half_pot) is not bool:
            raise TypeError('allow_half_pot must be bool')

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(',', ':')).encode()
        return hashlib.sha256(encoded).hexdigest()[:24]


def holdem_infoset_key(raw: dict, actions: tuple[str, ...], config: AbstractionConfig) -> str:
    """Only use information supplied by engine.observe(current_actor).

    The engine observation is assumed actor-safe; do not inspect the engine's
    .hole_cards, ._deck, or ._rng. Strict allowlist in the key generator.
    """
    hero = raw['player_id']
    hole = tuple(raw['hand'])
    board = tuple(raw['public_cards'])
    if len(hole) != 2 or not 0 <= hero < 6:
        raise ValueError('invalid acting seat / private hand')
    if len(set((*hole, *board))) != 2 + len(board):
        raise ValueError('duplicate visible cards')
    street = raw['street']
    if street == 'preflop':
        if board:
            raise ValueError('preflop cannot expose board')
        private = ['exact', *sorted(hole)]
    elif street in ('flop', 'turn', 'river'):
        digest = hashlib.sha256('|'.join(sorted(hole) + list(board)).encode()).digest()
        bucket = int.from_bytes(digest[:8], 'big') % config.postflop_buckets
        private = ['prototype_bucket', bucket]
    else:
        raise ValueError('unknown street')
    # Full public information is represented, but no private rival cards.
    public = [
        'e04-holdem-v1', config.fingerprint, hero, street, private, list(board),
        raw['button'], list(raw['stacks']), list(raw['street_bets']),
        list(raw['total_contributions']), list(raw['status']),
        raw['pot'], raw['to_call'], raw['min_raise_to'],
        list(raw['action_history']), list(raw['action_raise_to']), list(actions),
    ]
    return json.dumps(public, separators=(',', ':'), ensure_ascii=True)
