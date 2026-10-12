"""HUNL200 VRPO-Lite v1 actor/centralized-critic encoders.

Actor input is *only* Phase-5 InformationState; full training data is accepted
only by the separate encode_critic() entry point. History is never truncated.
No neural-network import is needed to encode a hand.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass

from holdem_agent_phase5.schema import InformationState
from holdem_agent_phase6.adapter import CriticState
from . import vocabulary as v

HISTORY_RE = re.compile(
    r'(preflop|flop|turn|river)\|seat=([01])\|'
    r'(fold|check|call|raise_to|all_in)\|raise_to=(-|[0-9]+)\Z'
)
STATUSES = ('active', 'folded', 'allin', 'out')
ACTOR_FEATURE_DIM = 20  # Phase-6 12 floats + 2 seats * 4 status flags
CRITIC_FEATURE_DIM = 24  # Explicitly documented in PHASE7_ENCODING.md
NUM_ACTIONS = 7


@dataclass(frozen=True, slots=True)
class EncodedActor:
    token_ids: tuple[int, ...]
    token_values: tuple[float, ...]
    features: tuple[float, ...]
    legal_action_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.token_ids or len(self.token_ids) != len(self.token_values):
            raise ValueError('actor token/value length mismatch')
        if len(self.features) != ACTOR_FEATURE_DIM or len(self.legal_action_mask) != NUM_ACTIONS:
            raise ValueError('unexpected actor encoding dimension')
        if not any(self.legal_action_mask):
            raise ValueError('nonterminal policy must have legal actions')


@dataclass(frozen=True, slots=True)
class EncodedCritic:
    token_ids: tuple[int, ...]
    token_values: tuple[float, ...]
    features: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.token_ids or len(self.token_ids) != len(self.token_values):
            raise ValueError('critic token/value length mismatch')
        if len(self.features) != CRITIC_FEATURE_DIM:
            raise ValueError('unexpected critic encoding dimension')


def _onehot_statuses(statuses: tuple[str, ...] | list[str]) -> tuple[float, ...]:
    if len(statuses) != 2 or any(s not in STATUSES for s in statuses):
        raise ValueError('expected exactly two recognized player statuses')
    return tuple(float(s == label) for s in statuses for label in STATUSES)


def _cards(cards: tuple[str, ...], *, count: tuple[int, ...]) -> list[int]:
    if len(cards) not in count:
        raise ValueError(f'expected card count in {count}, got {len(cards)}')
    return [v.card_id(c) for c in cards]


def _validate_distinct(cards: tuple[str, ...]) -> None:
    if len(set(cards)) != len(cards):
        raise ValueError('duplicate visible or training card')


def _history_tokens(history: tuple[str, ...], big_blind: int) -> tuple[list[int], list[float]]:
    ids: list[int] = []
    vals: list[float] = []

    def add(token: int, amount: float = 0.0) -> None:
        ids.append(token)
        vals.append(amount)

    for record in history:
        match = HISTORY_RE.fullmatch(record)
        if match is None:
            raise ValueError(f'unsupported Phase-6 action history record: {record!r}')
        street, seat, action, amount = match.groups()
        if (action == 'raise_to') != (amount != '-'):
            raise ValueError('raise_to requires exact public amount; other actions use dash')
        add(v.EVENT)
        add(v.STREET_TO_ID[street])
        add(v.seat_id(int(seat)))
        add(v.ACTION_TO_ID[action])
        if amount != '-':
            add(v.AMOUNT, int(amount) / big_blind)
    return ids, vals


def _make_tokens(
    hole_cards: tuple[tuple[str, ...], ...], board: tuple[str, ...],
    street: str, history: tuple[str, ...], *, big_blind: int,
    max_tokens: int, terminal: bool = False,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if street not in v.STREET_TO_ID:
        raise ValueError(f'invalid street: {street!r}')
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError('max_tokens must be positive integer')
    if len(hole_cards) not in (1, 2):
        raise ValueError('actor needs one set of cards; critic needs two')
    for cards in hole_cards:
        _cards(cards, count=(2,))
    _cards(board, count=(0, 3, 4, 5))
    _validate_distinct(tuple(c for cards in hole_cards for c in cards) + board)
    ids = [v.BOS]
    if len(hole_cards) == 1:
        ids.extend([v.PRIVATE, *_cards(hole_cards[0], count=(2,))])
    else:
        # Reserved privilege boundary: both seats appear only in critic encoding.
        ids.extend([v.PRIVATE, v.SEAT_0, *_cards(hole_cards[0], count=(2,))])
        ids.extend([v.OPPONENT_PRIVATE, v.SEAT_1, *_cards(hole_cards[1], count=(2,))])
    ids.extend([v.BOARD, *_cards(board, count=(0, 3, 4, 5)), v.STREET, v.STREET_TO_ID[street], v.HISTORY])
    values = [0.0] * len(ids)
    extra_ids, extra_values = _history_tokens(history, big_blind)
    ids.extend(extra_ids)
    values.extend(extra_values)
    if terminal:
        ids.append(v.TERMINAL)
        values.append(0.0)
    if len(ids) > max_tokens:
        raise ValueError(f'observation needs {len(ids)} tokens, exceeds max_tokens={max_tokens}; history not truncated')
    return tuple(ids), tuple(values)


def encode_actor(observation: InformationState, *, big_blind: int = 100, max_tokens: int = 256) -> EncodedActor:
    """Lossless Phase-6 history, public numeric features, and legal-action mask."""
    if type(big_blind) is not int or big_blind <= 0:
        raise ValueError('big_blind must be positive int')
    if len(observation.features) != 12 or len(observation.legal_action_mask) != NUM_ACTIONS:
        raise ValueError('expected Phase-6 HUNL200 actor contract (12 features, 7 actions)')
    if observation.player_id not in (0, 1) or observation.features[0] != float(observation.player_id):
        raise ValueError('player id and feature[0] mismatch')
    status_json = dict(observation.public_state).get('status')
    if status_json is None:
        raise ValueError('Phase-6 public_state.status is required')
    try:
        statuses = json.loads(status_json)
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid JSON status') from exc
    status_features = _onehot_statuses(statuses)
    tokens, values = _make_tokens(
        (observation.private_cards,), observation.public_cards, observation.stage,
        observation.history, big_blind=big_blind, max_tokens=max_tokens,
    )
    features = tuple(float(f) for f in observation.features) + status_features
    if not all(math.isfinite(x) for x in features):
        raise ValueError('non-finite actor features')
    return EncodedActor(tokens, values, features, observation.legal_action_mask)


def encode_critic(state: CriticState, *, big_blind: int = 100, max_tokens: int = 256) -> EncodedCritic:
    """Privileged centralized Q-Critic encoding. NEVER pass into the actor."""
    if type(big_blind) is not int or big_blind <= 0:
        raise ValueError('big_blind must be positive int')
    if len(state.hole_cards) != 2 or any(len(x) != 2 for x in state.hole_cards):
        raise ValueError('training state must have two complete hole-card sets')
    if any(len(x) != 2 for x in (state.stacks, state.contributions, state.street_bets, state.status)):
        raise ValueError('critic requires two seats')
    if state.button not in (0, 1) or (state.player_to_act not in (0, 1, -1)):
        raise ValueError('invalid critic seat')
    if state.is_terminal != (state.player_to_act == -1):
        raise ValueError('terminal state must have player_to_act=-1')
    if state.street not in v.STREET_TO_ID:
        raise ValueError('invalid street')
    numbers = (*state.stacks, *state.contributions, *state.street_bets, state.pot)
    if any(type(x) is not int or x < 0 for x in numbers) or state.pot != sum(state.contributions):
        raise ValueError('invalid centralized chips/pot')
    ids, values = _make_tokens(
        state.hole_cards, state.board, state.street, state.history,
        big_blind=big_blind, max_tokens=max_tokens, terminal=state.is_terminal,
    )
    features = (
        float(state.button == 0), float(state.button == 1),
        float(state.player_to_act == 0), float(state.player_to_act == 1),
        *(float(state.street == s) for s in v.STREETS),
        *(stack / big_blind for stack in state.stacks),
        *(c / big_blind for c in state.contributions),
        *(bet / big_blind for bet in state.street_bets),
        state.pot / big_blind,
        *_onehot_statuses(state.status),
        float(state.is_terminal),
    )
    return EncodedCritic(ids, values, tuple(features))
