# Full Python code — Volume 7/28

Part of [integrated report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Immutable source commit: `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`.

## src/holdem_ai/encoding/observation.py

Blob `1f9bd6f584feadde7710ce8e4c2f767daba960b0` · 8671 bytes.

<a id="py-073"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/encoding/observation.py" sha="1f9bd6f584feadde7710ce8e4c2f767daba960b0" -->

````````python
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

from holdem_ai.core.schema import InformationState
from holdem_ai.environment.hunl200_adapter import CriticState
from holdem_ai.encoding import vocabulary as v

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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/encoding/observation.py" -->

## src/holdem_ai/encoding/tensors.py

Blob `c49183bfc82b32709ef4a333150f0339dbab4c04` · 2352 bytes.

<a id="py-074"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/encoding/tensors.py" sha="c49183bfc82b32709ef4a333150f0339dbab4c04" -->

````````python
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

````````

<!-- END_PY_SOURCE path="src/holdem_ai/encoding/tensors.py" -->

## src/holdem_ai/encoding/vocabulary.py

Blob `24ebedf3aaa273b53eff63fc70f110748e0d9ac9` · 1074 bytes.

<a id="py-075"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/encoding/vocabulary.py" sha="24ebedf3aaa273b53eff63fc70f110748e0d9ac9" -->

````````python
"""Stable VRPO-Lite v1 token vocabulary (NOT the original VRPO paper encoding)."""
from __future__ import annotations

PAD = 0
BOS = 1
PRIVATE = 2
OPPONENT_PRIVATE = 3  # Training-only token: never used by the actor.
BOARD = 4
STREET = 5
HISTORY = 6
EVENT = 7
AMOUNT = 8
TERMINAL = 9
SEAT_0 = 10
SEAT_1 = 11

SUITS = 'sdch'
RANKS = '23456789TJQKA'
CARDS = tuple(s + r for s in SUITS for r in RANKS)
CARD_TO_ID = {card: 12 + i for i, card in enumerate(CARDS)}
STREETS = ('preflop', 'flop', 'turn', 'river')
STREET_TO_ID = {street: 64 + i for i, street in enumerate(STREETS)}
ACTIONS = ('fold', 'check', 'call', 'raise_to', 'all_in')
ACTION_TO_ID = {name: 68 + i for i, name in enumerate(ACTIONS)}
VOCAB_SIZE = 73


def card_id(card: str) -> int:
    try:
        return CARD_TO_ID[card]
    except (TypeError, KeyError) as exc:
        raise ValueError(f'invalid canonical card: {card!r}') from exc


def seat_id(seat: int) -> int:
    if type(seat) is not int or seat not in (0, 1):
        raise ValueError('seat must be 0 or 1')
    return SEAT_0 if seat == 0 else SEAT_1

````````

<!-- END_PY_SOURCE path="src/holdem_ai/encoding/vocabulary.py" -->

## src/holdem_ai/environment/__init__.py

Blob `d3cec59006cfffd5a6fdf42f3080c9102bc7788c` · 46 bytes.

<a id="py-076"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/__init__.py" sha="d3cec59006cfffd5a6fdf42f3080c9102bc7788c" -->

````````python
"""Poker hand and tournament environments."""

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/__init__.py" -->

## src/holdem_ai/environment/actions.py

Blob `a9081ef6a3c46c7c186f09801a004be4a880e827` · 3395 bytes.

<a id="py-077"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/actions.py" sha="a9081ef6a3c46c7c186f09801a004be4a880e827" -->

````````python
"""Independent no-limit betting constraints (amounts are integer chips).

Pure rule helpers. RLCard's 5-action abstraction is NOT modified by these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BetContext:
    """Current street context for one decision.

    actor_commitment: chips already bet this street by the acting player.
    highest_commitment: highest contribution by any player on this street.
    last_full_raise: most recent full raise increment (or BB initially).
    remaining_stack: acting player's chips available before decision.
    may_raise: false when action has not been reopened by a short all-in.
    """

    actor_commitment: int
    highest_commitment: int
    last_full_raise: int
    remaining_stack: int
    big_blind: int
    may_raise: bool = True

    def __post_init__(self) -> None:
        parts = (
            self.actor_commitment,
            self.highest_commitment,
            self.last_full_raise,
            self.remaining_stack,
            self.big_blind,
        )
        if any(not isinstance(v, int) for v in parts):
            raise TypeError('bet sizes must be integer chip values')
        if self.actor_commitment < 0 or self.highest_commitment < self.actor_commitment:
            raise ValueError('invalid street commitments')
        if self.remaining_stack < 0 or self.big_blind <= 0 or self.last_full_raise < self.big_blind:
            raise ValueError('invalid stack, blind, or last full raise')

    @property
    def amount_to_call(self) -> int:
        return min(self.remaining_stack, self.highest_commitment - self.actor_commitment)

    @property
    def max_total(self) -> int:
        return self.actor_commitment + self.remaining_stack

    @property
    def min_full_raise_to(self) -> int:
        if self.highest_commitment == 0:
            return self.big_blind
        return self.highest_commitment + self.last_full_raise

    @property
    def can_raise(self) -> bool:
        return self.may_raise and self.max_total > self.highest_commitment


def validate_raise_to(ctx: BetContext, total: int) -> str:
    """Validate total street commitment and classify full vs short all-in.

    A short all-in raise is allowed but does not automatically reopen betting.
    Caller tracks which opponents have already acted and `may_raise`.
    """
    if not isinstance(total, int):
        raise TypeError('raise-to amount must be an integer')
    if not ctx.may_raise:
        raise ValueError('betting has not reopened for this player')
    if total <= ctx.highest_commitment:
        raise ValueError('raise must exceed the current highest bet')
    if total > ctx.max_total:
        raise ValueError('raise exceeds available stack')
    if total >= ctx.min_full_raise_to:
        return 'full_raise'
    if total == ctx.max_total:
        return 'short_all_in_raise'
    raise ValueError(f'raise below minimum {ctx.min_full_raise_to}; only all-in is allowed')


def legal_basic_actions(ctx: BetContext) -> tuple[str, ...]:
    """Reference actions; separates check and call unlike RLCard's CHECK_CALL."""
    if ctx.remaining_stack == 0:
        return ()
    need = ctx.highest_commitment - ctx.actor_commitment
    options = ['fold', 'check' if need == 0 else 'call']
    if ctx.can_raise:
        options.append('raise')
    return tuple(options)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/actions.py" -->

## src/holdem_ai/environment/bet_abstraction.py

Blob `732e91e5c725553f6c71ba42d593df6e16e48273` · 5821 bytes.

<a id="py-078"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/bet_abstraction.py" sha="732e91e5c725553f6c71ba42d593df6e16e48273" -->

````````python
"""Optional integer-chip action abstraction for engines with arbitrary raise-to amounts.

This layer NEVER determines poker legality: the engine supplies legal flags and
min/max raise-to boundaries. Rounding, capping and duplicate suppression only
construct engine-safe candidates within those supplied bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

BET_ACTION_LABELS = (
    "fold", "check", "call", "raise_half_pot", "raise_pot", "raise_double_pot", "all_in",
)


@dataclass(frozen=True, slots=True)
class EngineBetAction:
    kind: Literal["fold", "check", "call", "raise_to", "all_in"]
    raise_to: int | None = None

    def __post_init__(self) -> None:
        if self.kind == "raise_to":
            if self.raise_to is None or self.raise_to < 0:
                raise ValueError("raise_to actions require nonnegative absolute street bet target")
        elif self.raise_to is not None:
            raise ValueError("only raise_to has a raise_to amount")


@dataclass(frozen=True, slots=True)
class NoLimitBetContext:
    """Poker-engine output, not an independent rules implementation.

    pot: total pot chips BEFORE player's next action (including opponents' bets)
    committed: player's chips already contributed on THIS street
    to_call: chips needed to match current street bet (can exceed stack)
    stack: player's remaining chips
    min_raise_to: engine-legal minimum *total contribution this street*
    max_raise_to: engine-legal maximum *total contribution this street*

    Raise flags must reflect re-opening, caps, tournament stack constraints and
    engine policy. A short all-in raise may be allowed below min_raise_to.
    """

    pot: int
    committed: int
    to_call: int
    stack: int
    min_raise_to: int | None
    max_raise_to: int | None
    can_fold: bool
    can_check: bool
    can_call: bool
    can_raise: bool

    def __post_init__(self) -> None:
        quantities = (self.pot, self.committed, self.to_call, self.stack)
        if any(type(x) is not int or x < 0 for x in quantities):
            raise ValueError("chip quantities must be nonnegative integers")
        for x in (self.min_raise_to, self.max_raise_to):
            if x is not None and (type(x) is not int or x < 0):
                raise ValueError("raise bounds must be nonnegative integers or None")
        if self.can_raise and (self.min_raise_to is None or self.max_raise_to is None):
            raise ValueError("can_raise requires explicit raise bounds")
        if self.can_check and self.to_call != 0:
            raise ValueError("check cannot be legal with outstanding call")
        if self.can_call and self.to_call <= 0:
            raise ValueError("call must be positive; use check when to_call=0")
        if self.max_raise_to is not None and self.max_raise_to > self.committed + self.stack:
            raise ValueError("max_raise_to exceeds all-in stack")


@dataclass(frozen=True, slots=True)
class DiscreteBetActions:
    """Fixed 7-action policy head; None entries are masked out."""

    actions: tuple[EngineBetAction | None, ...]

    @property
    def mask(self) -> tuple[bool, ...]:
        return tuple(a is not None for a in self.actions)

    def resolve(self, action_index: int) -> EngineBetAction:
        if isinstance(action_index, bool) or not isinstance(action_index, int):
            raise TypeError("action_index must be int")
        if not 0 <= action_index < len(self.actions) or self.actions[action_index] is None:
            raise ValueError("action is masked as illegal")
        action = self.actions[action_index]
        assert action is not None
        return action


def abstract_bet_actions(ctx: NoLimitBetContext) -> DiscreteBetActions:
    """Build action mask and engine action mapping using engine-provided bounds.

    For fraction f of the pot AFTER calling, the proposed total street
    contribution is c + call + round(f * (pot + call)).
    The 0.5, 1, 2 pot fractions are rounded to integer chips; this is an
    action-size *abstraction*, NOT a proof of poker-rule completeness.
    """
    actions: list[EngineBetAction | None] = [None] * len(BET_ACTION_LABELS)
    if ctx.can_fold:
        actions[0] = EngineBetAction("fold")
    if ctx.can_check:
        actions[1] = EngineBetAction("check")
    if ctx.can_call:
        actions[2] = EngineBetAction("call")

    if ctx.can_raise and ctx.stack > ctx.to_call:
        assert ctx.min_raise_to is not None and ctx.max_raise_to is not None
        maximum = ctx.max_raise_to
        called_to = ctx.committed + ctx.to_call
        all_in_to = ctx.committed + ctx.stack
        # All-in may be a legal short raise below the normal minimum. The
        # engine's 'can_raise' MUST already account for action reopening.
        if maximum == all_in_to and maximum > called_to:
            actions[6] = EngineBetAction("all_in")
        # Engine bounds define ordinary valid full raises. Short all-ins are
        # represented only by the explicit all_in action above.
        if ctx.min_raise_to <= maximum and maximum > called_to:
            seen_targets: set[int] = set()
            for index, (numer, denom) in enumerate(((1, 2), (1, 1), (2, 1)), start=3):
                extra = (numer * (ctx.pot + ctx.to_call) + denom // 2) // denom
                target = min(max(called_to + extra, ctx.min_raise_to), maximum)
                if target <= called_to or target >= all_in_to or target in seen_targets:
                    continue
                actions[index] = EngineBetAction("raise_to", target)
                seen_targets.add(target)
    if not any(x is not None for x in actions):
        raise ValueError("environment supplied no legal action for this turn")
    return DiscreteBetActions(tuple(actions))

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/bet_abstraction.py" -->

## src/holdem_ai/environment/blind_manager.py

Blob `1cfc0bffb3a2c9411296d5a85d32930ecc8ff586` · 2482 bytes.

<a id="py-079"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/blind_manager.py" sha="1cfc0bffb3a2c9411296d5a85d32930ecc8ff586" -->

````````python
"""Deterministic hand-count blind schedule for single-table tournaments.

The current hand engine has no antes; this manager intentionally supports only
small and big blinds. A final configured level repeats indefinitely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BlindLevel:
    small_blind: int
    big_blind: int

    def __post_init__(self) -> None:
        if (type(self.small_blind) is not int or type(self.big_blind) is not int
                or not 0 < self.small_blind < self.big_blind):
            raise ValueError('blinds must be integer chips satisfying 0 < SB < BB')


DEFAULT_BLIND_LEVELS = (
    BlindLevel(5, 10), BlindLevel(10, 20), BlindLevel(20, 40),
    BlindLevel(40, 80), BlindLevel(80, 160), BlindLevel(160, 320),
    BlindLevel(320, 640), BlindLevel(640, 1280),
    BlindLevel(1280, 2560), BlindLevel(2560, 5120),
    BlindLevel(5120, 10240),
)


class BlindManager:
    def __init__(
        self,
        levels: Sequence[BlindLevel | tuple[int, int]] | None = None,
        hands_per_level: int = 10,
    ) -> None:
        if type(hands_per_level) is not int or hands_per_level <= 0:
            raise ValueError('hands_per_level must be positive integer')
        if levels is None:
            levels = DEFAULT_BLIND_LEVELS
        parsed = []
        for level in levels:
            if isinstance(level, BlindLevel):
                parsed.append(level)
            elif isinstance(level, (list, tuple)) and len(level) == 2:
                parsed.append(BlindLevel(*level))
            else:
                raise ValueError('each blind level must be (SB, BB)')
        if not parsed:
            raise ValueError('blind schedule cannot be empty')
        for before, after in zip(parsed, parsed[1:]):
            if (after.small_blind < before.small_blind
                    or after.big_blind < before.big_blind):
                raise ValueError('blind levels must not decrease')
        self.levels = tuple(parsed)
        self.hands_per_level = hands_per_level

    def at_hand(self, hand_number: int) -> tuple[int, BlindLevel]:
        """Return (0-based level index, level) for a 1-based hand number."""
        if type(hand_number) is not int or hand_number < 1:
            raise ValueError('hand_number must be >= 1')
        index = min((hand_number - 1) // self.hands_per_level, len(self.levels) - 1)
        return index, self.levels[index]

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/blind_manager.py" -->

## src/holdem_ai/environment/corrected_game.py

Blob `a3dcc0506cafc433f4d4b97e0e039213f2403fdb` · 5690 bytes.

<a id="py-080"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/corrected_game.py" sha="a3dcc0506cafc433f4d4b97e0e039213f2403fdb" -->

````````python
"""RLCard 1.2.0 game subclass fixing heads-up seats and betting round.

Upstream RLCard source is never modified. Integration is version-guarded.
"""
from __future__ import annotations

from importlib.metadata import version

from holdem_ai.environment.corrected_round import HeadsUpBettingRound
from holdem_ai.environment.hand_evaluator import rank_showdown
from holdem_ai.environment.pot_manager import build_side_pots, settle_pots


def make_corrected_rlcard_env(*, starting_stack=200, seed=42, button=0, allow_step_back=False):
    if version('rlcard') != '1.2.0':
        raise RuntimeError('Corrected HUNL adapter is verified only for rlcard==1.2.0')
    if not isinstance(starting_stack, int) or starting_stack < 2:
        raise ValueError('starting_stack must be >= 2')
    if button not in (0, 1):
        raise ValueError('button must be player 0 or 1')

    import rlcard
    from copy import deepcopy
    from rlcard.games.nolimitholdem import Game as UpstreamGame
    from rlcard.games.nolimitholdem.round import Action
    from rlcard.games.limitholdem import PlayerStatus

    class CorrectedGame(UpstreamGame):
        """Locally substituted game, preserving the RLCard Env API."""

        def init_game(self):
            # Use RLCard's seeded deck, card distribution and player creation.
            super().init_game()
            if self.num_players != 2:
                raise ValueError('CorrectedGame supports only two players')
            # RLCard initially posts blinds using its multiplayer formula,
            # which is wrong for heads-up. Restore the chips before reposting.
            for player in self.players:
                player.remained_chips += player.in_chips
                player.in_chips = 0
                player.status = PlayerStatus.ALIVE
            btn = int(self.dealer_id)
            bb = 1 - btn
            self.players[btn].bet(chips=self.small_blind)
            self.players[bb].bet(chips=self.big_blind)
            for player in self.players:
                if player.remained_chips == 0:
                    player.status = PlayerStatus.ALLIN
            self.game_pointer = btn
            self.round = HeadsUpBettingRound(
                self.players, self.big_blind, self.dealer, Action,
                PlayerStatus.ALIVE, PlayerStatus.ALLIN, PlayerStatus.FOLDED,
            )
            self.round.start_new_round(self.game_pointer, [p.in_chips for p in self.players])
            # Dealer pot must match the new blind contributions, otherwise
            # the first pot-sized action would be computed incorrectly.
            self.dealer.pot = sum(p.in_chips for p in self.players)
            return self.get_state(self.game_pointer), self.game_pointer

        def step(self, action):
            if action == Action.FOLD:
                if action not in self.get_legal_actions():
                    raise ValueError('Illegal fold')
                if self.allow_step_back:
                    self.history.append((
                        deepcopy(self.round), self.game_pointer, self.round_counter,
                        deepcopy(self.dealer), deepcopy(self.public_cards),
                        deepcopy(self.players),
                    ))
                actor = self.game_pointer
                self.round.proceed_round(self.players, action)
                self.game_pointer = 1 - actor
                self.round.game_pointer = self.game_pointer
                # The hand terminates immediately, retaining the board as-is.
                # Unlike the upstream game, no extra cards are dealt on a fold.
                return self.get_state(self.game_pointer), self.game_pointer
            return super().step(action)

        def step_back(self):
            result = super().step_back()
            if result:
                self.round.players = self.players
                self.round.dealer = self.dealer
            return result

        def get_payoffs(self):
            if not self.is_over():
                raise RuntimeError('Cannot calculate payoffs before the hand ends')
            folded = [p.status == PlayerStatus.FOLDED for p in self.players]
            contributions = [int(p.in_chips) for p in self.players]
            breakdown = build_side_pots(contributions, folded)
            if sum(not f for f in folded) == 1:
                # Fold winner needs no completed five-card board.
                strengths = [None if f else (0,) for f in folded]
            else:
                if len(self.public_cards) != 5:
                    raise AssertionError('Showdown requires five community cards')
                board = [c.get_index() for c in self.public_cards]
                strengths = [rank_showdown([c.get_index() for c in p.hand], board) for p in self.players]
            awards = settle_pots(breakdown, strengths, button=int(self.dealer_id))
            payoffs = [awards[i] - contributions[i] for i in range(2)]
            if sum(payoffs) != 0:
                raise AssertionError('Chip conservation violated')
            return payoffs

    env = rlcard.make('no-limit-holdem', config={
        'seed': seed,
        'game_num_players': 2,
        'chips_for_each': starting_stack,
        'dealer_id': button,
        'allow_step_back': allow_step_back,
    })
    upstream = env.game
    corrected = CorrectedGame(allow_step_back=allow_step_back, num_players=2)
    corrected.configure({
        'game_num_players': 2,
        'chips_for_each': starting_stack,
        'dealer_id': button,
    })
    # Retain RLCard Env's configured RNG, used by all dealt cards.
    corrected.np_random = upstream.np_random
    env.game = corrected
    return env

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/corrected_game.py" -->

## src/holdem_ai/environment/corrected_round.py

Blob `cfa15eaf85fdc2f240a7df349e804b21188538ed` · 5593 bytes.

<a id="py-081"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/corrected_round.py" sha="cfa15eaf85fdc2f240a7df349e804b21188538ed" -->

````````python
"""Two-player no-limit betting round for the RLCard 1.2.0 action abstraction.

This module intentionally keeps the upstream integer action IDs (0..4).
Bet sizing uses raise-to targets derived from the pot *after* calling.
A short all-in does not reopen betting for an opponent who already acted.
"""
from __future__ import annotations

from enum import IntEnum


class ActionId(IntEnum):
    FOLD = 0
    CHECK_CALL = 1
    RAISE_HALF_POT = 2
    RAISE_POT = 3
    ALL_IN = 4


class HeadsUpBettingRound:
    """Minimal Round-compatible engine; accepts actual RLCard player objects."""

    def __init__(self, players, big_blind: int, dealer, action_enum, alive, allin, folded):
        if len(players) != 2:
            raise ValueError('HeadsUpBettingRound requires exactly two players')
        self.players = players
        self.num_players = 2
        self.init_raise_amount = big_blind
        self.big_blind = big_blind
        self.dealer = dealer
        self.action_enum = action_enum
        self.alive = alive
        self.allin = allin
        self.folded = folded
        self.raised = [0, 0]
        self.last_full_raise = big_blind
        self.game_pointer = 0
        self.pending: set[int] = set()
        self.acted_since_full_raise: set[int] = set()

    def start_new_round(self, game_pointer, raised=None):
        self.raised = list(raised) if raised is not None else [0, 0]
        self.last_full_raise = self.big_blind
        self.acted_since_full_raise = set()
        live = {i for i, p in enumerate(self.players) if p.status == self.alive and p.remained_chips > 0}
        # With one player all-in, no postflop betting can occur; preflop
        # still needs to offer the remaining player a call/fold decision.
        self.pending = live if raised is not None or len(live) == 2 else set()
        self.game_pointer = game_pointer

    def _to_call(self, actor):
        return max(self.raised) - self.raised[actor]

    def _raise_to(self, actor, action):
        current = self.raised[actor]
        to_call = self._to_call(actor)
        pot_after_call = sum(p.in_chips for p in self.players) + to_call
        factor = 0.5 if action == self.action_enum.RAISE_HALF_POT else 1.0
        raise_increment = int(pot_after_call * factor)
        return current + to_call + raise_increment

    def _can_raise(self, actor):
        opponent = self.players[1 - actor]
        if opponent.status != self.alive or opponent.remained_chips == 0:
            return False  # No legal raise into a player who cannot respond.
        if actor in self.acted_since_full_raise and self._to_call(actor) > 0:
            return False
        p = self.players[actor]
        return p.remained_chips > self._to_call(actor)

    def get_nolimit_legal_actions(self, players=None):
        actor = self.game_pointer
        p = self.players[actor]
        if actor not in self.pending or p.status != self.alive or p.remained_chips <= 0:
            return []
        legal = [self.action_enum.FOLD, self.action_enum.CHECK_CALL]
        if self._can_raise(actor):
            old_high = max(self.raised)
            min_raise_to = (old_high + self.last_full_raise) if old_high else self.big_blind
            max_to = self.raised[actor] + p.remained_chips
            if max_to > old_high:
                legal.append(self.action_enum.ALL_IN)
            for action in (self.action_enum.RAISE_HALF_POT, self.action_enum.RAISE_POT):
                target = self._raise_to(actor, action)
                if min_raise_to <= target <= max_to and target > old_high:
                    legal.append(action)
        return sorted(legal, key=lambda a: a.value)

    def proceed_round(self, players, action):
        actor = self.game_pointer
        if action not in self.get_nolimit_legal_actions(players):
            raise ValueError(f'illegal action {action} for actor={actor}')
        p = self.players[actor]
        old_high = max(self.raised)
        before = self.raised[actor]
        if action == self.action_enum.FOLD:
            p.status = self.folded
            self.pending.clear()
        else:
            if action == self.action_enum.CHECK_CALL:
                chips = min(self._to_call(actor), p.remained_chips)
            elif action == self.action_enum.ALL_IN:
                chips = p.remained_chips
            else:
                chips = self._raise_to(actor, action) - before
            p.bet(chips=chips)
            self.raised[actor] = before + chips
            if p.remained_chips == 0:
                p.status = self.allin
            self.pending.discard(actor)
            self.acted_since_full_raise.add(actor)
            delta = self.raised[actor] - old_high
            if delta > 0:
                if delta >= self.last_full_raise:
                    self.last_full_raise = delta
                    self.acted_since_full_raise = {actor}
                # A short all-in still requires any opponent who owes chips
                # to respond, but it does not restore their raise rights.
                other = 1 - actor
                if self.players[other].status == self.alive and self.players[other].remained_chips > 0:
                    self.pending.add(other)
        # Filter out opponents who can no longer act.
        self.pending = {i for i in self.pending if self.players[i].status == self.alive}
        self.game_pointer = (1 - actor) if (1 - actor) in self.pending else actor
        return self.game_pointer

    def is_over(self):
        return not self.pending or any(p.status == self.folded for p in self.players)

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/corrected_round.py" -->

## src/holdem_ai/environment/game_state.py

Blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` · 0 bytes.

<a id="py-082"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/game_state.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/game_state.py" -->

## src/holdem_ai/environment/hand_engine.py

Blob `c03e1ac3f67a9782f7c805a44c08e412faaa544f` · 4869 bytes.

<a id="py-083"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/hand_engine.py" sha="c03e1ac3f67a9782f7c805a44c08e412faaa544f" -->

````````python
"""One-hand HUNL interface. Corrected RLCard adapter is the default.

Scope: heads-up only; this is not a six-player tournament engine.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Any, Mapping, Sequence

from holdem_ai.agents.base_agent import Agent


@dataclass(frozen=True)
class HandResult:
    payoffs_chips: tuple[float, float]
    payoffs_bb: tuple[float, float]
    action_history: tuple[tuple[int, int], ...]


class HandEngine:
    def __init__(
        self,
        starting_stack: int = 200,
        seed: int | None = 42,
        allow_step_back: bool = False,
        rules_mode: str = 'corrected',
        button: int = 0,
    ) -> None:
        if starting_stack < 2:
            raise ValueError('starting_stack must be at least 2 chips.')
        if button not in (0, 1):
            raise ValueError('button must be 0 or 1')
        if rules_mode not in ('corrected', 'upstream'):
            raise ValueError('rules_mode must be corrected or upstream')

        if rules_mode == 'corrected':
            from holdem_ai.environment.corrected_game import make_corrected_rlcard_env
            self.env = make_corrected_rlcard_env(
                starting_stack=starting_stack, seed=seed,
                button=button, allow_step_back=allow_step_back,
            )
        else:
            import rlcard
            self.env = rlcard.make('no-limit-holdem', config={
                'seed': seed,
                'game_num_players': 2,
                'chips_for_each': starting_stack,
                'dealer_id': button,
                'allow_step_back': allow_step_back,
            })
        self.rules_mode = rules_mode
        self.num_players = 2
        self.big_blind = int(self.env.game.big_blind)
        self._next_button = button
        self._state: Mapping[str, Any] | None = None
        self._player_id: int | None = None
        self._history: list[tuple[int, int]] = []

    def reset(self) -> tuple[Mapping[str, Any], int]:
        self._history.clear()
        # Rotate seats every hand; RLCard does not rotate a fixed dealer_id.
        self.env.game.dealer_id = self._next_button
        self._next_button = 1 - self._next_button
        self._state, self._player_id = self.env.reset()
        return self._state, self._player_id

    @property
    def state(self) -> Mapping[str, Any]:
        if self._state is None:
            raise RuntimeError('Call reset() before requesting state.')
        return self._state

    @property
    def current_player(self) -> int:
        if self._player_id is None:
            raise RuntimeError('Call reset() before requesting current_player.')
        return self._player_id

    @property
    def legal_actions(self) -> tuple[int, ...]:
        if self.is_over():
            return ()
        return tuple(int(a) for a in self.state['legal_actions'])

    def is_over(self) -> bool:
        return bool(self.env.is_over())

    def step(self, action_id: int) -> tuple[Mapping[str, Any], int]:
        if self._state is None:
            raise RuntimeError('Call reset() before step().')
        if self.is_over():
            raise RuntimeError('The hand has already finished.')
        if action_id not in self.legal_actions:
            raise ValueError(f'Illegal action_id={action_id}; legal actions are {self.legal_actions}')
        actor = self.current_player
        state, player = self.env.step(int(action_id))
        self._history.append((actor, int(action_id)))
        self._state, self._player_id = state, player
        return self._state, self._player_id

    def get_result(self) -> HandResult:
        if not self.is_over():
            raise RuntimeError('The hand is not over yet.')
        payoffs = tuple(float(p) for p in self.env.get_payoffs())
        if len(payoffs) != 2 or not isclose(sum(payoffs), 0.0, rel_tol=0, abs_tol=1e-6):
            raise AssertionError(f'Invalid zero-sum payoff: {payoffs}')
        return HandResult(
            payoffs_chips=(payoffs[0], payoffs[1]),
            payoffs_bb=(payoffs[0] / self.big_blind, payoffs[1] / self.big_blind),
            action_history=tuple(self._history),
        )

    def play_hand(self, agents: Sequence[Agent], max_actions: int = 500) -> HandResult:
        if len(agents) != 2:
            raise ValueError('A HUNL hand needs exactly two agents.')
        if max_actions < 1:
            raise ValueError('max_actions must be positive.')
        state, player_id = self.reset()
        n = 0
        while not self.is_over():
            if n >= max_actions:
                raise RuntimeError('Hand exceeded max_actions; inspect engine/rules.')
            action_id = agents[player_id].select_action(state)
            state, player_id = self.step(action_id)
            n += 1
        return self.get_result()

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/hand_engine.py" -->

## src/holdem_ai/environment/hand_evaluator.py

Blob `10c593213a90c118dab2e6a8666cf5431843be40` · 3274 bytes.

<a id="py-084"></a>

<!-- BEGIN_PY_SOURCE path="src/holdem_ai/environment/hand_evaluator.py" sha="10c593213a90c118dab2e6a8666cf5431843be40" -->

````````python
"""Independent exact seven-card Hold'em evaluator for showdown verification.

Card notation: 'sK', 'd7', 'cT', 'hA' (suit first). Larger tuple wins.
This is a reference implementation, not an optimized production evaluator.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Sequence

RANKS = '23456789TJQKA'
SUITS = 'shdc'
Rank = tuple[int, ...]


def parse_card(card: str) -> tuple[str, int]:
    if not isinstance(card, str) or len(card) != 2:
        raise ValueError('card must have two characters: e.g. sK')
    suit, rank = card[0].lower(), card[1].upper()
    if suit not in SUITS or rank not in RANKS:
        raise ValueError(f'invalid Holdem card: {card!r}')
    return suit, RANKS.index(rank) + 2


def _straight_high(values: Sequence[int]) -> int | None:
    unique = set(values)
    if 14 in unique:
        unique.add(1)  # A-2-3-4-5 wheel
    for high in range(14, 4, -1):
        if all(v in unique for v in range(high - 4, high + 1)):
            return high
    return None


def rank_five(cards: Sequence[str]) -> Rank:
    if len(cards) != 5:
        raise ValueError('rank_five requires exactly 5 cards')
    parsed = [parse_card(x) for x in cards]
    if len(set(parsed)) != 5:
        raise ValueError('duplicate cards are not allowed')
    ranks = [rank for _, rank in parsed]
    suits = [suit for suit, _ in parsed]
    count = Counter(ranks)
    groups = sorted(((freq, rank) for rank, freq in count.items()), reverse=True)
    flush = len(set(suits)) == 1
    straight = _straight_high(ranks)
    if flush and straight is not None:
        return (8, straight)
    if groups[0][0] == 4:
        four = groups[0][1]
        kicker = max(r for r in ranks if r != four)
        return (7, four, kicker)
    if [g[0] for g in groups] == [3, 2]:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *sorted(ranks, reverse=True))
    if straight is not None:
        return (4, straight)
    if groups[0][0] == 3:
        triple = groups[0][1]
        return (3, triple, *sorted((r for r in ranks if r != triple), reverse=True))
    if groups[0][0] == 2 and groups[1][0] == 2:
        high_pair = max(groups[0][1], groups[1][1])
        low_pair = min(groups[0][1], groups[1][1])
        kicker = max(r for r in ranks if r not in (high_pair, low_pair))
        return (2, high_pair, low_pair, kicker)
    if groups[0][0] == 2:
        pair = groups[0][1]
        return (1, pair, *sorted((r for r in ranks if r != pair), reverse=True))
    return (0, *sorted(ranks, reverse=True))


def rank_seven(cards: Sequence[str]) -> Rank:
    """Exact max over C(7,5)=21 five-card combinations."""
    if len(cards) != 7:
        raise ValueError('rank_seven requires exactly 7 cards')
    canonical = [f'{parse_card(card)[0]}{card[1].upper()}' for card in cards]
    if len(set(canonical)) != 7:
        raise ValueError('duplicate cards are not allowed')
    return max(rank_five(hand) for hand in combinations(canonical, 5))


def rank_showdown(hole: Sequence[str], board: Sequence[str]) -> Rank:
    if len(hole) != 2 or len(board) != 5:
        raise ValueError('showdown requires 2 hole cards and 5 board cards')
    return rank_seven((*hole, *board))

````````

<!-- END_PY_SOURCE path="src/holdem_ai/environment/hand_evaluator.py" -->
