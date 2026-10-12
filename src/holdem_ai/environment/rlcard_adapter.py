"""Conservative RLCard Env adapter. No import-time dependency on rlcard/numpy."""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from holdem_ai.core.schema import InformationState, Terminal, Turn

# These are public fields only. NEVER pass raw_obs or engine state to a policy.
_PUBLIC_FIELDS = ("all_chips", "stakes", "pot", "total_chips", "current_player")


def _float_features(obj: Any) -> tuple[float, ...]:
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        value = float(obj)
        if not math.isfinite(value):
            raise ValueError("observation must be finite")
        return (value,)
    if isinstance(obj, (tuple, list)):
        return tuple(value for x in obj for value in _float_features(x))
    raise TypeError("RLCard obs must contain numeric feature data")


def _json_safe(value: Any) -> str:
    """Encode public JSON fields, including nested NumPy arrays/scalars.

    RLCard can return np.int64 values *inside* lists and dictionaries.  A
    top-level .tolist() call does not reach those nested values.  The encoder's
    ``default`` callback handles them recursively without importing NumPy.
    Unknown objects remain rejected rather than being stringified.
    """

    def convert_numpy(obj: Any) -> Any:
        tolist = getattr(obj, "tolist", None)
        if callable(tolist):
            return tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
        default=convert_numpy,
    )


def observe_rlcard_state(
    state: Mapping[str, Any],
    player_id: int,
    num_actions: int,
    *,
    history: Sequence[str] = (),
) -> InformationState:
    """Build player POV only from RLCard's get_state(player_id) projection.

    Required: ``obs`` and legal_actions (mapping of action_id => metadata).
    Public and own-card fields are separately whitelisted from raw_obs.
    Missing history means this is NOT enough for perfect-recall CFR.
    """
    if not isinstance(state, Mapping):
        raise TypeError("RLCard state must be a mapping")
    if num_actions <= 0:
        raise ValueError("num_actions must be positive")
    legal = state.get("legal_actions")
    if not isinstance(legal, Mapping):
        raise TypeError("expected legal_actions to be a mapping of integer action IDs")
    mask = [False] * num_actions
    for action_id in legal.keys():
        if isinstance(action_id, bool) or not isinstance(action_id, int):
            raise TypeError("RLCard legal action keys must be integer action IDs")
        if not 0 <= action_id < num_actions:
            raise ValueError(f"action_id {action_id} outside action space")
        mask[action_id] = True
    raw = state.get("raw_obs", {})
    if not isinstance(raw, Mapping):
        raw = {}
    hand = raw.get("hand", ())
    board = raw.get("public_cards", ())
    if not isinstance(hand, (tuple, list)) or not isinstance(board, (tuple, list)):
        raise TypeError("own hand / community cards must be lists of cards")
    public_values = tuple(
        (name, _json_safe(raw[name])) for name in _PUBLIC_FIELDS if name in raw
    )
    return InformationState(
        player_id=player_id,
        features=_float_features(state["obs"]),
        legal_action_mask=tuple(mask),
        private_cards=tuple(str(card) for card in hand),
        public_cards=tuple(str(card) for card in board),
        stage=str(raw.get("stage", "unknown")),
        public_state=public_values,
        history=tuple(history),
    )


class RLCardTurnDriver:
    """Adapter for RLCard .reset()/.step() -> (state, player_id).

    Provide explicit canonical action_labels for the ENVIRONMENT version; do
    not assume one no-limit action vocabulary works for arbitrary bet sizing.
    A custom 2-6 player/tournament engine should implement TurnBasedEnvironment
    directly, reusing observe_rlcard_state if it exposes the RLCard state API.
    """

    def __init__(
        self,
        env: Any,
        action_labels: Sequence[str],
        *,
        history_fn: Callable[[Any, int, Mapping[str, Any]], Sequence[str]] | None = None,
        terminal_rewards_fn: Callable[[Any], Sequence[float]] | None = None,
    ) -> None:
        if not action_labels or len(set(action_labels)) != len(action_labels):
            raise ValueError("action_labels must be nonempty and unique")
        self.env = env
        self.action_labels = tuple(action_labels)
        self.history_fn = history_fn
        self.terminal_rewards_fn = terminal_rewards_fn
        self._current: Turn | Terminal | None = None

    def reset(self, seed: int | None = None) -> Turn | Terminal:
        if seed is not None:
            seeder = getattr(self.env, "seed", None)
            if not callable(seeder):
                raise TypeError("environment cannot be seeded; do not silently claim reproducibility")
            seeder(seed)
        result = self.env.reset()
        return self._receive(result)

    def step(self, action_index: int) -> Turn | Terminal:
        if not isinstance(self._current, Turn):
            raise RuntimeError("reset before step and never step after terminal")
        self._current.observation.assert_legal(action_index)
        result = self.env.step(action_index)
        return self._receive(result)

    def _is_over(self) -> bool:
        checker = getattr(self.env, "is_over", None)
        if callable(checker):
            return bool(checker())
        game = getattr(self.env, "game", None)
        checker = getattr(game, "is_over", None)
        if callable(checker):
            return bool(checker())
        raise TypeError("RLCard env needs is_over() or game.is_over()")

    def _receive(self, result: Any) -> Turn | Terminal:
        if self._is_over():
            if self.terminal_rewards_fn is not None:
                payoffs = self.terminal_rewards_fn(self.env)
            else:
                getter = getattr(self.env, "get_payoffs", None)
                if not callable(getter):
                    raise TypeError("terminal payoff getter is required")
                payoffs = getter()
            self._current = Terminal(tuple(float(x) for x in payoffs))
            return self._current
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise TypeError("RLCard .reset/.step must return (player_state, player_id)")
        state, seat = result
        if isinstance(seat, bool) or not isinstance(seat, int) or seat < 0:
            raise TypeError("player id must be a nonnegative integer")
        history = self.history_fn(self.env, seat, state) if self.history_fn else ()
        self._current = Turn(observe_rlcard_state(state, seat, len(self.action_labels), history=history))
        return self._current
