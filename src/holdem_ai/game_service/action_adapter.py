"""Adapt the existing seven-action HUNL200 policy head to GUI action offers.

IMPORTANT: policy indices are NOT the Phase-3 engine action IDs. A browser
sends a seven-head index; only HUNL200Adapter is allowed to execute it.
Arbitrary human raise-to amounts are deferred to GUI-1.3, where the adapter
will need an explicit public method rather than mutating its private fields.
"""
from __future__ import annotations

import json
from typing import Any

from holdem_ai.core.schema import InformationState
from holdem_ai.environment.bet_abstraction import (
    BET_ACTION_LABELS,
    NoLimitBetContext,
    abstract_bet_actions,
)


def legal_gui_actions(observation: InformationState, engine: Any) -> list[dict[str, Any]]:
    """Provide legal labels and exact engine-validated raise-to targets.

    Only invoked for the acting HUMAN, using the same observable information
    available to the actor; never consumes an opponent's hidden hole cards.
    A mask mismatch is an integration error, not a reason to guess a bet size.
    """
    if engine.is_over() or engine.current_player != observation.player_id:
        raise ValueError("GUI action choices require the acting player's turn")
    raw = engine.observe(observation.player_id)["raw_obs"]
    i = observation.player_id
    engine_actions = set(engine.legal_actions)
    can_raise = 4 in engine_actions
    committed = int(raw["street_bets"][i])
    stack = int(raw["stacks"][i])
    to_call = int(raw["to_call"])
    ctx = NoLimitBetContext(
        pot=int(raw["pot"]),
        committed=committed,
        to_call=to_call,
        stack=stack,
        min_raise_to=int(raw["min_raise_to"]) if can_raise else None,
        max_raise_to=committed + stack if can_raise else None,
        can_fold=0 in engine_actions,
        can_check=1 in engine_actions and to_call == 0,
        can_call=1 in engine_actions and to_call > 0,
        can_raise=can_raise,
    )
    choices = abstract_bet_actions(ctx)
    if choices.mask != observation.legal_action_mask:
        raise AssertionError("GUI and agent legal-action masks disagree")
    return [
        {
            "index": index,
            "label": BET_ACTION_LABELS[index],
            "kind": action.kind,
            "raise_to": action.raise_to if action.kind == "raise_to" else (
                committed + stack if action.kind == "all_in" else None
            ),
        }
        for index, action in enumerate(choices.actions)
        if action is not None
    ]


def validate_human_action(observation: InformationState, action_index: int) -> int:
    """Validate without trusting a browser-supplied legal mask."""
    observation.assert_legal(action_index)
    return action_index
