"""Explicit whitelist of game data safe to serialize to the human browser.

This is a TRUSTED SERVER boundary: we can read both seats' stacks and the
human's private cards, but we must never serialize the future deck, burned
cards, training critic state or live opponent hole cards. The bot separately
receives its own InformationState directly from HUNL200Adapter.
"""
from __future__ import annotations

from typing import Any

from holdem_ai.core.schema import Terminal, Turn

from .action_adapter import legal_gui_actions


def project_human_view(
    adapter: Any, transition: Turn | Terminal,
    *, hero_seat: int, hand_number: int, revision: int,
    action_log: list[dict[str, Any]], completed_hands: int,
    cumulative_bb: float, stack_mode: str = "independent",
) -> dict[str, Any]:
    """Return a fresh JSON-serializable object with an intentional field list."""
    if hero_seat not in (0, 1):
        raise ValueError("HUNL has exactly two physical seats")
    engine = adapter.engine  # trusted server code only; never return this object
    over = isinstance(transition, Terminal)
    opponent = 1 - hero_seat
    street = engine.street
    result = engine.get_result() if over else None
    total_chips = 2 * adapter.stack_bb * adapter.big_blind
    if sum(engine.initial_stacks) != total_chips:
        raise AssertionError("invalid match chip sum")
    match_over = bool(stack_mode == "continuous" and over and min(result.final_stacks) == 0)
    winner_seat = (result.final_stacks.index(max(result.final_stacks)) if match_over else None)
    # The human always sees their own cards, even while waiting or after folding.
    view: dict[str, Any] = {
        "hand_number": hand_number,
        "revision": revision,
        "phase": "complete" if over else "playing",
        "game_type": "HUNL200",
        "stack_mode": stack_mode,
        "starting_stacks": list(engine.initial_stacks),
        "match_over": match_over,
        "winner_seat": winner_seat,
        "hero_seat": hero_seat,
        "button": engine.button,
        "small_blind": engine.small_blind,
        "big_blind": engine.big_blind,
        "street": street,
        "board": list(engine.board),
        "hero_cards": list(engine.hole_cards[hero_seat]),
        "seats": [
            {"seat": i, "role": "human" if i == hero_seat else "bot",
             "stack": int(result.final_stacks[i] if over else engine.stacks[i]), "street_bet": int(engine.street_bets[i]),
             "total_contribution": int(engine.contributions[i]),
             "status": str(engine.status[i])}
            for i in (0, 1)
        ],
        "pot": int(sum(engine.contributions)),
        "to_act": None if over else int(transition.player_id),
        "your_turn": not over and transition.player_id == hero_seat,
        "actions": [],
        "to_call": None,
        "min_raise_to": None,
        "max_raise_to": None,
        "can_raise_to": False,
        "action_log": [dict(event) for event in action_log],
        "completed_hands": completed_hands,
        "cumulative_bb": cumulative_bb,
        "bb_per_100": 100.0 * cumulative_bb / completed_hands if completed_hands else 0.0,
        "hand_result": None,
    }
    if not over and transition.player_id == hero_seat:
        obs = transition.observation
        # Verify that acting-player observation is the human's and not the bot's.
        if obs.player_id != hero_seat:
            raise AssertionError("human GUI cannot receive bot observations")
        raw = engine.observe(hero_seat)["raw_obs"]
        view["to_call"] = int(raw["to_call"])
        can_raise = 4 in engine.legal_actions
        view["can_raise_to"] = can_raise
        if can_raise:
            view["min_raise_to"] = int(raw["min_raise_to"])
            view["max_raise_to"] = int(raw["street_bets"][hero_seat]) + int(raw["stacks"][hero_seat])
        view["actions"] = legal_gui_actions(obs, engine)
    if over:
        assert result is not None
        if sum(result.payoffs_chips) != 0 or sum(result.final_stacks) != total_chips:
            raise AssertionError("chip conservation violated")
        is_showdown = not any(result.folded) and len(result.board) == 5
        view["hand_result"] = {
            "payoffs_chips": list(result.payoffs_chips),
            "payoffs_bb": list(result.payoffs_bb),
            "final_stacks": list(result.final_stacks),
            "awards": list(result.awards),
            "showdown": is_showdown,
            # Revealing the opponent is allowed only after a genuine showdown.
            "opponent_cards": list(engine.hole_cards[opponent]) if is_showdown else None,
        }
    return view
