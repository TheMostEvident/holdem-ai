"""Tournament ENV 1.3: four reproducible, observation-only heuristic opponents.

No neural weights, RL training, opponent hole cards, Monte Carlo, or claims of
improved win-rate. ``improved`` names the user-facing development slot; actual
performance must be benchmarked separately against frozen reference bots.

Only Phase-3 public observe(actor) plus the actor's own private hand is read.
Action IDs are the native five-action tournament abstraction (0..4).
"""
from __future__ import annotations

from itertools import combinations
import random
from typing import Any, Mapping

from holdem_ai.environment.hand_evaluator import parse_card, rank_five

PROFILES: tuple[str, ...] = (
    "tight_aggressive", "balanced_value", "pot_odds", "controlled_pressure",
)


def _preflop_strength(hole: tuple[str, str]) -> float:
    (_, x), (_, y) = (parse_card(card) for card in hole)
    high, low = max(x, y), min(x, y)
    if x == y:
        return min(0.99, 0.54 + (high - 2) / 12.0 * 0.43)
    suited = hole[0][0].lower() == hole[1][0].lower()
    gap = high - low
    high_value = (high - 2) / 12.0
    low_value = (low - 2) / 12.0
    score = 0.13 + 0.40 * high_value + 0.24 * low_value
    if suited:
        score += 0.07
    if gap <= 2:
        score += 0.07 if gap == 1 else 0.035
    if high == 14:
        score += 0.04
    return max(0.0, min(0.94, score))


def _postflop_strength(hole: tuple[str, str], board: tuple[str, ...]) -> float:
    cards = (*hole, *board)
    if len(cards) < 5:
        raise ValueError("flop must contain at least three board cards")
    best = max(rank_five(combo) for combo in combinations(cards, 5))
    category = best[0]
    # Coarse made-hand heuristic, not showdown equity against multiple rivals.
    base = (0.18, 0.49, 0.70, 0.80, 0.88, 0.92, 0.96, 0.985, 0.997)[category]
    if category == 0:
        base += 0.13 * ((best[1] - 2) / 12.0)
    elif category == 1:
        base += 0.09 * ((best[1] - 2) / 12.0)
    if len(board) == 5:
        # Common board-only straights, flushes, etc are not exclusive strength.
        board_rank = rank_five(board)
        if best == board_rank:
            base = min(base, 0.49)
    # Transparent, cheap heuristic draws; no hidden/deck-peeking equity.
    suits = [parse_card(c)[0] for c in cards]
    if len(board) < 5 and max(suits.count(s) for s in set(suits)) >= 4:
        base += 0.07
    return min(0.999, base)


class ImprovedTournamentBot:
    """Native legal-action rule policy; four distinguishable playing styles."""

    def __init__(self, profile: str, *, seed: int | None = None):
        if profile not in PROFILES:
            raise ValueError(f"unknown improved bot profile: {profile}")
        self.profile = profile
        self._rng = random.Random(seed)

    def select_action(self, observation: Mapping[str, Any]) -> int:
        legal = tuple(int(a) for a in observation["legal_actions"])
        if not legal or any(a not in range(5) for a in legal):
            raise ValueError("tournament observation has invalid legal action indices")
        raw = observation["raw_obs"]
        hole = tuple(raw["hand"])
        board = tuple(raw["public_cards"])
        if len(hole) != 2 or len(board) not in (0, 3, 4, 5):
            raise ValueError("acting-player hand or board is malformed")
        strength = _preflop_strength(hole) if not board else _postflop_strength(hole, board)
        stack = int(raw["stacks"][raw["player_id"]])
        call = min(stack, max(0, int(raw["to_call"])))
        pot = max(0, int(raw["pot"]))
        pot_odds = call / max(1, pot + call)
        live_rivals = sum(
            1 for i, status in enumerate(raw["status"])
            if i != raw["player_id"] and status not in ("out", "folded")
        )
        # More multiway opponents -> stronger hands needed for value betting.
        multiway = min(0.12, max(0, live_rivals - 1) * 0.022)
        # 1.3.1: The 1.3 thresholds over-folded/over-called against 5 live
        # rivals. These are *heuristic* activation thresholds, NOT equity or
        # empirical win-rate estimates. Tune via separate seeded tournaments.
        cfg = {
            "tight_aggressive": (0.675, 0.49, 0.12),
            "balanced_value": (0.565, 0.40, 0.16),
            "pot_odds": (0.635, 0.34, 0.06),
            "controlled_pressure": (0.495, 0.395, 0.22),
        }
        raise_threshold, continue_threshold, aggression = cfg[self.profile]
        jitter = (self._rng.random() - 0.5) * 0.035
        value = strength + jitter - multiway
        # Short effective stacks permit a value-oriented shove, but never a
        # random automatic 200BB all-in simply because action 4 is legal.
        if 4 in legal and stack <= max(1, pot) and value >= 0.87:
            return 4
        # Important: half-pot can be *illegal* while a pot raise is legal
        # (e.g. minimum legal raise > half-pot). Never gate pot-sized raises
        # on the availability of action 2.
        if value >= raise_threshold and (2 in legal or 3 in legal):
            if 3 in legal and (
                2 not in legal or value >= raise_threshold + 0.14 or
                (self.profile == "controlled_pressure" and self._rng.random() < aggression)
            ):
                return 3
            if 2 in legal:
                return 2
        if 1 in legal and (call == 0 or value >= continue_threshold or
                           (self.profile == "pot_odds" and value + 0.08 >= pot_odds)):
            return 1
        if 0 in legal:
            return 0
        if 1 in legal:
            return 1
        # Engine-only fallback: every returned action must belong to legal.
        return legal[0]
