"""Phase 17: matched-seed, variable-stack HUNL continuous-match evaluation.

This is an EVALUATION-ONLY harness. Phase 6 HUNL200Adapter.reset() and PPO/
Q-Boost training are never modified. A match carries stacks between hands,
rotates the button, and stops at bankruptcy or the hand horizon.

Only each player's InformationState reaches .act(); all other information is
used for post-action accounting, not policy decisions. No unrevealed cards are
included in reports. Seed matching does not create exact duplicate poker.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Protocol

from holdem_ai.core.schema import InformationState, Terminal, Turn
from holdem_ai.game_service.interactive_adapter import InteractiveHUNLAdapter

ACTION_NAMES = (
    "fold", "check", "call", "raise_half_pot", "raise_pot",
    "raise_double_pot", "all_in",
)
STACK_BANDS = ("lt_40", "40_to_99", "100_to_199", "ge_200")


class Policy(Protocol):
    def act(self, observation: InformationState, rng: random.Random) -> int: ...


@dataclass(frozen=True)
class ContinuousEvaluationConfig:
    groups: int = 8
    hands_per_match: int = 30
    candidate_stacks_bb: tuple[int, ...] = (50, 100, 200, 300, 350)
    total_stack_bb: int = 400
    big_blind: int = 100
    small_blind: int = 50
    seed: int = 17_000_000
    max_decisions_per_hand: int = 200

    def __post_init__(self) -> None:
        for name in ("groups", "hands_per_match", "total_stack_bb", "big_blind",
                     "small_blind", "seed", "max_decisions_per_hand"):
            value = getattr(self, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"{name} must be a valid integer")
        if self.groups < 2:
            raise ValueError("groups >=2 are needed for an uncertainty interval")
        if not 0 < self.small_blind < self.big_blind:
            raise ValueError("require 0 < small_blind < big_blind")
        if self.total_stack_bb != 400:
            # GUI-1.6 adapter fixes the match bank to 2 * 200BB. Reject rather
            # than silently vary an experimental constraint.
            raise ValueError("GUI-1.6 continuous adapter requires total_stack_bb=400")
        if not isinstance(self.candidate_stacks_bb, tuple) or not self.candidate_stacks_bb:
            raise ValueError("candidate_stacks_bb must be a nonempty tuple")
        if any(type(x) is not int or not 0 < x < self.total_stack_bb
               for x in self.candidate_stacks_bb):
            raise ValueError("each candidate stack must be 1..399 BB")
        if len(set(self.candidate_stacks_bb)) != len(self.candidate_stacks_bb):
            raise ValueError("duplicate stack scenarios are not permitted")


def _stack_band(effective_remaining_bb: float) -> str:
    if effective_remaining_bb < 40:
        return STACK_BANDS[0]
    if effective_remaining_bb < 100:
        return STACK_BANDS[1]
    if effective_remaining_bb < 200:
        return STACK_BANDS[2]
    return STACK_BANDS[3]


class _ActionTally:
    """Conditional action rates: numerator and legal-opportunity denominator."""

    def __init__(self) -> None:
        self.legal = {b: [0] * 7 for b in STACK_BANDS}
        self.taken = {b: [0] * 7 for b in STACK_BANDS}

    def record(self, observation: InformationState, index: int, *,
               remaining_stacks: tuple[int, int], big_blind: int) -> None:
        # The hand engine is inspected by the EVALUATOR for analytics only.
        # Never forward it (or hidden cards) to the acting policy.
        stacks = remaining_stacks
        if len(stacks) != 2:
            raise AssertionError("Phase 17 requires heads-up public observations")
        band = _stack_band(min(stacks) / big_blind)
        for action, allowed in enumerate(observation.legal_action_mask):
            if allowed:
                self.legal[band][action] += 1
        self.taken[band][index] += 1

    def merge(self, other: _ActionTally) -> None:
        for band in STACK_BANDS:
            for action in range(7):
                self.legal[band][action] += other.legal[band][action]
                self.taken[band][action] += other.taken[band][action]

    def report(self) -> dict:
        return {
            band: {
                "actions_taken": sum(self.taken[band]),
                "conditional_action_rate": {
                    name: (round(self.taken[band][i] / self.legal[band][i], 6)
                           if self.legal[band][i] else None)
                    for i, name in enumerate(ACTION_NAMES)
                },
                "legal_opportunities": dict(zip(ACTION_NAMES, self.legal[band])),
                "chosen_counts": dict(zip(ACTION_NAMES, self.taken[band])),
            }
            for band in STACK_BANDS
        }


def _group_interval(samples: list[float]) -> dict[str, float]:
    """One independent SHUFFLE SEED GROUP is the statistical replicate."""
    n = len(samples)
    if n < 2 or not all(math.isfinite(x) for x in samples):
        raise ValueError("need two or more finite seed-group estimates")
    mean = math.fsum(samples) / n
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in samples) / (n - 1))
    half = 1.96 * sd / math.sqrt(n)
    return {
        "mean_bb100": round(mean, 6),
        "ci95_lower_bb100": round(mean - half, 6),
        "ci95_upper_bb100": round(mean + half, 6),
        "std_seed_group_bb100": round(sd, 6),
        "standard_error_bb100": round(sd / math.sqrt(n), 6),
    }


def _play_match(
    candidate: Policy, opponent: Policy, *, cfg: ContinuousEvaluationConfig,
    candidate_seat: int, starting_button: int, candidate_stack_bb: int,
    group_seed: int, tally: _ActionTally,
) -> dict:
    """One match from a fixed endowment; no top-ups, no stack reset."""
    if candidate_seat not in (0, 1) or starting_button not in (0, 1):
        raise ValueError("seat/button must be 0 or 1")
    other = 1 - candidate_seat
    start = [0, 0]
    start[candidate_seat] = candidate_stack_bb * cfg.big_blind
    start[other] = (cfg.total_stack_bb - candidate_stack_bb) * cfg.big_blind
    stacks = tuple(start)
    rng_deck = random.Random(group_seed)
    # Independent streams per policy/seat/layout; *deck seeds*, not action
    # histories, are matched across the four seat/button layouts.
    rng_candidate = random.Random((group_seed << 4) + candidate_seat * 2 + starting_button + 1)
    rng_opponent = random.Random((group_seed << 4) + candidate_seat * 2 + starting_button + 100)
    hands = 0
    showdowns = 0
    hand_wins = 0
    hand_losses = 0
    hand_ties = 0
    decisions = 0
    for hand in range(cfg.hands_per_match):
        if min(stacks) <= 0:
            break
        adapter = InteractiveHUNLAdapter(
            big_blind=cfg.big_blind, small_blind=cfg.small_blind,
            stack_bb=cfg.total_stack_bb // 2,
            button=(starting_button + hand) % 2,
        )
        transition = adapter.reset_with_stacks(stacks, seed=rng_deck.getrandbits(64))
        count = 0
        while isinstance(transition, Turn):
            count += 1
            if count > cfg.max_decisions_per_hand:
                raise RuntimeError("max_decisions_per_hand reached; no truncated payoff was recorded")
            observation = transition.observation
            acting_seat = observation.player_id
            policy = candidate if acting_seat == candidate_seat else opponent
            rng = rng_candidate if acting_seat == candidate_seat else rng_opponent
            action = policy.act(observation, rng)
            # Strictly reject illegal actions rather than auto-repairing.
            observation.assert_legal(action)
            if acting_seat == candidate_seat:
                tally.record(observation, action,
                             remaining_stacks=tuple(adapter.engine.stacks),
                             big_blind=cfg.big_blind)
            transition = adapter.step(action)
            decisions += 1
        if not isinstance(transition, Terminal):
            raise AssertionError("expected a terminal transition")
        result = adapter.engine.get_result()
        updated = tuple(int(x) for x in result.final_stacks)
        if sum(updated) != cfg.total_stack_bb * cfg.big_blind:
            raise AssertionError("chip conservation violated")
        if any(updated[i] - stacks[i] != result.payoffs_chips[i] for i in (0, 1)):
            raise AssertionError("payout does not match start/end stacks")
        payoff = int(result.payoffs_chips[candidate_seat])
        hand_wins += payoff > 0
        hand_losses += payoff < 0
        hand_ties += payoff == 0
        stacks = updated
        hands += 1
        if len(result.board) == 5 and not any(result.folded):
            showdowns += 1
    if not hands:
        raise AssertionError("positive starting stacks must play at least one hand")
    net_bb = (stacks[candidate_seat] - start[candidate_seat]) / cfg.big_blind
    return {
        "hands": hands,
        "net_bb": net_bb,
        "final_candidate_bb": stacks[candidate_seat] / cfg.big_blind,
        "candidate_bust": stacks[candidate_seat] == 0,
        "opponent_bust": stacks[other] == 0,
        "candidate_ahead": net_bb > 0,
        "candidate_behind": net_bb < 0,
        "showdowns": showdowns,
        "hand_wins": hand_wins,
        "hand_losses": hand_losses,
        "hand_ties": hand_ties,
        "decisions": decisions,
    }


def evaluate_continuous_stacks(
    candidate: Policy, opponent: Policy, *,
    config: ContinuousEvaluationConfig | None = None,
    control: Policy | None = None,
    on_group=None,
) -> dict:
    """Compare a candidate (and optional control) under paired shuffle seeds.

    Groups are independent, each four layouts (seat 0/1, BTN 0/1).
    Within one group the bb/100 estimator pools chips and hands across layouts
    before the group-level CI; early-bust matches have fewer hands.
    """
    cfg = config or ContinuousEvaluationConfig()
    if candidate is opponent or (control is not None and control is opponent):
        raise ValueError("use separate policy objects for candidate/control and opponent")
    results = []
    for start_bb in cfg.candidate_stacks_bb:
        primary_tally = _ActionTally()
        control_tally = _ActionTally() if control is not None else None
        primary_blocks: list[dict] = []
        control_blocks: list[dict] = []
        primary_matches: list[dict] = []
        control_matches: list[dict] = []
        for group_idx in range(cfg.groups):
            group_seed = cfg.seed + group_idx
            for who, tally, matches, blocks in (
                (candidate, primary_tally, primary_matches, primary_blocks),
                (control, control_tally, control_matches, control_blocks),
            ):
                if who is None:
                    continue
                group_matches = [
                    _play_match(
                        who, opponent, cfg=cfg, candidate_seat=seat,
                        starting_button=button, candidate_stack_bb=start_bb,
                        group_seed=group_seed, tally=tally,
                    )
                    for button in (0, 1) for seat in (0, 1)
                ]
                matches.extend(group_matches)
                total_hands = sum(m["hands"] for m in group_matches)
                net = math.fsum(m["net_bb"] for m in group_matches)
                blocks.append({
                    "seed": group_seed,
                    "hands": total_hands,
                    "net_bb": round(net, 6),
                    "bb100": round(100.0 * net / total_hands, 12),
                })
            if on_group is not None:
                on_group(start_bb, group_idx + 1, cfg.groups)

        def summarize(blocks: list[dict], matches: list[dict], tally: _ActionTally) -> dict:
            return {
                **_group_interval([b["bb100"] for b in blocks]),
                "groups": cfg.groups,
                "matches": len(matches),
                "hands": sum(x["hands"] for x in matches),
                "mean_net_bb_per_match": round(math.fsum(x["net_bb"] for x in matches) / len(matches), 6),
                "mean_hands_per_match": round(math.fsum(x["hands"] for x in matches) / len(matches), 6),
                "match_profit_fraction": round(sum(x["candidate_ahead"] for x in matches) / len(matches), 6),
                "candidate_bust_fraction": round(sum(x["candidate_bust"] for x in matches) / len(matches), 6),
                "opponent_bust_fraction": round(sum(x["opponent_bust"] for x in matches) / len(matches), 6),
                "hand_win_fraction": round(sum(x["hand_wins"] for x in matches) / sum(x["hands"] for x in matches), 6),
                "hand_loss_fraction": round(sum(x["hand_losses"] for x in matches) / sum(x["hands"] for x in matches), 6),
                "hand_tie_fraction": round(sum(x["hand_ties"] for x in matches) / sum(x["hands"] for x in matches), 6),
                "showdown_fraction": round(sum(x["showdowns"] for x in matches) / sum(x["hands"] for x in matches), 6),
                "candidate_action_profile_by_effective_remaining_stack": tally.report(),
                "seed_groups": blocks,
            }

        scenario = {
            "candidate_start_bb": start_bb,
            "opponent_start_bb": cfg.total_stack_bb - start_bb,
            "candidate": summarize(primary_blocks, primary_matches, primary_tally),
        }
        if control is not None:
            assert control_tally is not None
            scenario["control"] = summarize(control_blocks, control_matches, control_tally)
            paired_deltas = [a["bb100"] - b["bb100"]
                             for a, b in zip(primary_blocks, control_blocks)]
            scenario["paired_delta"] = {
                **_group_interval(paired_deltas),
                "group_deltas_bb100": [round(v, 6) for v in paired_deltas],
            }
        results.append(scenario)
    return {
        "phase": 17,
        "protocol": "continuous_400bb_four_layout_seed_group",
        "config": {**asdict(cfg), "candidate_stacks_bb": list(cfg.candidate_stacks_bb)},
        "scenarios": results,
        "interpretation": (
            "Per-seed four-layout paired design; within-group deck reuse does NOT "
            "guarantee exact duplicate poker. Approximate normal 95% intervals on "
            "seed-group bb/100, not GTO exploitability. Early bankruptcies "
            "shorten match lengths. This is continuous match evaluation, NOT "
            "Phase16 independent 200BB-hand evaluation."
        ),
    }
