"""Summarize on-policy HUNL terminal rewards and termination actions per update.

No opponent private cards or full engine state is used. Showdown is NOT
inferred solely from a non-fold ending; report non-fold explicitly.
"""
from __future__ import annotations

import math
from collections import Counter

from holdem_ai.training.ppo import TrajectoryRows


def summarize_rollout_rewards(rows: TrajectoryRows) -> dict:
    payoffs = [float(r[0]) for r in rows.hand_rewards]
    if len(payoffs) != rows.hands or not payoffs or any(not math.isfinite(x) for x in payoffs):
        raise ValueError("invalid hand rewards")
    n = len(payoffs)
    mean = math.fsum(payoffs) / n
    var = math.fsum((v - mean) ** 2 for v in payoffs) / n
    out = {
        "hands": n,
        "seat0_mean_reward_bb": round(mean, 6),
        "seat0_reward_std_bb": round(math.sqrt(var), 6),
        "mean_absolute_terminal_reward_bb": round(math.fsum(abs(v) for v in payoffs) / n, 6),
        "min_seat0_reward_bb": round(min(payoffs), 6),
        "max_seat0_reward_bb": round(max(payoffs), 6),
        "decisions_per_hand_mean": (round(sum(rows.decisions_per_hand) / n, 6)
                                     if rows.decisions_per_hand else None),
        "fold_terminated_hands": None,
        "fold_termination_fraction": None,
        "non_fold_terminated_hands": None,
        "showdown_fraction": None,  # Not observable from terminal action alone.
    }
    if rows.terminal_actions:
        counts = Counter(rows.terminal_actions)
        fold = counts[0]
        out["fold_terminated_hands"] = fold
        out["fold_termination_fraction"] = round(fold / n, 6)
        out["non_fold_terminated_hands"] = n - fold
    return out
