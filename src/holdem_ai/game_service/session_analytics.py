"""GUI-1.5: bounded, hero-visible completed-hand ledger and online metrics.

Only receive the already-sanitized project_human_view() result. The analytics
layer never accesses deck, critic state or the opponent's unrevealed cards.
Statistics cover ALL completed hands; detailed records are capped in memory.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import math
from typing import Any


class SessionAnalytics:
    def __init__(self, *, history_limit: int = 500) -> None:
        if type(history_limit) is not int or history_limit < 1:
            raise ValueError('history_limit must be a positive integer')
        self._records: deque[dict[str, Any]] = deque(maxlen=history_limit)
        self.count = 0
        self.wins = 0
        self.losses = 0
        self.break_even = 0
        self.showdowns = 0
        self.total_chips = 0
        self.total_bb = 0.0
        self.mean_bb = 0.0
        self.m2_bb = 0.0
        self.peak_bb = 0.0
        self.max_drawdown_bb = 0.0

    def append_completed(self, human_view: dict[str, Any]) -> None:
        """Accept exactly one terminal, sanitized view per hand."""
        result = human_view.get('hand_result')
        if human_view.get('phase') != 'complete' or result is None:
            raise ValueError('analytics only accepts completed hands')
        hero = human_view['hero_seat']
        chips = int(result['payoffs_chips'][hero])
        bb = float(result['payoffs_bb'][hero])
        if not math.isfinite(bb) or not math.isclose(bb * human_view['big_blind'], chips, abs_tol=1e-5):
            raise ValueError('inconsistent payoff units')
        if result['opponent_cards'] is not None and not result['showdown']:
            raise ValueError('folded opponent cards must remain hidden')
        expected_hand = self.count + 1
        if human_view['hand_number'] != expected_hand:
            raise ValueError('completed hand numbers must be contiguous')
        outcome = 'win' if chips > 0 else 'loss' if chips < 0 else 'even'
        self.count += 1
        self.wins += outcome == 'win'
        self.losses += outcome == 'loss'
        self.break_even += outcome == 'even'
        self.showdowns += bool(result['showdown'])
        self.total_chips += chips
        self.total_bb += bb
        difference = bb - self.mean_bb
        self.mean_bb += difference / self.count
        self.m2_bb += difference * (bb - self.mean_bb)
        self.peak_bb = max(self.peak_bb, self.total_bb)
        self.max_drawdown_bb = max(self.max_drawdown_bb, self.peak_bb - self.total_bb)
        record = {
            'hand_number': self.count,
            'starting_stacks': list(human_view.get('starting_stacks', [200 * human_view['big_blind']] * 2)),
            'button': human_view['button'],
            'street': human_view['street'],
            'big_blind': human_view['big_blind'],
            'small_blind': human_view['small_blind'],
            'hero_cards': list(human_view['hero_cards']),
            'board': list(human_view['board']),
            'action_log': deepcopy(human_view['action_log']),
            'hero_net_chips': chips,
            'hero_net_bb': bb,
            'cumulative_bb': self.total_bb,
            'outcome': outcome,
            'result': deepcopy(result),
        }
        self._records.append(record)

    def summary(self) -> dict[str, Any]:
        n = self.count
        # A simple normal-approximation interval, not a guarantee of independent
        # samples or statistical significance against a general opponent pool.
        deviation = math.sqrt(max(0.0, self.m2_bb) / (n - 1)) if n > 1 else None
        radius = 1.96 * deviation / math.sqrt(n) * 100 if deviation is not None else None
        return {
            'completed_hands': n,
            'wins': self.wins,
            'losses': self.losses,
            'break_even': self.break_even,
            'showdowns': self.showdowns,
            'net_chips': self.total_chips,
            'net_bb': self.total_bb,
            'bb_per_100': 100 * self.total_bb / n if n else 0.0,
            'mean_bb_per_hand': self.mean_bb if n else 0.0,
            'std_bb_per_hand': deviation,
            'ci95_lower_bb100': 100 * self.mean_bb - radius if radius is not None else None,
            'ci95_upper_bb100': 100 * self.mean_bb + radius if radius is not None else None,
            'max_drawdown_bb': self.max_drawdown_bb,
            'retained_hand_records': len(self._records),
            'records_truncated': n > len(self._records),
            'note': ('Descriptive local-session results. Approximate hand-level 95% CI assumes '
                     'independent hands; not a GTO, exploitability or population-strength estimate.'),
        }

    def history(self, *, offset: int = 0, limit: int = 50) -> dict[str, Any]:
        if type(offset) is not int or offset < 0:
            raise ValueError('offset must be a nonnegative integer')
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('limit must be between 1 and 100')
        records = list(reversed(self._records))
        items = deepcopy(records[offset:offset + limit])
        return {
            'items': items,
            'offset': offset,
            'limit': limit,
            'total': len(records),
            'completed_hands': self.count,
            'has_more': offset + len(items) < len(records),
            'records_truncated': self.count > len(records),
        }
