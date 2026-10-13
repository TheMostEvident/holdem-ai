"""Show conditional preflop heuristic raise mix (NOT actual in-game raise frequency).

Execute from project root:
  uv run --extra gui python -m scripts.tournament_env131_diagnose
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations

from holdem_ai.tournament_service.improved_bot import ImprovedTournamentBot, PROFILES


def main() -> None:
    deck = [f'{s}{r}' for s in 'shdc' for r in '23456789TJQKA']
    print('Preflop heuristic check: 1,326 possible 2-card combos')
    print('Assumption: 6 active seats, call 10, pot 45, all raises legal, stack 20,000')
    print('These rates are *conditional policy choices*, NOT measured game win/raise rates.')
    for profile in PROFILES:
        bot = ImprovedTournamentBot(profile, seed=2026)
        counts: Counter[int] = Counter()
        for hole in combinations(deck, 2):
            obs = dict(legal_actions={a: None for a in range(5)}, raw_obs={
                'hand': list(hole), 'public_cards': [], 'player_id': 2,
                'to_call': 10, 'pot': 45, 'stacks': [20000]*6, 'status':['active']*6,
            })
            counts[bot.select_action(obs)] += 1
        pct = 100 * (counts[2]+counts[3])/1326
        print(f'{profile:<21} Fold={counts[0]:4} Check/Call={counts[1]:4} '
              f'HalfPot={counts[2]:4} Pot={counts[3]:4} AllIn={counts[4]:4}  '
              f'RaiseShare={pct:5.1f}%')
    print('Seat 2 remains an UNTRAINED Tournament Agent placeholder using a heuristic fallback.')


if __name__ == '__main__':
    main()
