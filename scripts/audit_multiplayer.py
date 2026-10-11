"""Quick random-benchmark smoke audit. Does not certify poker rules."""
from __future__ import annotations

import argparse
from collections import Counter

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', type=int, default=6)
    parser.add_argument('--hands', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 2 <= args.players <= 6 or args.hands <= 0:
        parser.error('players must be 2..6 and hands > 0')
    showdown = 0
    wins = Counter()
    total_actions = 0
    for h in range(args.hands):
        engine = MultiPlayerHandEngine(num_players=args.players, seed=args.seed + h,
                                       button=h % args.players)
        agents = [RandomAgent(args.seed + h * 100 + i) for i in range(args.players)]
        result = engine.play_hand(agents)
        assert sum(result.payoffs_chips) == 0
        assert sum(result.final_stacks) == args.players * 200
        assert min(result.final_stacks) >= 0
        assert sum(result.contributions) == sum(result.awards)
        showdown += len(result.board) == 5
        total_actions += len(result.action_history)
        if max(result.payoffs_chips) > 0:
            for i, p in enumerate(result.payoffs_chips):
                if p == max(result.payoffs_chips):
                    wins[i] += 1
    print(f'PASS: {args.hands}/{args.hands} hands, {args.players} seats, seed={args.seed}')
    print(f'Showdowns: {showdown} | Total actions: {total_actions}')
    print(f'Max-payoff seat counts (ties counted for all): {dict(wins)}')
    print('Scope: chip conservation and runtime smoke check; not a full rules certification.')


if __name__ == '__main__':
    main()
