"""Single-table tournament smoke audit; run `uv run python scripts/audit_tournament.py`."""
from __future__ import annotations

import argparse
from collections import Counter

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.tournament_env import TournamentEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', type=int, default=6)
    parser.add_argument('--tournaments', type=int, default=100)
    parser.add_argument('--stack', type=int, default=1000)
    parser.add_argument('--hands-per-level', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 2 <= args.players <= 6 or args.tournaments <= 0 or args.stack <= 0:
        parser.error('players must be 2..6; tournaments and stack must be positive')
    wins = Counter()
    durations = []
    total_hands = 0
    total_eliminations = 0
    for t in range(args.tournaments):
        env = TournamentEnv(num_players=args.players, starting_stack=args.stack,
                            hands_per_level=args.hands_per_level, seed=args.seed + t)
        agents = [RandomAgent(args.seed + 1_000 * t + i) for i in range(args.players)]
        outcome = env.play_tournament(agents)
        assert sum(outcome.final_stacks) == args.players * args.stack
        assert outcome.final_stacks[outcome.winner] == args.players * args.stack
        assert sum(1 for x in outcome.final_stacks if x > 0) == 1
        assert len(outcome.history) == outcome.hands_played
        assert sum(len(record.eliminated) for record in outcome.history) == args.players - 1
        assert all(sum(r.hand_result.payoffs_chips) == 0 for r in outcome.history)
        wins[outcome.winner] += 1
        durations.append(outcome.hands_played)
        total_hands += outcome.hands_played
        total_eliminations += args.players - 1
    print(f'PASS: {args.tournaments}/{args.tournaments} tournaments, {args.players} seats, seed={args.seed}')
    print(f'Hands: {total_hands} | Mean hands/tournament: {sum(durations) / len(durations):.2f}')
    print(f'Eliminations: {total_eliminations} | Winner seats: {dict(sorted(wins.items()))}')
    print('Scope: termination, stack continuity, seat elimination and chip conservation.')
    print('Not a complete rules/side-pot/ICM certification.')


if __name__ == '__main__':
    main()
