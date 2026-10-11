"""Run one 2-6 seat tournament (random-agent baseline) and show eliminations."""
from __future__ import annotations

import argparse

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.tournament_env import TournamentEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', type=int, default=6)
    parser.add_argument('--stack', type=int, default=1000)
    parser.add_argument('--hands-per-level', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    env = TournamentEnv(num_players=args.players, starting_stack=args.stack,
                        hands_per_level=args.hands_per_level, seed=args.seed)
    agents = [RandomAgent(args.seed + i) for i in range(args.players)]
    result = env.play_tournament(agents)
    print(f'Tournament complete: champion=P{result.winner}, hands={result.hands_played}')
    for record in result.history:
        if args.verbose or record.eliminated:
            print(f'Hand {record.hand_number:3d} | BTN=P{record.button} '
                  f'blinds={record.blinds.small_blind}/{record.blinds.big_blind} '
                  f'| stacks={record.stacks_after} '
                  f'| busted={list(record.eliminated)}')
    print(f'Final ranks (seat order): {result.ranks}')
    print(f'Final stacks: {result.final_stacks}')
    print(f'Win rewards: {result.rewards}')


if __name__ == '__main__':
    main()
