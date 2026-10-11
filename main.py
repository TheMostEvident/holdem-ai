"""CLI smoke test: two RandomAgents play HUNL via the RLCard adapter."""

import argparse

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine


def main() -> None:
    parser = argparse.ArgumentParser(description='holdem-ai: first RLCard HUNL integration')
    parser.add_argument('--hands', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stack', type=int, default=200)
    args = parser.parse_args()
    if args.hands < 1:
        parser.error('--hands must be positive')

    engine = HandEngine(starting_stack=args.stack, seed=args.seed)
    agents = (RandomAgent(seed=args.seed), RandomAgent(seed=args.seed + 1))
    total_bb = [0.0, 0.0]

    print(f'RLCard HUNL: {args.hands} hands, {args.stack} chips per player, BB={engine.big_blind}')
    for hand_index in range(1, args.hands + 1):
        result = engine.play_hand(agents)
        for i in range(2):
            total_bb[i] += result.payoffs_bb[i]
        print(
            f'Hand {hand_index:>3}: '
            f'P0={result.payoffs_bb[0]:+7.2f} BB, '
            f'P1={result.payoffs_bb[1]:+7.2f} BB, '
            f'actions={len(result.action_history)}'
        )
    print(f'P0 mean performance: {100 * total_bb[0] / args.hands:+.2f} bb/100')
    print('Note: RLCard HUNL rules are not yet independently validated.')


if __name__ == '__main__':
    main()
