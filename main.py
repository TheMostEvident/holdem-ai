"""Play independent heads-up hands with corrected positions by default."""
import argparse

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine


def main():
    parser = argparse.ArgumentParser(description='holdem-ai corrected HUNL smoke test')
    parser.add_argument('--hands', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--stack', type=int, default=200)
    parser.add_argument('--rules', choices=['corrected', 'upstream'], default='corrected')
    args = parser.parse_args()
    if args.hands < 1:
        parser.error('--hands must be positive')
    engine = HandEngine(starting_stack=args.stack, seed=args.seed, rules_mode=args.rules)
    agents = (RandomAgent(seed=args.seed), RandomAgent(seed=args.seed + 1))
    total = 0.0
    print(f'HUNL rules={args.rules}, hands={args.hands}, stack={args.stack}, BB={engine.big_blind}')
    for hand in range(1, args.hands + 1):
        result = engine.play_hand(agents)
        total += result.payoffs_bb[0]
        print(f'Hand {hand:>3}: P0={result.payoffs_bb[0]:+7.2f} BB, '
              f'P1={result.payoffs_bb[1]:+7.2f} BB, actions={len(result.action_history)}')
    print(f'P0 observed bb/100: {100 * total / args.hands:+.2f} (not a performance estimate for tiny samples)')
    print('This is a development baseline; not certified for real-money or tournament use.')


if __name__ == '__main__':
    main()
