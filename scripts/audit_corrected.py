"""uv run python scripts/audit_corrected.py --hands 100 --seed 42"""
import argparse

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine
from holdem_ai.environment.rules_audit import audit_preflop, summarize


def main():
    parser = argparse.ArgumentParser(description='Strict rule checks for corrected HUNL adapter')
    parser.add_argument('--hands', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.hands < 1:
        parser.error('--hands must be positive')
    engine = HandEngine(seed=args.seed)
    for button in (0, 1):
        engine.env.game.dealer_id = button
        engine.env.reset()
        checks = audit_preflop(engine.env.game)
        print(f'BUTTON={button}')
        print(summarize(checks))
        if not all(c.passed for c in checks):
            return 1
    for button in (0, 1):
        check_engine = HandEngine(seed=args.seed, button=button)
        _, player = check_engine.reset()
        if player != button:
            raise AssertionError('Button must act first preflop')
        check_engine.step(1)
        check_engine.step(1)
        assert check_engine.env.game.stage.name == 'FLOP'
        assert check_engine.current_player == 1 - button
        print(f'PASS bb_acts_first_postflop (button={button})')

    agents = (RandomAgent(seed=args.seed), RandomAgent(seed=args.seed + 1))
    for h in range(args.hands):
        result = engine.play_hand(agents)
        if sum(result.payoffs_chips) != 0:
            raise AssertionError(f'Zero-sum violation on hand {h+1}')
    print(f'PASS {args.hands}/{args.hands} random hands: zero-sum and no runtime error')
    print('STRICT RESULT: PASS (covered scenarios only; not a complete poker rules certification)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
