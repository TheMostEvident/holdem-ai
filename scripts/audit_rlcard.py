"""Run against installed RLCard: uv run python scripts/audit_rlcard.py --hands 100.

Exits 1 on rule violations, distinguishing strict audit from green pytest unit
suite. Does not change the original RLCard engine or the Phase-1 HandEngine.
"""

import argparse
import sys

import rlcard

from holdem_ai.agents.random_agent import RandomAgent
from holdem_ai.environment.hand_engine import HandEngine
from holdem_ai.environment.rules_audit import audit_preflop, summarize


def main() -> int:
    parser = argparse.ArgumentParser(description='Diagnostic audit of RLCard HUNL, not a repair tool')
    parser.add_argument('--hands', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.hands < 1:
        parser.error('--hands must be positive')

    raw = rlcard.make('no-limit-holdem', config={
        'seed': args.seed, 'game_num_players': 2, 'chips_for_each': 200,
    })
    raw.reset()
    checks = audit_preflop(raw.game)
    print('=== RLCard 1.2.0: STRICT HEADS-UP PREFLOP AUDIT ===')
    print(summarize(checks))

    engine = HandEngine(starting_stack=200, seed=args.seed)
    agents = (RandomAgent(seed=args.seed), RandomAgent(seed=args.seed + 1))
    failures = 0
    for idx in range(args.hands):
        try:
            result = engine.play_hand(agents)
            if abs(sum(result.payoffs_chips)) > 1e-6:
                failures += 1
        except Exception as exc:
            failures += 1
            print(f'ERROR hand={idx + 1}: {type(exc).__name__}: {exc}')
    print(f'PAYOFF: {args.hands - failures}/{args.hands} hands passed zero-sum/runtime check')
    print('Note: payoff checks do NOT establish correct betting, showdown or side-pot rules.')
    issues = sum(not check.passed for check in checks) + failures
    print(f'STRICT RESULT: {"PASS" if issues == 0 else "FAIL"} ({issues} checks/issues)')
    return 1 if issues else 0


if __name__ == '__main__':
    sys.exit(main())
