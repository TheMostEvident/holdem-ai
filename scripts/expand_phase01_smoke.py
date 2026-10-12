"""Real-engine Expand Phase 1 acceptance smoke: each table target 2..6 seats.

Usage from repository root:
  uv run --extra dev python -m scripts.expand_phase01_smoke --hands-per-n 1000
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import time
from holdem_ai.expand.environment import Action, DynamicTable, TableConfig


def run(*, hands_per_n: int, seed: int, output: str | None = None,
        journal_directory: str | None = None) -> dict:
    if type(hands_per_n) is not int or hands_per_n <= 0:
        raise ValueError('hands-per-n must be a positive integer')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be nonnegative integer')
    rng = random.Random(seed)
    stats = []
    start = time.monotonic()
    for max_seated in range(2, 7):
        filename = Path(journal_directory) / f'e01_{max_seated}.jsonl' if journal_directory else None
        table = DynamicTable(TableConfig(), session_id=f'e01_{seed}_{max_seated}', journal_path=filename)
        next_id = 0
        hands = 0  # hands played with exactly max_seated participants
        warmup_hands = 0
        decisions = 0
        busts = 0
        raises = 0
        joins = 0
        attempts = 0
        while hands < hands_per_n:
            attempts += 1
            if attempts > hands_per_n * 30 + 500:
                raise RuntimeError(f'could not open {hands_per_n} hands for {max_seated} seats')
            occupied = sum(s.player_id is not None for s in table.snapshot().seats)
            for _ in range(max_seated - occupied):
                pid = f'p{next_id:08d}'
                next_id += 1
                buyin = rng.randint(20, 300) * 100
                table.request_join(pid, buyin)
                joins += 1
            hand_seed = rng.getrandbits(63)
            hand = table.start_hand(hand_seed)
            if hand is None:
                continue
            for _ in range(250):
                player_id = table.current_player_id
                if player_id is None:
                    break
                raw = table.observe_current()
                legal = raw['legal_actions']
                to_call = raw['raw_obs']['to_call']
                draw = rng.random()
                if 4 in legal and draw < 0.008:
                    action = Action('all_in')
                    raises += 1
                elif 3 in legal and draw < 0.06:
                    action = Action('raise_pot')
                    raises += 1
                elif 2 in legal and draw < 0.10:
                    action = Action('raise_half_pot')
                    raises += 1
                elif to_call and draw < 0.18:
                    action = Action('fold')
                else:
                    action = Action('call' if to_call else 'check')
                table.act(player_id, action)
                decisions += 1
            else:
                raise RuntimeError('250 decisions exceeded in one hand')
            settled = table.settle_hand()
            if sum(settled.payoffs_chips) != 0 or sum(settled.final_stacks) != sum(settled.starting_stacks):
                raise AssertionError('hand chip invariant violated')
            busts += len(settled.busted_player_ids)
            if len(hand.participants) == max_seated:
                hands += 1
            else:
                warmup_hands += 1
        stats.append({
            'target_seats': max_seated, 'hands': hands, 'warmup_hands': warmup_hands,
            'all_hands': hands + warmup_hands, 'decisions': decisions,
            'join_requests': joins, 'busts': busts, 'raises': raises,
            'final_table_chips': table.snapshot().total_chips,
            'buyins_minus_cashouts': table.total_buyins - table.total_cashouts,
        })
    result = {'expand_phase': 1, 'mode': 'cash', 'hands_per_target': hands_per_n,
              'targets': [2, 3, 4, 5, 6], 'target_hands': sum(x['hands'] for x in stats),
              'total_hands': sum(x['all_hands'] for x in stats),
              'seed': seed, 'elapsed_seconds': round(time.monotonic()-start, 3),
              'source': 'real Base Phase 3 MultiPlayerHandEngine', 'results': stats}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Never overwrite prior experiments.
        with path.open('x', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write('\n')
    return result


def main():
    p = argparse.ArgumentParser(description='Expand Phase 1 real 2..6 player hand smoke')
    p.add_argument('--hands-per-n', type=int, default=10)
    p.add_argument('--seed', type=int, default=170001)
    p.add_argument('--output-json', default=None)
    p.add_argument('--journal-directory', default=None)
    args = p.parse_args()
    print(json.dumps(run(hands_per_n=args.hands_per_n, seed=args.seed,
                         output=args.output_json, journal_directory=args.journal_directory),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
