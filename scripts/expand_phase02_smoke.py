"""Expand E2: real-engine 2..6 person acting observation & action smoke.

This script does NOT train or certify a 6-max poker player. It only checks
that existing Phase 3 rules and E1/E2 actor observations cooperate.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from holdem_ai.expand.environment import DynamicTable
from holdem_ai.expand.information import Phase2TableAdapter


def run(*, hands_per_n: int, seed: int, output_json: str) -> dict:
    if type(hands_per_n) is not int or hands_per_n < 1 or type(seed) is not int or seed < 0:
        raise ValueError('positive hands-per-n and nonnegative seed required')
    from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine  # check dependency
    assert MultiPlayerHandEngine is not None
    results = []
    for target_n in range(2,7):
        table = DynamicTable(session_id=f'e02-smoke-n{target_n}')
        env = Phase2TableAdapter(table)
        for i in range(target_n):
            table.request_join(f'p{i}', 10000, preferred_seat=i)
        counted, setup_hands, actions, max_hist, missing = 0, 0, 0, 0, 0
        attempts = 0
        while counted < hands_per_n:
            if attempts > hands_per_n + 100:
                raise RuntimeError(f'failed to admit {target_n} players')
            hand = env.start_hand(seed + target_n*100000 + attempts)
            attempts += 1
            if hand is None:
                raise RuntimeError('not enough players for table')
            live_n = len(hand.participants)
            if live_n != target_n:
                setup_hands += 1
            steps = 0
            while table.current_player_id is not None:
                if steps >= 500:
                    raise RuntimeError('hand exceeded 500 actions')
                state = env.observe()
                x = env.encode()
                assert state.num_dealt_in == live_n
                assert state.hero_seat in dict(hand.participants)
                assert len(x.legal_mask) == 7 and len(x.seat_mask) == 6
                assert sum(x.seat_mask) == target_n
                assert len(x.global_features) > 100
                assert sum(x.legal_mask) >= 1
                max_hist = max(max_hist, len(state.action_history))
                chosen = 'call' if state.to_call_bb > 0 else 'check'
                if not state.legal_actions.allows(chosen):
                    missing += 1
                    raise AssertionError('engine rejects legal check/call')
                env.act_abstract(chosen)
                steps += 1
                actions += 1
            result = env.settle_hand()
            assert sum(result.payoffs_chips) == 0
            assert sum(result.final_stacks) == sum(result.starting_stacks)
            if live_n == target_n:
                counted += 1
        results.append({'players': target_n, 'hands': counted, 'entry_setup_hands': setup_hands,
                        'actor_decisions_including_setup': actions, 'max_history': max_hist,
                        'illegal_or_missing_check_call': missing})
    report = {'phase': 'Expand Phase 2', 'kind':'real_engine_info_state_smoke',
              'total_target_hands': sum(x['hands'] for x in results),
              'total_setup_hands': sum(x['entry_setup_hands'] for x in results),
              'training_updates': 0,'seed':seed,'results':results}
    dest=Path(output_json)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open('x',encoding='utf-8') as fh:
        json.dump(report,fh,ensure_ascii=False,indent=2)
        fh.write('\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return report


def main():
    p = argparse.ArgumentParser(description='Expand Phase 2 2..6 player info-state smoke; no learning')
    p.add_argument('--hands-per-n',type=int,default=10)
    p.add_argument('--seed',type=int,default=180201)
    p.add_argument('--output-json',default='runs/expand_phase02/smoke_10.json')
    args=p.parse_args()
    run(hands_per_n=args.hands_per_n,seed=args.seed,output_json=args.output_json)


if __name__=='__main__':
    main()
