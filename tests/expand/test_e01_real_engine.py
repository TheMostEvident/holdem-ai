"""These require the user's real Base Phase 3 modules (skipped in standalone patch)."""
import pytest
from holdem_ai.expand.environment import Action, DynamicTable

pytest.importorskip('holdem_ai.environment.multi_player_engine')
from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine
from holdem_ai.environment.pot_manager import build_side_pots, settle_pots


def to_showdown(table, *, max_decisions=200):
    for _ in range(max_decisions):
        pid = table.current_player_id
        if pid is None:
            return table.settle_hand()
        to_call = table.observe_current()['raw_obs']['to_call']
        table.act(pid, Action('call' if to_call else 'check'))
    raise AssertionError('hand did not terminate')


def test_e01_real_engine_two_through_six_participants():
    t = DynamicTable()
    for i in range(6):
        t.request_join(f'p{i}', 10000, i)
    saw = set()
    for hand_num in range(40):
        handle = t.start_hand(hand_num + 1)
        assert handle is not None
        n = len(handle.participants)
        saw.add(n)
        assert len({p for _, p in handle.participants}) == n
        assert 2 <= n <= 6
        assert handle.button_seat in dict(handle.participants)
        assert handle.big_blind_seat in dict(handle.participants)
        settlement = to_showdown(t)
        assert sum(settlement.payoffs_chips) == 0
        assert sum(settlement.final_stacks) == sum(settlement.starting_stacks)
        assert not t.snapshot().hand_in_progress
    assert saw == {2, 3, 4, 5, 6}


def test_e01_real_engine_short_allin_sidepots():
    engine = MultiPlayerHandEngine(
        stacks=(20000, 9000, 3000, 0, 0, 0), button=0,
        small_blind=50, big_blind=100, seed=84000)
    engine.reset()
    assert engine.current_player == 0
    engine.step(4)  # button's full-stack raise
    while not engine.is_over():
        engine.step(1)  # short all-in call permitted by Phase 3 engine
    result = engine.get_result()
    assert sum(result.final_stacks) == 32000
    assert sum(result.payoffs_chips) == 0
    assert sum(result.contributions) == sum(result.awards)
    assert result.contributions[1] == 9000
    assert result.contributions[2] == 3000


def test_e01_reference_pots_odd_chips_and_folded_money():
    breakdown = build_side_pots((500, 200, 100), (True, False, False))
    assert breakdown.uncalled_returns == (300, 0, 0)
    assert sum(p.amount for p in breakdown.pots) == 500
    awards = settle_pots(breakdown, (None, (2,), (1,)), button=0)
    assert awards == (300, 500, 0)
    odd = build_side_pots((101, 101, 101), (False, True, False))
    awards = settle_pots(odd, ((7,), None, (7,)), button=0)
    assert awards == (151, 0, 152)


def test_e01_real_engine_replay_deterministic(tmp_path):
    logfile = tmp_path / 'real.jsonl'
    t = DynamicTable(session_id='replay', journal_path=logfile)
    t.request_join('x', 12000)
    t.request_join('y', 20000)
    for seed in range(10, 15):
        t.start_hand(seed)
        to_showdown(t)
    from holdem_ai.expand.environment import replay_journal
    restored = replay_journal(logfile)
    assert restored.ledger.records == t.ledger.records
    assert restored.snapshot() == t.snapshot()
