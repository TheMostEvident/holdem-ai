import json
import pytest

from conftest import FakeHandEngine, BustHandEngine
from holdem_ai.expand.environment import (TableConfig, CashRuleSet, DynamicTable,
                                          Action, replay_journal, TournamentRuleSet)
import holdem_ai.expand.environment.table_manager as tm


@pytest.fixture(autouse=True)
def simulate_base_seats(monkeypatch):
    """Base seat order verified separately in real engine integration tests."""
    def blinds(button, participants):
        ordered = [p for p in ((button + k) % 6 for k in range(1, 7)) if p in participants]
        return (button, ordered[0]) if len(participants) == 2 else (ordered[0], ordered[1])
    monkeypatch.setattr(tm, 'blind_seats', blinds)


def make_table(tmp_path=None, cls=FakeHandEngine):
    path = tmp_path / 'ledger.jsonl' if tmp_path else None
    return DynamicTable(engine_factory=cls, journal_path=path, session_id='t')


def join_and_start(table, *players, seed=1):
    for name in players:
        table.request_join(name, 10000)
    handle = table.start_hand(seed)
    assert handle is not None
    return handle


def finish(table):
    table.act(table.current_player_id, Action('check'))
    return table.settle_hand()


def test_config_rejects_unsupported_modes_and_amounts():
    cfg = TableConfig()
    assert cfg.big_blind == 100
    assert CashRuleSet(cfg).min_buyin == 2000
    assert CashRuleSet(cfg).max_buyin == 30000
    with pytest.raises(ValueError): TableConfig(max_seats=5)
    with pytest.raises(ValueError): TableConfig(rake_bps=1)
    with pytest.raises(ValueError): TableConfig(join_rule='post_to_enter')
    with pytest.raises(ValueError): CashRuleSet(cfg).validate_buyin(True)
    with pytest.raises(ValueError): CashRuleSet(cfg).validate_buyin(1999)
    with pytest.raises(ValueError): Action('raise_to')
    with pytest.raises(NotImplementedError): DynamicTable(TableConfig(mode='tournament'))
    with pytest.raises(NotImplementedError): TournamentRuleSet().require_implemented()


def test_initial_bootstrap_fixed_seats_and_cash_conservation():
    t = make_table()
    eid = t.request_join('a', 10000, preferred_seat=5)
    assert eid.endswith('00000001')
    t.request_join('b', 20000, preferred_seat=2)
    assert t.snapshot().seats[5].seat_status == 'reserved'
    assert t.snapshot().total_chips == 0
    hand = t.start_hand(42)
    assert len(hand.participants) == 2
    assert {i for i, _ in hand.participants} == {2, 5}
    assert t.snapshot().total_chips == 30000
    assert t.snapshot().hand_in_progress
    outcome = finish(t)
    assert sum(outcome.payoffs_chips) == 0
    assert sum(outcome.final_stacks) == 30000
    assert t.snapshot().total_chips == 30000
    assert not t.snapshot().hand_in_progress


def test_join_during_hand_never_participates_until_boundary():
    t = make_table()
    join_and_start(t, 'a', 'b')
    first = t.snapshot().total_chips
    t.request_join('c', 10000)
    assert t.snapshot().seats[2].seat_status == 'reserved'
    assert t.snapshot().total_chips == first
    assert len(t._hand_roster) == 2
    with pytest.raises(RuntimeError): t.apply_boundary_events()
    finish(t)
    assert t.snapshot().total_chips == first + 10000
    assert t.snapshot().seats[2].seat_status == 'waiting_bb'


def test_leave_during_hand_and_id_reuse_of_same_seat():
    t = make_table()
    join_and_start(t, 'a', 'b')
    t.request_leave('a')
    assert t.snapshot().seats[0].seat_status == 'pending_leave'
    assert 0 in t._hand_roster
    with pytest.raises(ValueError): t.request_leave('a')
    finish(t)
    assert t.snapshot().seats[0].seat_status == 'vacant'
    assert t.total_cashouts == 10000
    assert t.snapshot().total_chips == 10000
    t.request_join('new', 10000, 0)
    hand = t.start_hand(2)
    assert set(pid for _, pid in hand.participants) == {'new', 'b'}


def test_bust_requires_final_settlement_and_rebuy_can_rescue():
    t = make_table(cls=BustHandEngine)
    join_and_start(t, 'a', 'b')
    # Zero final stack exists only after terminal settlement.
    t.request_rebuy('a', 5000)
    settled = finish(t)
    assert settled.busted_player_ids == ('a',)
    assert t.snapshot().seats[0].player_id == 'a'
    assert t.snapshot().seats[0].stack_chips == 5000
    assert t.snapshot().seats[0].seat_status == 'waiting_bb'
    assert t.snapshot().total_chips == 25000


def test_bust_removes_seat_when_no_rebuy():
    t = make_table(cls=BustHandEngine)
    join_and_start(t, 'a', 'b')
    settled = finish(t)
    assert settled.busted_player_ids == ('a',)
    assert t.snapshot().seats[0].seat_status == 'vacant'
    assert t.snapshot().total_chips == 20000
    assert t.start_hand(2) is None


def test_duplicate_join_seat_collision_invalid_rebuy_and_bad_action():
    t = make_table()
    t.request_join('a', 20000, 2)
    with pytest.raises(ValueError): t.request_join('a', 10000, 3)
    with pytest.raises(ValueError): t.request_join('b', 10000, 2)
    with pytest.raises(ValueError): t.request_rebuy('a', 1000)
    t.request_join('b', 20000, 3)
    t.start_hand(1)
    with pytest.raises(ValueError): t.act('b', Action('check'))
    with pytest.raises(ValueError): t.act('a', Action('call'))
    with pytest.raises(ValueError): t.act('a', Action('raise_to', raise_to_chips=0))
    finish(t)
    with pytest.raises(ValueError): t.request_rebuy('a', 15000)  # top-up cap exceeded


def test_sitout_resume_rejoins_at_bb_or_bootstrap():
    t = make_table()
    join_and_start(t, 'a', 'b', 'c')
    finish(t)
    t.request_sitout('b')
    t.apply_boundary_events()
    assert t.snapshot().seats[1].seat_status == 'sitting_out'
    h = t.start_hand(2)
    assert len(h.participants) == 2
    t.request_resume('b')
    finish(t)
    assert t.snapshot().seats[1].seat_status == 'waiting_bb'
    for k in range(3, 15):
        h = t.start_hand(k)
        if 1 in dict(h.participants):
            assert h.big_blind_seat == 1 or len(h.participants) == 2
            finish(t)
            break
        finish(t)
    else:
        pytest.fail('waiting player never got big blind')


def test_stale_reserved_leave_and_concurrent_cash_commands():
    t = make_table()
    t.request_join('a', 10000, 0)
    t.request_leave('a')
    t.apply_boundary_events()
    assert t.snapshot().seats[0].seat_status == 'vacant'
    assert t.total_buyins == t.total_cashouts == 10000
    t.request_join('b', 10000, 0)
    t.request_join('c', 10000, 1)
    t.apply_boundary_events()
    t.request_rebuy('b', 5000)
    with pytest.raises(ValueError): t.request_rebuy('b', 20000)
    t.request_leave('b')
    with pytest.raises(ValueError): t.request_rebuy('b', 1000)
    t.apply_boundary_events()
    assert t.snapshot().seats[0].seat_status == "vacant"
    assert t.total_buyins - t.total_cashouts == t.snapshot().total_chips


def test_replay_full_session_and_tamper_detection(tmp_path):
    t = make_table(tmp_path)
    join_and_start(t, 'a', 'b', seed=13)
    t.request_join('c', 10000)
    finish(t)
    t.start_hand(14)
    finish(t)
    t.request_leave('b')
    t.apply_boundary_events()
    re = replay_journal(tmp_path / 'ledger.jsonl', engine_factory=FakeHandEngine)
    assert re.snapshot() == t.snapshot()
    assert re.ledger.records == t.ledger.records
    source = (tmp_path / 'ledger.jsonl').read_text(encoding='utf-8').splitlines()
    edited = [json.loads(x) for x in source]
    for event in edited:
        if event['event'] == 'hand_settled':
            event['final_stacks'][0] += 1
            break
    bad = tmp_path / 'bad.jsonl'
    bad.write_text('\n'.join(json.dumps(x) for x in edited) + '\n', encoding='utf-8')
    with pytest.raises(ValueError): replay_journal(bad, engine_factory=FakeHandEngine)


def test_replay_journal_prevents_accidental_overwrite(tmp_path):
    t = make_table(tmp_path)
    assert t.snapshot().hand_number == 0
    with pytest.raises(FileExistsError): make_table(tmp_path)


def test_all_in_call_translation_for_short_stack():
    t = make_table()
    join_and_start(t, 'a', 'b')
    t._hand.observe = lambda seat: {'raw_obs': {'to_call': 50000}}
    t.act('a', Action('all_in'))
    assert t._hand._actions == [1]
    t.settle_hand()


def test_waiting_player_only_enters_on_big_blind_except_bootstrap():
    t = make_table()
    join_and_start(t, 'a', 'b', 'c', 'd')
    finish(t)
    # The first two players bootstrap, later arrivals pay BB before playing.
    admitted = set()
    for seed in range(2, 20):
        old = {i for i, seat in enumerate(t.snapshot().seats) if seat.seat_status == 'seated'}
        hand = t.start_hand(seed)
        new = set(dict(hand.participants)) - old
        if new:
            assert len(new) == 1
            assert new == {hand.big_blind_seat}
            admitted.update(new)
        finish(t)
        if admitted == {2, 3}:
            break
    assert admitted == {2, 3}


def test_raise_to_is_total_street_wager_and_replayable(tmp_path):
    t = make_table(tmp_path)
    join_and_start(t, 'a', 'b')
    t.act('a', Action('raise_to', raise_to_chips=1500))
    assert t._hand._actions == [(5, 1500), 5]
    t.settle_hand()
    restored = replay_journal(tmp_path / 'ledger.jsonl', engine_factory=FakeHandEngine)
    assert restored.snapshot() == t.snapshot()
