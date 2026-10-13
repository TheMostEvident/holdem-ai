# Complete Python source volume 18/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## tests/agents/__init__.py

Original blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (0 bytes).

<!-- BEGIN_PY_SOURCE path="tests/agents/__init__.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="tests/agents/__init__.py" -->

## tests/environment/__init__.py

Original blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (0 bytes).

<!-- BEGIN_PY_SOURCE path="tests/environment/__init__.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="tests/environment/__init__.py" -->

## tests/environment/test_betting.py

Original blob `f487b67984a09ae27645e028cdc42a4558db5fd9` (1888 bytes).

<!-- BEGIN_PY_SOURCE path="tests/environment/test_betting.py" sha="f487b67984a09ae27645e028cdc42a4558db5fd9" -->

````````python
import pytest

from holdem_ai.environment.actions import BetContext, legal_basic_actions, validate_raise_to


def test_bb_min_raise_preflop():
    ctx = BetContext(actor_commitment=1, highest_commitment=2,
                     last_full_raise=2, remaining_stack=99, big_blind=2)
    assert ctx.amount_to_call == 1
    assert ctx.min_full_raise_to == 4
    assert validate_raise_to(ctx, 4) == 'full_raise'
    with pytest.raises(ValueError, match='below minimum'):
        validate_raise_to(ctx, 3)


def test_postflop_bet_minimum_and_stack_cap():
    ctx = BetContext(actor_commitment=0, highest_commitment=0,
                     last_full_raise=2, remaining_stack=7, big_blind=2)
    assert legal_basic_actions(ctx) == ('fold', 'check', 'raise')
    assert validate_raise_to(ctx, 2) == 'full_raise'
    with pytest.raises(ValueError):
        validate_raise_to(ctx, 8)


def test_short_all_in_is_allowed_but_is_not_a_full_raise():
    ctx = BetContext(actor_commitment=0, highest_commitment=10,
                     last_full_raise=10, remaining_stack=14, big_blind=2)
    assert validate_raise_to(ctx, 14) == 'short_all_in_raise'
    with pytest.raises(ValueError):
        validate_raise_to(ctx, 13)
    already_acted = BetContext(actor_commitment=10, highest_commitment=14,
                               last_full_raise=10, remaining_stack=100, big_blind=2,
                               may_raise=False)
    assert legal_basic_actions(already_acted) == ('fold', 'call')
    with pytest.raises(ValueError, match='not reopened'):
        validate_raise_to(already_acted, 30)


def test_short_call_forces_all_in_not_raise():
    ctx = BetContext(actor_commitment=2, highest_commitment=100,
                     last_full_raise=20, remaining_stack=9, big_blind=2)
    assert ctx.amount_to_call == 9
    assert legal_basic_actions(ctx) == ('fold', 'call')
    assert not ctx.can_raise

````````

<!-- END_PY_SOURCE path="tests/environment/test_betting.py" -->

## tests/environment/test_elimination.py

Original blob `e69de29bb2d1d6434b8b29ae775ad8c2e48c5391` (0 bytes).

<!-- BEGIN_PY_SOURCE path="tests/environment/test_elimination.py" sha="e69de29bb2d1d6434b8b29ae775ad8c2e48c5391" -->

````````python

````````

<!-- END_PY_SOURCE path="tests/environment/test_elimination.py" -->

## tests/environment/test_hand_evaluator.py

Original blob `3525e30ed6a42068db3ea868363edcf2485a0171` (1708 bytes).

<!-- BEGIN_PY_SOURCE path="tests/environment/test_hand_evaluator.py" sha="3525e30ed6a42068db3ea868363edcf2485a0171" -->

````````python
import pytest

from holdem_ai.environment.hand_evaluator import rank_five, rank_seven, rank_showdown
from holdem_ai.environment.pot_manager import build_side_pots, settle_pots


def test_hierarchy_of_poker_hands():
    candidates = [
        ['sA', 'hK', 'd9', 'c6', 's3'], # high card
        ['sA', 'hA', 'd9', 'c6', 's3'], # pair
        ['sA', 'hA', 'd9', 'c9', 's3'], # two pair
        ['sA', 'hA', 'dA', 'c6', 's3'], # trips
        ['s2', 'h3', 'd4', 'c5', 's6'], # straight
        ['sA', 'sJ', 's8', 's6', 's3'], # flush
        ['sA', 'hA', 'dA', 'c6', 's6'], # full house
        ['sA', 'hA', 'dA', 'cA', 's3'], # four of a kind
        ['s9', 'sT', 'sJ', 'sQ', 'sK'], # straight flush
    ]
    scores = [rank_five(hand) for hand in candidates]
    assert scores == sorted(scores)
    assert len(set(scores)) == 9


def test_wheel_straight_loses_to_six_high():
    assert rank_five(['sA', 'h2', 'd3', 'c4', 's5']) == (4, 5)
    assert rank_five(['s2', 'h3', 'd4', 'c5', 's6']) > (4, 5)


def test_two_triplets_form_highest_full_house():
    assert rank_seven(['sA', 'hA', 'dA', 'sK', 'hK', 'dK', 'c2']) == (6, 14, 13)


def test_board_plays_and_split_at_showdown():
    board = ['sT', 'sJ', 'sQ', 'sK', 'sA']
    a = rank_showdown(['h2', 'c3'], board)
    b = rank_showdown(['h4', 'd5'], board)
    assert a == b == (8, 14)
    breakdown = build_side_pots((20, 20), (False, False))
    assert settle_pots(breakdown, (a, b), button=0) == (20, 20)


def test_invalid_duplicate_or_bad_card():
    with pytest.raises(ValueError, match='duplicate'):
        rank_seven(['sA', 'sA', 'd5', 'c3', 'h8', 'c9', 'cT'])
    with pytest.raises(ValueError):
        rank_five(['sA', 'sK', 'sT', 'sJ', 'bad'])

````````

<!-- END_PY_SOURCE path="tests/environment/test_hand_evaluator.py" -->

## tests/environment/test_position.py

Original blob `0b18b09ca4bd46d12d12e74521c5885a4700037a` (942 bytes).

<!-- BEGIN_PY_SOURCE path="tests/environment/test_position.py" sha="0b18b09ca4bd46d12d12e74521c5885a4700037a" -->

````````python
import pytest

from holdem_ai.environment.seat_manager import heads_up_seats, post_blinds, rotate_heads_up_button


@pytest.mark.parametrize('button', [0, 1])
def test_heads_up_button_is_sb_and_first_preflop(button):
    seats = heads_up_seats(button)
    assert seats.small_blind == button
    assert seats.preflop_first == button
    assert seats.big_blind == 1 - button
    assert seats.postflop_first == 1 - button
    assert rotate_heads_up_button(button) == 1 - button


def test_blinds_are_charged_to_correct_seats():
    posted, remaining = post_blinds((100, 100), button=0, sb=1, bb=2)
    assert posted == (1, 2)
    assert remaining == (99, 98)
    posted, remaining = post_blinds((100, 1), button=0, sb=1, bb=2)
    assert posted == (1, 1)
    assert remaining == (99, 0)


@pytest.mark.parametrize('button', [-1, 2])
def test_invalid_button_is_rejected(button):
    with pytest.raises(ValueError):
        heads_up_seats(button)

````````

<!-- END_PY_SOURCE path="tests/environment/test_position.py" -->

## tests/environment/test_side_pot.py

Original blob `5339700d71c80b20b9cbea8c2a2389dc5fec6af3` (1850 bytes).

<!-- BEGIN_PY_SOURCE path="tests/environment/test_side_pot.py" sha="5339700d71c80b20b9cbea8c2a2389dc5fec6af3" -->

````````python
import pytest

from holdem_ai.environment.pot_manager import build_side_pots, settle_pots


def test_three_way_all_in_has_correct_main_and_side_pots():
    p = build_side_pots((100, 200, 500), (False, False, False))
    assert [(x.amount, x.eligible) for x in p.pots] == [
        (300, (0, 1, 2)), (200, (1, 2)),
    ]
    assert p.uncalled_returns == (0, 0, 300)
    # Player 0 wins main; player 1 wins side; 300 uncalled returned to player 2.
    assert settle_pots(p, ((9,), (8,), (7,)), button=0) == (300, 200, 300)


def test_folded_player_contribution_stays_in_pot():
    p = build_side_pots((100, 100, 100), (False, True, False))
    assert p.pots[0].amount == 300
    assert p.pots[0].eligible == (0, 2)
    assert settle_pots(p, ((10,), None, (7,)), button=0) == (300, 0, 0)


def test_unmatched_wager_returns_to_contributor():
    p = build_side_pots((100, 20), (False, True))
    assert [(pot.amount, pot.eligible) for pot in p.pots] == [(40, (0,))]
    assert p.uncalled_returns == (80, 0)
    assert settle_pots(p, ((8,), None), button=0) == (120, 0)


def test_odd_chip_goes_to_first_winner_left_of_button():
    # Three-way pot 3 chips + a second player only ties once but wins 1 odd.
    p = build_side_pots((1, 1, 1), (False, False, False))
    # Tied players are seats 0 and 2; button 0 -> seat 2 is first left of BTN.
    assert settle_pots(p, ((8,), (7,), (8,)), button=0) == (1, 0, 2)


def test_conservation_and_reject_invalid_states():
    p = build_side_pots((51, 68, 72, 12), (False, False, True, False))
    result = settle_pots(p, ((2,), (4,), None, (9,)), button=1)
    assert sum(result) == 51 + 68 + 72 + 12
    with pytest.raises(ValueError, match='matching fold flags'):
        build_side_pots((1, 2), (False,))
    with pytest.raises(ValueError, match='at least one'):
        build_side_pots((1, 1), (True, True))

````````

<!-- END_PY_SOURCE path="tests/environment/test_side_pot.py" -->

## tests/expand/conftest.py

Original blob `f35659850cedf50a9fef7afe6fea7a1f93028eaa` (2814 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/conftest.py" sha="f35659850cedf50a9fef7afe6fea7a1f93028eaa" -->

````````python
"""Isolated stub for table-management tests; real-engine tests are separate."""
from __future__ import annotations
from pathlib import Path
import sys
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[2] / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class FakeHandEngine:
    """Not a poker simulator. Tests table events independent of Base Phase 3."""
    def __init__(self, *, stacks, button, small_blind, big_blind, seed):
        self.initial_stacks = tuple(stacks)
        self.stacks = list(stacks)
        self.contributions = [0] * 6
        participants = [i for i, stack in enumerate(stacks) if stack > 0]
        self.button = button
        def nxt(i):
            return next((i + x) % 6 for x in range(1, 7) if (i + x) % 6 in participants)
        if len(participants) == 2:
            sb, bb = button, nxt(button)
        else:
            sb = nxt(button)
            bb = nxt(sb)
        self.seats = SimpleNamespace(small_blind=sb, big_blind=bb)
        self.current_player = participants[0]
        self._done = False
        self._actions = []
        self._result = None

    def reset(self):
        return self.observe(self.current_player), self.current_player

    def is_over(self):
        return self._done

    def observe(self, player_id):
        assert player_id == self.current_player and not self._done
        return {'legal_actions': {0: None, 1: None},
                'raw_obs': {'to_call': 0, 'hand': ['cA', 'dK'], 'pot': 0}}

    def step(self, action_id):
        self._actions.append(action_id)
        self._done = True
        self._result = SimpleNamespace(final_stacks=tuple(self.stacks),
                                       payoffs_chips=(0,) * 6,
                                       board=(), action_history=((self.current_player, action_id),),
                                       action_raise_to=(None,))

    def step_raise_to(self, amount):
        self._actions.append((5, amount))
        self.step(5)

    def get_result(self):
        assert self._done
        return self._result


class BustHandEngine(FakeHandEngine):
    def step(self, action_id):
        super().step(action_id)
        occupied = [i for i, x in enumerate(self.initial_stacks) if x]
        loser, winner = occupied[:2]
        transferred = self.initial_stacks[loser]
        out = list(self.initial_stacks)
        out[loser] = 0
        out[winner] += transferred
        payoffs = [0] * 6
        payoffs[loser] = -transferred
        payoffs[winner] = transferred
        self._result = SimpleNamespace(final_stacks=tuple(out), payoffs_chips=tuple(payoffs),
                                       board=(), action_history=((self.current_player, action_id),),
                                       action_raise_to=(None,))

````````

<!-- END_PY_SOURCE path="tests/expand/conftest.py" -->

## tests/expand/e03_fake_engine.py

Original blob `a6b089d01ec085d50d81cc2607d9bae799401cfb` (3340 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/e03_fake_engine.py" sha="a6b089d01ec085d50d81cc2607d9bae799401cfb" -->

````````python
"""Deterministic test-only 2-6 player game. NOT a Hold'em rules engine."""
from types import SimpleNamespace


class FastFakeEngine:
    def __init__(self,*,stacks,button,small_blind,big_blind,seed):
        self.initial_stacks=tuple(stacks)
        self.stacks=list(stacks)
        self.contributions=[0]*6
        self.street_bets=[0]*6
        self.status=['active' if s else 'out' for s in stacks]
        self._seats=[j for j,x in enumerate(stacks) if x>0]
        self.button=button
        self.seed=seed
        def nxt(x):
            return next((x+k)%6 for k in range(1,7) if (x+k)%6 in self._seats)
        sb=button if len(self._seats)==2 else nxt(button)
        bb=nxt(button) if len(self._seats)==2 else nxt(sb)
        self.seats=SimpleNamespace(small_blind=sb,big_blind=bb)
        self.current_player=self._seats[0]
        self.history=[]
        self.raises=[]
        self._done=False
    def reset(self):
        return self.observe(self.current_player),self.current_player
    def observe(self,pid):
        assert pid==self.current_player
        cards=['cA','dK','cT','dJ','h3','s7']
        return {'legal_actions':{0:None,1:None},'raw_obs':{'hand':[cards[pid],cards[(pid+1)%6]],
         'public_cards':[],'street':'preflop','button':self.button,
         'player_id':pid,'stacks':tuple(self.stacks),'street_bets':tuple(self.street_bets),
         'total_contributions':tuple(self.contributions),'status':tuple(self.status),
         'pot':0,'to_call':0,'min_raise_to':2,'action_history':tuple(self.history),
         'action_raise_to':tuple(self.raises)}}
    def step(self,a):
        assert a in (0,1)
        self.history.append((self.current_player,a))
        self.raises.append(None)
        i=self._seats.index(self.current_player)
        if i+1==len(self._seats):
            self._done=True
            self.current_player=-1
        else:self.current_player=self._seats[i+1]
    def is_over(self):return self._done
    def get_result(self):
        assert self._done
        awards=[0]*6
        # Deterministic, zero-sum, signed hand payoff to each physical seat.
        if self.seed%2:
            if self._seats[1]!=self._seats[0]:
                awards[self._seats[0]]+=10
                awards[self._seats[1]]-=10
        else:
            awards[self._seats[0]]-=10
            awards[self._seats[1]]+=10
        return SimpleNamespace(final_stacks=tuple(x+y for x,y in zip(self.stacks,awards)),
           payoffs_chips=tuple(awards),board=(),action_history=tuple(self.history),
           action_raise_to=tuple(self.raises))

# Test-only isolated shim: absent Base Phase3 files in sandbox. Never installed
# in source distribution and never used when the real engine is importable.
try:
    from holdem_ai.environment.multi_player_seats import multi_player_seats
except ModuleNotFoundError:
    import holdem_ai.expand.environment.multiway_adapter as _adapter
    def _test_blind_seats(button, active):
        def nxt(seat):
            return next((seat + k) % 6 for k in range(1,7) if (seat + k) % 6 in active)
        if len(active)==2:
            return button, nxt(button)
        return nxt(button), nxt(nxt(button))
    _adapter.blind_seats=_test_blind_seats
    import holdem_ai.expand.environment.table_manager as _table_manager
    _table_manager.blind_seats=_test_blind_seats

````````

<!-- END_PY_SOURCE path="tests/expand/e03_fake_engine.py" -->

## tests/expand/e07_fake_tournament.py

Original blob `90b9901f572b978db1404a5f9f172ed60fafbb29` (2004 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/e07_fake_tournament.py" sha="90b9901f572b978db1404a5f9f172ed60fafbb29" -->

````````python
"""A deterministic side-effect-free test double, NEVER a poker engine."""
from types import SimpleNamespace


class KnockoutEngine:
    def __init__(self, *, stacks, button, small_blind, big_blind, seed):
        self.initial=tuple(stacks)
        self.stacks=list(stacks)
        self.active=[i for i,s in enumerate(stacks) if s>0]
        self.button=button
        self.big_blind=big_blind
        self.small_blind=small_blind
        self.seed=seed
        self.current_player=self.active[0]
        self.done=False
        def nxt(pos):return next((pos+i)%6 for i in range(1,7) if (pos+i)%6 in self.active)
        sb=button if len(self.active)==2 else nxt(button)
        bb=nxt(button) if len(self.active)==2 else nxt(sb)
        self.seats=SimpleNamespace(small_blind=sb,big_blind=bb)
    def reset(self):return self.observe(self.current_player),self.current_player
    def observe(self,p):
        assert p==self.current_player
        return {'legal_actions':{0:None,1:None},
            'raw_obs':{'hand':['cA','dK'],'public_cards':[], 'street':'preflop',
            'button':self.button,'player_id':p,'stacks':tuple(self.stacks),
            'street_bets':(0,)*6,'total_contributions':(0,)*6,
            'status':tuple('active' if i in self.active else 'out' for i in range(6)),
            'pot':0,'to_call':0,'min_raise_to':self.big_blind,
            'action_history':(), 'action_raise_to':()}}
    def step(self,a):
        if a not in (0,1):raise ValueError('fake legal action')
        self.current_player=-1
        self.done=True
    def is_over(self):return self.done
    def get_result(self):
        assert self.done
        # Knock out one (nonwinner) seat; deterministic and zero-sum.
        loser=self.active[0]
        winner=self.active[-1]
        final=list(self.stacks)
        final[winner]+=final[loser]
        final[loser]=0
        payoffs=tuple(a-b for a,b in zip(final,self.stacks))
        return SimpleNamespace(final_stacks=tuple(final),payoffs_chips=payoffs)

````````

<!-- END_PY_SOURCE path="tests/expand/e07_fake_tournament.py" -->

## tests/expand/test_e01_real_engine.py

Original blob `08febe74e010fb26b91358749a0dfb400b612fb4` (3209 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e01_real_engine.py" sha="08febe74e010fb26b91358749a0dfb400b612fb4" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/expand/test_e01_real_engine.py" -->

## tests/expand/test_e01_smoke_unit.py

Original blob `33648e36aa11203ff0bcaddf17a73e2782cc059a` (1308 bytes).

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e01_smoke_unit.py" sha="33648e36aa11203ff0bcaddf17a73e2782cc059a" -->

````````python
"""CLI contract tested with a fake engine; the real smoke is a Windows gate."""
import json
import pytest
from conftest import FakeHandEngine
from holdem_ai.expand.environment import DynamicTable
import holdem_ai.expand.environment.table_manager as tm
from scripts import expand_phase01_smoke as cli


def test_e01_smoke_cli_counts_exact_2_to_6_players(tmp_path, monkeypatch):
    def blinds(button, participants):
        ordered = [p for p in ((button + k) % 6 for k in range(1, 7)) if p in participants]
        return (button, ordered[0]) if len(participants) == 2 else (ordered[0], ordered[1])
    monkeypatch.setattr(tm, 'blind_seats', blinds)
    monkeypatch.setattr(cli, 'DynamicTable', lambda *args, **kwargs: DynamicTable(
        *args, engine_factory=FakeHandEngine, **kwargs))
    output = tmp_path / 'smoke.json'
    result = cli.run(hands_per_n=3, seed=101, output=str(output))
    assert result['target_hands'] == 15
    assert [x['hands'] for x in result['results']] == [3, 3, 3, 3, 3]
    assert result['total_hands'] >= 15
    assert all(x['final_table_chips'] == x['buyins_minus_cashouts'] for x in result['results'])
    assert json.loads(output.read_text(encoding='utf-8')) == result
    with pytest.raises(FileExistsError):
        cli.run(hands_per_n=3, seed=101, output=str(output))

````````

<!-- END_PY_SOURCE path="tests/expand/test_e01_smoke_unit.py" -->
