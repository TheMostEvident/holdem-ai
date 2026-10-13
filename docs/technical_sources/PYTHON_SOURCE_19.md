# Complete Python source volume 19/28

[Integrated technical report](../POKER_AI_FINAL_INTEGRATED_TECHNICAL_REPORT.md). Pinned source commit `6a8ae368ddc02f814140d3770bc5993eaf9a15e4`. All code literal and complete.

## tests/expand/test_e01_table_unit.py

Original blob `db06a5084523584c1ccad775f0854a246e7200d1` (9209 bytes).

<a name="py-217"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e01_table_unit.py" sha="db06a5084523584c1ccad775f0854a246e7200d1" -->

````````python
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

````````

<!-- END_PY_SOURCE path="tests/expand/test_e01_table_unit.py" -->

## tests/expand/test_e02_adapter_fake.py

Original blob `ad72a612293b51780854faef45df1b785b5b484d` (3925 bytes).

<a name="py-218"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e02_adapter_fake.py" sha="ad72a612293b51780854faef45df1b785b5b484d" -->

````````python
from types import SimpleNamespace
import pytest
from holdem_ai.expand.environment import Action, DynamicTable
from holdem_ai.expand.information import Phase2TableAdapter


class PublicRichEngine:
    def __init__(self, *, stacks, button, small_blind, big_blind, seed):
        self.initial_stacks=tuple(stacks)
        self.stacks=list(stacks)
        self.contributions=[0]*6
        self.bets=[0]*6
        self.status=['active' if x>0 else 'out' for x in stacks]
        self.button=button
        occupied=[i for i,x in enumerate(stacks) if x>0]
        self.current_player=occupied[0]
        def nxt(i):
            return next((i+k)%6 for k in range(1,7) if (i+k)%6 in occupied)
        self.seats=SimpleNamespace(small_blind=button if len(occupied)==2 else nxt(button),
                                    big_blind=nxt(button) if len(occupied)==2 else nxt(nxt(button)))
        self._order=occupied
        self._history=[]
        self._raise_history=[]
        self.street='preflop'
        self.board=[]
        self._done=False

    def reset(self):
        return self.observe(self.current_player), self.current_player

    def is_over(self):
        return self._done

    def observe(self, player_id):
        assert player_id==self.current_player and not self._done
        return {'legal_actions':{0:None,1:None},'raw_obs':{
            'hand':['cA','dK'], 'public_cards':list(self.board),
            'street':self.street,'button':self.button,'player_id':player_id,
            'stacks':tuple(self.stacks),'street_bets':tuple(self.bets),
            'total_contributions':tuple(self.contributions),'status':tuple(self.status),
            'pot':0,'to_call':0,'min_raise_to':2,
            'action_history':tuple(self._history),'action_raise_to':tuple(self._raise_history)}}

    def step(self, action_id):
        assert action_id==1
        player=self.current_player
        self._history.append((player,action_id))
        self._raise_history.append(None)
        if len(self._history)==1:
            self.street='flop'
            self.board=['c2','d4','h5']
            self.current_player=self._order[1]
        else:
            self._done=True
            self.current_player=-1

    def get_result(self):
        assert self._done
        return SimpleNamespace(final_stacks=tuple(self.stacks),payoffs_chips=(0,)*6,
          board=tuple(self.board),action_history=tuple(self._history),
          action_raise_to=tuple(self._raise_history))


def mk_table():
    table=DynamicTable(engine_factory=PublicRichEngine)
    table.request_join('a',10000,0)
    table.request_join('b',10000,1)
    return table


def test_e02_tracker_carries_street_across_actions():
    a=Phase2TableAdapter(mk_table())
    a.start_hand(1)
    before=a.observe()
    assert before.action_history==()
    a.act_abstract('check')
    after=a.observe()
    assert after.street=='flop'
    assert after.action_history[0].street=='preflop'
    assert after.action_history[0].kind=='check'
    a.act_abstract('check')
    settlement=a.settle_hand()
    assert settlement.hand_number==1
    assert not a.table.snapshot().hand_in_progress
    a.start_hand(2)
    assert a.observe().action_history==()


def test_e02_rejected_action_does_not_corrupt_tracker():
    a=Phase2TableAdapter(mk_table())
    a.start_hand(1)
    with pytest.raises(ValueError):
        a.act_abstract('call')
    assert len(a.observe().action_history)==0


def test_e02_external_action_detected_without_leak():
    a=Phase2TableAdapter(mk_table())
    a.start_hand(1)
    # Caller bypasses adapter; no guessed/mislabelled history permitted.
    a.table.act(a.table.current_player_id,Action('check'))
    with pytest.raises(ValueError,match='mismatched'):
        a.observe()


def test_e02_attaching_midhand_rejected():
    table=mk_table()
    table.start_hand(1)
    with pytest.raises(ValueError,match='mid-hand'):
        Phase2TableAdapter(table)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e02_adapter_fake.py" -->

## tests/expand/test_e02_policy.py

Original blob `76f96314b3ebbbefafddc48f57f24e6730e5c68c` (2999 bytes).

<a name="py-219"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e02_policy.py" sha="76f96314b3ebbbefafddc48f57f24e6730e5c68c" -->

````````python
import pytest

torch = pytest.importorskip('torch')
from holdem_ai.expand.information import (make_information_state,encode_observation)
from holdem_ai.expand.information.policy_v2 import PolicyV2
from test_e02_state_and_actions import obs


def fixture():
    return encode_observation(make_information_state(
        obs(), big_blind_chips=100,
        seat_presence=(True,True,True,False,False,False)))


def test_e02_policy_logits_gradients_and_illegal_zero():
    torch.manual_seed(1)
    model = PolicyV2(width=32, heads=4, layers=1)
    x=fixture()
    logits=model([x,x])
    assert logits.shape==(2,7)
    probs=logits.softmax(-1)
    assert probs[0,1].item()==0.0
    assert torch.allclose(probs.sum(-1), torch.ones(2), atol=1e-6)
    loss=-logits.log_softmax(-1)[:,3].mean()
    loss.backward()
    assert model.policy_head[-1].weight.grad is not None
    assert model.policy_head[-1].weight.grad.abs().sum().item()>0
    assert model.seat_project[0].weight.grad.abs().sum().item()>0
    assert model.history_project[0].weight.grad is None or model.history_project[0].weight.grad.abs().sum().item()==0


def test_e02_policy_seed_and_legal_sampling():
    model=PolicyV2(width=32,heads=4,layers=1)
    x=fixture()
    assert model.select_action(x,rng_seed=123)==model.select_action(x,rng_seed=123)
    assert model.select_action(x,rng_seed=123) != 'check'
    assert model.select_action(x,rng_seed=123,greedy=True) != 'check'
    with pytest.raises(ValueError):
        model.select_action(x,rng_seed=-1)


def test_e02_seat_permutation_equivariance():
    from dataclasses import replace
    # Rotation of physical labels by 2 with identical relative pos and
    # hero/global features preserves set-transformer output (no absolute IDs).
    x=fixture()
    rotated=replace(x,
        seat_features=x.seat_features[-2:]+x.seat_features[:-2],
        seat_mask=x.seat_mask[-2:]+x.seat_mask[:-2])
    model=PolicyV2(width=32,heads=4,layers=1)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model([x]),model([rotated]),atol=1e-5)


def test_e02_public_history_affects_policy_and_trains_gru():
    from dataclasses import replace
    from holdem_ai.expand.information.schema import PublicAction
    state=make_information_state(obs(),big_blind_chips=100,
      seat_presence=(True,True,True,False,False,False))
    raw=obs()
    raw['raw_obs']['action_history']=((1,1),)
    raw['raw_obs']['action_raise_to']=(None,)
    history=(PublicAction(actor_seat=1,street='preflop',kind='call',
              raise_to_bb=None,to_call_bb=0.5,pot_bb=2.5),)
    state2=make_information_state(raw,big_blind_chips=100,
      seat_presence=(True,True,True,False,False,False),history=history)
    h=encode_observation(state2)
    assert h.history_mask[0]
    model=PolicyV2(width=32,heads=4,layers=1)
    logits=model([h])
    assert torch.isfinite(logits[:,0]).all()
    (-logits.log_softmax(-1)[:,3]).sum().backward()
    assert model.history_project[0].weight.grad.abs().sum()>0

````````

<!-- END_PY_SOURCE path="tests/expand/test_e02_policy.py" -->

## tests/expand/test_e02_real_engine.py

Original blob `d423a12726bde3afcdc68930c833c109c68bf890` (2882 bytes).

<a name="py-220"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e02_real_engine.py" sha="d423a12726bde3afcdc68930c833c109c68bf890" -->

````````python
"""Only run where Base Phase3 is installed (Windows user's repository)."""
import pytest
pytest.importorskip('holdem_ai.environment.multi_player_engine')

from holdem_ai.expand.environment import DynamicTable
from holdem_ai.expand.information import Phase2TableAdapter
from holdem_ai.expand.information.policy_v2 import PolicyV2


@pytest.mark.parametrize('target_n',range(2,7))
def test_e02_real_engine_dynamic_roster_and_observation(target_n):
    table=DynamicTable(session_id=f'e02-real-{target_n}')
    adapter=Phase2TableAdapter(table)
    for i in range(target_n):
        table.request_join(f'p{i}',10000,preferred_seat=i)
    sizes=set()
    for k in range(45):
        hand=adapter.start_hand(5020+target_n*100+k)
        assert hand is not None
        sizes.add(len(hand.participants))
        for _ in range(400):
            pid=table.current_player_id
            if pid is None:
                break
            state=adapter.observe()
            assert state.hero_seat in dict(hand.participants)
            assert state.num_dealt_in==len(hand.participants)
            obs=adapter.encode()
            assert sum(obs.legal_mask)>=1
            assert sum(obs.seat_mask)==target_n  # waiting BB occupied too
            if state.to_call_bb>0:
                adapter.act_abstract('call')
            else:
                adapter.act_abstract('check')
        else:
            pytest.fail('live game exceeded 400 player decisions')
        settled=adapter.settle_hand()
        assert sum(settled.payoffs_chips)==0
        assert sum(settled.final_stacks)==sum(settled.starting_stacks)
        if target_n in sizes:
            break
    assert target_n in sizes


def test_e02_all_legal_action_translations_accepted_by_real_engine():
    from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine
    from holdem_ai.expand.information import make_information_state, to_table_action
    for n in range(2,7):
        stacks=tuple([10000]*n+[0]*(6-n))
        engine=MultiPlayerHandEngine(stacks=stacks,button=0,small_blind=50,big_blind=100,seed=73000+n)
        obs,_=engine.reset()
        state=make_information_state(obs,big_blind_chips=100,
                                     seat_presence=tuple(i<n for i in range(6)))
        for name in ('fold','check','call','raise_half_pot','raise_pot','raise_double_pot','all_in'):
            if not state.legal_actions.allows(name):
                continue
            check=MultiPlayerHandEngine(stacks=stacks,button=0,small_blind=50,big_blind=100,seed=73000+n)
            check.reset()
            action=to_table_action(state.legal_actions,name)
            if action.kind=='raise_to':
                check.step_raise_to(action.raise_to_chips)
            else:
                check.step({'fold':0,'check':1,'call':1,'all_in':4}[action.kind])
            assert len(check.action_history)==1

````````

<!-- END_PY_SOURCE path="tests/expand/test_e02_real_engine.py" -->

## tests/expand/test_e02_state_and_actions.py

Original blob `ee4f3c7e7f02c129de85b9e6e4c5354fa89ae885` (7263 bytes).

<a name="py-221"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e02_state_and_actions.py" sha="ee4f3c7e7f02c129de85b9e6e4c5354fa89ae885" -->

````````python
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict
import pytest

from holdem_ai.expand.information import (make_information_state, legal_action_spec,
    to_table_action, ACTION_NAMES, TrainingOnlyCriticState, encode_observation)
from holdem_ai.expand.information.encoder import SEAT_DIM, GLOBAL_DIM, HISTORY_DIM, MAX_HISTORY


def obs(*, hero=0, to_call=100, stack=10000, pot=150, street='preflop',
        legal=(0, 1, 2, 3, 4)):
    stacks = [10000,10000,10000,0,0,0]
    stacks[hero] = stack
    invested = [0, 50, 100, 0,0,0]
    bets = list(invested)
    if hero == 0:
        assert to_call==100
    if hero == 1:
        assert to_call==50
    if hero == 2:
        assert to_call==0
    return {
        'legal_actions': {i: None for i in legal},
        'raw_obs': {'hand':['cA','hK'],'public_cards':[],
                    'street':street,'button':0,'player_id':hero,
                    'stacks':tuple(stacks),'street_bets':tuple(bets),
                    'total_contributions':tuple(invested),'status':('active','active','active','out','out','out'),
                    'pot':pot,'to_call':to_call,'min_raise_to':200,
                    'action_history':(), 'action_raise_to':(),
                    # These must never reach actor even if supplied to builder.
                    'opponent_holes':(('dA','sA'),), 'future_deck':('c2','h2'), 'seed':123},
        'hidden_engine_state': {'hole_cards': 'DANGER'},
    }


def mk():
    return make_information_state(obs(), big_blind_chips=100,
        seat_presence=(True,True,True,False,False,False))


def test_e02_only_acting_player_and_public_fields():
    state = mk()
    assert state.hero_hole == ('cA', 'hK')
    assert state.num_dealt_in == 3
    assert state.seats[3].relative_position == 3
    assert not state.seats[3].occupied
    assert not state.seats[3].dealt_in
    assert state.seats[0].is_hero
    assert 'opponent' not in repr(state)
    assert 'future_deck' not in repr(state)
    assert 'hidden_engine_state' not in repr(state)
    assert 'seed' not in repr(state)
    encoded = encode_observation(state)
    assert len(encoded.seat_features) == 6
    assert all(len(row) == SEAT_DIM for row in encoded.seat_features)
    assert len(encoded.global_features) == GLOBAL_DIM
    assert len(encoded.history_features) == MAX_HISTORY
    assert all(len(x) == HISTORY_DIM for x in encoded.history_features)
    assert sum(encoded.seat_mask) == 3
    assert sum(encoded.history_mask) == 0


def test_e02_invariance_to_opponent_hole_and_future_deck():
    a, b = obs(), obs()
    b['raw_obs']['opponent_holes'] = [('c3','d3'),('c6','h6')]
    b['raw_obs']['future_deck'] = ['hT']
    b['raw_obs']['seed'] = 9999
    b['hidden_engine_state'] = {"critic": "nothing"}
    left = make_information_state(a,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    right = make_information_state(b,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    assert left == right
    assert encode_observation(left) == encode_observation(right)
    with pytest.raises((AttributeError, FrozenInstanceError)):
        left.hero_hole = ('dA','hA')


def test_e02_street_hole_and_board_validation():
    o = obs()
    o['raw_obs']['public_cards'] = ['sA']
    with pytest.raises(ValueError, match='board'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    o=obs()
    o['raw_obs']['hand'] = ['cA','cA']
    with pytest.raises(ValueError, match='duplicate'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    o=obs()
    o['raw_obs']['street'] = 'river'
    with pytest.raises(ValueError, match='street/board'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)


def test_e02_bad_public_chip_state_rejected():
    o = obs()
    o['raw_obs']['pot'] = 151
    with pytest.raises(ValueError,match='pot/contributions'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    o = obs()
    o['raw_obs']['status'] = ('active','active','active','active','out','out')
    with pytest.raises(ValueError,match='registry'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)


def test_e02_action_translation_exact_pot_math():
    state = mk()
    spec = state.legal_actions
    assert spec.mask == (True, False, True, True, True, True, True)
    assert spec.raise_to_chips[3] == 225
    assert spec.raise_to_chips[4] == 350
    assert spec.raise_to_chips[5] == 600
    assert spec.raise_to_chips[6] == 10000
    assert to_table_action(spec, 'raise_half_pot').raise_to_chips == 225
    assert to_table_action(spec, 'raise_pot').raise_to_chips == 350
    assert to_table_action(spec, 'raise_double_pot').raise_to_chips == 600
    assert to_table_action(spec, 'all_in').kind == 'all_in'
    with pytest.raises(ValueError,match='illegal'):
        to_table_action(spec,'check')


def test_e02_short_allin_call_is_not_duplicate_allin_action():
    o=obs(stack=50,legal=(0,1))
    state=make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    spec=state.legal_actions
    assert spec.allows('call')
    assert not spec.allows('all_in')
    assert not spec.allows('raise_double_pot')


def test_e02_check_vs_call_legality():
    o=obs(hero=2,to_call=0)
    state=make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)
    assert state.legal_actions.allows('check')
    assert not state.legal_actions.allows('call')


def test_e02_deduplicate_wager_target():
    o=obs(stack=225)
    # Half-pot raise to == all-in, canonical half-pot action only
    spec=legal_action_spec(o['raw_obs'],tuple(o['legal_actions']),100)
    assert spec.allows('raise_half_pot')
    assert not spec.allows('all_in')


def test_e02_all_sitting_out_retained_as_occupied_token():
    o=obs()
    state=make_information_state(o,big_blind_chips=100,
          seat_presence=(True,True,True,True,False,False))
    assert state.seats[3].occupied and not state.seats[3].dealt_in
    assert encode_observation(state).seat_mask[3]


def test_e02_no_training_critic_in_encoder():
    with pytest.raises(TypeError,match='InformationState'):
        encode_observation(TrainingOnlyCriticState(full_hand_state={'secret': 'x'}))


def test_e02_empty_action_raises():
    o=obs(legal=())
    with pytest.raises(ValueError,match='unsupported engine'):
        make_information_state(o,big_blind_chips=100,seat_presence=(True,)*3+(False,)*3)


def test_e02_table_occupancy_status_are_separate_and_consistent():
    o=obs()
    with pytest.raises(ValueError,match='presence/status'):
        make_information_state(o,big_blind_chips=100,
          seat_presence=(True,True,True,False,False,False),
          seat_statuses=('seated','seated','seated','waiting_bb','vacant','vacant'))
    state=make_information_state(o,big_blind_chips=100,
          seat_presence=(True,True,True,True,False,False),
          seat_statuses=('seated','seated','seated','sitting_out','vacant','vacant'))
    assert state.seats[3].table_status=='sitting_out'
    assert not state.seats[3].dealt_in
    x=encode_observation(state)
    assert x.seat_features[3][-2] == 1.0  # sitting-out token

````````

<!-- END_PY_SOURCE path="tests/expand/test_e02_state_and_actions.py" -->

## tests/expand/test_e03_cli.py

Original blob `3819fc0f3ede349b8413d104d223715d9797a839` (883 bytes).

<a name="py-222"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e03_cli.py" sha="3819fc0f3ede349b8413d104d223715d9797a839" -->

````````python
import pytest
from scripts.expand_phase03_train import build_parser,make_config


def test_e03_cli_has_safe_defaults_and_restricted_resume():
    p=build_parser()
    default=make_config(p.parse_args([]))
    assert default.rollout.num_players==(2,3,4,5,6)
    assert default.rollout.starting_stack_bb==100
    assert make_config(p.parse_args(['--resume'])) is None
    with pytest.raises(ValueError,match='resume loads immutable'):
        make_config(p.parse_args(['--resume','--hands-per-update','8']))


def test_e03_custom_curriculum():
    p=build_parser()
    cfg=make_config(p.parse_args(['--hands-per-update','4','--num-players','3,6',
      '--random-stack-range-bb','30,200','--shared-fraction','0.4','--snapshot-fraction','0.2']))
    assert cfg.rollout.hands_per_update==4
    assert cfg.rollout.num_players==(3,6)
    assert cfg.rollout.random_stack_range_bb==(30,200)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e03_cli.py" -->

## tests/expand/test_e03_real_engine.py

Original blob `5e9027db334ea0a751ed135a7b03ac59ac55a311` (999 bytes).

<a name="py-223"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e03_real_engine.py" sha="5e9027db334ea0a751ed135a7b03ac59ac55a311" -->

````````python
"""Real 2-6 seat engine integration: deferred until the Windows full repo."""
import pytest
pytest.importorskip('holdem_ai.environment.multi_player_engine')

from holdem_ai.expand.rl import (ActorCritic, RolloutConfig, OpponentMix,
                                  MultiAgentCollector, prepare_gae)


@pytest.mark.parametrize('n',(2,3,4,5,6))
def test_e03_full_hands_real_base_engine(n):
    import torch
    torch.manual_seed(3)
    actor=ActorCritic(width=16,heads=2,layers=1)
    collector=MultiAgentCollector(actor,
       RolloutConfig(hands_per_update=2,num_players=(n,),max_warmup_hands=100,
                     mix=OpponentMix(shared_policy_fraction=0,snapshot_fraction=0,
                         scripted_names=('calling_station',))))
    result=collector.collect(seed=40000+n,policy_version='real-u0')
    assert result.hands_by_n[n]==2
    assert result.illegal_actions==0
    assert all(abs(sum(x))<1e-8 for x in result.hand_rewards)
    assert len(prepare_gae(result.transitions))>0

````````

<!-- END_PY_SOURCE path="tests/expand/test_e03_real_engine.py" -->

## tests/expand/test_e03_returns_and_policy.py

Original blob `cf384ca17a12a27d4fcf11e5f9bab98843484a4b` (3915 bytes).

<a name="py-224"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e03_returns_and_policy.py" sha="cf384ca17a12a27d4fcf11e5f9bab98843484a4b" -->

````````python
import math
import random
from dataclasses import replace
import pytest
import torch
from holdem_ai.expand.information.schema import ACTION_NAMES
from holdem_ai.expand.information.encoder import encode_observation
from holdem_ai.expand.rl import (ActorCritic, OpponentMix, RolloutConfig, PPOConfig,
    MultiAgentCollector, prepare_gae, optimize_ppo)
from holdem_ai.expand.rl.population import scripted_action, choose_role
from e03_fake_engine import FastFakeEngine


def mk_model(seed=12):
    torch.manual_seed(seed)
    return ActorCritic(width=16,heads=2,layers=1)


def test_population_probabilities_and_guards():
    with pytest.raises(ValueError): OpponentMix(shared_policy_fraction=.7,snapshot_fraction=.7)
    mix=OpponentMix(shared_policy_fraction=0.3,snapshot_fraction=0.5)
    rng=random.Random(1)
    assert 'snapshot' in {choose_role(rng,mix,has_snapshots=True) for _ in range(50)}
    assert 'snapshot' not in {choose_role(rng,mix,has_snapshots=False) for _ in range(50)}


def test_rollout_all_2_to_6_and_reward_assignment():
    config=RolloutConfig(hands_per_update=5,num_players=(2,3,4,5,6),max_warmup_hands=100,
                         mix=OpponentMix(shared_policy_fraction=1,snapshot_fraction=0))
    c=MultiAgentCollector(mk_model(),config,engine_factory=FastFakeEngine)
    for n in range(2,7):
        result=c.collect(seed=100+n,policy_version='frozen-u0',num_hands=2,force_n=n)
        assert result.total_hands==2
        assert result.hands_by_n[n]==2
        assert result.illegal_actions==0
        assert result.transitions
        assert all(len(t.observation.seats)==6 for t in result.transitions)
        assert all(math.isfinite(t.action_logprob) for t in result.transitions)
        assert all(abs(sum(rewards))<1e-10 for rewards in result.hand_rewards)
        grouped={}
        for t in result.transitions:
            grouped.setdefault((t.session_id,t.hand_id,t.seat_id),[]).append(t)
        assert all(sum(t.terminal_hand for t in stream)==1 for stream in grouped.values())
        samples=prepare_gae(result.transitions)
        assert len(samples)==len(result.transitions)
        assert all(math.isfinite(s.advantage) for s in samples)


def test_missing_terminal_rejected():
    c=MultiAgentCollector(mk_model(),RolloutConfig(hands_per_update=1),engine_factory=FastFakeEngine)
    t=c.collect(seed=21,policy_version='u0',force_n=2).transitions
    with pytest.raises(ValueError,match='unterminated'):
        prepare_gae(tuple(replace(x,terminal_hand=False) for x in t))


def test_policy_mask_gradient_and_ppo_stats():
    model=mk_model()
    config=RolloutConfig(hands_per_update=2,num_players=(2,))
    data=MultiAgentCollector(model,config,engine_factory=FastFakeEngine).collect(seed=123,policy_version='u0')
    old={k:v.clone() for k,v in model.state_dict().items()}
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-3)
    result=optimize_ppo(model,optimizer,prepare_gae(data.transitions),PPOConfig(epochs=1,minibatch_size=2))
    assert result['samples']==len(data.transitions)
    assert math.isfinite(result['entropy'])
    assert any(not torch.equal(old[k],new) for k,new in model.state_dict().items())
    t=data.transitions[0]
    probs=model.actor([t.encoded])[0].softmax(-1).detach().tolist()
    assert sum(probs)==pytest.approx(1)
    assert all(probs[i]==0 for i,x in enumerate(t.encoded.legal_mask) if not x)
    with pytest.raises(TypeError):model.value([{'hidden_cards':['cA','dA']}])


def test_deterministic_rollout_with_frozen_policy():
    c=MultiAgentCollector(mk_model(),RolloutConfig(hands_per_update=3),engine_factory=FastFakeEngine)
    x=c.collect(seed=414,policy_version='u0')
    y=c.collect(seed=414,policy_version='u0')
    assert [t.action for t in x.transitions]==[t.action for t in y.transitions]
    assert [t.reward_bb for t in x.transitions]==[t.reward_bb for t in y.transitions]
    assert x.hands_by_n==y.hands_by_n

````````

<!-- END_PY_SOURCE path="tests/expand/test_e03_returns_and_policy.py" -->

## tests/expand/test_e03_trainer.py

Original blob `190e08ee85730422ea730f00ba7f4fba91e3e3ed` (3203 bytes).

<a name="py-225"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e03_trainer.py" sha="190e08ee85730422ea730f00ba7f4fba91e3e3ed" -->

````````python
from dataclasses import replace
import json
import pytest
import torch
from holdem_ai.expand.rl import (ActorCritic, RolloutConfig, PPOConfig, OpponentMix)
from holdem_ai.expand.rl.trainer import (TrainConfig, train,load_trained_actor,
                                          evaluate_checkpoint, config_from_dict)
from e03_fake_engine import FastFakeEngine


def config():
    return TrainConfig(width=16,heads=2,layers=1,seed=11,rollout_seed=320000001,
                       validation_seed=330000001,eval_every=1,eval_hands=2,
                       snapshot_every=1,max_snapshots=2,
                       rollout=RolloutConfig(hands_per_update=2,num_players=(2,3),
                           mix=OpponentMix(shared_policy_fraction=0.2,snapshot_fraction=0.5)),
                       ppo=PPOConfig(epochs=1,minibatch_size=8))


def test_e03_checkpoint_resume_snapshot_validation(tmp_path):
    root=tmp_path/'a'
    a=train(root,config=config(),additional_updates=1,engine_factory=FastFakeEngine)
    assert a['completed_updates']==1
    assert (root/'latest.pt').exists() and (root/'best.pt').exists()
    assert (root/'snapshots/update_0000001.pt').exists()
    assert (root/'validation_anchor.json').exists()
    assert (root/'run.json').exists()
    b=train(root,resume=True,additional_updates=1,engine_factory=FastFakeEngine)
    assert b['completed_updates']==2
    assert (root/'snapshots/update_0000002.pt').exists()
    assert load_trained_actor(root/'latest.pt').actor.width==16
    assert load_trained_actor(root/'best.pt').actor.width==16
    assert len((root/'metrics.jsonl').read_text().strip().splitlines())==3
    with pytest.raises(FileExistsError):
        train(root,config=config(),additional_updates=1,engine_factory=FastFakeEngine)
    with pytest.raises(ValueError,match='resume config'):
        train(root,config=replace(config(),eval_hands=3),resume=True,additional_updates=1,
              engine_factory=FastFakeEngine)
    results=evaluate_checkpoint(root/'best.pt',hands=2,seed=340000000,
             num_players=(2,3),opponents=('random',),engine_factory=FastFakeEngine)
    assert len(results['results'])==2
    assert results['results']['3p_random']['hands']==2


def test_e03_resume_exact_reproducible(tmp_path):
    cfg=config()
    first=tmp_path/'first'
    second=tmp_path/'second'
    train(first,config=cfg,additional_updates=2,engine_factory=FastFakeEngine)
    train(second,config=cfg,additional_updates=1,engine_factory=FastFakeEngine)
    train(second,resume=True,additional_updates=1,engine_factory=FastFakeEngine)
    x=torch.load(first/'latest.pt',weights_only=True)
    y=torch.load(second/'latest.pt',weights_only=True)
    assert x['completed_updates']==y['completed_updates']==2
    assert all(torch.equal(v,y['model'][k]) for k,v in x['model'].items())
    assert x['snapshots']==y['snapshots']


def test_e03_checkpoint_tamper_rejected(tmp_path):
    root=tmp_path/'tamper'
    train(root,config=config(),additional_updates=1,engine_factory=FastFakeEngine)
    (root/'best.pt').write_bytes(b'fake best model')
    with pytest.raises(ValueError,match='hash'):
        train(root,resume=True,additional_updates=1,engine_factory=FastFakeEngine)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e03_trainer.py" -->

## tests/expand/test_e04_checkpoint.py

Original blob `49836fbeb0a82f2155ad983e636edf7d5b486a08` (2163 bytes).

<a name="py-226"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e04_checkpoint.py" sha="49836fbeb0a82f2155ad983e636edf7d5b486a08" -->

````````python
import copy
import gzip
import json
from pathlib import Path
import pytest
from holdem_ai.expand.pluribus.blueprint import (KuhnGame,LeducGame,MCCFRConfig,MCCFRSolver,
                                                   save_checkpoint,load_checkpoint)


def test_resume_exact_weights_and_round(tmp_path):
    a=MCCFRSolver(KuhnGame(2),MCCFRConfig(seed=9))
    b=MCCFRSolver(KuhnGame(2),MCCFRConfig(seed=9))
    a.train(200)
    b.train(100)
    target=tmp_path/'model.json.gz'
    save_checkpoint(target,b)
    restored=load_checkpoint(target)
    restored.train(100)
    assert a.completed_iterations==restored.completed_iterations
    assert a.store.as_dict()==restored.store.as_dict()


def test_checkpoint_roundtrip_leduc(tmp_path):
    s=MCCFRSolver(LeducGame(3),MCCFRConfig(seed=20))
    s.train(40)
    path=tmp_path/'l.json.gz'
    one=save_checkpoint(path,s)
    assert len(one['sha256'])==64
    loaded=load_checkpoint(path)
    assert loaded.game.num_players==3 and loaded.game.name=='leduc'
    assert loaded.store.as_dict()==s.store.as_dict()


def test_checksum_detects_tampering(tmp_path):
    s=MCCFRSolver(KuhnGame(2))
    s.train(5)
    path=tmp_path/'bad.json.gz'
    save_checkpoint(path,s)
    with gzip.open(path,'rt') as stream:content=json.load(stream)
    content['payload']['completed_iterations']=9999
    with gzip.open(path,'wt') as stream:json.dump(content,stream)
    with pytest.raises(ValueError,match='checksum'):
        load_checkpoint(path)


def test_checkpoint_incompatible_actions_rejected(tmp_path):
    s=MCCFRSolver(KuhnGame(2))
    s.train(8)
    path=tmp_path/'test.json.gz'
    save_checkpoint(path,s)
    with gzip.open(path,'rt') as stream:outer=json.load(stream)
    first=next(iter(outer['payload']['entries']))
    outer['payload']['entries'][first]['actions']=['fold']
    import hashlib
    payload=outer['payload']
    raw=json.dumps(payload,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()
    outer['sha256']=hashlib.sha256(raw).hexdigest()
    with gzip.open(path,'wt') as stream:json.dump(outer,stream)
    with pytest.raises(ValueError,match='array lengths'):
        load_checkpoint(path)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e04_checkpoint.py" -->

## tests/expand/test_e04_cli.py

Original blob `3855f9ce47556ae242a5b9247693124ea80edc03` (1169 bytes).

<a name="py-227"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e04_cli.py" sha="3855f9ce47556ae242a5b9247693124ea80edc03" -->

````````python
import json
import argparse
import pytest
from scripts.expand_phase04_mccfr import parser,run
from scripts.expand_phase04_evaluate import main as evaluate_main


def test_cli_kuhn_smoke_resume(tmp_path):
    root=tmp_path/'first'
    args=parser().parse_args(['--game','kuhn','--iterations','12','--root',str(root),
                              '--eval-hands','8'])
    one=run(args)
    assert one['completed_iterations']==12 and one['evaluation']['hands']==8
    resumed=run(parser().parse_args(['--root',str(root),'--resume','--iterations','8']))
    assert resumed['completed_iterations']==20
    with pytest.raises(FileExistsError):run(args)


def test_cli_leduc_smoke(tmp_path):
    root=tmp_path/'leduc'
    result=run(parser().parse_args(['--game','leduc','--iterations','4','--root',str(root)]))
    assert result['completed_iterations']==4


def test_cli_budget_limit_saves_empty_model(tmp_path):
    result=run(parser().parse_args(['--game','kuhn','--iterations','20','--node-budget','1',
                                    '--root',str(tmp_path/'limited')]))
    assert result['stopped_by_budget'] is not None
    assert result['completed_iterations']==0

````````

<!-- END_PY_SOURCE path="tests/expand/test_e04_cli.py" -->

## tests/expand/test_e04_holdem_mock.py

Original blob `047ff8de4d58a61bf6eba247383548bfd734b027` (2673 bytes).

<a name="py-228"></a>

<!-- BEGIN_PY_SOURCE path="tests/expand/test_e04_holdem_mock.py" sha="047ff8de4d58a61bf6eba247383548bfd734b027" -->

````````python
import random
import pytest
from holdem_ai.expand.pluribus.abstraction import AbstractionConfig, holdem_infoset_key
from holdem_ai.expand.pluribus.blueprint import HoldemGame,MCCFRSolver,MCCFRConfig,save_checkpoint,load_checkpoint
from e03_fake_engine import FastFakeEngine

@pytest.mark.parametrize('n',[2,3,4,5,6])
def test_real_interface_on_fast_deterministic_engine(n):
    game=HoldemGame(n,engine_factory=FastFakeEngine)
    state=game.start(random.Random(n))
    assert len(state.player_ids)==n and len(set(state.player_ids))==n
    key=state.infoset(state.current_player)
    assert 'e04-holdem' in key
    solver=MCCFRSolver(game,MCCFRConfig(seed=n,max_nodes_per_iteration=10000))
    stats=solver.train(12)
    assert stats[-1]['iteration']==12
    assert any(sum(e['strategy_sum'])>0 for e in solver.store.entries.values())


def test_private_rival_cards_do_not_affect_hero_infoset():
    game=HoldemGame(3,engine_factory=FastFakeEngine)
    state=game.start(random.Random(1))
    k=state.infoset(state.current_player)
    state.engine.some_private_opponent_cards = ['sA','cK']
    state.engine._future_deck = ['hA']
    assert k==state.infoset(state.current_player)


def test_postflop_card_buckets_and_abstraction_id():
    from copy import deepcopy
    game=HoldemGame(2,engine_factory=FastFakeEngine)
    state=game.start(random.Random(10))
    raw=state.engine.observe(state.current_player)['raw_obs']
    raw=deepcopy(raw)
    raw['street']='flop'
    raw['public_cards']=['hA','d2','s7']
    ident=holdem_infoset_key(raw,('fold','check'),AbstractionConfig(postflop_buckets=16))
    assert 'prototype_bucket' in ident
    assert AbstractionConfig(16).fingerprint != AbstractionConfig(32).fingerprint
    assert 'cA' not in ident # bucket, not exact private card


def test_holdem_save_and_restore(tmp_path):
    solver=MCCFRSolver(HoldemGame(3,engine_factory=FastFakeEngine), MCCFRConfig(seed=123))
    solver.train(9)
    path=tmp_path/'p.json.gz'
    save_checkpoint(path,solver)
    loaded=load_checkpoint(path,engine_factory=FastFakeEngine)
    assert loaded.game.abstraction.fingerprint==solver.game.abstraction.fingerprint
    assert loaded.store.as_dict()==solver.store.as_dict()
    loaded.train(2)
    assert loaded.completed_iterations==11


def test_no_illegal_abstract_actions_on_mock():
    state=HoldemGame(6,engine_factory=FastFakeEngine).start(random.Random(5))
    while not state.is_terminal():
        for x in state.legal_actions():
            assert x in ('fold','check','call','raise_half_pot','raise_pot','all_in')
        state=state.next_state(state.legal_actions()[-1])
    assert sum(state.terminal_payoffs())==pytest.approx(0)

````````

<!-- END_PY_SOURCE path="tests/expand/test_e04_holdem_mock.py" -->
