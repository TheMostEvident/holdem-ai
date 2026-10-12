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
