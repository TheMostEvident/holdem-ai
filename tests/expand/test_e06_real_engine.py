"""Runs only once Base Phase 3 engine is present in user's real repository."""
import pytest
pytest.importorskip('holdem_ai.environment.multi_player_engine')
from holdem_ai.expand.environment import DynamicTable
from holdem_ai.expand.information import Phase2TableAdapter
from holdem_ai.expand.opponent import OpponentRegistry,public_decision_from_state


def test_public_event_hook_with_real_hand_engine():
    table=DynamicTable()
    tenures={}
    for i in range(4):
        tenures[f'p{i}']=table.request_join(f'p{i}',10000,i)
    bridge=Phase2TableAdapter(table)
    reg=OpponentRegistry()
    used=0
    for hand_seed in range(100,115):
        hand=bridge.start_hand(hand_seed)
        assert hand is not None
        for seat,pid in hand.participants:
            if seat not in reg.active_by_seat:reg.bind_seat(seat,pid,tenures[pid])
        for action_index in range(180):
            if table.current_player_id is None:break
            info=bridge.observe()
            action='check' if info.legal_actions.allows('check') else 'call'
            pid=table.current_player_id
            event=public_decision_from_state(info,player_id=pid,player_session_id=tenures[pid],
                hand_id=hand.hand_id,action_index=action_index,action=action,
                position=('button' if info.hero_seat==hand.button_seat else
                    'small_blind' if info.hero_seat==hand.small_blind_seat else
                    'big_blind' if info.hero_seat==hand.big_blind_seat else 'other'))
            reg.observe(event)
            bridge.act_abstract(action)
            used+=1
        else:raise AssertionError('engine decisions exceeded')
        result=bridge.settle_hand()
        assert sum(result.payoffs_chips)==0
        for seat,pid in hand.participants:reg.models[(pid,tenures[pid])].end_hand(hand.hand_id)
    assert used>0
    assert sum(len(m.seen_events) for m in reg.models.values())==used
