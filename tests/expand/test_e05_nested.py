import pytest
from holdem_ai.expand.pluribus.search import NestedSearchSession

def test_off_tree_re_root_and_committed_own_actions():
    s=NestedSearchSession(hero=0)
    s.observe_action(actor=1,street='preflop',action='raise_to',raise_to=75,abstract_actions=('fold','call','raise_pot'))
    assert s.off_tree_count==1
    s.observe_action(actor=0,street='preflop',action='call',infoset_key='hero:AA')
    with pytest.raises(ValueError):
        s.observe_action(actor=0,street='preflop',action='fold',infoset_key='hero:AA')
    s.observe_action(actor=1,street='flop',action='check')
    assert s.street_resets==1
    assert s.own_commitments[('preflop','hero:AA')]=='call'
    assert len(s.public_trace())==3

def test_reject_bad_off_tree_size():
    s=NestedSearchSession(0)
    with pytest.raises(ValueError):s.observe_action(actor=1,action='raise_to',street='flop',raise_to=-1)
