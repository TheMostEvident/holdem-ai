import math
import pytest
from holdem_ai.expand.opponent import OpponentConfig,OpponentRegistry, PublicDecision, spr_bin

A=('fold','call','raise_pot')

def event(i,action='call',*,player='alice',tenure='join1',seat=2,street='preflop',position='button',legal=A,hand=None,pre=0):
    return PublicDecision(player,tenure,hand or f'h{i}',i,seat,street,position,'4to10',legal,action,pre)

def test_spr_buckets_and_validation():
    assert [spr_bin(x) for x in (0,1,4,10)]==['lt1','1to4','4to10','ge10']
    for v in (-1,math.inf,math.nan):
        with pytest.raises(ValueError):spr_bin(v)
    with pytest.raises(ValueError):event(1,'all_in')
    with pytest.raises(ValueError):event(1,legal=('call','fold'))
    with pytest.raises(ValueError):event(1,seat=6)

def test_session_isolation_across_same_seat():
    r=OpponentRegistry()
    r.bind_seat(2,'alice','join1')
    for i in range(40):r.observe(event(i,'fold'))
    old=r.evidence(2,event(1).context)
    assert old.posterior['fold']>.95
    r.unbind_seat(2)
    r.bind_seat(2,'bob','join2')
    fresh=r.evidence(2,event(1).context)
    assert fresh.raw_events==0 and fresh.posterior['fold']==pytest.approx(1/3)
    with pytest.raises(ValueError):r.observe(event(41,'fold'))
    r.unbind_seat(2)
    r.bind_seat(2,'alice','join3')
    assert r.evidence(2,event(1).context).raw_events==0
    assert ('alice','join1') in r.models

def test_no_seat_lease_overlap():
    r=OpponentRegistry();r.bind_seat(0,'a','t1')
    with pytest.raises(ValueError):r.bind_seat(1,'a','t1')

def test_legal_context_and_street_conditioned():
    r=OpponentRegistry();r.bind_seat(2,'alice','join1')
    for i in range(25):r.observe(event(i,'fold'))
    alt=event(27,'call',street='river')
    assert r.evidence(2,alt.context).raw_events==0
    alt2=event(28,'call',legal=('fold','call'))
    assert r.evidence(2,alt2.context).raw_events==0
    alt3=event(29,'call',position='early')
    assert r.evidence(2,alt3.context).raw_events==0

def test_duplicate_decisions_fail_closed():
    r=OpponentRegistry();r.bind_seat(2,'alice','join1')
    x=event(1)
    r.observe(x)
    with pytest.raises(ValueError):r.observe(x)

def test_dirichlet_smoothing_and_decay():
    r=OpponentRegistry();r.bind_seat(2,'alice','join1')
    x=event(0)
    assert r.evidence(2,x.context).posterior['fold']==pytest.approx(1/3)
    for i in range(55):r.observe(event(i,'fold'))
    p=r.evidence(2,x.context).posterior
    assert p['fold']>p['call']>0
    assert sum(p.values())==pytest.approx(1)
    assert r.evidence(2,x.context).effective_recent<r.evidence(2,x.context).effective_long

def test_style_shift_detection_and_recovery():
    cfg=OpponentConfig(long_decay=.999,recent_decay=.89,min_long=25,min_recent=4,
                       shift_kl_threshold=.08,shift_consecutive=2,shift_hold_events=12)
    r=OpponentRegistry(cfg);r.bind_seat(2,'alice','join1')
    active_before=False
    for i in range(110):
        out=r.observe(event(i,'fold'))
        active_before |= out.shift_active
    assert not active_before
    signals=[]
    for i in range(110,185):
        out=r.observe(event(i,'raise_pot'))
        signals.append(out.shift_active)
    assert any(signals), 'must detect large probability shift'
    assert out.shift_events>0

def test_vpip_pfr_3bet_and_multi_decisions_one_hand():
    r=OpponentRegistry();r.bind_seat(2,'alice','join1')
    r.observe(event(1,'call',hand='h1'))
    r.observe(event(2,'raise_pot',hand='h1',pre=1))
    r.models[('alice','join1')].end_hand('h1')
    r.models[('alice','join1')].end_hand('h1')
    r.observe(event(3,'fold',hand='h2'))
    r.models[('alice','join1')].end_hand('h2')
    out=r.models[('alice','join1')].diagnostics()
    assert out['preflop_hands']==2
    assert out['vpip_rate']==pytest.approx(.5)
    assert out['pfr_rate']==pytest.approx(.5)
    assert out['threebet_rate']==pytest.approx(.5)

def test_capacity_no_silent_eviction():
    cfg=OpponentConfig(max_event_ids=2)
    r=OpponentRegistry(cfg);r.bind_seat(2,'alice','join1')
    r.observe(event(0));r.observe(event(1))
    with pytest.raises(RuntimeError):r.observe(event(2))

def test_model_does_not_contain_opponent_private_cards():
    r=OpponentRegistry();r.bind_seat(2,'alice','join1')
    r.observe(event(1))
    assert not any(k in r.models[('alice','join1')].__dict__ for k in ('hole','deck','board','critic'))
