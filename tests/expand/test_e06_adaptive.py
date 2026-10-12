import pytest
from holdem_ai.expand.opponent import (OpponentConfig,OpponentRegistry,PublicDecision,SafeAdaptivePolicy,
                                       GuardConfig,kl_divergence,tv_distance)

LEGAL=('fold','call','raise_pot')
BASE={'fold':.2,'call':.6,'raise_pot':.2}
PROPOSAL={'fold':.05,'call':.05,'raise_pot':.90}

def fitted(n=75):
    cfg=OpponentConfig(min_adapt_samples=12)
    r=OpponentRegistry(cfg);r.bind_seat(1,'opp','tenure-1')
    x=None
    for i in range(n):
        x=r.observe(PublicDecision('opp','tenure-1',f'h{i}',i,1,'preflop','button','4to10',LEGAL,'call'))
    return cfg,x

def test_no_evidence_preserves_base():
    gate=SafeAdaptivePolicy()
    ans=gate.adjust(BASE,PROPOSAL,None,legal=LEGAL)
    assert ans.probabilities==BASE and not ans.used_proposal

def test_low_samples_no_adapt():
    cfg,e=fitted(4)
    ans=SafeAdaptivePolicy(cfg).adjust(BASE,PROPOSAL,e,legal=LEGAL)
    assert ans.probabilities==BASE and ans.reason=='insufficient_evidence'

def test_trust_kl_tv_bounds_and_probability_simplex():
    cfg,e=fitted()
    limits=GuardConfig(kl_budget=.002,tv_budget=.011,max_mix=.8)
    ans=SafeAdaptivePolicy(cfg,limits).adjust(BASE,PROPOSAL,e,legal=LEGAL)
    assert ans.used_proposal
    assert 0<ans.mixture_weight<.8
    assert sum(ans.probabilities.values())==pytest.approx(1)
    assert kl_divergence(ans.probabilities,BASE)<=limits.kl_budget+1e-10
    assert tv_distance(ans.probabilities,BASE)<=limits.tv_budget+1e-10
    assert ans.probabilities['raise_pot']>BASE['raise_pot']

def test_zero_support_cannot_create_probability():
    cfg,e=fitted()
    base={'fold':0.,'call':1.,'raise_pot':0.}
    out=SafeAdaptivePolicy(cfg).adjust(base,PROPOSAL,e,legal=LEGAL)
    assert out.probabilities==base
    assert not out.used_proposal

def test_changepoint_fallback():
    cfg,e=fitted()
    from dataclasses import replace
    e=replace(e,shift_active=True)
    ans=SafeAdaptivePolicy(cfg).adjust(BASE,PROPOSAL,e,legal=LEGAL)
    assert ans.reason=='change_point_fallback'
    assert ans.probabilities==BASE

def test_no_proposal_unchanged_even_with_evidence():
    cfg,e=fitted()
    ans=SafeAdaptivePolicy(cfg).adjust(BASE,None,e,legal=LEGAL)
    assert ans.reason=='no_exploitation_proposal' and not ans.used_proposal

def test_illegal_action_or_invalid_distribution_rejected():
    cfg,e=fitted()
    with pytest.raises(ValueError):SafeAdaptivePolicy(cfg).adjust(BASE,{'fold':.1,'call':.9,'all_in':.0},e,legal=LEGAL)
    with pytest.raises(ValueError):SafeAdaptivePolicy(cfg).adjust(BASE,{'fold':.1,'call':-.9,'raise_pot':1.8},e,legal=LEGAL)

def test_context_mismatch_is_fallback():
    cfg,e=fitted()
    ans=SafeAdaptivePolicy(cfg).adjust({'fold':.5,'call':.5}, {'fold':.2,'call':.8},e,
        legal=('fold','call'),expected_opponent_context=('river','button','4to10',LEGAL))
    assert ans.reason=='opponent_context_mismatch'


def test_hero_legal_and_opponent_legal_may_differ():
    cfg,e=fitted()
    legal=('fold','call')
    ans=SafeAdaptivePolicy(cfg).adjust({'fold':.5,'call':.5}, {'fold':.1,'call':.9},e,legal=legal,
                                       expected_opponent_context=e.context)
    assert ans.used_proposal and set(ans.probabilities)==set(legal)
