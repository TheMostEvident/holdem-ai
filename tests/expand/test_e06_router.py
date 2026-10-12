import pytest
from holdem_ai.expand.opponent import (OpponentRegistry,OpponentConfig,PublicDecision,StyleExpertRouter,
                                       SafeAdaptivePolicy)

OPP_LEGAL=('fold','call','raise_pot')
HERO_LEGAL=('check','raise_half_pot','raise_pot')
EXPERTS={
 'fold_heavy':{'check':.1,'raise_half_pot':.3,'raise_pot':.6},
 'calling':{'check':.2,'raise_half_pot':.6,'raise_pot':.2},
 'aggressive':{'check':.65,'raise_half_pot':.30,'raise_pot':.05},
}

def fitted(action):
    cfg=OpponentConfig(min_adapt_samples=12)
    reg=OpponentRegistry(cfg);reg.bind_seat(4,'a','tenure1')
    evidence=None
    for i in range(65):
        evidence=reg.observe(PublicDecision('a','tenure1',f'h{i}',i,4,'flop','cutoff','1to4',OPP_LEGAL,action))
    return cfg,evidence

@pytest.mark.parametrize('action,style', [('fold','fold_heavy'),('call','calling'),('raise_pot','aggressive')])
def test_router_uses_different_experts_for_styles(action,style):
    cfg,evidence=fitted(action)
    route=StyleExpertRouter(cfg).propose(evidence,EXPERTS,hero_legal=HERO_LEGAL)
    assert route.style==style and route.reason=='routed_existing_expert'
    assert route.proposal==EXPERTS[style]
    base={'check':.6,'raise_half_pot':.2,'raise_pot':.2}
    # Hero and opponent legal action signatures differ, which MUST be allowed.
    guarded=SafeAdaptivePolicy(cfg).adjust(base,route.proposal,evidence,legal=HERO_LEGAL)
    assert guarded.used_proposal
    assert sum(guarded.probabilities.values())==pytest.approx(1)

def test_router_uncertain_fallback():
    assert StyleExpertRouter().propose(None,EXPERTS,hero_legal=HERO_LEGAL).proposal is None
    cfg,e=fitted('fold')
    route=StyleExpertRouter(cfg).propose(e,{},hero_legal=HERO_LEGAL)
    assert route.reason=='missing_expert' and route.proposal is None
