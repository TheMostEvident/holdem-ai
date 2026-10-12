import pytest
from holdem_ai.expand.pluribus.search import CONTINUATIONS,continuation_policy

@pytest.mark.parametrize('variant',CONTINUATIONS)
def test_all_continuations_have_valid_legal_distribution(variant):
    acts=('fold','check','call','raise_half_pot','raise_pot','all_in')
    probs=continuation_policy(acts,[1]*len(acts),variant)
    assert set(probs)==set(acts)
    assert all(0<=v<=1 for v in probs.values())
    assert sum(probs.values())==pytest.approx(1)

def test_fivefold_weighting_is_over_original_mass():
    acts=('fold','call','raise_pot')
    b=[0.2,0.3,0.5]
    fold=continuation_policy(acts,b,'fold_bias')
    assert fold['fold']==pytest.approx(1/1.8)
    call=continuation_policy(acts,b,'call_bias')
    assert call['call']==pytest.approx(1.5/2.2)
    r=continuation_policy(acts,b,'raise_bias')
    assert r['raise_pot']==pytest.approx(2.5/3)
    assert continuation_policy(acts,b,'blueprint')['call']==pytest.approx(.3)

def test_raise_bias_affects_all_raise_sizes_not_check():
    p=continuation_policy(('check','raise_half_pot','raise_pot','all_in'),(1,1,1,1),'raise_bias')
    assert p['raise_half_pot']==p['raise_pot']==p['all_in']
    assert p['raise_pot']==pytest.approx(5*p['check'])

@pytest.mark.parametrize('bad',[(-1,1),(0,0),(float('nan'),1)])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError): continuation_policy(('fold','call'),bad,'blueprint')
