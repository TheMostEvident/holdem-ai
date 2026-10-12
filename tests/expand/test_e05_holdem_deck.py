import random
import pytest
from holdem_ai.expand.pluribus.search.belief import DECK,HoldemObservationBelief

@pytest.mark.parametrize('n',[2,3,4,5,6])
def test_card_layout_exact_hero_visible_board_and_no_collision(n):
    occupied=([0,2,3,4,5] if n<=5 else [0,1,2,3,4,5])[:n]
    button=occupied[0]
    hero=occupied[-1]
    raw={'player_id':hero,'button':button,'hand':['sA','cK'],
         'public_cards':['d2','h3','s4','c5','d6'],
         'street':'river','stacks':[40 if i in occupied else 0 for i in range(6)],
         'street_bets':[0]*6,'action_history':(), 'action_raise_to':()}
    s=HoldemObservationBelief(raw,initial_stacks=tuple(raw['stacks']),
                              small_blind=1,big_blind=2)
    for seed in range(3):
        cards=s._random_deck(random.Random(seed))
        assert len(cards)==52 and set(cards)==DECK and len(set(cards))==52
        dealing=[]
        pos=button
        for _ in occupied:
            pos=next((pos+d)%6 for d in range(1,7) if (pos+d)%6 in occupied)
            dealing.append(pos)
        index=dealing.index(hero)
        assert cards[index]=='sA'
        assert cards[n+index]=='cK'
        assert cards[2*n+1:2*n+4]==['d2','h3','s4']
        assert cards[2*n+5]=='c5' and cards[2*n+7]=='d6'


def test_card_collisions_and_bad_hole_rejected():
    x={'player_id':0,'button':0,'hand':['sA','sA'], 'public_cards':[],
       'stacks':[20,20,0,0,0,0], 'street_bets':[0]*6,'action_history':(),
       'action_raise_to':()}
    with pytest.raises(ValueError):
        HoldemObservationBelief(x,initial_stacks=(20,20,0,0,0,0),small_blind=1,big_blind=2)
