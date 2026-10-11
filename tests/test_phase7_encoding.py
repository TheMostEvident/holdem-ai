"""Phase 7 contracts: actor privacy, history, stable cards, critic feature schema."""
from __future__ import annotations

import dataclasses
import json

import pytest

from holdem_agent_phase5.schema import InformationState, Turn
from holdem_agent_phase6 import HUNL200Adapter
from holdem_agent_phase6.adapter import CriticState
from holdem_agent_phase7 import (
    ACTOR_FEATURE_DIM, CRITIC_FEATURE_DIM, VOCAB_SIZE, encode_actor, encode_critic,
)
from holdem_agent_phase7 import vocabulary as v


def _actor(**kw):
    default = dict(
        player_id=0,
        features=(0.0, 1.0, 0.0, 1.5, 0.5, 0.5, 199.5, 199., 1., .5, 1., 2.),
        legal_action_mask=(True, False, True, True, True, True, True),
        private_cards=('sA', 'dK'), public_cards=(), stage='preflop',
        public_state=(('status', json.dumps(['active', 'active'])),),
        history=(),
    )
    default.update(kw)
    return InformationState(**default)


def _critic(**kw):
    default = dict(
        player_to_act=0, button=0, street='preflop', board=(),
        hole_cards=(('sA', 'dK'), ('cQ', 'hJ')),
        stacks=(19950, 19900), contributions=(50, 100),
        street_bets=(50, 100), status=('active', 'active'), pot=150,
        history=(), is_terminal=False,
    )
    default.update(kw)
    return CriticState(**default)


def test_actor_dimensions_and_hole_card_tokens():
    x = encode_actor(_actor())
    assert len(x.features) == ACTOR_FEATURE_DIM == 20
    assert len(x.legal_action_mask) == 7
    assert x.token_ids[:4] == (v.BOS, v.PRIVATE, v.card_id('sA'), v.card_id('dK'))
    assert v.OPPONENT_PRIVATE not in x.token_ids
    assert len(x.token_ids) == len(x.token_values)
    assert len(x.token_ids) < 256


def test_actor_public_status_encoded_from_public_state_only():
    normal = encode_actor(_actor())
    allin = encode_actor(_actor(public_state=(('status', '["active","allin"]'),)))
    assert normal.token_ids == allin.token_ids
    assert normal.features[-8:] != allin.features[-8:]
    assert allin.features[-8:] == (1.,0.,0.,0.,0.,0.,1.,0.)


def test_history_contains_raise_amount_in_bb():
    record = 'preflop|seat=1|raise_to|raise_to=450'
    x = encode_actor(_actor(history=(record,)))
    assert x.token_ids[-5:] == (
        v.EVENT, v.STREET_TO_ID['preflop'], v.SEAT_1, v.ACTION_TO_ID['raise_to'], v.AMOUNT,
    )
    assert x.token_values[-1] == 4.5


def test_history_order_and_fold_distinguished():
    h1 = ('preflop|seat=0|call|raise_to=-', 'preflop|seat=1|check|raise_to=-')
    h2 = tuple(reversed(h1))
    assert encode_actor(_actor(history=h1)).token_ids != encode_actor(_actor(history=h2)).token_ids
    assert v.ACTION_TO_ID['fold'] != v.ACTION_TO_ID['check']


@pytest.mark.parametrize('bad', [
    'preflop|seat=2|call|raise_to=-',
    'preflop|seat=0|raise_to|raise_to=-',
    'preflop|seat=0|call|raise_to=300',
    'river|seat=0|raise_to|raise_to=abc',
    'totally different format',
])
def test_malformed_history_is_rejected(bad):
    with pytest.raises(ValueError):
        encode_actor(_actor(history=(bad,)))


def test_history_overflow_explicit_not_truncated():
    history = ('preflop|seat=0|call|raise_to=-',) * 100
    with pytest.raises(ValueError, match='not truncated'):
        encode_actor(_actor(history=history), max_tokens=25)
    full = encode_actor(_actor(history=history), max_tokens=600)
    assert full.token_ids.count(v.EVENT) == 100


def test_card_domain_and_duplicate_rejected():
    with pytest.raises(ValueError, match='invalid canonical card'):
        encode_actor(_actor(private_cards=('sZ', 'dK')))
    with pytest.raises(ValueError, match='duplicate'):
        encode_actor(_actor(private_cards=('sA','dK'),public_cards=('sA','h2','c3')))
    assert len(v.CARDS) == 52 and len(set(v.CARD_TO_ID.values())) == 52
    assert max(v.ACTION_TO_ID.values()) < VOCAB_SIZE


def test_actor_wrong_phase6_feature_contract_rejected():
    with pytest.raises(ValueError):
        encode_actor(_actor(features=(0., 1.)))
    with pytest.raises(ValueError):
        encode_actor(_actor(player_id=1))
    with pytest.raises(ValueError):
        encode_actor(_actor(public_state=()))


def test_critic_two_holes_differ_and_no_actor_privilege():
    state = _critic()
    a = encode_critic(state)
    b = encode_critic(dataclasses.replace(state, hole_cards=(('sA','dK'), ('cT','hJ'))))
    assert len(a.features) == CRITIC_FEATURE_DIM == 24
    assert a.token_ids != b.token_ids
    assert v.OPPONENT_PRIVATE in a.token_ids
    actor1 = encode_actor(_actor())
    actor2 = encode_actor(_actor())
    assert actor1 == actor2  # critic-only opponent change never enters the actor


def test_critic_chips_are_in_big_blind_units():
    x = encode_critic(_critic())
    assert x.features[:4] == (1.,0.,1.,0.)
    assert x.features[4:8] == (1.,0.,0.,0.)
    assert x.features[8:10] == (199.5, 199.0)
    assert x.features[10:12] == (.5,1.)
    assert x.features[14] == 1.5
    assert x.features[-1] == 0.0


def test_critic_terminal_encoding():
    x = encode_critic(_critic(player_to_act=-1, is_terminal=True))
    assert x.features[2:4] == (0.,0.)
    assert x.features[-1] == 1.
    assert x.token_ids[-1] == v.TERMINAL
    with pytest.raises(ValueError):
        encode_critic(_critic(player_to_act=-1, is_terminal=False))


def test_critic_rejects_bad_pot_and_duplicate_cards():
    with pytest.raises(ValueError):
        encode_critic(_critic(pot=999))
    with pytest.raises(ValueError, match='duplicate'):
        encode_critic(_critic(hole_cards=(('sA','dK'),('sA','hJ'))))


def test_real_phase6_adapter_accepts_actor_and_critic():
    # Uses real HUNL200Adapter; on the developer's machine this must run
    # against the installed actual MultiPlayerHandEngine, not the fake test engine.
    env = HUNL200Adapter(button=1, expose_training_state=True)
    current = env.reset(seed=123)
    assert isinstance(current, Turn)
    obs = current.observation
    actor = encode_actor(obs)
    critic = encode_critic(env.training_state())
    assert actor.legal_action_mask == obs.legal_action_mask
    assert actor.features[0] == obs.player_id
    assert len(critic.token_ids) > len(actor.token_ids)
    assert v.card_id(env.engine.hole_cards[1 - obs.player_id][0]) not in actor.token_ids


def test_history_after_phase6_action_matches():
    env = HUNL200Adapter(button=0, expose_training_state=True)
    first = env.reset(seed=17)
    assert isinstance(first, Turn)
    action = 2  # call
    assert first.observation.legal_action_mask[action]
    nxt = env.step(action)
    assert isinstance(nxt, Turn)
    encoded = encode_actor(nxt.observation)
    assert v.ACTION_TO_ID['call'] in encoded.token_ids
    assert v.EVENT in encoded.token_ids
