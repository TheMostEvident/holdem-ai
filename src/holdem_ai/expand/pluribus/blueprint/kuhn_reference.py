"""Exact 2-player Kuhn exploitability (unabstracted 3-card game only).

For each player, enumerate deterministic response mappings on that player's
information sets; expectation over all six uniformly likely 3P2 deals and
opponent's averaged behavioral strategy. Small toy game only; NOT a 6-max
Nash/exploitability metric.
"""
from __future__ import annotations
from itertools import permutations, product
from .toy_games import ToyState


def _root(hole):
    return ToyState('kuhn',tuple(hole),None,0,(1,1),(0,0),(True,True),(0,1),False,())


def _value(state, player, store, mapping):
    if state.is_terminal():
        return state.terminal_payoffs()[player]
    acting=state.current_player
    actions=state.legal_actions()
    key=state.infoset(acting)
    if acting==player:
        return _value(state.next_state(mapping[key]),player,store,mapping)
    probs=store.policy(key,actions,average=True)
    return sum(p*_value(state.next_state(a),player,store,mapping)
               for p,a in zip(probs,actions))


def _collect(state, player, choices):
    if state.is_terminal():
        return
    if state.current_player==player:
        choices[state.infoset(player)]=state.legal_actions()
    for action in state.legal_actions():
        _collect(state.next_state(action),player,choices)


def kuhn_nash_conv(store):
    """Return max_resp(player0) + max_resp(player1); NashConv >= 0."""
    deals=tuple(permutations(range(3),2))
    values=[]
    for player in (0,1):
        choices={}
        for deal in deals:
            _collect(_root(deal),player,choices)
        keys=sorted(choices)
        best=float('-inf')
        for mapping_actions in product(*(choices[key] for key in keys)):
            mapping=dict(zip(keys,mapping_actions))
            value=sum(_value(_root(deal),player,store,mapping) for deal in deals)/len(deals)
            best=max(best,value)
        values.append(best)
    return {'nash_conv':sum(values),'best_response_values':values,
            'note':'Kuhn-only exact, not 6-max exploitability'}
