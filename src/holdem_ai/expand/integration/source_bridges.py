"""Opt-in E4/E5 source bridges with explicit infoset/belief contract.

There is NO generic safe conversion from E2 PublicAction to E4 exact engine-ID
history in E1-6; caller must supply a verified matching infoset-key factory.
Never infer opponent hole cards or reconstruct hidden engine internals here.
"""
from __future__ import annotations
from typing import Callable
from holdem_ai.expand.information.schema import InformationState, ACTION_NAMES


def from_e04_checkpoint(path: str, *, exact_infoset_key:Callable[[InformationState],str]):
    from holdem_ai.expand.pluribus.blueprint.checkpoint import load_checkpoint
    solved=load_checkpoint(path)
    if solved.game.name!='holdem':
        raise ValueError('cannot deploy toy-game blueprint into Holdem')
    if not callable(exact_infoset_key):raise TypeError('verified exact_infoset_key required')
    def distribution(state:InformationState):
        if state.num_dealt_in!=solved.game.num_players:
            raise ValueError('blueprint player count incompatible with observation')
        key=exact_infoset_key(state)
        entry=solved.store.entries.get(key)
        if entry is None:raise ValueError('unseen blueprint infoset; do not pretend uniform is trained')
        available=tuple(a for a,can in zip(ACTION_NAMES,state.legal_actions.mask) if can)
        if set(entry['actions'])!=set(available):
            raise ValueError('action abstraction incompatibility: exact action set differs')
        weights=solved.store.policy(key,tuple(entry['actions']),average=True)
        table=dict(zip(entry['actions'],weights))
        return {a:table[a] for a in available}
    return distribution,solved.game.abstraction.fingerprint


def from_e05_solver(solver, *, verified_root_belief:Callable[[InformationState],object]):
    """Budget-limited search source; throws on fallback/timeout for agent router."""
    if not callable(verified_root_belief):raise TypeError('verified root belief callable required')
    def distribution(state:InformationState):
        result=solver.search(verified_root_belief(state))
        if result.status!='solved':
            raise TimeoutError(f'search incomplete: {result.status}')
        legal=tuple(a for a,yes in zip(ACTION_NAMES,state.legal_actions.mask) if yes)
        if set(result.action_probabilities)!=set(legal):
            raise ValueError('search root action abstraction incompatible')
        return dict(result.action_probabilities)
    return distribution
