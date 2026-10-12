"""Allowlisted public-only events and fixed-cardinality legal-action context.

player_session_id is a *join tenure ID*, not a physical seat or table session.
A returning player must receive a new tenure, even when player_id is reused.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from holdem_ai.expand.information.schema import ACTION_NAMES, STREETS, InformationState

POSITIONS = ('button', 'small_blind', 'big_blind', 'early', 'middle', 'cutoff', 'other')
RAISE_ACTIONS = frozenset({'raise_half_pot','raise_pot','raise_double_pot','all_in'})


def spr_bin(spr: float) -> str:
    if not isinstance(spr, (float,int)) or isinstance(spr,bool) or not math.isfinite(spr) or spr < 0:
        raise ValueError('SPR must be finite nonnegative')
    return 'lt1' if spr < 1 else '1to4' if spr < 4 else '4to10' if spr < 10 else 'ge10'


@dataclass(frozen=True)
class PublicDecision:
    player_id: str
    player_session_id: str
    hand_id: str
    action_index: int
    seat_id: int
    street: str
    position: str
    spr_bucket: str
    legal: tuple[str, ...]
    action: str
    preflop_raises_before: int = 0

    def __post_init__(self):
        for x in (self.player_id,self.player_session_id,self.hand_id):
            if type(x) is not str or not x.strip():raise ValueError('nonempty identity and hand IDs required')
        if type(self.action_index) is not int or self.action_index < 0:raise ValueError('invalid action_index')
        if type(self.seat_id) is not int or not 0 <= self.seat_id < 6:raise ValueError('invalid seat')
        if self.street not in STREETS or self.position not in POSITIONS:
            raise ValueError('invalid street/position')
        if self.spr_bucket not in ('lt1','1to4','4to10','ge10'):
            raise ValueError('invalid SPR bucket')
        if (not isinstance(self.legal,tuple) or not self.legal
                or len(self.legal)!=len(set(self.legal))
                or any(a not in ACTION_NAMES for a in self.legal)
                or tuple(a for a in ACTION_NAMES if a in self.legal)!=self.legal):
            raise ValueError('legal actions must be unique in canonical order')
        if self.action not in self.legal:raise ValueError('observed action must have been legal')
        if type(self.preflop_raises_before) is not int or self.preflop_raises_before < 0:
            raise ValueError('invalid raise count')

    @property
    def identity(self) -> tuple[str,str]:
        return (self.player_id,self.player_session_id)

    @property
    def context(self) -> tuple[str,str,str,tuple[str,...]]:
        return (self.street,self.position,self.spr_bucket,self.legal)

    @property
    def event_id(self) -> tuple[str,str,str,int]:
        return (*self.identity,self.hand_id,self.action_index)


def public_decision_from_state(state: InformationState, *, player_id: str,
                               player_session_id: str, hand_id: str,
                               action_index: int, action: str, position: str,
                               preflop_raises_before: int = 0) -> PublicDecision:
    """Immediately project acting-player state into public fields ONLY.

    Caller must supply the seat's authenticated player/tenure identity and the
    publicly observable action. The state's private hero_hole is NEVER used.
    """
    if type(state) is not InformationState:
        raise TypeError('requires Phase2 InformationState, not an engine/private state')
    seat = state.seats[state.hero_seat]
    if not seat.dealt_in or not seat.occupied:raise ValueError('decision seat unavailable')
    legal = tuple(a for a,allowed in zip(ACTION_NAMES,state.legal_actions.mask) if allowed)
    stack = seat.stack_bb
    total_pot = state.pot_bb
    spr = stack / max(total_pot,1e-9)
    return PublicDecision(player_id,player_session_id,hand_id,action_index,
                          state.hero_seat,state.street,position,spr_bin(spr),legal,
                          action,preflop_raises_before)
