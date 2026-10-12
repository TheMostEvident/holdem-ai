"""Phase1/Phase2 public event bridge, called ONLY after a legal action commits.

Caller flow:
  before = phase2.observe()
  phase2.act_abstract(action)             # may raise; no Phase6 state written
  tracker.record_accepted_action(before, handle=..., ...)  # now commit evidence

Physical seat IDs and player-session IDs must come from the trusted Phase1
seat registry and unique request_join event token (not from hole cards).
"""
from __future__ import annotations
from typing import Mapping
from holdem_ai.expand.information.schema import InformationState, PublicAction
from holdem_ai.expand.environment.table_state import HandHandle, TableSnapshot
from .model import OpponentRegistry, ModelEvidence
from .schema import PublicDecision, public_decision_from_state, RAISE_ACTIONS


def position_for_seat(handle:HandHandle,seat:int)->str:
    occupied=set(x for x,_ in handle.participants)
    if seat not in occupied:raise ValueError('seat not in dealt hand')
    if seat==handle.button_seat:return 'button'
    if seat==handle.small_blind_seat:return 'small_blind'
    if seat==handle.big_blind_seat:return 'big_blind'
    others=[]
    i=handle.big_blind_seat
    while True:
        i=(i+1)%6
        if i==handle.button_seat:break
        if i in occupied and i not in (handle.small_blind_seat,handle.big_blind_seat):
            others.append(i)
    if seat not in others:raise ValueError('nonstandard hand position')
    if seat==others[-1]:return 'cutoff'
    if seat==others[0]:return 'early'
    return 'middle'


def count_prior_preflop_raises(history:tuple[PublicAction,...])->int:
    return sum(x.street=='preflop' and (x.kind in RAISE_ACTIONS or x.kind=='raise_to') for x in history)


class PublicOpponentTracker:
    def __init__(self, registry:OpponentRegistry|None=None):
        self.registry=registry or OpponentRegistry()

    def sync_seats(self,snapshot:TableSnapshot,tenure_by_player:Mapping[str,str])->None:
        """Sync public Phase1 roster, using unique join request IDs as tenures.

        Must be called AFTER applying table-boundary events. A rejoin gets a
        new tenure even if physical seat and display player ID are unchanged.
        """
        wanted={}
        for s in snapshot.seats:
            if s.player_id is None:continue
            if s.player_id not in tenure_by_player:
                raise ValueError('tenure token not supplied for seated player')
            wanted[s.seat_id]=(s.player_id,tenure_by_player[s.player_id])
        for seat,identity in list(self.registry.active_by_seat.items()):
            if wanted.get(seat)!=identity:self.registry.unbind_seat(seat)
        for seat,identity in wanted.items():
            self.registry.bind_seat(seat,*identity)

    def record_accepted_action(self, before:InformationState, *,
                               handle:HandHandle, player_id:str,player_session_id:str,
                               action_index:int, action:str)->ModelEvidence:
        """Commit action evidence ONLY after the game engine accepts the action.

        `action` must be one of the Phase2 abstract action names and legal in
        the *before* observation. Arbitrary exact off-tree bet amounts cannot
        be assigned to this discrete action set without a separate translator.
        """
        if handle.terminal or before.hero_seat not in dict(handle.participants):
            raise ValueError('not an eligible hand decision')
        if dict(handle.participants)[before.hero_seat]!=player_id:
            raise ValueError('player ID mismatch with hand roster')
        evidence=public_decision_from_state(before,player_id=player_id,player_session_id=player_session_id,
                            hand_id=handle.hand_id,action_index=action_index,action=action,
                            position=position_for_seat(handle,before.hero_seat),
                            preflop_raises_before=count_prior_preflop_raises(before.action_history))
        return self.registry.observe(evidence)

    def close_hand(self,handle:HandHandle)->None:
        for _,player_id in handle.participants:
            for (pid,session),model in self.registry.models.items():
                if pid==player_id and handle.hand_id in model.hands:
                    model.end_hand(handle.hand_id)
