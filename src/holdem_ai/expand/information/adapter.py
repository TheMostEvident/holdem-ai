"""Phase2 interface for Phase1 DynamicTable; records public action street exactly.

Table is still the sole game-rule authority. This wrapper never changes its
live hand engine. A wrapper attached mid-hand fails closed rather than forging
an action history with the wrong street.
"""
from __future__ import annotations
from .builder import make_information_state
from .encoder import encode_observation
from .actions import to_table_action
from .schema import InformationState, PublicAction
from holdem_ai.expand.environment.table_state import Action


class Phase2TableAdapter:
    def __init__(self, table):
        self.table = table
        if table.snapshot().hand_in_progress:
            raise ValueError('attach Phase2TableAdapter before start_hand, not mid-hand')
        self._history: list[PublicAction] = []
        self._hand_id: str | None = None

    def start_hand(self, seed: int):
        if self.table.snapshot().hand_in_progress:
            raise RuntimeError('cannot start a second hand')
        result = self.table.start_hand(seed)
        self._history.clear()
        self._hand_id = result.hand_id if result is not None else None
        return result

    def observe(self) -> InformationState:
        snap = self.table.snapshot()
        if not snap.hand_in_progress or snap.current_actor_seat is None:
            raise RuntimeError('no acting player; cannot form observation')
        obs = self.table.observe_current()
        if obs['raw_obs']['player_id'] != snap.current_actor_seat:
            raise AssertionError('acting player differs from table snapshot')
        return make_information_state(
            obs, big_blind_chips=snap.config.big_blind,
            seat_presence=tuple(s.player_id is not None for s in snap.seats),
            seat_statuses=tuple(s.seat_status for s in snap.seats),
            history=tuple(self._history),
        )

    def encode(self):
        return encode_observation(self.observe())

    def act_abstract(self, name: str):
        state = self.observe()
        return self.act(to_table_action(state.legal_actions, name))

    def act(self, action: Action):
        if not isinstance(action, Action):
            raise TypeError('required phase1 Action')
        before = self.observe()
        bb = before.big_blind_chips
        raw_kind = action.kind
        if raw_kind == 'raise_half_pot':
            # Integer exact same as engine, not floating approximation.
            raw = self.table.observe_current()['raw_obs']
            target_chips = raw['street_bets'][before.hero_seat] + raw['to_call'] + (raw['pot']+raw['to_call'])//2
        elif raw_kind == 'raise_pot':
            raw = self.table.observe_current()['raw_obs']
            target_chips = raw['street_bets'][before.hero_seat] + raw['to_call'] + (raw['pot']+raw['to_call'])
        elif raw_kind == 'raise_to':
            target_chips = action.raise_to_chips
        elif raw_kind == 'all_in':
            raw = self.table.observe_current()['raw_obs']
            target_chips = raw['street_bets'][before.hero_seat] + raw['stacks'][before.hero_seat]
        else:
            target_chips = None
        player_id = self.table.current_player_id
        # Mutate underlying hand FIRST; never track rejected/illegal actions.
        step = self.table.act(player_id, action)
        event = PublicAction(
            actor_seat=before.hero_seat, street=before.street, kind=raw_kind,
            raise_to_bb=(target_chips/bb if target_chips is not None else None),
            to_call_bb=before.to_call_bb, pot_bb=before.pot_bb,
        )
        self._history.append(event)
        return step

    def settle_hand(self):
        result = self.table.settle_hand()
        self._hand_id = None
        self._history.clear()
        return result
