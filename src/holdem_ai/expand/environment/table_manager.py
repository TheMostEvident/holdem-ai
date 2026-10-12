"""Expand Phase 1: dynamic 6-seat cash sessions on the unchanged Phase 3 hand engine.

Events change *eligible rosters* only between hands. The hand engine owns
chips committed to pots; the table manager owns external buyin/cashout flow.
This module intentionally does not implement RL, PPO, or MCCFR.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Any
import json

from .cash_rules import CashRuleSet
from .event_ledger import EventLedger
from .multiway_adapter import blind_seats, make_hand_engine, moving_button_after
from .table_state import (Action, HandHandle, PlayerSeat, Settlement,
                          StepEvent, TableConfig, TableSnapshot)


@dataclass
class _Seat:
    player_id: str | None = None
    stack: int = 0
    status: str = 'vacant'


@dataclass(frozen=True)
class _QueueEntry:
    event_id: str
    kind: str
    player_id: str
    seat: int
    chips: int = 0


class DynamicTable:
    """Cash-only E1 manager. Tournament mode is a hard failure until E7.

    `engine_factory` is injectable for isolated tests; default uses the actual
    Base Phase 3 engine, without patching it. Not thread-safe: serialize table
    actions through one owner process, as with a poker table event loop.
    """

    def __init__(self, config: TableConfig | None = None, *, session_id: str = 'expand-e01',
                 journal_path: str | Path | None = None,
                 engine_factory: Callable[..., Any] | None = None):
        self.config = config if config is not None else TableConfig()
        if self.config.mode != 'cash':
            raise NotImplementedError('Tournament play is Expand Phase 7; E1 is cash only')
        if not isinstance(session_id, str) or not session_id:
            raise ValueError('nonempty session_id is required')
        self.rules = CashRuleSet(self.config)
        self.session_id = session_id
        self.ledger = EventLedger(journal_path)
        self._engine_factory = engine_factory or make_hand_engine
        self._seats = [_Seat() for _ in range(6)]
        self._pending: list[_QueueEntry] = []
        self._event_id = 0
        self._hand_number = 0
        self._button: int | None = None
        self._hand: Any | None = None
        self._hand_seed: int | None = None
        self._hand_id: str | None = None
        self._hand_roster: dict[int, str] = {}
        self._hand_initial: tuple[int, ...] | None = None
        self.total_buyins = 0
        self.total_cashouts = 0
        self.ledger.append('table_created', config=asdict(self.config), session_id=self.session_id)

    def _next_event_id(self) -> str:
        self._event_id += 1
        return f'{self.session_id}:event:{self._event_id:08d}'

    def _require_player(self, player_id: str) -> int:
        if not isinstance(player_id, str) or not player_id:
            raise ValueError('player_id must be nonempty string')
        for i, seat in enumerate(self._seats):
            if seat.player_id == player_id:
                return i
        raise ValueError('unknown player_id')

    def _open_hand(self) -> bool:
        return self._hand is not None

    def _validate_seat(self, seat_id: int) -> None:
        if type(seat_id) is not int or not 0 <= seat_id < 6:
            raise ValueError('seat must be integer in 0..5')

    def _money_check(self) -> None:
        if self._open_hand():
            live = sum(self._hand.stacks) + sum(self._hand.contributions) + sum(
                seat.stack for i, seat in enumerate(self._seats) if i not in self._hand_roster)
        else:
            live = sum(seat.stack for seat in self._seats)
        if live != self.total_buyins - self.total_cashouts:
            raise AssertionError(f'cash conservation: {live} != {self.total_buyins}-{self.total_cashouts}')
        if any(s.stack < 0 for s in self._seats):
            raise AssertionError('negative seat stack')
        ids = [s.player_id for s in self._seats if s.player_id is not None]
        if len(ids) != len(set(ids)) or len(self._seats) != 6:
            raise AssertionError('invalid seat registry')

    def request_join(self, player_id: str, buyin_chips: int,
                     preferred_seat: int | None = None) -> str:
        if not isinstance(player_id, str) or not player_id:
            raise ValueError('nonempty player_id is required')
        if any(seat.player_id == player_id for seat in self._seats):
            raise ValueError('player already occupies or reserved a seat')
        self.rules.validate_buyin(buyin_chips)
        if preferred_seat is None:
            empty = [i for i, s in enumerate(self._seats) if s.status == 'vacant']
            if not empty:
                raise ValueError('table full')
            preferred_seat = empty[0]
        self._validate_seat(preferred_seat)
        seat = self._seats[preferred_seat]
        if seat.status != 'vacant':
            raise ValueError('seat already occupied or reserved')
        eid = self._next_event_id()
        self._seats[preferred_seat] = _Seat(player_id=player_id, stack=0, status='reserved')
        self._pending.append(_QueueEntry(eid, 'join', player_id, preferred_seat, buyin_chips))
        self.ledger.append('request', event_id=eid, kind='join', player_id=player_id,
                           seat=preferred_seat, chips=buyin_chips)
        return eid

    def request_leave(self, player_id: str) -> str:
        i = self._require_player(player_id)
        if any(e.player_id == player_id and e.kind == 'leave' for e in self._pending):
            raise ValueError('leave already pending')
        eid = self._next_event_id()
        self._pending.append(_QueueEntry(eid, 'leave', player_id, i))
        # Status-only change; live hand roster remains immutable.
        if self._seats[i].status != 'reserved':
            self._seats[i].status = 'pending_leave'
        self.ledger.append('request', event_id=eid, kind='leave', player_id=player_id, seat=i, chips=0)
        return eid

    def request_rebuy(self, player_id: str, amount_chips: int) -> str:
        i = self._require_player(player_id)
        seat = self._seats[i]
        if seat.status in ('reserved', 'pending_leave') or any(
                e.player_id == player_id and e.kind == 'leave' for e in self._pending):
            raise ValueError('rebuy is unavailable for reserved/leaving player')
        pending_rebuys = sum(e.chips for e in self._pending if e.player_id == player_id and e.kind == 'rebuy')
        # During a hand, the live stack is the correct basis, but buyin cap is
        # evaluated against the original hand-start stack to avoid pot-dependent
        # mid-hand top-up qualification.
        basis = self._hand_initial[i] if self._open_hand() and i in self._hand_roster else seat.stack
        self.rules.validate_rebuy(basis + pending_rebuys, amount_chips)
        eid = self._next_event_id()
        self._pending.append(_QueueEntry(eid, 'rebuy', player_id, i, amount_chips))
        self.ledger.append('request', event_id=eid, kind='rebuy', player_id=player_id,
                           seat=i, chips=amount_chips)
        return eid

    def request_sitout(self, player_id: str) -> str:
        return self._queue_status(player_id, 'sitout')

    def request_resume(self, player_id: str) -> str:
        return self._queue_status(player_id, 'resume')

    def _queue_status(self, player_id: str, kind: str) -> str:
        i = self._require_player(player_id)
        seat = self._seats[i]
        if seat.status in ('reserved', 'pending_leave') or any(
                e.player_id == player_id and e.kind in ('sitout', 'resume', 'leave') for e in self._pending):
            raise ValueError('conflicting pending status request')
        if kind == 'sitout' and seat.status not in ('seated', 'waiting_bb'):
            raise ValueError('player cannot sit out')
        if kind == 'resume' and seat.status != 'sitting_out':
            raise ValueError('player is not sitting out')
        eid = self._next_event_id()
        self._pending.append(_QueueEntry(eid, kind, player_id, i))
        self.ledger.append('request', event_id=eid, kind=kind, player_id=player_id, seat=i, chips=0)
        return eid

    def apply_boundary_events(self) -> None:
        if self._open_hand():
            raise RuntimeError('cannot apply pending changes during a hand')
        self.ledger.append('boundary_call')
        self._apply_pending()

    def _apply_pending(self) -> None:
        if self._open_hand():
            raise RuntimeError('cannot modify roster during an open hand')
        pending, self._pending = self._pending, []
        for entry in pending:
            seat = self._seats[entry.seat]
            if seat.player_id != entry.player_id:
                # A prior leave in this same boundary removes later commands.
                self.ledger.append('boundary_skip', event_id=entry.event_id, reason='player_departed')
                continue
            if entry.kind == 'join':
                if seat.status != 'reserved':
                    raise AssertionError('reservation mismatch')
                seat.stack = entry.chips
                seat.status = 'waiting_bb'
                self.total_buyins += entry.chips
                self.ledger.append('boundary_join', event_id=entry.event_id,
                                   player_id=entry.player_id, seat=entry.seat, buyin=entry.chips)
            elif entry.kind == 'leave':
                chips = seat.stack
                self.total_cashouts += chips
                self._seats[entry.seat] = _Seat()
                self.ledger.append('boundary_leave', event_id=entry.event_id,
                                   player_id=entry.player_id, seat=entry.seat, cashout=chips)
            elif entry.kind == 'rebuy':
                seat.stack += entry.chips
                self.total_buyins += entry.chips
                if seat.status in ('busted', 'waiting_bb') or seat.stack == entry.chips:
                    seat.status = 'waiting_bb'
                self.ledger.append('boundary_rebuy', event_id=entry.event_id,
                                   player_id=entry.player_id, seat=entry.seat, chips=entry.chips)
            elif entry.kind == 'sitout':
                seat.status = 'sitting_out'
                self.ledger.append('boundary_sitout', event_id=entry.event_id, seat=entry.seat)
            elif entry.kind == 'resume':
                seat.status = 'waiting_bb'
                self.ledger.append('boundary_resume', event_id=entry.event_id, seat=entry.seat)
            else:
                raise AssertionError('invalid boundary command')
        # Final stacks of 0 imply actual cash bust only AFTER the hand settles.
        # A queued rebuy can save a busted player's original seat.
        for i, seat in enumerate(self._seats):
            if seat.player_id is not None and seat.stack == 0 and seat.status != 'reserved':
                player_id = seat.player_id
                self._seats[i] = _Seat()
                self.ledger.append('boundary_bust', seat=i, player_id=player_id)
        self._money_check()

    def _choose_roster(self) -> tuple[int, tuple[int, ...], tuple[int, ...]] | None:
        active = tuple(i for i, s in enumerate(self._seats)
                       if s.status == 'seated' and s.stack > 0)
        waiting = tuple(i for i, s in enumerate(self._seats)
                        if s.status == 'waiting_bb' and s.stack > 0)
        activated: list[int] = []
        if len(active) < 2:
            # Bootstrap a stalled table. A two-person table must start even
            # though there was no prior BB rotation to wait for.
            need = 2 - len(active)
            if len(waiting) < need:
                return None
            activated.extend(waiting[:need])
            active = tuple(sorted(active + tuple(activated)))
            waiting = tuple(i for i in waiting if i not in activated)
            button = moving_button_after(self._button, active)
            return button, active, tuple(activated)

        button = moving_button_after(self._button, active)
        # Admit at most one additional waiting player, and only if they will
        # post the big blind in THIS hand. Never grant free entry at BTN/CO.
        for candidate in waiting:
            roster = tuple(sorted(active + (candidate,)))
            _, bb = blind_seats(button, roster)
            if bb == candidate:
                activated.append(candidate)
                active = roster
                break
        return button, active, tuple(activated)

    def start_hand(self, seed: int) -> HandHandle | None:
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be nonnegative integer')
        if self._open_hand():
            raise RuntimeError('previous hand must be settled first')
        # Log the external command BEFORE boundary effects for strict replay.
        self.ledger.append('command_start_hand', seed=seed)
        # Commit queued table operations at the boundary, before selecting
        # participants; joins never occur mid-hand.
        self._apply_pending()
        selection = self._choose_roster()
        if selection is None:
            self.ledger.append('waiting_for_players', seated=sum(s.status == 'seated' for s in self._seats),
                               waiting=sum(s.status == 'waiting_bb' for s in self._seats))
            return None
        button, roster, admitted = selection
        for i in admitted:
            self._seats[i].status = 'seated'
        for i in admitted:
            self.ledger.append('big_blind_admission', seat=i, player_id=self._seats[i].player_id,
                               bootstrap=len(roster) == 2)
        stacks = tuple(self._seats[i].stack if i in roster else 0 for i in range(6))
        engine = self._engine_factory(stacks=stacks, button=button,
                                      small_blind=self.config.small_blind,
                                      big_blind=self.config.big_blind, seed=seed)
        engine.reset()
        self._hand = engine
        self._hand_seed = seed
        self._hand_number += 1
        self._hand_id = f'{self.session_id}:hand:{self._hand_number:08d}'
        self._hand_roster = {i: self._seats[i].player_id for i in roster}
        self._hand_initial = stacks
        self._button = button
        self._money_check()
        self.ledger.append('hand_started', hand_id=self._hand_id, hand_number=self._hand_number,
                           button=button, sb=engine.seats.small_blind, bb=engine.seats.big_blind,
                           participants=[[i, self._seats[i].player_id] for i in roster],
                           starting_stacks=list(stacks), seed=seed)
        return HandHandle(hand_id=self._hand_id, hand_number=self._hand_number,
                          button_seat=button, small_blind_seat=engine.seats.small_blind,
                          big_blind_seat=engine.seats.big_blind,
                          participants=tuple((i, self._seats[i].player_id) for i in roster),
                          seed=seed, terminal=engine.is_over())

    @property
    def current_player_id(self) -> str | None:
        if not self._open_hand() or self._hand.is_over():
            return None
        return self._hand_roster[self._hand.current_player]

    def observe_current(self) -> dict:
        if self.current_player_id is None:
            raise RuntimeError('no pending decision')
        # Base engine returns only the acting player's hole cards. This raw
        # representation is temporary: E2 adds a formal InformationState.
        return dict(self._hand.observe(self._hand.current_player))

    def act(self, player_id: str, action: Action) -> StepEvent:
        if not self._open_hand() or self._hand.is_over():
            raise RuntimeError('no current hand action')
        if player_id != self.current_player_id:
            raise ValueError('not the acting player')
        if not isinstance(action, Action):
            raise TypeError('action must be Action')
        actor = self._hand.current_player
        observation = self._hand.observe(actor)['raw_obs']
        to_call = observation['to_call']
        if action.kind == 'check' and to_call != 0:
            raise ValueError('cannot check when facing a bet')
        if action.kind == 'call' and to_call == 0:
            raise ValueError('cannot call when check is free')
        kind_action = {
            'fold': 0, 'check': 1, 'call': 1,
            'raise_half_pot': 2, 'raise_pot': 3,
        }
        if action.kind == 'raise_to':
            self._hand.step_raise_to(action.raise_to_chips)
        elif action.kind == 'all_in':
            # In the Base engine, short all-in calls are represented by action
            # 1 (check/call), while action 4 is an aggressive all-in raise.
            chosen = 1 if self._hand.stacks[actor] <= to_call else 4
            self._hand.step(chosen)
        else:
            self._hand.step(kind_action[action.kind])
        self._money_check()
        self.ledger.append('command_action', hand_id=self._hand_id, player_id=player_id,
                           seat=actor, action=asdict(action))
        nxt = None if self._hand.is_over() else self._hand.current_player
        return StepEvent(hand_id=self._hand_id, actor_seat=actor, player_id=player_id,
                         action=action, terminal=self._hand.is_over(), next_actor_seat=nxt)

    def settle_hand(self) -> Settlement:
        if not self._open_hand() or not self._hand.is_over():
            raise RuntimeError('settle requires a terminal, unsettled hand')
        result = self._hand.get_result()
        initial = self._hand_initial
        if (len(result.final_stacks) != 6 or len(result.payoffs_chips) != 6
                or sum(result.final_stacks) != sum(initial)
                or sum(result.payoffs_chips) != 0
                or any(x < 0 for x in result.final_stacks)):
            raise AssertionError('invalid or non-zero-sum hand settlement')
        busts = tuple(self._hand_roster[i] for i in self._hand_roster if result.final_stacks[i] == 0)
        settled = Settlement(hand_id=self._hand_id, hand_number=self._hand_number,
                             starting_stacks=initial, final_stacks=tuple(result.final_stacks),
                             payoffs_chips=tuple(result.payoffs_chips),
                             payoffs_bb=tuple(x / self.config.big_blind for x in result.payoffs_chips),
                             busted_player_ids=busts, button_seat=self._button)
        self.ledger.append('command_settle', hand_id=self._hand_id)
        self.ledger.append('hand_settled', hand_id=self._hand_id,
                           final_stacks=list(result.final_stacks), payoffs=list(result.payoffs_chips),
                           busts=list(busts), board=list(result.board),
                           action_history=[list(x) for x in result.action_history],
                           action_raise_to=list(result.action_raise_to))
        for i in self._hand_roster:
            self._seats[i].stack = result.final_stacks[i]
            if self._seats[i].status == 'seated' and result.final_stacks[i] == 0:
                self._seats[i].status = 'busted'
        self._hand = None
        self._hand_id = None
        self._hand_roster = {}
        self._hand_initial = None
        self._hand_seed = None
        # Hand boundary is automatic: commit queued joins, departures, rebuys.
        self._apply_pending()
        return settled

    def snapshot(self) -> TableSnapshot:
        rows = []
        for i, seat in enumerate(self._seats):
            stack = self._hand.stacks[i] if self._open_hand() and i in self._hand_roster else seat.stack
            rows.append(PlayerSeat(seat_id=i, player_id=seat.player_id,
                                   stack_chips=stack, seat_status=seat.status))
        return TableSnapshot(config=self.config, seats=tuple(rows), button_seat=self._button,
                             hand_number=self._hand_number, session_id=self.session_id,
                             hand_in_progress=self._open_hand(),
                             current_actor_seat=self._hand.current_player if self.current_player_id is not None else None,
                             pending_event_count=len(self._pending),
                             total_chips=self.total_buyins - self.total_cashouts)


def replay_journal(path: str | Path, *, engine_factory: Callable[..., Any] | None = None) -> DynamicTable:
    """Re-execute external commands and verify all emitted effects exactly.

    NOTE: Journal contains shuffle seeds; it is privileged experiment data and
    must never be supplied to a learning actor as an observation.
    """
    events = EventLedger.read(path)
    if not events or events[0]['event'] != 'table_created':
        raise ValueError('missing journal header')
    original = events[0]
    table = DynamicTable(TableConfig(**original['config']),
                         session_id=original['session_id'], engine_factory=engine_factory)
    # Effects are regenerated, not blindly trusted. Compare as we advance.
    cursor = 1
    while cursor < len(events):
        record = events[cursor]
        tag = record['event']
        if tag == 'request':
            kind, pid = record['kind'], record['player_id']
            if kind == 'join':
                eid = table.request_join(pid, record['chips'], record['seat'])
            elif kind == 'leave':
                eid = table.request_leave(pid)
            elif kind == 'rebuy':
                eid = table.request_rebuy(pid, record['chips'])
            elif kind == 'sitout':
                eid = table.request_sitout(pid)
            elif kind == 'resume':
                eid = table.request_resume(pid)
            else:
                raise ValueError(f'unknown recorded request: {kind}')
            if eid != record['event_id']:
                raise ValueError('event ID mismatch')
        elif tag == 'boundary_call':
            table.apply_boundary_events()
        elif tag == 'command_start_hand':
            table.start_hand(record['seed'])
        elif tag == 'command_action':
            table.act(record['player_id'], Action(**record['action']))
        elif tag == 'command_settle':
            table.settle_hand()
        else:
            raise ValueError(f'unexpected journal command at seq {record["seq"]}: {tag}')
        produced = table.ledger.records
        if produced[:len(produced)] != events[:len(produced)]:
            raise ValueError(f'journal replay divergence near seq {cursor + 1}')
        cursor = len(produced)
    if table.ledger.records != events:
        raise ValueError('replayed event stream differs')
    return table
