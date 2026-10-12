"""Tiny 2/3-player toy poker games for reference testing MCCFR.

Kuhn: one betting round, 1-chip bet, no raises, 3-card deck for 2p,
4-card deck for 3p. Leduc: two streets, paired ranks J/Q/K, fixed 1/2
chip bet, max one bet per street, one shared card. No peek at rival cards.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ToyState:
    game: Literal['kuhn', 'leduc']
    hole: tuple[int, ...]  # Entire deck is engine-owned, NOT returned in infoset.
    board: int | None  # Chance outcome pre-sampled; hidden in preflop infoset.
    street: int
    contributions: tuple[int, ...]
    invested: tuple[int, ...]
    active: tuple[bool, ...]
    pending: tuple[int, ...]
    raised: bool
    history: tuple[tuple[int, int, str], ...]

    @property
    def num_players(self):
        return len(self.hole)

    @property
    def current_player(self) -> int:
        return self.pending[0] if self.pending else -1

    def is_terminal(self) -> bool:
        return not self.pending

    def legal_actions(self) -> tuple[str, ...]:
        if self.is_terminal():
            return ()
        i = self.current_player
        if self.invested[i] < max(self.invested):
            return ('fold', 'call')
        if self.raised:
            # can only reach here if no outstanding action; next street/terminal.
            raise AssertionError('betting round should be closed')
        return ('check', 'bet')

    def infoset(self, player: int) -> str:
        if player != self.current_player:
            raise ValueError('only acting player may view private observation')
        import json
        card = self.hole[player] if self.game == 'kuhn' else self.hole[player] // 2
        visible_board = self.board // 2 if self.game == 'leduc' and self.street == 1 else None
        return json.dumps([self.game, self.num_players, player, card, visible_board,
                           list(self.active), list(self.contributions),
                           list(self.history), list(self.legal_actions())], separators=(',', ':'))

    def next_state(self, action: str) -> 'ToyState':
        if action not in self.legal_actions():
            raise ValueError('illegal toy action')
        i = self.current_player
        active = list(self.active)
        contrib = list(self.contributions)
        invested = list(self.invested)
        pending = list(self.pending[1:])
        raised = self.raised
        if action == 'fold':
            active[i] = False
        elif action == 'call':
            difference = max(invested) - invested[i]
            invested[i] += difference
            contrib[i] += difference
        elif action == 'bet':
            stake = (2 if self.game == 'leduc' and self.street == 1 else 1)
            invested[i] += stake
            contrib[i] += stake
            raised = True
            # Previous checkers must now respond to the outstanding bet.
            pending = [p for p in range(self.num_players) if p != i and active[p]]
            pending.sort(key=lambda p: (p - i) % self.num_players)
        newhist = (*self.history, (self.street, i, action))
        if sum(active) == 1:
            return ToyState(self.game,self.hole,self.board,self.street,tuple(contrib),
                            tuple(invested),tuple(active),(),raised,newhist)
        if pending:
            return ToyState(self.game,self.hole,self.board,self.street,tuple(contrib),
                            tuple(invested),tuple(active),tuple(pending),raised,newhist)
        if self.game == 'leduc' and self.street == 0:
            return ToyState(self.game,self.hole,self.board,1,tuple(contrib),
                            tuple(0 for _ in contrib),tuple(active),
                            tuple(p for p in range(self.num_players) if active[p]),False,newhist)
        return ToyState(self.game,self.hole,self.board,self.street,tuple(contrib),
                        tuple(invested),tuple(active),(),raised,newhist)

    def terminal_payoffs(self) -> tuple[float, ...]:
        if not self.is_terminal():
            raise RuntimeError('no terminal payoff before end')
        active = [p for p, yes in enumerate(self.active) if yes]
        if len(active) == 1:
            winners = active
        elif self.game == 'kuhn':
            best = max(self.hole[i] for i in active)
            winners = [i for i in active if self.hole[i] == best]
        else:
            assert self.board is not None
            boardrank = self.board // 2
            ranks = {i: (int(self.hole[i] // 2 == boardrank), self.hole[i] // 2) for i in active}
            best = max(ranks.values())
            winners = [i for i in active if ranks[i] == best]
        pot = sum(self.contributions)
        result = [-float(c) for c in self.contributions]
        for winner in winners:
            result[winner] += pot / len(winners)
        if abs(sum(result)) > 1e-9:
            raise AssertionError('toy poker zero-sum violated')
        return tuple(result)


class KuhnGame:
    def __init__(self, num_players: int = 2):
        if num_players not in (2,3):
            raise ValueError('KuhnGame supports 2 or 3 players')
        self.num_players = num_players
        self.name = 'kuhn'

    def start(self, rng) -> ToyState:
        cards = rng.sample(range(self.num_players + 1), self.num_players)
        n = self.num_players
        return ToyState('kuhn',tuple(cards),None,0,(1,)*n,(0,)*n,(True,)*n,
                        tuple(range(n)),False,())


class LeducGame:
    def __init__(self, num_players: int = 2):
        if num_players not in (2,3):
            raise ValueError('LeducGame supports 2 or 3 players')
        self.num_players = num_players
        self.name = 'leduc'

    def start(self, rng) -> ToyState:
        cards = rng.sample(range(6),self.num_players+1)
        n = self.num_players
        return ToyState('leduc',tuple(cards[:n]),cards[-1],0,(1,)*n,(0,)*n,
                        (True,)*n,tuple(range(n)),False,())
