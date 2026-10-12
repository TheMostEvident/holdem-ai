"""Public-knowledge-only roots for toy poker and collision-free Hold'em sampling."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import math
import random


class InconsistentPublicHistory(ValueError):
    pass


@dataclass(frozen=True)
class ToyRootBelief:
    """Reconstruct a toy game *from public actions + one hero private card*.

    No access to the observed world's rival cards. Opponent actions are replayed
    as public facts. For this MVP, unobserved hands are sampled uniformly given
    public cards (action likelihood updates are not automatic).
    """
    game: object
    hero: int
    hero_card: int
    history: tuple[str,...] = ()
    board_rank: int | None = None
    max_attempts: int = 10000

    def sample(self,rng:random.Random):
        for _ in range(self.max_attempts):
            state=self.game.start(rng)
            own=state.hole[self.hero] if state.game=='kuhn' else state.hole[self.hero]//2
            if own!=self.hero_card: continue
            try:
                for a in self.history:
                    state=state.next_state(a)
            except (ValueError,AssertionError):
                continue
            visible = state.board//2 if state.game=='leduc' and state.street==1 else None
            if self.board_rank is not None and visible!=self.board_rank: continue
            if state.is_terminal(): continue
            return state
        raise InconsistentPublicHistory('cannot sample root satisfying toy public constraints')

    @property
    def fingerprint(self):
        return repr((self.game.name,self.game.num_players,self.hero,self.hero_card,self.history,self.board_rank))


SUITS='cdhs'
RANKS='23456789TJQKA'
DECK=frozenset(s+r for s in SUITS for r in RANKS)


def _card(c):
    if not isinstance(c,str) or c not in DECK:
        raise ValueError(f'invalid canonical card {c!r}')
    return c


class HoldemObservationBelief:
    """Sample complete Phase3 engine worlds consistent with public observations.

    Inputs are an acting player's raw_obs (not an engine instance), start-of-
    hand stacks, blind sizes, and action history already visible in raw_obs.
    Hole cards for opponents and unexposed board/burn cards are NEW samples.
    All dealt cards are collision-free. Any replay mismatch is rejected.
    """
    def __init__(self, raw_obs:dict, *, initial_stacks:tuple[int,...],
                 small_blind:int, big_blind:int, engine_factory:Callable | None=None,
                 max_attempts:int=200, opponent_likelihood:Callable | None=None,
                 abstraction=None):
        self.raw = dict(raw_obs)
        self.stacks = tuple(initial_stacks)
        self.sb = small_blind
        self.bb = big_blind
        self.factory = engine_factory
        self.max_attempts = max_attempts
        self.opponent_likelihood = opponent_likelihood
        from ..abstraction import AbstractionConfig
        self.abstraction = abstraction if abstraction is not None else AbstractionConfig()
        if not isinstance(self.abstraction, AbstractionConfig):raise TypeError('abstraction must be AbstractionConfig')
        if len(self.stacks)!=6 or sum(s>0 for s in self.stacks)<2:
            raise ValueError('six physical seats, at least 2 occupied required')
        if any(type(s) is not int or s<0 for s in self.stacks):
            raise ValueError('invalid starting stacks')
        if not (type(small_blind) is type(big_blind) is int and 0<small_blind<big_blind):
            raise ValueError('invalid blinds')
        self.hero=int(self.raw['player_id']); self.button=int(self.raw['button'])
        self.hole=tuple(_card(c) for c in self.raw['hand'])
        self.board=tuple(_card(c) for c in self.raw['public_cards'])
        if len(self.hole)!=2 or len(self.board) not in (0,3,4,5) or len(set(self.hole+self.board))!=len(self.hole+self.board):
            raise ValueError('duplicate or invalid known cards')
        self.occupied=tuple(i for i,s in enumerate(self.stacks) if s>0)
        if self.hero not in self.occupied or self.button not in self.occupied:
            raise ValueError('hero and button must be seated')
        self.actions=tuple(self.raw['action_history'])
        self.raise_to=tuple(self.raw['action_raise_to'])
        if len(self.actions)!=len(self.raise_to): raise ValueError('invalid action trace')
        if type(max_attempts) is not int or max_attempts<1: raise ValueError('max_attempts must be positive')
        if len(self.raw['stacks'])!=6 or len(self.raw['street_bets'])!=6:
            raise ValueError('invalid observation vectors')
        # Publicly visible fingerprint only (hero cards are own permitted info).
        self.fingerprint=repr((self.hero,self.hole,self.board,self.button,self.stacks,
                               self.actions,self.raise_to,self.raw['street'],tuple(self.raw['stacks'])))

    def _random_deck(self,rng):
        order=[]; nextseat=self.button
        for _ in self.occupied:
            nextseat=next(i for offset in range(1,7) if (i:=(nextseat+offset)%6) in self.occupied)
            order.append(nextseat)
        known={order.index(self.hero):self.hole[0],len(order)+order.index(self.hero):self.hole[1]}
        n=len(order)
        if len(self.board)>=3:
            for j,c in enumerate(self.board[:3]):known[2*n+1+j]=c
        if len(self.board)>=4:known[2*n+5]=self.board[3]
        if len(self.board)==5:known[2*n+7]=self.board[4]
        # 2n hole, burn(2n), flop(2n+1..3), burn(2n+4), turn(2n+5), burn(2n+6), river(2n+7)
        remaining=list(DECK-set(known.values()))
        remaining.sort(); rng.shuffle(remaining)
        it=iter(remaining)
        return [known.get(j) if j in known else next(it) for j in range(52)]

    def sample(self,rng:random.Random):
        factory=self.factory
        if factory is None:
            from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine
            factory=MultiPlayerHandEngine
        for _ in range(self.max_attempts):
            engine=factory(stacks=self.stacks,button=self.button,
                           small_blind=self.sb,big_blind=self.bb,deck=self._random_deck(rng))
            engine.reset()
            weight=1.0
            try:
                for (seat,action),amount in zip(self.actions,self.raise_to):
                    if engine.is_over() or engine.current_player!=seat:
                        raise InconsistentPublicHistory('action seat mismatch')
                    if self.opponent_likelihood is not None and seat!=self.hero:
                        p=float(self.opponent_likelihood(engine.observe(seat),action,amount))
                        if not math.isfinite(p) or p<0 or p>1:
                            raise ValueError('opponent likelihood outside [0,1]')
                        weight*=p
                    if action==5:
                        engine.step_raise_to(amount)
                    else:
                        engine.step(action)
                if engine.is_over() or engine.current_player!=self.hero:
                    raise InconsistentPublicHistory('hero turn mismatch')
                out=engine.observe(self.hero)['raw_obs']
                for key in ('hand','public_cards','street','button','player_id','stacks',
                            'street_bets','total_contributions','status','pot','to_call',
                            'action_history','action_raise_to'):
                    if out[key]!=self.raw[key]:
                        raise InconsistentPublicHistory(f'public replay mismatch: {key}')
                # Reject likelihood weighted particles without changing their hidden states.
                if self.opponent_likelihood is not None and rng.random()>=weight:
                    continue
                from ..blueprint.holdem import HoldemState
                return HoldemState(engine,self.abstraction,self.bb)
            except (InconsistentPublicHistory,ValueError,AssertionError):
                continue
        raise InconsistentPublicHistory('could not reconstruct public observation')
