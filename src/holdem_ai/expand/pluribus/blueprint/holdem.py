"""Small-resource 2--6 seat Hold'em MCCFR adapter using the Phase3 engine.

No hidden-card access while forming infoset. Complete chance deals occur in
engine constructor/reset; trees branch via copy.deepcopy on legal actions.
This is a bounded research prototype; entire 6-max full tree is intractable.
"""
from __future__ import annotations
from dataclasses import dataclass
import copy
from typing import Callable
from ..abstraction.cards import AbstractionConfig, holdem_infoset_key


@dataclass
class HoldemState:
    engine: object
    abstraction: AbstractionConfig
    big_blind: int

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(i for i,stack in enumerate(self.engine.initial_stacks) if stack > 0)

    @property
    def current_player(self) -> int:
        return self.engine.current_player

    def is_terminal(self):
        return self.engine.is_over()

    def _obs(self):
        if self.is_terminal():
            raise RuntimeError('terminal hand')
        return self.engine.observe(self.current_player)

    def _mapping(self):
        obs = self._obs()
        ids = obs['legal_actions']
        raw = obs['raw_obs']
        items = []
        if 0 in ids:
            items.append(('fold',0))
        if 1 in ids:
            items.append(('check' if raw['to_call'] == 0 else 'call',1))
        if self.abstraction.allow_half_pot and 2 in ids:
            items.append(('raise_half_pot',2))
        if 3 in ids:
            items.append(('raise_pot',3))
        if 4 in ids:
            # Avoid multiple abstract actions that produce the same actual amount.
            hero = self.current_player
            maxbet = raw['street_bets'][hero] + raw['stacks'][hero]
            tocall = raw['to_call']
            pot_after_call = raw['pot'] + tocall
            target_pot = raw['street_bets'][hero] + tocall + pot_after_call
            target_half = raw['street_bets'][hero] + tocall + pot_after_call//2
            other_targets = ({target_half} if 2 in ids and self.abstraction.allow_half_pot else set())
            if 3 in ids:
                other_targets.add(target_pot)
            if maxbet not in other_targets:
                items.append(('all_in',4))
        return dict(items)

    def legal_actions(self):
        return tuple(self._mapping())

    def infoset(self, player: int):
        if player != self.current_player:
            raise ValueError('not your information state')
        return holdem_infoset_key(self._obs()['raw_obs'],self.legal_actions(),self.abstraction)

    def next_state(self,action: str):
        mapping = self._mapping()
        if action not in mapping:
            raise ValueError('illegal abstract action')
        engine = copy.deepcopy(self.engine)
        engine.step(mapping[action])
        return HoldemState(engine,self.abstraction,self.big_blind)

    def terminal_payoffs(self):
        if not self.is_terminal():
            raise RuntimeError('not terminal')
        values = self.engine.get_result().payoffs_chips
        if len(values) != 6:
            raise AssertionError('must use six physical seats')
        return tuple(v/self.big_blind for v in values)


class HoldemGame:
    def __init__(self, num_players: int = 2, *, big_blind: int = 2, stack_bb: int = 30,
                 abstraction: AbstractionConfig | None = None,
                 engine_factory: Callable | None = None):
        if type(num_players) is not int or not 2 <= num_players <= 6:
            raise ValueError('num_players must be 2..6')
        if type(big_blind) is not int or big_blind < 2:
            raise ValueError('big_blind must be >=2')
        if type(stack_bb) is not int or not 2 <= stack_bb <= 300:
            raise ValueError('stack_bb must be 2..300')
        self.name = 'holdem'
        self.num_players = num_players
        self.big_blind = big_blind
        self.stack_bb = stack_bb
        self.abstraction = abstraction or AbstractionConfig()
        self.engine_factory = engine_factory

    def start(self,rng) -> HoldemState:
        factory = self.engine_factory
        if factory is None:
            from holdem_ai.environment.multi_player_engine import MultiPlayerHandEngine
            factory = MultiPlayerHandEngine
        participants = sorted(rng.sample(range(6), self.num_players))
        stacks = [self.stack_bb*self.big_blind if i in participants else 0 for i in range(6)]
        button = rng.choice(participants)
        engine = factory(stacks=stacks,button=button,small_blind=self.big_blind//2,
                         big_blind=self.big_blind,seed=rng.randrange(2**63))
        engine.reset()
        return HoldemState(engine,self.abstraction,self.big_blind)
