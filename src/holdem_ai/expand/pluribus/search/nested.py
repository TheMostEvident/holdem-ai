"""Public-trace session for re-searching at each decision / off-tree action.

Phase5 does not solve the entire previous street from its beginning: this
replays factual public history, then re-solves the *current* information set.
Own committed actions are immutable once registered.
"""
from __future__ import annotations
from dataclasses import dataclass,field


@dataclass
class NestedSearchSession:
    hero: int
    street: str='preflop'
    history: list[tuple[int,str,int|None]]=field(default_factory=list)
    own_commitments: dict[tuple[str,str],str]=field(default_factory=dict)
    off_tree_count: int=0
    street_resets: int=0

    def observe_action(self,*,actor:int,action:str,street:str,infoset_key:str|None=None,
                       raise_to:int|None=None,abstract_actions:tuple[str,...]=()):
        if not isinstance(action,str) or not action:raise ValueError('action required')
        if action=='raise_to' and (type(raise_to) is not int or raise_to<=0):
            raise ValueError('off-tree raise_to requires positive integer target')
        if street!=self.street:
            self.street=street;self.street_resets+=1
        commitment=(street,infoset_key or '')
        if actor==self.hero:
            if not infoset_key:raise ValueError('hero commitment requires infoset')
            if commitment in self.own_commitments:
                if self.own_commitments[commitment]!=action:
                    raise ValueError('cannot revise committed own action')
                raise ValueError('duplicate hero public action')
            self.own_commitments[commitment]=action
        if action=='raise_to' and action not in abstract_actions:
            self.off_tree_count+=1
        self.history.append((actor,action,raise_to))

    def public_trace(self)->tuple[tuple[int,str,int|None],...]:
        return tuple(self.history)
