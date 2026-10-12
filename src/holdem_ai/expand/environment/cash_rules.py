"""E1 cash table admission rules: fixed blinds, no rake, wait for BB."""
from __future__ import annotations
from dataclasses import dataclass
from .table_state import TableConfig


@dataclass(frozen=True)
class CashRuleSet:
    config: TableConfig

    def __post_init__(self):
        if self.config.mode != 'cash':
            raise ValueError('CashRuleSet requires cash mode')

    @property
    def min_buyin(self) -> int:
        return self.config.big_blind * self.config.min_buyin_bb

    @property
    def max_buyin(self) -> int:
        return self.config.big_blind * self.config.max_buyin_bb

    def validate_buyin(self, chips: int) -> None:
        if type(chips) is not int or not self.min_buyin <= chips <= self.max_buyin:
            raise ValueError(f'buyin must be integer chips in [{self.min_buyin}, {self.max_buyin}]')

    def validate_rebuy(self, stack: int, chips: int) -> None:
        if type(chips) is not int or chips <= 0 or stack + chips > self.max_buyin:
            raise ValueError('top-up must be positive and resulting stack <= max buyin')
