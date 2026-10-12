"""Safe, explicit experimental snapshot facility (not a CFR correctness guarantee)."""
from __future__ import annotations

from copy import deepcopy
from typing import TypeVar

T = TypeVar("T")


def snapshot(environment: T) -> T:
    """Deepcopy engine AND RNG state, if its implementation supports deepcopy.

    Do not assume a poker engine is branchable: test reproducibility and
    mutation isolation for each engine's card RNG, seat RNG and state history.
    """
    try:
        replica = deepcopy(environment)
    except Exception as exc:
        raise RuntimeError("engine snapshot failed; provide engine-specific clone API") from exc
    if replica is environment:
        raise RuntimeError("snapshot returned the original environment object")
    return replica
