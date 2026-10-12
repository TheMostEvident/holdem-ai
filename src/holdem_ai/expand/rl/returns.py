"""Per-hand, per-seat GAE: decision streams are never conflated across players."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

from holdem_ai.expand.information.schema import InformationState
from holdem_ai.expand.information.encoder import EncodedObservation


@dataclass(frozen=True)
class MultiAgentTransition:
    session_id: str
    hand_id: str
    player_id: str
    seat_id: int
    observation: InformationState
    encoded: EncodedObservation
    action: str
    action_logprob: float
    value: float
    reward_bb: float
    terminal_hand: bool
    terminal_session: bool
    policy_version: str
    num_players: int


@dataclass(frozen=True)
class PreparedSample:
    transition: MultiAgentTransition
    advantage: float
    return_bb: float


def prepare_gae(transitions: Sequence[MultiAgentTransition], *, gamma: float = 1.0,
                lam: float = 0.95) -> tuple[PreparedSample, ...]:
    if not (0 <= gamma <= 1 and 0 <= lam <= 1):
        raise ValueError('gamma and lambda must be in [0,1]')
    if not transitions:
        raise ValueError('empty transition batch')
    groups: dict[tuple[str, str, int], list[int]] = {}
    for k, t in enumerate(transitions):
        key = t.session_id, t.hand_id, t.seat_id
        groups.setdefault(key, []).append(k)
    out: list[PreparedSample | None] = [None]*len(transitions)
    for key, positions in groups.items():
        if not transitions[positions[-1]].terminal_hand:
            raise ValueError(f'unterminated player trajectory {key}')
        if sum(t.terminal_hand for t in (transitions[k] for k in positions)) != 1:
            raise ValueError('player trajectory has inconsistent terminal flag')
        if any(transitions[k].terminal_hand for k in positions[:-1]):
            raise ValueError('terminal action occurs before last player decision')
        gae = 0.0
        for offset in range(len(positions)-1, -1, -1):
            k=positions[offset]
            t=transitions[k]
            nxt = 0.0 if offset == len(positions)-1 else transitions[positions[offset+1]].value
            # The hand has no intermediate chip reward; the last decision
            # receives the entire seat's signed settled BB payoff.
            delta = t.reward_bb + gamma * nxt * (not t.terminal_hand) - t.value
            gae = delta + gamma * lam * (not t.terminal_hand) * gae
            out[k] = PreparedSample(t, gae, gae+t.value)
    return tuple(out)  # type: ignore[return-value]
