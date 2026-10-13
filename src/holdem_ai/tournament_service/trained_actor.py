"""Tournament ENV 1.4: frozen Expand Phase3 multi-seat PPO inference bridge.

The checkpoint was trained on a 2-6 player *cash* table, not the knockout
TournamentEnv. Tournament transfer is experimental. No training or mutation.
Never load HUNL200 2-player weights here and never pass privileged state.
"""
from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence


CHECKPOINT_ENV = "HOLDEM_AI_TOURNAMENT_CHECKPOINT"
MODE_ENV = "HOLDEM_AI_TOURNAMENT_POLICY_MODE"
DEVICE_ENV = "HOLDEM_AI_TOURNAMENT_DEVICE"
FORMAT = "expand-e03-v1"

# The current Phase3 hand engine has 5 legal actions; Expand Phase3 has 7.
# Native: 0 fold, 1 check_call, 2 half pot, 3 pot, 4 all-in.
# Expand: 0 fold, 1 check, 2 call, 3 half pot, 4 pot, 5 double pot, 6 all-in.
EXPAND_TO_NATIVE: tuple[int | None, ...] = (0, 1, 1, 2, 3, None, 4)


def status() -> dict[str, object]:
    """Describe local configuration; actual model structure verified at create."""
    config = os.environ.get(CHECKPOINT_ENV, "").strip()
    selected = Path(config).expanduser() if config else None
    return {
        "configured": bool(config),
        "file_found": bool(selected and selected.is_file()),
        "checkpoint_name": selected.name if selected else None,
        "format": FORMAT,
        "model_family": "Expand Phase3 multi-seat PPO (cash-game transfer)",
        "device": os.environ.get(DEVICE_ENV, "cpu"),
        "mode": os.environ.get(MODE_ENV, "sampled"),
        "validated": False,  # a file's presence does not prove compatibility
    }


def _sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_public_state(observation: Mapping[str, Any]):
    """Construct the Phase3 actor state using only the acting seat's observe().

    The separate public_action_log comes from the GUI session, where street and
    pre-action pot/to_call are captured *before* the engine mutates state.
    """
    from holdem_ai.expand.information.schema import (
        InformationState, LegalActionSpec, PublicAction, PublicSeatFeatures,
    )

    raw = observation["raw_obs"]
    tournament = observation["tournament"]
    actor = int(raw["player_id"])
    button = int(raw["button"])
    bb = int(tournament["big_blind"])
    if bb <= 0 or actor not in range(6):
        raise ValueError("invalid tournament actor or big blind")
    legal_native = {int(a) for a in observation["legal_actions"]}
    if not legal_native or not legal_native.issubset(set(range(5))):
        raise ValueError("invalid native tournament legal actions")
    raw_call = int(raw["to_call"])
    stacks = tuple(int(x) for x in raw["stacks"])
    statuses = tuple(str(x) for x in raw["status"])
    investments = tuple(int(x) for x in raw["total_contributions"])
    bets = tuple(int(x) for x in raw["street_bets"])
    if not all(len(x) == 6 for x in (stacks, statuses, investments, bets)):
        raise ValueError("expected six physical seat observations")
    max_raise = bets[actor] + stacks[actor]
    min_raise = int(raw["min_raise_to"])
    mask = (
        0 in legal_native,
        1 in legal_native and raw_call == 0,
        1 in legal_native and raw_call > 0,
        2 in legal_native,
        3 in legal_native,
        False,  # double-pot is NOT a native action: do not silently remap
        4 in legal_native,
    )
    if not any(mask):
        raise ValueError("no legal 7-action representation of tournament turn")
    amount = [None] * 7
    # Exact engine chip arithmetic, not floating pot fractions.
    amount[3] = bets[actor] + raw_call + (int(raw["pot"]) + raw_call) // 2 if mask[3] else None
    amount[4] = bets[actor] + raw_call + (int(raw["pot"]) + raw_call) if mask[4] else None
    amount[6] = max_raise if mask[6] else None
    legal = LegalActionSpec(
        mask=mask, raise_to_chips=tuple(amount),
        min_raise_to_chips=min_raise, max_raise_to_chips=max_raise,
        to_call_chips=raw_call,
    )
    seats = tuple(
        PublicSeatFeatures(
            occupied=st != "out", dealt_in=st != "out",
            folded=st == "folded", all_in=st == "allin",
            seat_status=st, table_status="seated" if st != "out" else "sitting_out",
            stack_bb=stacks[i] / bb, invested_bb=investments[i] / bb,
            street_bet_bb=bets[i] / bb,
            relative_position=(i - button) % 6,
            is_hero=i == actor,
        ) for i, st in enumerate(statuses)
    )
    public_events = observation.get("public_action_log", ())
    names = {0: "fold", 1: "check", 2: "raise_half_pot", 3: "raise_pot",
             4: "all_in", 5: "raise_to"}
    history = []
    for event in public_events:
        index = int(event["index"])
        kind = names[index]
        if index == 1 and int(event["to_call_before"]) > 0:
            kind = "call"
        target = event.get("raise_to")
        history.append(PublicAction(
            actor_seat=int(event["seat"]), street=str(event["street"]), kind=kind,
            raise_to_bb=None if target is None else int(target) / bb,
            to_call_bb=int(event["to_call_before"]) / bb,
            pot_bb=int(event["pot_before"]) / bb,
        ))
    return InformationState(
        hero_seat=actor, hero_hole=tuple(raw["hand"]),
        board=tuple(raw["public_cards"]), street=str(raw["street"]),
        button_seat=button, seats=seats,
        action_history=tuple(history), legal_actions=legal,
        pot_bb=int(raw["pot"]) / bb, to_call_bb=raw_call / bb,
        big_blind_chips=bb, num_dealt_in=sum(s != "out" for s in statuses),
    )


class TrainedTournamentActor:
    """Readonly trained Expand PPO agent, physical seat assigned by session."""

    def __init__(self, model: Any, *, fingerprint: str, seed: int | None = None,
                 mode: str = "sampled"):
        if mode not in ("sampled", "greedy"):
            raise ValueError("Tournament policy mode must be sampled or greedy")
        self.model = model
        self.mode = mode
        self._rng = random.Random(seed)
        self.fingerprint = fingerprint
        self.name = f"Expand Phase3 PPO [{fingerprint[:12]}]"

    @classmethod
    def from_environment(cls, *, seed: int | None = None) -> "TrainedTournamentActor":
        value = os.environ.get(CHECKPOINT_ENV, "").strip()
        if not value:
            raise ValueError(f"trained tournament mode requires {CHECKPOINT_ENV}")
        path = Path(value).expanduser()
        if not path.is_file():
            raise ValueError(f"configured Tournament checkpoint is missing: {path.name}")
        device = os.environ.get(DEVICE_ENV, "cpu")
        mode = os.environ.get(MODE_ENV, "sampled")
        if device not in ("cpu", "cuda") or mode not in ("sampled", "greedy"):
            raise ValueError("Tournament device must be cpu/cuda and policy mode sampled/greedy")
        # Import lazily; the default 5-baseline/improved modes do not require torch.
        from holdem_ai.expand.rl.trainer import load_trained_actor
        try:
            model = load_trained_actor(path, device=device)
            fingerprint = _sha256(path)
        except (KeyError, ValueError, TypeError, RuntimeError, OSError) as exc:
            raise ValueError("checkpoint rejected: expected a compatible, trusted Expand Phase3 PPO checkpoint; HUNL 2-seat weights are not supported") from exc
        return cls(model, fingerprint=fingerprint, seed=seed, mode=mode)

    def select_action(self, observation: Mapping[str, Any]) -> int:
        from holdem_ai.expand.information.encoder import encode_observation
        state = _prepare_public_state(observation)
        # Using the actor head ONLY; the model critic is not queried.
        probabilities = tuple(float(p) for p in self.model.actor.action_probabilities(
            encode_observation(state)))
        if len(probabilities) != 7 or any(not math.isfinite(p) or p < 0 for p in probabilities):
            raise RuntimeError("invalid trained policy distribution")
        mask = state.legal_actions.mask
        weights = [p if allowed else 0.0 for p, allowed in zip(probabilities, mask)]
        total = math.fsum(weights)
        if total <= 0:
            raise RuntimeError("no probability assigned to a legal tournament action")
        if self.mode == "greedy":
            selected = max((i for i, ok in enumerate(mask) if ok), key=lambda i: weights[i])
        else:
            point = self._rng.random() * total
            cumulative = 0.0
            selected = next(i for i, p in enumerate(weights) if p > 0)  # numeric fallback
            for i, value in enumerate(weights):
                cumulative += value
                if point < cumulative and value > 0:
                    selected = i
                    break
        native = EXPAND_TO_NATIVE[selected]
        if native is None or native not in observation["legal_actions"]:
            raise RuntimeError("trained agent selected an invalid native tournament action")
        return native
