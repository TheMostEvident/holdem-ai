# Poker AI — Verified Python source excerpts and function contracts

These are **literal code excerpts fetched from repository main**, not generated pseudocode. Each section specifies its original path. The excerpts are intentionally partial: consult the full linked source file for behavior beyond the quoted lines. Accompanying explanations distinguish source observations from broader interpretation.

## Base Environment P3 — native betting engine

**Actual source:** [`src/holdem_ai/environment/multi_player_engine.py`](https://github.com/TheMostEvident/holdem-ai/blob/main/src/holdem_ai/environment/multi_player_engine.py)

**Role and process:** The engine fixes five native action indices, keeps physical seats, and sets chip and deck validation. Its reset and subsequent actions implement a single hand; persistence between hands is delegated to the tournament or GUI service.

```python
"""Standalone 2-6 player no-limit Hold'em *hand* environment.

Keeps RLCard's five integer action IDs, but does not reuse its unsupported
multiway betting engine. Uses the project's exact reference showdown evaluator
and side-pot settlement. A hand resets stacks; tournaments come in Phase 4.

All amounts are integral chips, and raise_to always means the total amount
invested *on the current street*, not the additional chips to push.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from holdem_ai.environment.hand_evaluator import RANKS, SUITS, rank_showdown
from holdem_ai.environment.multi_player_seats import multi_player_seats, next_occupied
from holdem_ai.environment.pot_manager import build_side_pots, settle_pots

FOLD = 0
CHECK_CALL = 1
RAISE_HALF_POT = 2
RAISE_POT = 3
ALL_IN = 4


@dataclass(frozen=True)
class MultiPlayerHandResult:
    payoffs_chips: tuple[int, ...]
    payoffs_bb: tuple[float, ...]
    final_stacks: tuple[int, ...]
    contributions: tuple[int, ...]
    awards: tuple[int, ...]
    board: tuple[str, ...]
    folded: tuple[bool, ...]
    button: int
    action_history: tuple[tuple[int, int], ...]
    action_raise_to: tuple[int | None, ...]


class MultiPlayerHandEngine:
    """Play one hand using fixed physical seats and a moving-button model.

    `stacks` may contain zeroes for eliminated seats, provided >=2 remain.
    `deck` is only for deterministic tests; default RNG shuffles all 52 cards.
    Does not provide step_back or tournament-level chip persistence.
    """

    def __init__(self, stacks: Sequence[int] | None = None, *,
                 num_players: int = 6, starting_stack: int = 200,
                 small_blind: int = 1, big_blind: int = 2,
                 button: int = 0, seed: int | None = 42,
```

## Base Agent P9 — PPO baseline

**Actual source:** [`src/holdem_ai/training/ppo.py`](https://github.com/TheMostEvident/holdem-ai/blob/main/src/holdem_ai/training/ppo.py)

**Role and process:** The class and dataclasses show a frozen on-policy, terminal-reward PPO contract, where an actor never sees privileged critic state. Configuration validates learning rate, epsilon, batch size and other safety parameters.

```python
"""On-policy PPO baseline for Phase-8 HUNL200 networks (NOT VRPO/Q-Boosting).

Poker is undiscounted, zero-sum, and terminal-reward-only: every decision by
seat i in a finished hand has return G_i equal to that seat's final payoff BB.
A privileged Q-critic is allowed to construct a TRAINING-ONLY, action-independent
state baseline V_i(s) = sum_a pi_old(a|I_i) * Q_i(s,a). No critic state enters
Actor.forward or NeuralActorPolicy.act. Fitted Q targets only observed actions:
there are no counterfactual action labels in a single self-play trajectory.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.environment.hunl200_adapter import HUNL200Adapter
from holdem_ai.training.global_rollout import collect_global_episode
from holdem_ai.encoding.tensors import batch_actors, batch_critics
from holdem_ai.encoding.observation import encode_actor, encode_critic
from holdem_ai.encoding.observation import EncodedActor, EncodedCritic
from holdem_ai.models.transformer import CentralizedQCritic, TransformerActor, chosen_q
from holdem_ai.agents.neural_policy import NeuralActorPolicy


@dataclass(frozen=True)
class PPOConfig:
    hands_per_update: int = 16
    epochs: int = 2
    minibatch_size: int = 32
    learning_rate: float = 3e-4
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    grad_clip: float = 0.5
    inference_batch_size: int = 64
    max_decisions: int = 200
    big_blind: int = 100
    seed: int = 42

    def __post_init__(self) -> None:
        for field in ('hands_per_update', 'epochs', 'minibatch_size',
                      'inference_batch_size', 'max_decisions', 'big_blind'):
            value = getattr(self, field)
            if type(value) is not int or value < 1:
                raise ValueError(f'{field} must be a positive integer')
        if type(self.seed) is not int:
```

## Expand E4 — External-Sampling MCCFR

**Actual source:** [`src/holdem_ai/expand/pluribus/blueprint/mccfr.py`](https://github.com/TheMostEvident/holdem-ai/blob/main/src/holdem_ai/expand/pluribus/blueprint/mccfr.py)

**Role and process:** The source defines a game-state protocol, memory/node/depth budgets, regret matching and atomic iteration updates. Traverser branches are enumerated while chance/non-traverser actions are sampled.

```python
"""Multi-player external-sampling MCCFR with linear weights and atomic iterations.

On every traversal: chance is sampled by game.start; non-traverser actions
are sampled from their current policy; traverser actions are fully branched.
We add the traverser's own-reach-weighted strategy to the average when its
infosets are traversed. This estimator samples *opponent/chance reach* and
explicitly multiplies *own reach*, avoiding opponent-hole-card exposure.

Regret deltas and average deltas are only committed after traversal succeeds;
node/infoset cap failures cannot leave half an MCCFR iteration applied.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict
import hashlib
import math
import random
from typing import Protocol


class SearchBudgetExceeded(RuntimeError):
    pass


class StateProtocol(Protocol):
    @property
    def current_player(self) -> int: ...
    def is_terminal(self) -> bool: ...
    def legal_actions(self) -> tuple[str, ...]: ...
    def infoset(self, player: int) -> str: ...
    def next_state(self, action: str): ...
    def terminal_payoffs(self) -> tuple[float, ...]: ...


@dataclass(frozen=True)
class MCCFRConfig:
    seed: int = 400_000_004
    max_nodes_per_iteration: int = 20_000
    max_infosets: int = 200_000
    max_depth: int = 256
    linear_weighting: bool = True
    # Negative regret pruning intentionally not enabled in Phase 4 baseline.
    # Validate baseline convergence before developing biased optional pruning.
    algorithm: str = 'external_sampling_linear_mccfr_v1'

    def __post_init__(self):
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('seed must be nonnegative integer')
        for name in ('max_nodes_per_iteration','max_infosets','max_depth'):
            if type(getattr(self,name)) is not int or getattr(self,name) <= 0:
                raise ValueError(f'{name} must be positive integer')
        if type(self.linear_weighting) is not bool:
```

## Tournament ENV 1.4 — frozen multi-seat adapter

**Actual source:** [`src/holdem_ai/tournament_service/trained_actor.py`](https://github.com/TheMostEvident/holdem-ai/blob/main/src/holdem_ai/tournament_service/trained_actor.py)

**Role and process:** The adapter explicitly identifies its cash-game transfer status, declares native and Expand action IDs, and returns a status object that does not claim validated model structure from file existence.

```python
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
```

## HUNL GUI 1.0/1.6 — session manager

**Actual source:** [`src/holdem_ai/game_service/session_manager.py`](https://github.com/TheMostEvident/holdem-ai/blob/main/src/holdem_ai/game_service/session_manager.py)

**Role and process:** This stateful session separates transport from game logic, handles independent or continuous stacks, uses an RLock and revision checks, and prevents starting another hand before the current one ends.

```python
"""Local HUNL200 human-vs-agent game sessions; transport agnostic.

No server or React dependency. Training remains independent 200BB; local
GUI sessions may carry stacks between hands. Synchronous revision-checked play.
"""
from __future__ import annotations

import random
import secrets
import threading
from typing import Any

from holdem_ai.agents.policies import RandomLegalPolicy
from holdem_ai.core.schema import Terminal, Turn
from .interactive_adapter import InteractiveHUNLAdapter

from .action_adapter import validate_human_action
from .bot_adapter import BotActionAdapter, GUIBotPolicy
from .state_projection import project_human_view
from .session_analytics import SessionAnalytics


class HUNLGameSession:
    """An in-memory multi-hand heads-up session: human=seat 0, bot=seat 1.

    A bot policy can be RandomLegalPolicy, a diagnostic agent or a compatible
    NeuralActorPolicy. A trusted server must load checkpoints separately.
    """

    def __init__(
        self, *, bot_policy: GUIBotPolicy | None = None,
        big_blind: int = 100, small_blind: int = 50,
        stack_bb: int = 200, hero_seat: int = 0,
        seed: int | None = None, max_bot_actions: int = 1000,
        stack_mode: str = "independent",
    ) -> None:
        if hero_seat != 0:
            raise ValueError("GUI-1.0 fixes human seat to 0")
        if type(max_bot_actions) is not int or max_bot_actions < 1:
            raise ValueError("max_bot_actions must be positive integer")
        if stack_mode not in ("independent", "continuous"):
            raise ValueError("stack_mode must be independent or continuous")
        self.stack_mode = stack_mode
        # Validate configuration BEFORE accepting a game session.
        InteractiveHUNLAdapter(big_blind=big_blind, small_blind=small_blind, stack_bb=stack_bb)
        self.big_blind = big_blind
        self.small_blind = small_blind
        self.stack_bb = stack_bb
        self.hero_seat = hero_seat
        self.max_bot_actions = max_bot_actions
        self._seed_rng = random.Random(seed)
        self.bot = BotActionAdapter(bot_policy or RandomLegalPolicy(), random.Random(self._seed_rng.getrandbits(64)))
```

## Base Agent P17 — resumable PPO retrain

**Actual source:** [`src/holdem_ai/experiments/ppo_five_hour.py`](https://github.com/TheMostEvident/holdem-ai/blob/main/src/holdem_ai/experiments/ppo_five_hour.py)

**Role and process:** The typed configuration distinguishes intended active training duration, updates per call, replay/validation/critic seeds, KL guard thresholds and resume behavior. This is the training scheduler, not the policy's proof of strength.

```python
"""Phase17: resumable 5-hour PPO retraining (AGENT ONLY, new run).

Reuses validated Phase16 on-policy A0/A3 update, exact post-update KL and
matched-deck frozen-opponent evaluation. Existing engine, GUI, PPO math,
Phase14 source and Phase15/16 reports remain untouched.

The wall-clock budget accumulates ACTIVE invocation time, excluding downtime
between sessions. Training stops at an update boundary; validation at the
last update may extend past the requested wall-clock budget.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch

from holdem_ai.evaluation.checkpoint_loader import load_actor
from holdem_ai.experiments.longrun import _exclusive_lock
from holdem_ai.experiments.ppo_stress import (
    StressConfig, _baseline, _one_update, _validation,
    _json_atomic, _sha,
)
from holdem_ai.training.ppo_session import load_checkpoint


PROTOCOL = "phase17_five_hour_ppo_v1"


@dataclass(frozen=True)
class FiveHourConfig:
    source_checkpoint: str
    root: str
    device: str = "cpu"
    duration_hours: float = 5.0
    variant: str = "a0"       # A0 monitoring by default; A3 opt-in gate
    hands_per_update: int = 128
    minibatch_size: int = 128
    epochs: int = 1
    env_batch_size: int = 32
    eval_every: int = 25
    eval_groups: int = 80
    critic_eval_hands: int = 128
    save_every: int = 10
    max_updates: int = 5000
    rollout_seed: int = 700_000_000
    validation_seed: int = 80_000_000
```
