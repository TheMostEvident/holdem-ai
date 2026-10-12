"""Trusted local checkpoint selection for GUI-1.4 HUNL evaluation.

Clients choose an ID, never a checkpoint path. The GUI neither accepts uploaded
weights nor downloads weights, and never exposes local filesystem paths.
Torch / the Phase-16 loader are imported only when a neural session is made.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Mapping

from holdem_ai.agents.diagnostic_opponents import build_builtin

BUILTIN_OPPONENTS = ("random", "calling_station", "tight", "pot_pressure")

MODEL_PATH_VARIABLES = {
    "ppo": "HOLDEM_AI_PPO_CHECKPOINT",
    "qboost": "HOLDEM_AI_QBOOST_CHECKPOINT",
}
LABELS = {
    "random": "Random policy",
    "calling_station": "Calling station",
    "tight": "Tight policy",
    "pot_pressure": "Pot pressure",
    "ppo": "Trained PPO",
    "qboost": "Trained Q-Boost PPO",
}


@dataclass(frozen=True, slots=True)
class BotInfo:
    id: str
    label: str
    trained: bool
    available: bool
    message: str

    def as_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "trained": self.trained,
                "available": self.available, "message": self.message}


class CheckpointCatalog:
    """Two server-configured, trusted model slots plus four baselines.

    One policy object is shared across game sessions; each session owns an
    independent RNG. NeuralActorPolicy holds a frozen eval-only Actor, so
    no training, optimizer update, or policy mutation occurs in this service.
    """

    def __init__(self, *, environment: Mapping[str, str] | None = None) -> None:
        env = dict(os.environ if environment is None else environment)
        self.paths = {key: env.get(var, "").strip() for key, var in MODEL_PATH_VARIABLES.items()}
        self.device = env.get("HOLDEM_AI_GUI_DEVICE", "cpu").strip().lower()
        self.mode = env.get("HOLDEM_AI_GUI_POLICY_MODE", "sampled").strip().lower()
        if self.device not in ("cpu", "cuda"):
            raise ValueError("HOLDEM_AI_GUI_DEVICE must be cpu or cuda")
        if self.mode not in ("sampled", "greedy"):
            raise ValueError("HOLDEM_AI_GUI_POLICY_MODE must be sampled or greedy")
        self._lock = threading.RLock()
        self._cached: dict[str, tuple[tuple[str, int, int], object]] = {}

    def _model_file(self, name: str) -> Path | None:
        raw = self.paths[name]
        return Path(raw).expanduser().resolve() if raw else None

    def options(self) -> list[dict]:
        results = [BotInfo(name, LABELS[name], False, True, "Ready to play").as_dict()
                   for name in BUILTIN_OPPONENTS]
        for name in MODEL_PATH_VARIABLES:
            path = self._model_file(name)
            exists = bool(path and path.is_file())
            message = ("Checkpoint found; compatibility checked when the game starts"
                       if exists else "Checkpoint file not configured or missing")
            results.append(BotInfo(name, LABELS[name], True, exists, message).as_dict())
        return results

    def build_policy(self, name: str):
        if name in BUILTIN_OPPONENTS:
            return build_builtin(name)
        if name not in MODEL_PATH_VARIABLES:
            raise ValueError("Unknown bot selection")
        path = self._model_file(name)
        if path is None or not path.is_file():
            raise ValueError(f"{LABELS[name]} checkpoint is not configured or missing. Configure {MODEL_PATH_VARIABLES[name]}.")
        try:
            metadata = path.stat()
            key = (str(path), metadata.st_mtime_ns, metadata.st_size)
        except OSError as exc:
            raise ValueError("Unable to inspect configured checkpoint") from exc
        with self._lock:
            if name in self._cached and self._cached[name][0] == key:
                return self._cached[name][1]
            try:
                # Phase 16's shared loader uses torch.load(weights_only=True),
                # NetworkConfig, and strict Actor weights. No Python pickle load.
                from holdem_ai.evaluation.checkpoint_loader import load_actor
                from holdem_ai.agents.neural_policy import NeuralActorPolicy

                actor = load_actor(path, device=self.device)
                if actor.config.dropout != 0:
                    raise ValueError("evaluation model must have dropout=0")
                policy = NeuralActorPolicy(actor, device=self.device,
                                           big_blind=100, greedy=self.mode == "greedy")
            except (ImportError, OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Could not load {LABELS[name]} checkpoint. Confirm it is a trusted, "
                    "compatible Phase 9-15 Actor file and the required PyTorch device is available."
                ) from exc
            self._cached[name] = (key, policy)
            return policy
