"""Read-only Phase 15 PPO optimizer diagnostics from existing JSONL logs."""
from __future__ import annotations

import json
import math
from pathlib import Path

FIELDS = ("entropy", "approx_kl", "clip_fraction", "grad_norm_actor",
          "grad_norm_critic", "actor_loss", "critic_loss")


def audit_metrics(path: str | Path) -> dict:
    updates = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") != "update" or record.get("algorithm") != "ppo":
            continue
        values = {}
        for name in FIELDS:
            value = record.get(name)
            if value is not None:
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"nonfinite {name} at line {number}")
                values[name] = float(value)
        updates.append({"update": record["update"], **values})
    if not updates:
        raise ValueError("no PPO update metrics found")
    out = {}
    for field in FIELDS:
        values = [row[field] for row in updates if field in row]
        out[field] = ({
            "observations": len(values),
            "first": values[0], "last": values[-1],
            "min": min(values), "max": max(values),
            "mean": round(sum(values) / len(values), 7),
        } if values else None)
    return {
        "updates": len(updates), "first_update": updates[0]["update"],
        "last_update": updates[-1]["update"], "fields": out,
        "last_20_updates": updates[-20:],
        "note": "Metrics are optimizer-minibatch aggregates, not fixed-state statistics. "
                "Learning rate/entropy coefficient come from run.json ppo_config; "
                "these summaries alone cannot identify a causal collapse mechanism.",
    }
