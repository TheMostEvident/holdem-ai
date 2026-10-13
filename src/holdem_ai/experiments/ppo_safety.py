"""Phase 15 PPO stability safety policy.

Entropy below a fixed threshold is a warning, not proof of collapse.
KL breaches can still cause a hard stop after configured consecutive updates.
This function has no side effects and is testable independently.
"""
from __future__ import annotations


def safety_assessment(*, entropy: float, approx_kl: float,
                      entropy_threshold: float, max_approx_kl: float,
                      entropy_stop_enabled: bool, consecutive: int) -> dict:
    if type(entropy_stop_enabled) is not bool or consecutive < 0:
        raise ValueError("invalid safety settings")
    low_entropy = entropy < entropy_threshold
    high_kl = approx_kl > max_approx_kl
    safety_violation = high_kl or (low_entropy and entropy_stop_enabled)
    return {
        "low_entropy": low_entropy,
        "high_kl": high_kl,
        "consecutive_safety_warnings": consecutive + 1 if safety_violation else 0,
    }
