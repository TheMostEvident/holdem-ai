"""Regression tests for Phase 15 *read-only* policy/optimizer diagnostics."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from holdem_ai.evaluation.fixed_state_policy_audit import compare_actors_on_states
from holdem_ai.evaluation.ppo_metric_audit import audit_metrics


class StubActor:
    def __init__(self, fold_logit: float) -> None:
        self.fold_logit = fold_logit
        self.training = True

    def eval(self):
        self.training = False
        return self

    def train(self, flag=True):
        self.training = flag
        return self

    def distribution(self, batch):
        b, a = batch.legal_action_mask.shape
        logits = torch.zeros(b, a)
        logits[:, 0] = self.fold_logit
        logits = logits.masked_fill(~batch.legal_action_mask, -1.0e9)
        return torch.distributions.Categorical(logits=logits)


def _row(*, stage=0, button=1.0, mask=(True, False, True, False, False, False, False)):
    features = [0.0] * 20
    features[1] = button
    features[2] = float(stage)
    return SimpleNamespace(
        token_ids=(1,), token_values=(0.0,), features=tuple(features),
        legal_action_mask=mask,
    )


def test_fixed_state_fold_probability_change_is_not_trajectory_shift():
    rows = (_row(), _row(stage=1, button=0.0),
            _row(mask=(False, True, False, False, False, False, False)))
    before, after = StubActor(0.0), StubActor(2.0)
    results = compare_actors_on_states({"before": before, "after": after}, rows)
    a, b = results["before"]["overall"], results["after"]["overall"]
    assert a["states"] == b["states"] == 3
    assert a["fold_legal_states"] == b["fold_legal_states"] == 2
    assert a["mean_fold_probability_given_legal"] == pytest.approx(0.5)
    assert b["mean_fold_probability_given_legal"] == pytest.approx(
        torch.sigmoid(torch.tensor(2.0)).item(), abs=1e-6)
    assert a["multi_legal_states"] == 2
    assert results["before"]["street:flop"]["states"] == 1
    assert before.training and after.training  # no lasting state mutation


def test_no_fold_opportunities_and_one_legal_action():
    rows = (_row(mask=(False, True, False, False, False, False, False)),)
    result = compare_actors_on_states({"x": StubActor(1)}, rows)["x"]["overall"]
    assert result["mean_fold_probability_given_legal"] is None
    assert result["multi_legal_states"] == 0
    assert result["mean_normalized_entropy_for_multi_legal"] is None


def test_optimizer_metric_audit(tmp_path):
    p = tmp_path / "metrics.jsonl"
    p.write_text("\n".join([
        json.dumps({"kind": "update", "algorithm": "ppo", "update": 1,
                    "entropy": 0.8, "approx_kl": 0.001, "clip_fraction": 0.1}),
        json.dumps({"kind": "validation", "update": 1}),
        json.dumps({"kind": "update", "algorithm": "ppo", "update": 2,
                    "entropy": 0.2, "approx_kl": 0.002, "clip_fraction": 0.2}),
    ]), encoding="utf-8")
    summary = audit_metrics(p)
    assert summary["updates"] == 2
    assert summary["fields"]["entropy"]["last"] == 0.2
    assert summary["fields"]["grad_norm_actor"] is None


def test_optimizer_metric_audit_rejects_nonfinite(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"kind":"update","algorithm":"ppo","update":1,"entropy":NaN}\n')
    with pytest.raises(ValueError, match="nonfinite entropy"):
        audit_metrics(p)
