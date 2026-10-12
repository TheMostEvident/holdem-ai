"""Regression tests: RLCard public state may include nested NumPy scalars."""

import json

import pytest

from holdem_ai.environment.rlcard_adapter import observe_rlcard_state


def _sample_state(raw_obs):
    return {
        "obs": [0.0, 1.0],
        "legal_actions": {0: None, 2: None},
        "raw_obs": {
            "hand": ["SA", "DK"],
            "public_cards": [],
            "stage": "PREFLOP",
            **raw_obs,
        },
    }


def test_nested_numpy_integer_scalars_in_rlcard_public_state():
    np = pytest.importorskip("numpy")
    state = _sample_state({
        "all_chips": [np.int64(8), np.int32(100)],
        "stakes": {"small_blind": np.int64(5), "big_blind": np.int64(10)},
        "pot": np.int64(30),
        "current_player": np.int64(0),
    })
    observation = observe_rlcard_state(state, player_id=0, num_actions=3)
    public = {name: json.loads(value) for name, value in observation.public_state}
    assert public == {
        "all_chips": [8, 100],
        "stakes": {"big_blind": 10, "small_blind": 5},
        "pot": 30,
        "current_player": 0,
    }
    assert observation.legal_action_mask == (True, False, True)
    assert "8" in observation.infoset_key()


def test_nested_numpy_array_and_floats_are_json_safe():
    np = pytest.importorskip("numpy")
    state = _sample_state({
        "all_chips": [np.array([10, 20]), np.float32(1.25)],
        "pot": np.array(40, dtype=np.int64),
    })
    observation = observe_rlcard_state(state, 0, 3)
    public = {name: json.loads(value) for name, value in observation.public_state}
    assert public == {"all_chips": [[10, 20], 1.25], "pot": 40}


def test_invalid_public_value_rejected_rather_than_leaked_as_repr():
    state = _sample_state({"pot": object()})
    with pytest.raises(TypeError, match="not JSON serializable"):
        observe_rlcard_state(state, 0, 3)


def test_nan_public_value_still_rejected():
    np = pytest.importorskip("numpy")
    state = _sample_state({"pot": [np.float32(float("nan"))]})
    with pytest.raises(ValueError, match="Out of range float"):
        observe_rlcard_state(state, 0, 3)
