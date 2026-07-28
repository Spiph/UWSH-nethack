import numpy as np
import pytest
import torch

from ups.model import NLE_ACTIONS, RecurrentNLEPolicy
from ups.statistics import confidence_interval, hierarchical_bootstrap


def test_policy_shapes_registry_and_lora() -> None:
    model = RecurrentNLEPolicy(glyph_vocab=128, crop_size=5, hidden_size=32, lora_rank=2)
    output = model(torch.zeros(3, 2, 5, 5), torch.zeros(3, 2, 27))
    assert output.logits.shape == (3, 2, len(NLE_ACTIONS))
    assert output.value.shape == (3, 2)
    assert output.state.shape == (1, 2, 32)
    assert "recurrent_core.gru" in model.module_registry


def test_hierarchical_bootstrap_is_deterministic() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    tasks = ["a", "a", "b", "b"]
    seeds = [0, 1, 0, 1]
    one = hierarchical_bootstrap(values, tasks, seeds, 50, np.random.default_rng(9))
    two = hierarchical_bootstrap(values, tasks, seeds, 50, np.random.default_rng(9))
    assert np.array_equal(one, two)
    low, high = confidence_interval(one)
    assert low <= high
    with pytest.raises(ValueError):
        confidence_interval(np.array([np.nan]))
