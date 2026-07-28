import numpy as np
import pytest
import torch

from ups.model import NLE_ACTIONS, RecurrentNLEPolicy
from ups.sol import crop_glyphs
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


def test_phase_zero_crop_uses_nethack_xy_coordinates_and_padding() -> None:
    glyphs = np.arange(9, dtype=np.int16).reshape(3, 3)
    center = crop_glyphs(glyphs, np.array([1, 1]), 3)
    assert np.array_equal(center, glyphs)
    corner = crop_glyphs(glyphs, np.array([0, 0]), 3)
    assert corner.shape == (3, 3)
    assert corner[1, 1] == glyphs[0, 0]
    with pytest.raises(ValueError, match="shape"):
        crop_glyphs(glyphs[None], np.array([0, 0]), 3)
