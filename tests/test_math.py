import numpy as np
import pytest

from ups.alignment import assert_supported_module, permute_conv_channels, permute_mlp
from ups.lora import compose_lora, merged_equivalent, rotate_factors
from ups.metrics import action_agreement, action_kl, linear_cka, normalized_value_rmse
from ups.nulls import (
    element_shuffle,
    gaussian_norm_matched,
    independent_low_rank,
    layer_shuffle,
    spectrum_matched_orientation,
)
from ups.subspace import (
    effective_rank,
    fit_svd,
    hosvd,
    hosvd_reconstruct,
    principal_angles,
    projection_distance,
    select_rank,
)


def test_lora_rotation_and_merge() -> None:
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(2, 5)), rng.normal(size=(4, 2))
    rotation = np.array([[2.0, 0.5], [0.0, 1.0]])
    new_a, new_b = rotate_factors(a, b, rotation)
    assert np.allclose(compose_lora(a, b, 8), compose_lora(new_a, new_b, 8))
    assert merged_equivalent(rng.normal(size=(4, 5)), a, b, rng.normal(size=(7, 5)), 8)
    with pytest.raises(ValueError):
        compose_lora(a, np.ones((4, 3)))


def test_svd_hosvd_and_geometry() -> None:
    rng = np.random.default_rng(2)
    samples = rng.normal(size=(20, 6))
    basis = fit_svd(samples, 1.0)
    assert np.allclose(basis.reconstruct(samples), samples)
    assert select_rank(np.array([3.0, 1.0]), 0.8) == 1
    assert select_rank(np.zeros(2)) == 0
    tensor = rng.normal(size=(3, 4, 2))
    factors = hosvd(tensor)
    assert np.allclose(hosvd_reconstruct(tensor, factors, [3, 4, 2]), tensor)
    frame = np.eye(5)[:, :2]
    assert np.allclose(principal_angles(frame, frame), 0)
    assert projection_distance(frame, frame) == pytest.approx(0)
    assert effective_rank(np.array([1.0, 1.0])) == pytest.approx(2)


def test_metrics() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(30, 8))
    features = rng.normal(size=(30, 5))
    assert action_kl(logits, logits) == pytest.approx(0, abs=1e-14)
    assert action_agreement(logits, logits) == 1
    assert linear_cka(features, features) == pytest.approx(1)
    assert normalized_value_rmse(logits[:, 0], logits[:, 0]) == 0


def test_null_invariants() -> None:
    rng = np.random.default_rng(4)
    update = rng.normal(size=(7, 5))
    assert np.linalg.norm(gaussian_norm_matched(update, rng)) == pytest.approx(
        np.linalg.norm(update)
    )
    matched = spectrum_matched_orientation(update, rng)
    assert np.allclose(
        np.linalg.svd(matched, compute_uv=False), np.linalg.svd(update, compute_uv=False)
    )
    low = independent_low_rank(update, 2, rng)
    assert np.linalg.matrix_rank(low, tol=1e-10) <= 2
    assert np.linalg.norm(low) == pytest.approx(np.linalg.norm(update))
    shuffled = element_shuffle(update, rng)
    assert np.array_equal(np.sort(shuffled.ravel()), np.sort(update.ravel()))
    layers = [np.full((2, 2), index) for index in range(3)]
    assert sorted(int(item[0, 0]) for item in layer_shuffle(layers, rng)) == [0, 1, 2]


def test_permutation_function_invariance_and_gru_exclusion() -> None:
    rng = np.random.default_rng(5)
    w1, bias, w2 = rng.normal(size=(4, 3)), rng.normal(size=4), rng.normal(size=(2, 4))
    permutation = np.array([3, 1, 0, 2])
    p1, pb, p2 = permute_mlp(w1, bias, w2, permutation)
    x = rng.normal(size=(10, 3))
    assert np.allclose(np.maximum(0, x @ w1.T + bias) @ w2.T, np.maximum(0, x @ p1.T + pb) @ p2.T)
    kernel, next_kernel = rng.normal(size=(4, 3, 3, 3)), rng.normal(size=(2, 4, 3, 3))
    pk, p_bias, pn = permute_conv_channels(kernel, bias, next_kernel, permutation)
    assert np.array_equal(pk, kernel[permutation])
    assert np.array_equal(p_bias, bias[permutation])
    assert np.array_equal(pn, next_kernel[:, permutation])
    with pytest.raises(NotImplementedError):
        assert_supported_module("recurrent_core.gru")
