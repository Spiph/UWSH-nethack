from __future__ import annotations

import copy
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from scripts.metaworld_geometry import (
    actor_vector,
    fit_centered_basis,
    hash_actor,
    load_actor_vector,
    project_oracle,
    random_orthonormal_basis,
    verify_policy_symmetry,
)

Array = npt.NDArray[Any]


class DummySACActor(torch.nn.Module):
    """SAC-like actor with two hidden layers and separate distribution heads."""

    def __init__(self) -> None:
        super().__init__()
        self.features_dim = 5
        self.latent_pi = torch.nn.Sequential(
            torch.nn.Linear(self.features_dim, 7),
            torch.nn.Tanh(),
            torch.nn.Linear(7, 6),
            torch.nn.Tanh(),
        )
        self.mu = torch.nn.Linear(6, 3)
        self.log_std = torch.nn.Linear(6, 3)

    def get_action_dist_params(
        self, observations: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        latent = self.latent_pi(observations)
        return self.mu(latent), self.log_std(latent), {}


def _outputs(actor: DummySACActor) -> tuple[torch.Tensor, torch.Tensor]:
    observations = torch.linspace(-1.0, 1.0, 35).reshape(7, 5)
    mean, log_std, _ = actor.get_action_dist_params(observations)
    return mean.detach().clone(), log_std.detach().clone()


def test_actor_vector_round_trip_restores_parameters_and_behavior() -> None:
    torch.manual_seed(17)
    actor = DummySACActor()
    original_hash = hash_actor(actor)
    vector, layout = actor_vector(actor)
    original_outputs = _outputs(actor)

    with torch.no_grad():
        for parameter in actor.parameters():
            parameter.add_(torch.randn_like(parameter))
    load_actor_vector(actor, vector, layout)

    restored_vector, restored_layout = actor_vector(actor)
    restored_outputs = _outputs(actor)
    assert restored_layout == layout
    np.testing.assert_array_equal(restored_vector, vector)
    assert hash_actor(actor) == original_hash
    for expected, actual in zip(original_outputs, restored_outputs, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("corruption", ["missing", "extra", "reordered", "shape", "offset"])
def test_invalid_layout_is_rejected_without_mutating_actor(corruption: str) -> None:
    actor = DummySACActor()
    vector, layout = actor_vector(actor)
    invalid_layout = copy.deepcopy(layout)
    if corruption == "missing":
        invalid_layout.pop()
    elif corruption == "extra":
        invalid_layout.append(
            {"name": "extra", "shape": [1], "start": len(vector), "end": len(vector) + 1}
        )
    elif corruption == "reordered":
        invalid_layout[0], invalid_layout[1] = invalid_layout[1], invalid_layout[0]
    elif corruption == "shape":
        invalid_layout[0]["shape"] = [1]
    else:
        invalid_layout[1]["start"] += 1
    before = actor_vector(actor)[0].copy()

    with pytest.raises(ValueError):
        load_actor_vector(actor, vector, invalid_layout)

    np.testing.assert_array_equal(actor_vector(actor)[0], before)


@pytest.mark.parametrize(
    "bad_vector",
    [np.array([np.nan]), np.array([np.inf]), np.array([1.0]), np.array([1.0 + 2.0j])],
)
def test_invalid_vector_is_rejected_atomically(bad_vector: Array) -> None:
    actor = DummySACActor()
    vector, layout = actor_vector(actor)
    before = vector.copy()
    candidate = bad_vector if bad_vector.size != vector.size else bad_vector.copy()
    if candidate.size == vector.size:
        candidate[0] = bad_vector[0]

    with pytest.raises(ValueError):
        load_actor_vector(actor, candidate, layout)

    np.testing.assert_array_equal(actor_vector(actor)[0], before)


def test_hidden_permutation_preserves_both_distribution_outputs_and_restores_weights() -> None:
    torch.manual_seed(29)
    actor = DummySACActor()
    vector, layout = actor_vector(actor)
    outputs = _outputs(actor)
    original_hash = hash_actor(actor)

    error = verify_policy_symmetry(actor, seed=44)

    assert error < 1e-6
    assert hash_actor(actor) == original_hash
    np.testing.assert_array_equal(actor_vector(actor)[0], vector)
    for expected, actual in zip(outputs, _outputs(actor), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actor_vector(actor)[1] == layout


def test_symmetry_restores_weights_if_output_evaluation_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = DummySACActor()
    before = actor_vector(actor)[0].copy()
    original = actor.get_action_dist_params
    calls = 0

    def failing_output(
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected failure")
        return original(observations)

    monkeypatch.setattr(actor, "get_action_dist_params", failing_output)
    with pytest.raises(RuntimeError, match="injected failure"):
        verify_policy_symmetry(actor, seed=3)
    np.testing.assert_array_equal(actor_vector(actor)[0], before)


@pytest.mark.parametrize("output_name", ["mean", "log_std"])
@pytest.mark.parametrize("phase", ["before", "after"])
def test_symmetry_rejects_nonfinite_distribution_outputs_and_restores(
    output_name: str, phase: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    actor = DummySACActor()
    before = actor_vector(actor)[0].copy()
    original = actor.get_action_dist_params
    calls = 0

    def nonfinite_output(
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        nonlocal calls
        calls += 1
        mean, log_std, other = original(observations)
        if (phase == "before" and calls == 1) or (phase == "after" and calls == 2):
            selected = mean if output_name == "mean" else log_std
            selected[0, 0] = torch.nan
        return mean, log_std, other

    monkeypatch.setattr(actor, "get_action_dist_params", nonfinite_output)
    with pytest.raises(ValueError, match="must be finite"):
        verify_policy_symmetry(actor, seed=3)
    np.testing.assert_array_equal(actor_vector(actor)[0], before)


def test_symmetry_detects_log_std_only_permutation_corruption(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor = DummySACActor()
    before = actor_vector(actor)[0].copy()
    original = actor.get_action_dist_params
    calls = 0

    def corrupted_log_std(
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        nonlocal calls
        calls += 1
        mean, log_std, other = original(observations)
        if calls == 2:
            log_std = log_std + 1e-3
        return mean, log_std, other

    monkeypatch.setattr(actor, "get_action_dist_params", corrupted_log_std)
    with pytest.raises(ValueError, match="max error"):
        verify_policy_symmetry(actor, seed=3)
    np.testing.assert_array_equal(actor_vector(actor)[0], before)


def test_symmetry_supports_float64_actor() -> None:
    actor = DummySACActor().double()
    assert verify_policy_symmetry(actor, seed=9) < 1e-12


def test_known_subspace_reconstructs_unseen_vector_and_discards_out_of_span_part() -> None:
    center = np.array([0.2, -0.7, 1.1, 0.4, -0.3])
    basis = np.array([[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0]])
    source = np.stack(
        [center + basis.T @ coordinates for coordinates in ([1, 0], [-1, 0], [0, 1], [0, -1])]
    )
    target = center + basis.T @ np.array([0.35, -0.8])
    fitted_center, fitted_basis, actual_rank = fit_centered_basis(source, rank=2)

    reconstructed = project_oracle(target, fitted_center, fitted_basis)
    np.testing.assert_allclose(reconstructed, target, atol=1e-12)
    assert actual_rank == 2

    outside = target.copy()
    outside[2] += 4.0
    projected = project_oracle(outside, fitted_center, fitted_basis)
    np.testing.assert_allclose(projected, target, atol=1e-12)


def test_two_source_centered_basis_has_rank_at_most_one() -> None:
    source = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, -1.0, 5.0, 2.0]])
    center, basis, actual_rank = fit_centered_basis(source, rank=1)
    assert actual_rank == 1
    assert basis.shape == (1, source.shape[1])
    np.testing.assert_allclose(center, source.mean(axis=0))
    with pytest.raises(ValueError, match="exceeds source rank"):
        fit_centered_basis(source, rank=2)


def test_rank_deficient_sources_report_actual_rank_and_reject_unavailable_rank() -> None:
    line = np.array([1.0, 2.0, -1.0])
    source = np.stack([np.zeros(3), line, 2 * line, -line])
    center, basis, actual_rank = fit_centered_basis(source, rank=1)
    assert actual_rank == 1
    np.testing.assert_allclose((source - center) @ basis.T @ basis, source - center, atol=1e-12)
    with pytest.raises(ValueError, match="exceeds source rank"):
        fit_centered_basis(source, rank=2)


def test_random_basis_is_deterministic_and_row_orthonormal() -> None:
    first = random_orthonormal_basis(parameter_count=11, rank=4, seed=812)
    second = random_orthonormal_basis(parameter_count=11, rank=4, seed=812)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first @ first.T, np.eye(4), atol=1e-12)


@pytest.mark.parametrize(
    ("source", "rank"),
    [
        (np.ones((1, 3)), 1),
        (np.array([[0.0, 1.0], [np.nan, 2.0]]), 1),
        (np.ones((3, 2)), 0),
        (np.ones((3, 2)), 1.5),
    ],
)
def test_basis_fit_rejects_invalid_source_or_rank(source: Array, rank: int) -> None:
    with pytest.raises(ValueError):
        fit_centered_basis(source, rank)


def test_basis_fit_rejects_complex_source() -> None:
    source = np.array([[0.0 + 1.0j, 2.0], [1.0, 3.0]])
    with pytest.raises(ValueError, match="real-valued"):
        fit_centered_basis(source, rank=1)


def test_projection_rejects_nonfinite_and_mismatched_inputs() -> None:
    basis = np.eye(2)
    with pytest.raises(ValueError):
        project_oracle(np.array([1.0, np.inf]), np.zeros(2), basis)
    with pytest.raises(ValueError):
        project_oracle(np.ones(3), np.zeros(2), basis)
    with pytest.raises(ValueError, match="orthonormal"):
        project_oracle(np.ones(2), np.zeros(2), np.ones((1, 2)))


def test_projection_rejects_complex_vector_center_and_basis() -> None:
    basis = np.eye(2)
    with pytest.raises(ValueError, match="real-valued"):
        project_oracle(np.array([1.0 + 1.0j, 2.0]), np.zeros(2), basis)
    with pytest.raises(ValueError, match="real-valued"):
        project_oracle(np.ones(2), np.array([0.0 + 1.0j, 0.0]), basis)
    with pytest.raises(ValueError, match="real-valued"):
        project_oracle(np.ones(2), np.zeros(2), np.array([[1.0 + 1.0j, 0.0]]))


@pytest.mark.parametrize(("parameter_count", "rank"), [(0, 1), (4, 0), (4, 5)])
def test_random_basis_rejects_invalid_dimensions(parameter_count: int, rank: int) -> None:
    with pytest.raises(ValueError):
        random_orthonormal_basis(parameter_count, rank, seed=1)
