"""Small, dependency-light geometry helpers for Meta-World SAC actors."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

Array = npt.NDArray[Any]
FloatArray = npt.NDArray[np.float64]


def actor_vector(actor: torch.nn.Module) -> tuple[FloatArray, list[dict[str, Any]]]:
    """Return actor parameters as a float64 vector and their ordered layout."""
    pieces: list[FloatArray] = []
    layout: list[dict[str, Any]] = []
    offset = 0
    for name, parameter in actor.named_parameters():
        array = parameter.detach().cpu().contiguous().numpy().astype(np.float64, copy=False)
        flat = array.reshape(-1)
        pieces.append(flat)
        layout.append(
            {"name": name, "shape": list(array.shape), "start": offset, "end": offset + flat.size}
        )
        offset += flat.size
    if not pieces:
        return np.empty(0, dtype=np.float64), layout
    return np.concatenate(pieces), layout


def _validated_load(
    actor: torch.nn.Module, vector: Array, layout: list[dict[str, Any]]
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    if not isinstance(vector, np.ndarray) or vector.ndim != 1:
        raise ValueError("actor vector must be a one-dimensional NumPy array")
    if (
        not np.issubdtype(vector.dtype, np.number)
        or np.iscomplexobj(vector)
        or not np.isfinite(vector).all()
    ):
        raise ValueError("actor vector must contain only finite numeric values")
    if not isinstance(layout, list):
        raise ValueError("layout must be a list")

    parameters = list(actor.named_parameters())
    expected: list[dict[str, Any]] = []
    offset = 0
    for name, parameter in parameters:
        size = parameter.numel()
        expected.append(
            {"name": name, "shape": list(parameter.shape), "start": offset, "end": offset + size}
        )
        offset += size
    if len(layout) != len(expected):
        raise ValueError("layout parameter count does not match actor")
    for index, (item, want) in enumerate(zip(layout, expected, strict=True)):
        if not isinstance(item, dict) or item != want:
            raise ValueError(f"layout entry {index} does not exactly match actor parameter layout")
        if want["start"] < 0 or want["end"] < want["start"]:
            raise ValueError("layout offsets must be contiguous and nonnegative")
    if offset != vector.size:
        raise ValueError(f"actor vector length {vector.size} does not match expected {offset}")

    staged: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    for (_, parameter), item in zip(parameters, expected, strict=True):
        values = vector[item["start"] : item["end"]].reshape(item["shape"])
        converted = torch.as_tensor(values, device=parameter.device, dtype=parameter.dtype)
        if not torch.isfinite(converted).all():
            raise ValueError(f"values overflow parameter dtype for {item['name']}")
        staged.append((parameter, converted))
    return staged


def load_actor_vector(actor: torch.nn.Module, vector: Array, layout: list[dict[str, Any]]) -> None:
    """Load a vector only after validating the complete ordered parameter layout."""
    staged = _validated_load(actor, vector, layout)
    previous = [parameter.detach().clone() for parameter, _ in staged]
    try:
        with torch.no_grad():
            for parameter, values in staged:
                parameter.copy_(values)
    except Exception:
        with torch.no_grad():
            for (parameter, _), original in zip(staged, previous, strict=True):
                parameter.copy_(original)
        raise


def hash_actor(actor: torch.nn.Module) -> str:
    """Hash ordered named parameter values, matching the pilot helper."""
    digest = hashlib.sha256()
    for name, parameter in actor.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def verify_policy_symmetry(actor: torch.nn.Module, seed: int) -> float:
    """Permute two hidden layers and verify mean and log-std are unchanged."""
    hidden = getattr(actor, "latent_pi", None)
    if not isinstance(hidden, torch.nn.Module):
        raise ValueError("actor latent_pi must be a torch module")
    linears = [layer for layer in hidden.modules() if isinstance(layer, torch.nn.Linear)]
    if len(linears) < 2:
        raise ValueError("expected at least two linear layers in the SAC actor")
    if not hasattr(actor, "features_dim") or not callable(
        getattr(actor, "get_action_dist_params", None)
    ):
        raise ValueError("actor must expose features_dim and get_action_dist_params")
    first, second = linears[:2]
    original = {key: value.detach().clone() for key, value in actor.state_dict().items()}
    device = next(actor.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    parameter_dtype = next(actor.parameters()).dtype
    observations = torch.randn(
        (64, actor.features_dim), generator=generator, device=device, dtype=parameter_dtype
    )
    try:
        with torch.no_grad():
            before = actor.get_action_dist_params(observations)
            if len(before) < 2:
                raise ValueError("action distribution parameters must include mean and log_std")
            before_mean, before_log_std = before[0].detach().clone(), before[1].detach().clone()
            if not torch.isfinite(before_mean).all() or not torch.isfinite(before_log_std).all():
                raise ValueError("mean and log_std must be finite before permutation")
            permutation = torch.randperm(first.out_features, generator=generator, device=device)
            first.weight.copy_(first.weight[permutation])
            first.bias.copy_(first.bias[permutation])
            second.weight.copy_(second.weight[:, permutation])
            after = actor.get_action_dist_params(observations)
            if len(after) < 2:
                raise ValueError("action distribution parameters must include mean and log_std")
            if not torch.isfinite(after[0]).all() or not torch.isfinite(after[1]).all():
                raise ValueError("mean and log_std must be finite after permutation")
            errors = (
                torch.max(torch.abs(before_mean - after[0])).item(),
                torch.max(torch.abs(before_log_std - after[1])).item(),
            )
            error = float(max(errors))
    finally:
        actor.load_state_dict(original, strict=True)
    if error > 1e-6:
        raise ValueError(f"function preserving hidden permutation failed: max error={error}")
    return error


def _finite_vector(values: Array, name: str) -> FloatArray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite one-dimensional vector")
    return array


def fit_centered_basis(source_vectors: Array, rank: int) -> tuple[FloatArray, FloatArray, int]:
    """Fit a centered PCA basis using only the supplied source rows."""
    if np.iscomplexobj(source_vectors):
        raise ValueError("source_vectors must be real-valued")
    source = np.asarray(source_vectors, dtype=np.float64)
    if source.ndim != 2 or source.shape[0] < 2 or source.shape[1] < 1:
        raise ValueError("source_vectors must be a 2D array with at least two nonempty rows")
    if not np.isfinite(source).all():
        raise ValueError("source_vectors must be finite")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)) or rank < 1:
        raise ValueError("rank must be a positive integer")
    center = source.mean(axis=0)
    centered = source - center
    _, singular_values, right_vectors = np.linalg.svd(centered, full_matrices=False)
    tolerance = (
        max(centered.shape) * np.finfo(np.float64).eps * singular_values[0]
        if singular_values.size
        else 0.0
    )
    actual_rank = int(np.count_nonzero(singular_values > tolerance))
    if rank > actual_rank:
        raise ValueError(f"requested rank {rank} exceeds source rank {actual_rank}")
    return center, right_vectors[:rank].copy(), actual_rank


def project_oracle(vector: Array, center: Array, basis: Array) -> FloatArray:
    """Project a target vector onto a row-oriented, centered orthonormal basis."""
    target = _finite_vector(vector, "vector")
    mean = _finite_vector(center, "center")
    if np.iscomplexobj(basis):
        raise ValueError("basis must be real-valued")
    directions = np.asarray(basis, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != target.size or mean.size != target.size:
        raise ValueError("vector, center, and basis dimensions do not match")
    if directions.shape[0] < 1 or not np.isfinite(directions).all():
        raise ValueError("basis must contain finite directions")
    if not np.allclose(directions @ directions.T, np.eye(directions.shape[0]), atol=1e-8):
        raise ValueError("basis rows must be orthonormal")
    return np.asarray(mean + (target - mean) @ directions.T @ directions, dtype=np.float64)


def random_orthonormal_basis(parameter_count: int, rank: int, seed: int) -> FloatArray:
    """Generate a deterministic row-oriented basis with orthonormal rows."""
    if isinstance(parameter_count, bool) or not isinstance(parameter_count, (int, np.integer)):
        raise ValueError("parameter_count must be a positive integer")
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise ValueError("rank must be a positive integer")
    if parameter_count < 1 or rank < 1 or rank > parameter_count:
        raise ValueError("rank must be between one and parameter_count")
    generator = np.random.default_rng(seed)
    gaussian = generator.standard_normal((parameter_count, rank))
    orthogonal, _ = np.linalg.qr(gaussian, mode="reduced")
    return orthogonal.T
