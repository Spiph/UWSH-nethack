"""Centered SVD/HOSVD, projection, and rank metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SVDBasis:
    mean: Array
    components: Array
    singular_values: Array

    def reconstruct(self, samples: Array) -> Array:
        centered = samples - self.mean
        return self.mean + (centered @ self.components.T) @ self.components


def select_rank(singular_values: Array, threshold: float = 0.95) -> int:
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")
    energy = singular_values**2
    if energy.size == 0 or float(energy.sum()) == 0:
        return 0
    return int(np.searchsorted(np.cumsum(energy) / energy.sum(), threshold) + 1)


def fit_svd(samples: Array, threshold: float = 0.95) -> SVDBasis:
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("samples must be a 2D population with at least two rows")
    mean = samples.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(samples - mean, full_matrices=False)
    rank = select_rank(singular_values, threshold)
    return SVDBasis(mean, vt[:rank], singular_values)


def mode_unfold(tensor: Array, mode: int) -> Array:
    return np.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def hosvd(tensor: Array) -> list[Array]:
    if tensor.ndim < 2:
        raise ValueError("HOSVD requires a tensor of order >= 2")
    return [
        np.linalg.svd(mode_unfold(tensor, mode), full_matrices=True)[0]
        for mode in range(tensor.ndim)
    ]


def mode_product(tensor: Array, matrix: Array, mode: int) -> Array:
    product = np.tensordot(matrix, tensor, axes=(1, mode))
    return np.moveaxis(product, 0, mode)


def hosvd_reconstruct(tensor: Array, factors: list[Array], ranks: list[int]) -> Array:
    if len(factors) != tensor.ndim or len(ranks) != tensor.ndim:
        raise ValueError("one factor and rank required per tensor mode")
    core = tensor
    for mode, factor in enumerate(factors):
        core = mode_product(core, factor[:, : ranks[mode]].T, mode)
    result = core
    for mode, factor in enumerate(factors):
        result = mode_product(result, factor[:, : ranks[mode]], mode)
    return result


def principal_angles(left: Array, right: Array) -> Array:
    q_left, _ = np.linalg.qr(left)
    q_right, _ = np.linalg.qr(right)
    values = np.linalg.svd(q_left.T @ q_right, compute_uv=False)
    return np.asarray(np.arccos(np.clip(values, -1, 1)), dtype=np.float64)


def projection_distance(left: Array, right: Array) -> float:
    q_left, _ = np.linalg.qr(left)
    q_right, _ = np.linalg.qr(right)
    return float(np.linalg.norm(q_left @ q_left.T - q_right @ q_right.T, ord="fro"))


def effective_rank(singular_values: Array) -> float:
    energy = singular_values**2
    if float(energy.sum()) == 0:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))
