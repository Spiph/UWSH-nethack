"""Deterministic matched null ensembles."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def gaussian_norm_matched(update: Array, rng: np.random.Generator) -> Array:
    sample = rng.normal(size=update.shape)
    norm = np.linalg.norm(sample)
    return sample * (np.linalg.norm(update) / norm) if norm else sample


def spectrum_matched_orientation(update: Array, rng: np.random.Generator) -> Array:
    if update.ndim != 2:
        raise ValueError("spectrum matching expects a matrix")
    _, singular_values, _ = np.linalg.svd(update, full_matrices=False)
    q_left, _ = np.linalg.qr(rng.normal(size=(update.shape[0], len(singular_values))))
    q_right, _ = np.linalg.qr(rng.normal(size=(update.shape[1], len(singular_values))))
    return np.asarray((q_left * singular_values) @ q_right.T, dtype=np.float64)


def independent_low_rank(update: Array, rank: int, rng: np.random.Generator) -> Array:
    if update.ndim != 2 or not 0 < rank <= min(update.shape):
        raise ValueError("invalid matrix or rank")
    left = rng.normal(size=(update.shape[0], rank))
    right = rng.normal(size=(rank, update.shape[1]))
    candidate = left @ right
    return np.asarray(
        candidate * (np.linalg.norm(update) / np.linalg.norm(candidate)), dtype=np.float64
    )


def element_shuffle(update: Array, rng: np.random.Generator) -> Array:
    flat = update.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(update.shape)


def layer_shuffle(updates: list[Array], rng: np.random.Generator) -> list[Array]:
    if len({item.shape for item in updates}) != 1:
        raise ValueError("layer shuffle requires shape-compatible layers")
    order = rng.permutation(len(updates))
    return [updates[index].copy() for index in order]
