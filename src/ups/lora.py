"""Canonical LoRA composition and numerical validation."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def compose_lora(a: Array, b: Array, alpha: float | None = None) -> Array:
    """Return the canonical update alpha/rank * B@A (PEFT convention)."""
    if a.ndim != 2 or b.ndim != 2 or b.shape[1] != a.shape[0]:
        raise ValueError("expected A[rank,in] and B[out,rank]")
    rank = a.shape[0]
    scale = 1.0 if alpha is None else alpha / rank
    return scale * (b @ a)


def rotate_factors(a: Array, b: Array, rotation: Array) -> tuple[Array, Array]:
    """Apply the exact LoRA gauge transform A'=R A, B'=B R^-1."""
    if rotation.shape != (a.shape[0], a.shape[0]):
        raise ValueError("rotation shape must equal rank by rank")
    return rotation @ a, b @ np.linalg.inv(rotation)


def merged_equivalent(weight: Array, a: Array, b: Array, x: Array, alpha: float) -> bool:
    scale = alpha / a.shape[0]
    direct = x @ weight.T + scale * ((x @ a.T) @ b.T)
    merged = x @ (weight + compose_lora(a, b, alpha)).T
    return bool(np.allclose(direct, merged, rtol=1e-10, atol=1e-10))
