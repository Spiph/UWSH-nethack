"""Git Re-Basin-style channel/unit coordinate descent.

GRU permutations are deliberately unsupported: recurrent comparisons use invariant
principal-angle, projection, and representation metrics.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

Array = npt.NDArray[np.float64]


def match_units(reference: Array, candidate: Array) -> npt.NDArray[np.int64]:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("unit matrices must be shape-compatible [units, features]")
    scores = reference @ candidate.T
    row, column = linear_sum_assignment(-scores)
    return np.asarray(column[np.argsort(row)], dtype=np.int64)


def permute_mlp(
    first_weight: Array,
    first_bias: Array,
    second_weight: Array,
    permutation: npt.NDArray[np.int64],
) -> tuple[Array, Array, Array]:
    """Permute hidden outputs and compensate the next layer's inputs."""
    return (
        first_weight[permutation],
        first_bias[permutation],
        second_weight[:, permutation],
    )


def permute_conv_channels(
    weight: Array, bias: Array, next_weight: Array, permutation: npt.NDArray[np.int64]
) -> tuple[Array, Array, Array]:
    if weight.ndim != 4 or next_weight.ndim != 4:
        raise ValueError("convolution kernels must be [out,in,h,w]")
    return weight[permutation], bias[permutation], next_weight[:, permutation]


def assert_supported_module(name: str) -> None:
    if "gru" in name.lower() or "recurrent" in name.lower():
        raise NotImplementedError("GRU symmetry is not resolved; use invariant metrics")
