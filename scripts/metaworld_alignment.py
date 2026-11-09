"""Exact two-hidden-layer SAC permutations and source-only weight matching.

No environment, policy training or GPU is used here. Verification probes are
fixed synthetic feature vectors, not target observations or teacher actions.
"""

from __future__ import annotations

import copy
import importlib
import math
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
Layout = list[dict[str, Any]]
SAC_PARAMETERS = (
    "latent_pi.0.weight",
    "latent_pi.0.bias",
    "latent_pi.2.weight",
    "latent_pi.2.bias",
    "mu.weight",
    "mu.bias",
    "log_std.weight",
    "log_std.bias",
)


def validate_layout(layout: Layout, size: int) -> Layout:
    offset, names = 0, set()
    for item in layout:
        if set(item) != {"name", "shape", "start", "end"}:
            raise ValueError("parameter layout fields differ from the actor NPZ schema")
        shape = item["shape"]
        if (
            not isinstance(item["name"], str)
            or not item["name"]
            or item["name"] in names
            or not isinstance(shape, list)
            or any(type(dim) is not int or dim < 1 for dim in shape)
            or type(item["start"]) is not int
            or type(item["end"]) is not int
            or item["start"] != offset
            or item["end"] != offset + math.prod(shape)
        ):
            raise ValueError("invalid, duplicate or noncontiguous parameter layout")
        names.add(item["name"])
        offset = item["end"]
    if not layout or offset != size:
        raise ValueError("parameter layout does not cover every actor parameter")
    return copy.deepcopy(layout)


def _parts(vector: Array, layout: Layout) -> dict[str, Array]:
    validate_layout(layout, vector.size)
    if set(item["name"] for item in layout) != set(SAC_PARAMETERS):
        raise ValueError("alignment supports exactly two SAC hidden layers and both output heads")
    parts = {
        item["name"]: vector[item["start"] : item["end"]].reshape(item["shape"]).copy()
        for item in layout
    }
    first, second = parts["latent_pi.0.weight"], parts["latent_pi.2.weight"]
    if (
        first.ndim != 2
        or second.ndim != 2
        or second.shape[1] != first.shape[0]
        or parts["latent_pi.0.bias"].shape != (first.shape[0],)
        or parts["latent_pi.2.bias"].shape != (second.shape[0],)
        or any(
            parts[f"{head}.weight"].ndim != 2
            or parts[f"{head}.weight"].shape[1] != second.shape[0]
            or parts[f"{head}.bias"].shape != (parts[f"{head}.weight"].shape[0],)
            for head in ("mu", "log_std")
        )
    ):
        raise ValueError("inconsistent feedforward actor layer dimensions")
    return parts


def _pack(parts: dict[str, Array], layout: Layout) -> Array:
    return np.concatenate([parts[item["name"]].reshape(-1) for item in layout])


def apply_sac_permutations(vector: Array, layout: Layout, permutations: list[list[int]]) -> Array:
    parts = _parts(vector, layout)
    if len(permutations) != 2:
        raise ValueError("one permutation is required for each of the two hidden layers")
    for permutation, layer in zip(permutations, (0, 2), strict=True):
        width = parts[f"latent_pi.{layer}.weight"].shape[0]
        if any(type(index) is not int for index in permutation) or sorted(permutation) != list(
            range(width)
        ):
            raise ValueError("hidden-unit permutation must be a bijection")
    first, second = permutations
    parts["latent_pi.0.weight"] = parts["latent_pi.0.weight"][first]
    parts["latent_pi.0.bias"] = parts["latent_pi.0.bias"][first]
    parts["latent_pi.2.weight"] = parts["latent_pi.2.weight"][second][:, first]
    parts["latent_pi.2.bias"] = parts["latent_pi.2.bias"][second]
    for head in ("mu", "log_std"):
        parts[f"{head}.weight"] = parts[f"{head}.weight"][:, second]
    return _pack(parts, layout)


def distribution_outputs(
    vector: Array, layout: Layout, observations: Array, activation: str
) -> tuple[Array, Array]:
    parts = _parts(vector, layout)
    if activation not in ("relu", "tanh"):
        raise ValueError("only pointwise relu/tanh feedforward actors are supported")
    if observations.ndim != 2 or observations.shape[1] != parts["latent_pi.0.weight"].shape[1]:
        raise ValueError("verification observations have the wrong feature dimensions")
    hidden = observations
    for layer in (0, 2):
        hidden = hidden @ parts[f"latent_pi.{layer}.weight"].T + parts[f"latent_pi.{layer}.bias"]
        hidden = np.maximum(hidden, 0) if activation == "relu" else np.tanh(hidden)
    mean = hidden @ parts["mu.weight"].T + parts["mu.bias"]
    log_std = hidden @ parts["log_std.weight"].T + parts["log_std.bias"]
    return np.asarray(mean, dtype=np.float64), np.asarray(log_std, dtype=np.float64)


def align_sac_vectors(
    vectors: Array,
    layout: Layout,
    *,
    anchor: int,
    activation: str,
    iterations: int,
    probe_seed: int,
) -> tuple[Array, list[dict[str, Any]]]:
    if type(iterations) is not int or not 1 <= iterations <= 100:
        raise ValueError("alignment iterations must be explicitly bounded in [1,100]")
    if not 0 <= anchor < vectors.shape[0]:
        raise ValueError("alignment anchor must be a remaining source actor")
    reference = _parts(vectors[anchor], layout)
    width0 = reference["latent_pi.0.weight"].shape[0]
    width1 = reference["latent_pi.2.weight"].shape[0]
    probes = np.random.default_rng(probe_seed).standard_normal(
        (64, reference["latent_pi.0.weight"].shape[1])
    )
    assignment = importlib.import_module("scipy.optimize").linear_sum_assignment
    result, reports = [], []
    for raw in vectors:
        current = raw.copy()
        accumulated = [list(range(width0)), list(range(width1))]
        for _ in range(iterations):
            changed = False
            for hidden_layer, width in enumerate((width0, width1)):
                candidate = _parts(current, layout)

                def signatures(parts: dict[str, Array], layer: int) -> Array:
                    arrays: tuple[Array, ...]
                    if layer == 0:
                        arrays = (
                            parts["latent_pi.0.weight"],
                            parts["latent_pi.0.bias"][:, None],
                            parts["latent_pi.2.weight"].T,
                        )
                    else:
                        arrays = (
                            parts["latent_pi.2.weight"],
                            parts["latent_pi.2.bias"][:, None],
                            parts["mu.weight"].T,
                            parts["log_std.weight"].T,
                        )
                    return np.concatenate(arrays, axis=1)

                rows, columns = assignment(
                    -(signatures(reference, hidden_layer) @ signatures(candidate, hidden_layer).T)
                )
                permutation = columns[np.argsort(rows)].tolist()
                step = [list(range(width0)), list(range(width1))]
                step[hidden_layer] = permutation
                current = apply_sac_permutations(current, layout, step)
                accumulated[hidden_layer] = [accumulated[hidden_layer][i] for i in permutation]
                changed = changed or permutation != list(range(width))
            if not changed:
                break
        before = distribution_outputs(raw, layout, probes, activation)
        after = distribution_outputs(current, layout, probes, activation)
        errors = [
            float(np.max(np.abs(left - right))) for left, right in zip(before, after, strict=True)
        ]
        if not np.isfinite(current).all() or not all(np.isfinite(errors)) or max(errors) > 1e-8:
            raise ValueError("alignment failed function preservation for mu/log_std heads")
        result.append(current)
        reports.append(
            {
                "permutations": accumulated,
                "verification": "fixed_synthetic_feature_probes; no target data",
                "probe_seed": probe_seed,
                "max_abs_mu_error": errors[0],
                "max_abs_log_std_error": errors[1],
            }
        )
    return np.stack(result), reports
