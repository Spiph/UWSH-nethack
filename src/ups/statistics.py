"""Leakage-safe grouped bootstrap utilities."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def hierarchical_bootstrap(
    values: Array,
    tasks: Sequence[str],
    seeds: Sequence[int],
    replicates: int,
    rng: np.random.Generator,
) -> Array:
    if values.ndim != 1 or not (len(values) == len(tasks) == len(seeds)):
        raise ValueError("values, tasks, and seeds must have equal one-dimensional length")
    unique_tasks = np.unique(tasks)
    results = np.empty(replicates)
    for replicate in range(replicates):
        task_draw = rng.choice(unique_tasks, size=len(unique_tasks), replace=True)
        selected: list[float] = []
        for task in task_draw:
            indices = np.flatnonzero(np.asarray(tasks) == task)
            task_seeds = np.unique(np.asarray(seeds)[indices])
            seed_draw = rng.choice(task_seeds, size=len(task_seeds), replace=True)
            for seed in seed_draw:
                matching = indices[np.asarray(seeds)[indices] == seed]
                selected.append(float(values[rng.choice(matching)]))
        results[replicate] = np.median(selected)
    return results


def confidence_interval(samples: Array, level: float = 0.95) -> tuple[float, float]:
    if samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("bootstrap samples must be finite and non-empty")
    tail = (1 - level) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return float(low), float(high)
