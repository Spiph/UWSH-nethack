"""Reproducible, seed-level statistical summaries for RL experiments.

The unit of replication in these helpers is a training seed (never an
episode).  They intentionally accept ordinary pandas data frames so that
reports can be generated from either evaluation parquet files or collected
learning-curve data.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]

Array = npt.NDArray[np.float64]


def _finite_frame(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame.empty:
        raise ValueError("statistics input is empty")
    for column in columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"column {column!r} must be numeric")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"column {column!r} contains non-finite values")


def _validate_units(
    frame: pd.DataFrame,
    environment_col: str,
    seed_col: str,
    value_col: str,
    checkpoint_col: str | None = None,
    expected_environments: Sequence[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> None:
    columns = [environment_col, seed_col, value_col]
    if checkpoint_col:
        columns.append(checkpoint_col)
    _finite_frame(frame, [seed_col, value_col] + ([checkpoint_col] if checkpoint_col else []))
    if frame[columns].isna().any().any():
        raise ValueError("statistics input contains incomplete rows")
    unit_columns = [environment_col, seed_col] + ([checkpoint_col] if checkpoint_col else [])
    if frame.duplicated(subset=unit_columns).any():
        raise ValueError("duplicate seed-level units; do not treat episodes as replications")
    if expected_environments is not None:
        actual = set(frame[environment_col].astype(str))
        if actual != set(map(str, expected_environments)):
            raise ValueError(
                f"incomplete environments: expected {set(expected_environments)}, got {actual}"
            )
    if expected_seeds is not None:
        actual = set(frame[seed_col].astype(int))
        if actual != set(expected_seeds):
            raise ValueError(f"incomplete seeds: expected {set(expected_seeds)}, got {actual}")
    if expected_environments is not None and expected_seeds is not None:
        observed_units = set(
            zip(frame[environment_col].astype(str), frame[seed_col].astype(int), strict=False)
        )
        expected_units = {
            (str(environment), int(seed))
            for environment in expected_environments
            for seed in expected_seeds
        }
        if observed_units != expected_units:
            raise ValueError("incomplete environment-by-seed replication units")


def hierarchical_bootstrap(
    values: Array,
    tasks: Sequence[str],
    seeds: Sequence[int],
    replicates: int,
    rng: np.random.Generator,
) -> Array:
    """Legacy task/seed bootstrap, retaining task resampling for subspace analyses."""
    if values.ndim != 1 or not (len(values) == len(tasks) == len(seeds)):
        raise ValueError("values, tasks, and seeds must have equal one-dimensional length")
    if replicates <= 0 or not np.isfinite(values).all():
        raise ValueError("values must be finite and replicates must be positive")
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
    if samples.size == 0 or not np.isfinite(samples).all() or not 0 < level < 1:
        raise ValueError("bootstrap samples must be finite and non-empty; level must be in (0, 1)")
    tail = (1 - level) / 2
    low, high = np.quantile(samples, [tail, 1 - tail])
    return float(low), float(high)


def _summary(values: Array, bootstrap_replicates: int, rng: np.random.Generator) -> dict[str, Any]:
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("seed values must be finite and non-empty")
    draws = rng.choice(values, size=(bootstrap_replicates, values.size), replace=True).mean(axis=1)
    low, high = confidence_interval(draws)
    return {
        "n_seeds": int(values.size),
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "standard_error": float(values.std(ddof=1) / np.sqrt(values.size))
        if values.size > 1
        else 0.0,
        "median": float(np.median(values)),
        "ci95_low": low,
        "ci95_high": high,
    }


def aggregate_seed_results(
    frame: pd.DataFrame,
    *,
    value_col: str = "value",
    environment_col: str = "environment",
    seed_col: str = "seed",
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
    expected_environments: Sequence[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Aggregate one seed-level value per environment without pooling episodes."""
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    _validate_units(
        frame,
        environment_col,
        seed_col,
        value_col,
        expected_environments=expected_environments,
        expected_seeds=expected_seeds,
    )
    rng = np.random.default_rng(bootstrap_seed)
    rows = []
    for environment, group in frame.groupby(environment_col, sort=True):
        stats = _summary(group[value_col].to_numpy(float), bootstrap_replicates, rng)
        rows.append({"environment": str(environment), "metric": value_col, **stats})
    return pd.DataFrame(rows)


def aggregate_learning_curves(
    frame: pd.DataFrame,
    *,
    value_col: str = "value",
    environment_col: str = "environment",
    seed_col: str = "seed",
    checkpoint_col: str = "checkpoint",
    checkpoints: Sequence[float] | None = None,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
    expected_environments: Sequence[str] | None = None,
    expected_seeds: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Aggregate curves on a common checkpoint grid, with seed-level CIs."""
    _validate_units(
        frame,
        environment_col,
        seed_col,
        value_col,
        checkpoint_col,
        expected_environments,
        expected_seeds,
    )
    available = set(frame[checkpoint_col].unique())
    grid = sorted(available if checkpoints is None else set(checkpoints))
    if not set(grid).issubset(available):
        raise ValueError("requested checkpoint grid is not present")
    expected = set(expected_seeds) if expected_seeds is not None else None
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    for (environment, checkpoint), group in frame[frame[checkpoint_col].isin(grid)].groupby(
        [environment_col, checkpoint_col], sort=True
    ):
        seeds = set(group[seed_col].astype(int))
        if expected is not None and seeds != expected:
            raise ValueError(f"incomplete seed curve at {environment}/{checkpoint}")
        rows.append(
            {
                "environment": str(environment),
                "checkpoint": checkpoint,
                "metric": value_col,
                **_summary(group[value_col].to_numpy(float), bootstrap_replicates, rng),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty or result.groupby("environment")["checkpoint"].nunique().min() != len(grid):
        raise ValueError("incomplete common checkpoint grid")
    return result


def bootstrap_power_diagnostic(
    baseline: Sequence[float],
    treatment: Sequence[float],
    *,
    minimum_lift: float,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Estimate two-sided detection power for a declared seed-level lift.

    Centered bootstrap draws estimate the null distribution of the difference
    in seed means.  Shifting that distribution by ``minimum_lift`` estimates
    how often the same two-sided bootstrap test would reject at alpha=0.05.
    This remains a diagnostic with five seeds, not proof that the study is
    adequately powered for every downstream subspace comparison.
    """
    a, b = np.asarray(baseline, float), np.asarray(treatment, float)
    if (
        a.ndim != 1
        or b.ndim != 1
        or a.size < 2
        or b.size < 2
        or not np.isfinite(a).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("baseline and treatment require >=2 finite seed-level values")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")
    if not np.isfinite(minimum_lift) or minimum_lift <= 0:
        raise ValueError("minimum_lift must be finite and positive")
    rng = np.random.default_rng(bootstrap_seed)
    centered_a = a - a.mean()
    centered_b = b - b.mean()
    null_draws = rng.choice(centered_b, (bootstrap_replicates, b.size), replace=True).mean(
        1
    ) - rng.choice(centered_a, (bootstrap_replicates, a.size), replace=True).mean(1)
    critical_value = float(np.quantile(np.abs(null_draws), 0.95))
    alternative_draws = null_draws + minimum_lift
    estimated_power = float(np.mean(np.abs(alternative_draws) > critical_value))
    observed_draws = rng.choice(b, (bootstrap_replicates, b.size), replace=True).mean(
        1
    ) - rng.choice(a, (bootstrap_replicates, a.size), replace=True).mean(1)
    low, high = confidence_interval(observed_draws)
    return {
        "method": "centered independent seed bootstrap power diagnostic",
        "estimand": "absolute difference in treatment and baseline seed means",
        "resampling": "sample centered training-seed values with replacement within each cohort",
        "alpha": 0.05,
        "target_power": 0.80,
        "decision_rule": (
            "two-sided rejection when absolute mean difference exceeds the "
            "bootstrapped 95th-percentile null critical value"
        ),
        "n_baseline_seeds": int(a.size),
        "n_treatment_seeds": int(b.size),
        "observed_lift": float(b.mean() - a.mean()),
        "minimum_lift": float(minimum_lift),
        "null_critical_value": critical_value,
        "estimated_power_at_minimum_lift": estimated_power,
        "meets_target_power": estimated_power >= 0.80,
        "ci95_low": low,
        "ci95_high": high,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": bootstrap_seed,
    }


def write_statistics_report(
    seed_results: pd.DataFrame,
    output_prefix: str | Path,
    *,
    learning_curves: pd.DataFrame | None = None,
    power: dict[str, Any] | None = None,
    individual_results: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Write machine-readable JSON and presentation-ready CSV/Parquet artifacts."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if seed_results.empty:
        raise ValueError("seed_results is empty")
    outputs = {
        "json": str(prefix.with_suffix(".json")),
        "csv": str(prefix.with_suffix(".csv")),
        "parquet": str(prefix.with_suffix(".parquet")),
    }
    seed_results.to_csv(outputs["csv"], index=False)
    seed_results.to_parquet(outputs["parquet"], index=False)
    payload: dict[str, Any] = {"seed_summaries": seed_results.to_dict(orient="records")}
    if individual_results is not None:
        if individual_results.empty:
            raise ValueError("individual_results is empty")
        payload["individual_results"] = individual_results.to_dict(orient="records")
        individual_path = prefix.with_name(prefix.name + "_individual.csv")
        individual_results.to_csv(individual_path, index=False)
        outputs["individual_csv"] = str(individual_path)
    if learning_curves is not None:
        curve_path = prefix.with_name(prefix.name + "_learning_curves.parquet")
        learning_curves.to_parquet(curve_path, index=False)
        payload["learning_curves"] = learning_curves.to_dict(orient="records")
        outputs["learning_curves_parquet"] = str(curve_path)
    if power is not None:
        payload["power_diagnostic"] = power
    prefix.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return outputs
