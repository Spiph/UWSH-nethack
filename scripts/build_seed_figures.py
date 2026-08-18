"""Render uncertainty-aware cohort figures from ``ups report`` artifacts.

The input contains one value per training seed at every retained checkpoint.
Plots therefore show both the raw seed trajectories and two different, clearly
labeled summaries: ±1 sample standard deviation (between-seed dispersion) and
the 95% bootstrap confidence interval for the seed mean (mean uncertainty).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS = {
    "success": ("Fixed-evaluation success", "Success rate", (0.0, 1.0)),
    "return": ("Fixed-evaluation return", "Mean episode return", None),
    "length": ("Fixed-evaluation episode length", "Mean episode length", (0.0, None)),
    "max_depth": ("NetHack dungeon progress", "Mean maximum dungeon depth", (0.0, None)),
}


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--statistics",
        type=Path,
        required=True,
        help="The seed_report.json emitted by `ups report`.",
    )
    command.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/progress-assets/confirmatory"),
        help="Directory for SVG and PNG figures.",
    )
    return command


def _load(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    curves = pd.DataFrame(payload.get("learning_curves", []))
    individual = pd.DataFrame(payload.get("individual_results", []))
    required_curves = {
        "environment",
        "checkpoint",
        "metric",
        "mean",
        "sample_std",
        "ci95_low",
        "ci95_high",
        "n_seeds",
    }
    required_individual = {"environment", "seed", "checkpoint", "metric", "value"}
    missing = required_curves.difference(curves.columns)
    if missing:
        raise ValueError(f"statistics file has no complete learning curves: {sorted(missing)}")
    missing = required_individual.difference(individual.columns)
    if missing:
        raise ValueError(f"statistics file has no seed-level values: {sorted(missing)}")
    if int(curves["n_seeds"].min()) < 2:
        raise ValueError("sample-standard-deviation bands require at least two training seeds")
    return curves, individual


def _limits(metric: str, lower: np.ndarray, upper: np.ndarray) -> tuple[float | None, float | None]:
    bounds = METRICS[metric][2]
    low = bounds[0] if bounds else None
    high = bounds[1] if bounds else None
    if low is None:
        low = float(np.nanmin(lower))
    if high is None:
        high = float(np.nanmax(upper))
    if low == high:
        padding = 0.5 if low == 0 else abs(low) * 0.1
        return low - padding, high + padding
    padding = (high - low) * 0.08
    return low - (0 if bounds and bounds[0] is not None else padding), high + (
        0 if bounds and bounds[1] is not None else padding
    )


def _render_metric(
    metric: str,
    curves: pd.DataFrame,
    individual: pd.DataFrame,
    output_dir: Path,
) -> None:
    metric_curves = curves[curves["metric"] == metric].copy()
    if metric_curves.empty:
        return
    metric_individual = individual[individual["metric"] == metric]
    environments = sorted(metric_curves["environment"].unique())
    columns = min(3, len(environments))
    rows = int(np.ceil(len(environments) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5.1 * columns, 3.8 * rows), squeeze=False)
    color = "#2563EB"
    for axis, environment in zip(axes.flat, environments, strict=False):
        curve = metric_curves[metric_curves["environment"] == environment].sort_values("checkpoint")
        x = curve["checkpoint"].to_numpy(float) / 1_000_000
        mean = curve["mean"].to_numpy(float)
        std = curve["sample_std"].to_numpy(float)
        ci_low = curve["ci95_low"].to_numpy(float)
        ci_high = curve["ci95_high"].to_numpy(float)
        lower, upper = mean - std, mean + std
        if metric == "success":
            lower, upper = np.clip(lower, 0.0, 1.0), np.clip(upper, 0.0, 1.0)
        seed_values = metric_individual[metric_individual["environment"] == environment]
        for _, seed_curve in seed_values.groupby("seed", sort=True):
            seed_curve = seed_curve.sort_values("checkpoint")
            axis.plot(
                seed_curve["checkpoint"] / 1_000_000,
                seed_curve["value"],
                color="#94A3B8",
                linewidth=1.0,
                alpha=0.75,
                zorder=1,
            )
        axis.fill_between(x, lower, upper, color=color, alpha=0.18, label="±1 sample SD", zorder=2)
        axis.fill_between(
            x, ci_low, ci_high, color="#0F172A", alpha=0.12, label="95% bootstrap CI", zorder=3
        )
        axis.plot(
            x,
            mean,
            color=color,
            linewidth=2.4,
            marker="o",
            markersize=3.5,
            label="Seed mean",
            zorder=4,
        )
        axis.set_title(environment.removesuffix("-v0"), fontsize=10, fontweight="bold")
        axis.set_xlabel("Environment steps (millions)")
        axis.set_ylabel(METRICS[metric][1])
        axis.set_ylim(*_limits(metric, lower, upper))
        axis.grid(axis="y", color="#E2E8F0")
    for axis in axes.flat[len(environments) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle(METRICS[metric][0] + " across independent training seeds", fontweight="bold")
    figure.text(
        0.5,
        0.04,
        "Thin gray lines are individual training seeds. SD shows replicate dispersion; "
        "the bootstrap interval estimates uncertainty in the seed mean.",
        ha="center",
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.92))
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"seed_{metric}_curves.svg", bbox_inches="tight")
    figure.savefig(output_dir / f"seed_{metric}_curves.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parser().parse_args()
    curves, individual = _load(args.statistics)
    for metric in METRICS:
        _render_metric(metric, curves, individual, args.output_dir)
    print(f"Wrote seed-level figures to {args.output_dir}")


if __name__ == "__main__":
    main()
