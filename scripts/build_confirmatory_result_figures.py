"""Build the complete Phase Zero behavioral-results figure set.

The source of truth is the retained fixed-evaluation JSON emitted for every
environment, training seed, and checkpoint. Uncertainty is computed over the
five independent training seeds—not over evaluation episodes.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "phase0.yaml"
DEFAULT_OUTPUT = ROOT / "docs" / "result-assets"

BLUE = "#2563EB"
NAVY = "#0F172A"
GREEN = "#059669"
ORANGE = "#EA580C"
RED = "#DC2626"
GRAY = "#64748B"
LIGHT_GRAY = "#CBD5E1"
SEED_COLORS = ["#2563EB", "#EA580C", "#059669", "#7C3AED", "#DB2777"]

SHORT_NAMES = {
    "MiniHack-Room-Random-5x5-v0": "Random",
    "MiniHack-Room-Dark-5x5-v0": "Dark",
    "MiniHack-Room-Monster-5x5-v0": "Monster",
    "MiniHack-Room-Trap-5x5-v0": "Trap",
    "MiniHack-Room-Ultimate-5x5-v0": "Ultimate",
    "MiniHack-MazeWalk-9x9-v0": "MazeWalk",
}

SUMMARY_PATTERN = re.compile(r"-seed(?P<seed>\d+)-step(?P<checkpoint>\d+)\.json$")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    command.add_argument("--bootstrap-seed", type=int, default=0)
    return command


def configure_style() -> None:
    plt.rcParams.update(
        {
            "axes.facecolor": "#FFFFFF",
            "axes.edgecolor": LIGHT_GRAY,
            "axes.grid": False,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "figure.facecolor": "#FFFFFF",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "legend.frameon": False,
            "savefig.facecolor": "#FFFFFF",
            "xtick.color": GRAY,
            "ytick.color": GRAY,
        }
    )


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_records(config: dict[str, Any]) -> pd.DataFrame:
    artifact_root = ROOT / config["artifact_root"]
    evaluation_root = artifact_root / "evaluations"
    environments = list(config["environments"])
    seeds = [int(seed) for seed in config["seeds"]]
    interval = int(config["training"]["evaluation_interval"])
    maximum = int(config["training"]["max_environment_steps"])
    expected_checkpoints = set(range(interval, maximum + interval, interval))
    expected_episodes = int(config["training"]["evaluation_episodes"])
    expected_pairs = {
        (environment, seed, checkpoint)
        for environment in environments
        for seed in seeds
        for checkpoint in expected_checkpoints
    }
    records: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for summary_path in sorted(evaluation_root.glob("*.json")):
        match = SUMMARY_PATTERN.search(summary_path.name)
        if match is None:
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        environment = payload.get("environment")
        seed = int(match.group("seed"))
        checkpoint = int(match.group("checkpoint"))
        key = (environment, seed, checkpoint)
        if key not in expected_pairs:
            continue
        if key in seen:
            raise ValueError(f"duplicate fixed evaluation for {key}")
        seen.add(key)
        if payload.get("episodes") != expected_episodes:
            raise ValueError(f"wrong episode count in {summary_path}")
        if payload.get("max_episode_steps") is not None:
            raise ValueError(f"bounded diagnostic evaluation in {summary_path}")
        checkpoint_path = ROOT / payload["checkpoint"]
        table_path = ROOT / payload["table"]
        if not checkpoint_path.is_file() or not table_path.is_file():
            raise ValueError(f"missing linked artifact for {summary_path}")
        records.append(
            {
                "environment": environment,
                "environment_label": SHORT_NAMES[environment],
                "seed": seed,
                "checkpoint": checkpoint,
                "success": float(payload["success_rate"]),
                "return": float(payload["mean_return"]),
                "length": float(payload["mean_length"]),
                "episodes": int(payload["episodes"]),
                "eval_seed": int(payload["eval_seed"]),
                "summary": str(summary_path.relative_to(ROOT)),
                "table": str(Path(payload["table"])),
                "checkpoint_path": str(Path(payload["checkpoint"])),
            }
        )
    missing = expected_pairs - seen
    if missing:
        raise ValueError(f"missing {len(missing)} fixed evaluations; first={sorted(missing)[:3]}")
    frame = pd.DataFrame(records)
    frame["environment"] = pd.Categorical(frame["environment"], environments, ordered=True)
    return frame.sort_values(["environment", "seed", "checkpoint"]).reset_index(drop=True)


def bootstrap_interval(
    values: np.ndarray, rng: np.random.Generator, replicates: int
) -> tuple[float, float]:
    draws = rng.choice(values, size=(replicates, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def aggregate(records: pd.DataFrame, config: dict[str, Any], bootstrap_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(bootstrap_seed)
    replicates = int(config["analysis"]["bootstrap_replicates"])
    rows: list[dict[str, Any]] = []
    for metric in ("success", "return", "length"):
        for (environment, checkpoint), group in records.groupby(
            ["environment", "checkpoint"], observed=True, sort=False
        ):
            values = group[metric].to_numpy(float)
            low, high = bootstrap_interval(values, rng, replicates)
            rows.append(
                {
                    "environment": str(environment),
                    "environment_label": SHORT_NAMES[str(environment)],
                    "checkpoint": int(checkpoint),
                    "metric": metric,
                    "mean": float(values.mean()),
                    "sample_std": float(values.std(ddof=1)),
                    "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
                    "median": float(np.median(values)),
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_seeds": len(values),
                }
            )
    return pd.DataFrame(rows)


def save(figure: plt.Figure, output_dir: Path, name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_dir / f"{name}.svg", bbox_inches="tight")
    figure.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_final_success(
    records: pd.DataFrame,
    curves: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    maximum = int(config["training"]["max_environment_steps"])
    threshold = float(config["training"]["minimum_success"])
    final = records[records["checkpoint"] == maximum]
    summary = curves[(curves["checkpoint"] == maximum) & (curves["metric"] == "success")]
    environments = list(config["environments"])
    seeds = [int(seed) for seed in config["seeds"]]
    figure, axis = plt.subplots(figsize=(12.6, 5.8))
    offsets = np.linspace(-0.16, 0.16, len(seeds))
    for index, environment in enumerate(environments):
        values = final[final["environment"] == environment].sort_values("seed")
        axis.scatter(
            index + offsets,
            values["success"] * 100,
            color=[SEED_COLORS[seeds.index(int(seed))] for seed in values["seed"]],
            s=52,
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        row = summary[summary["environment"] == environment].iloc[0]
        mean = float(row["mean"]) * 100
        axis.errorbar(
            index,
            mean,
            yerr=[[mean - float(row["ci95_low"]) * 100], [float(row["ci95_high"]) * 100 - mean]],
            color=NAVY,
            marker="D",
            markersize=7,
            capsize=5,
            linewidth=2,
            zorder=5,
        )
        competent = int((values["success"] >= threshold).sum())
        axis.text(index, 4.5, f"{competent}/5 competent", ha="center", color=NAVY, fontsize=10)
    axis.axhline(threshold * 100, color=RED, linestyle="--", linewidth=1.6)
    axis.text(
        len(environments) - 0.5,
        threshold * 100 + 2,
        f"frozen competence = {threshold:.0%}",
        color=RED,
        ha="right",
        fontsize=10,
    )
    axis.set_xticks(
        range(len(environments)), [SHORT_NAMES[environment] for environment in environments]
    )
    axis.set_ylim(0, 104)
    axis.set_ylabel("Success over 200 fixed episodes (%)")
    axis.set_title(
        "Five task populations converge; MazeWalk remains initialization-sensitive",
        fontsize=17,
        fontweight="bold",
    )
    axis.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.01,
        "Colored points are independent training seeds; diamonds are seed means "
        "and bars are 95% bootstrap intervals.",
        ha="center",
        color=GRAY,
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save(figure, output_dir, "final_success_by_environment")


def plot_competence_matrix(records: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> None:
    maximum = int(config["training"]["max_environment_steps"])
    threshold = float(config["training"]["minimum_success"])
    environments = list(config["environments"])
    seeds = [int(seed) for seed in config["seeds"]]
    final = records[records["checkpoint"] == maximum]
    matrix = np.array(
        [
            [
                float(
                    final[(final["environment"] == environment) & (final["seed"] == seed)][
                        "success"
                    ].iloc[0]
                )
                for seed in seeds
            ]
            for environment in environments
        ]
    )
    figure, axis = plt.subplots(figsize=(10.2, 6.2))
    image = axis.imshow(matrix * 100, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.1%}",
                ha="center",
                va="center",
                color="white" if value < 0.45 or value > 0.9 else NAVY,
                fontweight="bold",
                fontsize=12,
            )
            if value < threshold:
                axis.add_patch(
                    plt.Rectangle(
                        (column - 0.49, row - 0.49),
                        0.98,
                        0.98,
                        fill=False,
                        edgecolor=RED,
                        linewidth=3,
                    )
                )
    axis.set_xticks(range(len(seeds)), [f"Seed {seed}" for seed in seeds])
    axis.set_yticks(
        range(len(environments)), [SHORT_NAMES[environment] for environment in environments]
    )
    axis.set_title("Final competence across all 30 policies", fontsize=17, fontweight="bold")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.03)
    colorbar.set_label("Fixed-evaluation success (%)")
    figure.text(
        0.5,
        0.02,
        f"Red outlines mark policies below the prespecified {threshold:.0%} competence threshold.",
        ha="center",
        color=GRAY,
    )
    figure.tight_layout(rect=(0, 0.06, 1, 1))
    save(figure, output_dir, "final_competence_matrix")


def plot_metric_curves(
    records: pd.DataFrame,
    curves: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
    metric: str,
) -> None:
    environments = list(config["environments"])
    seeds = [int(seed) for seed in config["seeds"]]
    threshold = float(config["training"]["minimum_success"])
    titles = {
        "success": ("Fixed-evaluation success across five training seeds", "Success rate"),
        "return": ("Fixed-evaluation return across five training seeds", "Mean return"),
        "length": ("Episode length exposes inefficient and failed policies", "Mean actions"),
    }
    figure, axes = plt.subplots(2, 3, figsize=(15.2, 8.2), sharex=True)
    for axis, environment in zip(axes.flat, environments, strict=True):
        individual = records[records["environment"] == environment]
        for color, seed in zip(SEED_COLORS, seeds, strict=True):
            seed_rows = individual[individual["seed"] == seed].sort_values("checkpoint")
            axis.plot(
                seed_rows["checkpoint"] / 1_000_000,
                seed_rows[metric],
                color=color,
                linewidth=1.4,
                alpha=0.9,
                label=f"Seed {seed}",
            )
        curve = curves[
            (curves["environment"] == environment) & (curves["metric"] == metric)
        ].sort_values("checkpoint")
        x = curve["checkpoint"].to_numpy(float) / 1_000_000
        mean = curve["mean"].to_numpy(float)
        std = curve["sample_std"].to_numpy(float)
        lower, upper = mean - std, mean + std
        if metric == "success":
            lower, upper = np.clip(lower, 0, 1), np.clip(upper, 0, 1)
        axis.fill_between(x, lower, upper, color=BLUE, alpha=0.12, label="±1 sample SD")
        axis.fill_between(
            x,
            curve["ci95_low"],
            curve["ci95_high"],
            color=NAVY,
            alpha=0.10,
            label="95% bootstrap CI",
        )
        axis.plot(x, mean, color=NAVY, linewidth=2.6, label="Seed mean")
        if metric == "success":
            axis.axhline(threshold, color=RED, linestyle="--", linewidth=1.1)
            axis.set_ylim(-0.02, 1.03)
        elif metric == "return":
            minimum = min(float(individual[metric].min()), float(curve["ci95_low"].min()))
            axis.set_ylim(min(-0.05, minimum - 0.05), 1.04)
        else:
            axis.set_yscale("log")
            axis.set_ylim(1, max(220, float(individual[metric].max()) * 1.1))
        axis.set_title(SHORT_NAMES[environment], fontweight="bold")
        axis.set_xlabel("Environment steps (millions)")
        axis.set_ylabel(titles[metric][1])
        axis.grid(axis="y", color="#E2E8F0", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=8, fontsize=9)
    figure.suptitle(titles[metric][0], fontsize=18, fontweight="bold", y=0.99)
    figure.text(
        0.5,
        0.04,
        "Colored lines are independent seeds. Shading separates replicate spread (SD) "
        "from uncertainty in the seed mean (bootstrap CI).",
        ha="center",
        color=GRAY,
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.10, 1, 0.95))
    save(figure, output_dir, f"seed_{metric}_learning_curves")


def plot_maze_divergence(records: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> None:
    environment = "MiniHack-MazeWalk-9x9-v0"
    threshold = float(config["training"]["minimum_success"])
    seeds = [int(seed) for seed in config["seeds"]]
    maze = records[records["environment"] == environment]
    figure, axis = plt.subplots(figsize=(11.4, 5.8))
    for color, seed in zip(SEED_COLORS, seeds, strict=True):
        rows = maze[maze["seed"] == seed].sort_values("checkpoint")
        axis.plot(
            rows["checkpoint"] / 1_000_000,
            rows["success"] * 100,
            color=color,
            marker="o",
            markersize=4,
            linewidth=2.2,
            label=f"Seed {seed}: final {rows.iloc[-1]['success']:.1%}",
        )
    axis.axhline(threshold * 100, color=RED, linestyle="--", linewidth=1.5)
    axis.text(2.0, threshold * 100 + 2, f"competence = {threshold:.0%}", ha="right", color=RED)
    axis.set_xlim(0.1, 2.0)
    axis.set_ylim(0, 104)
    axis.set_xlabel("Environment steps (millions)")
    axis.set_ylabel("Success over 200 fixed episodes (%)")
    axis.set_title(
        "MazeWalk bifurcates: two seeds learn late; three remain below criterion",
        fontsize=17,
        fontweight="bold",
    )
    axis.grid(axis="y", color="#E2E8F0")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", ncol=2)
    figure.tight_layout()
    save(figure, output_dir, "maze_seed_divergence")


def plot_final_efficiency(
    records: pd.DataFrame,
    curves: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    maximum = int(config["training"]["max_environment_steps"])
    environments = list(config["environments"])
    seeds = [int(seed) for seed in config["seeds"]]
    final = records[records["checkpoint"] == maximum]
    summary = curves[(curves["checkpoint"] == maximum) & (curves["metric"] == "length")]
    figure, axis = plt.subplots(figsize=(12.0, 5.7))
    offsets = np.linspace(-0.16, 0.16, len(seeds))
    for index, environment in enumerate(environments):
        values = final[final["environment"] == environment].sort_values("seed")
        axis.scatter(
            index + offsets,
            values["length"],
            color=[SEED_COLORS[seeds.index(int(seed))] for seed in values["seed"]],
            s=48,
            zorder=4,
        )
        row = summary[summary["environment"] == environment].iloc[0]
        mean = float(row["mean"])
        axis.errorbar(
            index,
            mean,
            yerr=[[mean - float(row["ci95_low"])], [float(row["ci95_high"]) - mean]],
            color=NAVY,
            marker="D",
            markersize=7,
            capsize=5,
            linewidth=2,
            zorder=5,
        )
    axis.set_yscale("log")
    axis.set_xticks(
        range(len(environments)), [SHORT_NAMES[environment] for environment in environments]
    )
    axis.set_ylabel("Mean actions per episode (log scale)")
    axis.set_title(
        "Final policies are efficient except for failed MazeWalk runs",
        fontsize=17,
        fontweight="bold",
    )
    axis.grid(axis="y", color="#E2E8F0", which="both")
    axis.spines[["top", "right"]].set_visible(False)
    figure.text(
        0.5,
        0.01,
        "Colored points are independent seeds; diamonds are seed means and bars are "
        "95% bootstrap intervals.",
        ha="center",
        color=GRAY,
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    save(figure, output_dir, "final_episode_efficiency")


def write_tables(
    records: pd.DataFrame,
    curves: pd.DataFrame,
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records.to_csv(output_dir / "seed_checkpoint_results.csv", index=False)
    curves.to_csv(output_dir / "aggregate_learning_curves.csv", index=False)
    maximum = int(config["training"]["max_environment_steps"])
    threshold = float(config["training"]["minimum_success"])
    final = records[records["checkpoint"] == maximum].copy()
    final.to_csv(output_dir / "final_seed_results.csv", index=False)
    final_summary = curves[curves["checkpoint"] == maximum].to_dict(orient="records")
    non_maze = final[final["environment"] != "MiniHack-MazeWalk-9x9-v0"]
    maze = final[final["environment"] == "MiniHack-MazeWalk-9x9-v0"]
    payload = {
        "source": str(Path(config["artifact_root"]) / "evaluations"),
        "training_seeds": [int(seed) for seed in config["seeds"]],
        "environments": list(config["environments"]),
        "checkpoints_per_policy": int(maximum / config["training"]["evaluation_interval"]),
        "checkpoint_evaluations": len(records),
        "episodes_per_evaluation": int(config["training"]["evaluation_episodes"]),
        "evaluated_episodes": int(len(records) * config["training"]["evaluation_episodes"]),
        "final_policies": len(final),
        "competence_threshold": threshold,
        "competent_final_policies": int((final["success"] >= threshold).sum()),
        "complete_competent_task_cohorts": int(
            final.groupby("environment", observed=True)["success"]
            .apply(lambda values: bool((values >= threshold).all()))
            .sum()
        ),
        "non_maze_final_successes": round(non_maze["success"].sum() * 200),
        "non_maze_final_episodes": int(len(non_maze) * 200),
        "maze_final_successes": round(maze["success"].sum() * 200),
        "maze_final_episodes": int(len(maze) * 200),
        "final_environment_summary": final_summary,
    }
    (output_dir / "confirmatory_result_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parser().parse_args()
    configure_style()
    config = load_config(args.config)
    records = load_records(config)
    curves = aggregate(records, config, args.bootstrap_seed)
    plot_final_success(records, curves, config, args.output_dir)
    plot_competence_matrix(records, config, args.output_dir)
    for metric in ("success", "return", "length"):
        plot_metric_curves(records, curves, config, args.output_dir, metric)
    plot_maze_divergence(records, config, args.output_dir)
    plot_final_efficiency(records, curves, config, args.output_dir)
    write_tables(records, curves, config, args.output_dir)
    print(
        f"Wrote 7 figures (SVG+PNG) and result tables from {len(records)} fixed evaluations "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
