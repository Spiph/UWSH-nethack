"""Build presentation figures directly from recorded experiment artifacts."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
CONFIRMATORY_ARTIFACTS = ARTIFACTS / "phase0-confirmatory-r2"
OUTPUT = ROOT / "docs" / "progress-assets"
BLUE = "#2563EB"
ORANGE = "#EA580C"
GREEN = "#059669"
PURPLE = "#7C3AED"
GRAY = "#64748B"


def set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#0F172A",
            "axes.titlecolor": "#0F172A",
            "axes.titlesize": 17,
            "axes.titleweight": "bold",
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "grid.color": "#E2E8F0",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.color": "#475569",
            "ytick.color": "#475569",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUTPUT / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    rate = successes / trials
    denominator = 1 + z**2 / trials
    center = (rate + z**2 / (2 * trials)) / denominator
    margin = z * math.sqrt(rate * (1 - rate) / trials + z**2 / (4 * trials**2)) / denominator
    return center - margin, min(1.0, center + margin)


def current_evaluations() -> list[dict[str, object]]:
    root = ARTIFACTS / "phase0" / "evaluations"
    pattern = re.compile(r"seed(?P<seed>\d+)-step(?P<target>\d+)\.json$")
    records: list[dict[str, object]] = []
    for summary_path in sorted(root.glob("*seed*-step*.json")):
        match = pattern.search(summary_path.name)
        if match is None:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        table = pd.read_parquet(summary_path.with_suffix(".parquet"))
        records.append(
            {
                "seed": int(match.group("seed")),
                "target": int(match.group("target")),
                "actual": int(re.search(r"_(\d+)\.pth$", summary["checkpoint"]).group(1)),
                "successes": int(table["success"].sum()),
                "episodes": len(table),
                "success_rate": float(table["success"].mean()),
                "mean_steps": float(table["steps"].mean()),
                "median_steps": float(table["steps"].median()),
                "p95_steps": float(table["steps"].quantile(0.95)),
                "max_steps": int(table["steps"].max()),
                "table": table,
            }
        )
    return records


def scalar_series(experiment: str, tag: str) -> tuple[np.ndarray, np.ndarray]:
    summary = ARTIFACTS / "phase0" / "sample_factory" / experiment / ".summary"
    points: dict[int, tuple[float, float]] = {}
    for event_path in sorted(summary.glob("*/*tfevents*")):
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        if tag not in accumulator.Tags()["scalars"]:
            continue
        for event in accumulator.Scalars(tag):
            previous = points.get(event.step)
            if previous is None or event.wall_time >= previous[0]:
                points[event.step] = (event.wall_time, event.value)
    ordered = sorted((step, value) for step, (_, value) in points.items())
    return np.array([item[0] for item in ordered]), np.array([item[1] for item in ordered])


def plot_training_dynamics() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    series = [
        ("phase0-minihack-room-random-5x5-seed0", "Seed 0", BLUE),
        ("phase0-minihack-room-random-5x5-seed1", "Seed 1", ORANGE),
    ]
    for experiment, label, color in series:
        steps, reward = scalar_series(experiment, "reward/reward")
        axes[0].plot(
            steps / 1000, reward, marker="o", markersize=4, linewidth=2.2, label=label, color=color
        )
        steps, length = scalar_series(experiment, "len/len")
        axes[1].plot(
            steps / 1000, length, marker="o", markersize=4, linewidth=2.2, label=label, color=color
        )
    axes[0].axhline(
        0.75, color=GRAY, linestyle="--", linewidth=1.4, label="75% competence criterion"
    )
    axes[0].set(
        title="Training reward approaches ceiling",
        xlabel="Environment steps (thousands)",
        ylabel="Rolling mean episode reward",
        ylim=(0, 1.05),
    )
    axes[1].set(
        title="Policies learn much shorter routes",
        xlabel="Environment steps (thousands)",
        ylabel="Rolling mean episode length",
        ylim=(0, 50),
    )
    for axis in axes:
        axis.grid(True, axis="y")
        axis.legend(loc="best")
    fig.suptitle(
        "Two independent initializations learn the Random Room task",
        fontsize=20,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.04,
        "Training diagnostics are rolling averages over 100 episodes; "
        "fixed evaluation results are shown separately.",
        ha="center",
        color=GRAY,
    )
    fig.tight_layout()
    save(fig, "training_dynamics")


def plot_fixed_evaluation(records: list[dict[str, object]]) -> None:
    labels = [f"Seed {r['seed']}\n{int(r['target']) // 1000}k steps" for r in records]
    values = np.array([float(r["success_rate"]) for r in records])
    intervals = [wilson(int(r["successes"]), int(r["episodes"])) for r in records]
    lower = values - np.array([interval[0] for interval in intervals])
    upper = np.array([interval[1] for interval in intervals]) - values
    colors = [BLUE if int(r["seed"]) == 0 else ORANGE for r in records]
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    x = np.arange(len(records))
    axis.bar(x, values * 100, color=colors, width=0.58, alpha=0.9)
    axis.errorbar(
        x,
        values * 100,
        yerr=np.vstack([lower, upper]) * 100,
        fmt="none",
        ecolor="#0F172A",
        capsize=6,
        linewidth=1.5,
    )
    for index, record in enumerate(records):
        axis.text(
            index,
            float(record["success_rate"]) * 100 + 0.25,
            f"{record['successes']}/{record['episodes']}",
            ha="center",
            fontweight="bold",
            color="#0F172A",
        )
    axis.axhline(75, color=GRAY, linestyle="--", linewidth=1.3, label="75% competence criterion")
    axis.set(
        title="Independent policies reach 99.5-100% fixed-evaluation success",
        ylabel="Success over 200 held-out episodes (%)",
        xticks=x,
        xticklabels=labels,
        ylim=(70, 102),
    )
    axis.grid(True, axis="y")
    axis.legend(loc="lower right")
    fig.text(
        0.5,
        0.01,
        "Error bars: Wilson 95% intervals. All checkpoints evaluated on the "
        "same 200 environment seeds.",
        ha="center",
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fixed_evaluation_success")


def plot_episode_efficiency(records: list[dict[str, object]]) -> None:
    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    palette = [BLUE, GREEN, ORANGE, PURPLE]
    for color, record in zip(palette, records, strict=False):
        table = record["table"]
        values = np.sort(table["steps"].to_numpy())
        y = np.arange(1, len(values) + 1) / len(values)
        label = (
            f"Seed {record['seed']} · {int(record['target']) // 1000}k "
            f"(median {record['median_steps']:.0f})"
        )
        axis.step(values, y * 100, where="post", linewidth=2.4, color=color, label=label)
    axis.axvline(5, color=GRAY, linestyle="--", linewidth=1.2)
    axis.text(5.15, 17, "95% finish within\n5 actions at the\nstrongest checkpoints", color=GRAY)
    axis.set(
        title="Competent policies solve the task in about three actions",
        xlabel="Actions taken before episode end",
        ylabel="Episodes completed by this step (%)",
        xlim=(0, 14),
        ylim=(0, 101),
    )
    axis.grid(True)
    axis.legend(loc="lower right")
    fig.text(
        0.5,
        0.01,
        "The seed-0 100k curve ends at 99.5% in this view because its single "
        "failure lasted 100 steps.",
        ha="center",
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "episode_efficiency")


def plot_archived_stability() -> None:
    root = ARTIFACTS / "phase0-pilot-no-retention-20260729" / "evaluations"
    pattern = re.compile(r"step(?P<target>\d+)\.json$")
    records: list[tuple[int, float, float]] = []
    for summary_path in root.glob("*.json"):
        match = pattern.search(summary_path.name)
        if match is None:
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        table = pd.read_parquet(summary_path.with_suffix(".parquet"))
        records.append(
            (
                int(match.group("target")),
                float(summary["success_rate"]),
                float(table["steps"].median()),
            )
        )
    records.sort()
    steps = np.array([record[0] for record in records]) / 1_000_000
    success = np.array([record[1] for record in records]) * 100
    medians = np.array([record[2] for record in records])
    fig, axis = plt.subplots(figsize=(10.5, 5.2))
    axis.plot(
        steps, success, color=BLUE, marker="o", linewidth=2.5, label="Fixed-evaluation success"
    )
    axis.fill_between(steps, 0, success, color=BLUE, alpha=0.08)
    axis.set(
        title="Archived pilot: competence appears by 200k and remains stable through 2M steps",
        xlabel="Environment steps (millions)",
        ylabel="Success over 200 fixed episodes (%)",
        ylim=(0, 105),
        xlim=(0.05, 2.05),
    )
    axis.grid(True)
    second = axis.twinx()
    second.plot(
        steps, medians, color=ORANGE, marker="s", linewidth=2, label="Median episode length"
    )
    second.set(ylabel="Median episode length", ylim=(0, 105))
    axis.annotate(
        "42/200",
        xy=(steps[0], success[0]),
        xytext=(0.23, 42),
        arrowprops={"arrowstyle": "->", "color": GRAY},
        color="#0F172A",
        fontweight="bold",
    )
    axis.annotate(
        "3,800/3,800 successes\nfrom 200k through 2M",
        xy=(1.1, 100),
        xytext=(0.78, 72),
        arrowprops={"arrowstyle": "->", "color": GRAY},
        color="#0F172A",
        fontweight="bold",
    )
    handles1, labels1 = axis.get_legend_handles_labels()
    handles2, labels2 = second.get_legend_handles_labels()
    axis.legend(handles1 + handles2, labels1 + labels2, loc="lower right")
    fig.text(
        0.5,
        0.01,
        "Archived behavioral record: intermediate checkpoints were evaluated "
        "but not retained for later weight analysis.",
        ha="center",
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "archived_pilot_stability")


def plot_first_confirmatory_seed() -> None:
    """Plot completed fixed evaluations from the first in-progress cohort seed."""
    root = CONFIRMATORY_ARTIFACTS / "evaluations"
    pattern = re.compile(r"seed(?P<seed>\d+)-step(?P<target>\d+)\.json$")
    records: list[dict[str, float | int]] = []
    for summary_path in sorted(root.glob("phase0-minihack-room-random-5x5-seed*-step*.json")):
        match = pattern.search(summary_path.name)
        if match is None:
            continue
        table = pd.read_parquet(summary_path.with_suffix(".parquet"))
        records.append(
            {
                "seed": int(match.group("seed")),
                "target": int(match.group("target")),
                "successes": int(table["success"].sum()),
                "episodes": len(table),
                "success_rate": float(table["success"].mean()),
                "mean_steps": float(table["steps"].mean()),
                "median_steps": float(table["steps"].median()),
            }
        )
    if not records:
        fig, axis = plt.subplots(figsize=(10.5, 4.8))
        axis.set_axis_off()
        axis.text(
            0.5,
            0.62,
            "Revision-2 confirmation has not produced a fixed evaluation yet.",
            ha="center",
            va="center",
            fontsize=20,
            fontweight="bold",
            color="#0F172A",
        )
        axis.text(
            0.5,
            0.40,
            "The prior execution is retained as an excluded incident after a checkpoint-retention\n"
            "and task/seed-registry defect. Pilot results remain separate.",
            ha="center",
            va="center",
            fontsize=13,
            color=GRAY,
        )
        fig.tight_layout()
        save(fig, "confirmatory_first_seed_progress")
        return
    seed = min(int(record["seed"]) for record in records)
    records = sorted(
        (record for record in records if int(record["seed"]) == seed),
        key=lambda record: int(record["target"]),
    )
    x = np.array([float(record["target"]) for record in records]) / 1_000_000
    success = np.array([float(record["success_rate"]) for record in records])
    intervals = [wilson(int(record["successes"]), int(record["episodes"])) for record in records]
    lower = success - np.array([interval[0] for interval in intervals])
    upper = np.array([interval[1] for interval in intervals]) - success
    lengths = np.array([float(record["mean_steps"]) for record in records])
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    axes[0].errorbar(
        x,
        success * 100,
        yerr=np.vstack((lower, upper)) * 100,
        color=GREEN,
        marker="o",
        markersize=7,
        linewidth=2.4,
        capsize=5,
    )
    axes[0].set(
        title="Fixed-evaluation success",
        xlabel="Environment steps (millions)",
        ylabel="Success over 200 held-out episodes (%)",
    )
    confidence_lower = (success - lower) * 100
    confidence_upper = (success + upper) * 100
    if float(confidence_lower.min()) >= 95:
        axes[0].set_ylim(95, 100.5)
    else:
        axes[0].set_ylim(
            max(0.0, float(confidence_lower.min()) - 5),
            min(100.5, float(confidence_upper.max()) + 3),
        )
    axes[0].grid(True, axis="y")
    axes[1].plot(x, lengths, color=PURPLE, marker="o", markersize=7, linewidth=2.4)
    axes[1].set(
        title="Route efficiency",
        xlabel="Environment steps (millions)",
        ylabel="Mean actions before episode end",
        ylim=(0, max(4.0, float(lengths.max()) + 0.4)),
    )
    axes[1].grid(True, axis="y")
    fig.suptitle(
        f"Live confirmation: Random Room seed {seed}, first fixed evaluation at {x[-1]:.1f}M steps",
        fontsize=18,
        fontweight="bold",
        y=1.04,
    )
    fig.text(
        0.5,
        -0.04,
        (
            "One in-progress training seed; error bars are Wilson 95% intervals over 200 fixed "
            "episodes. The remaining four seeds are required before a population conclusion."
        ),
        ha="center",
        color=GRAY,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "confirmatory_first_seed_progress")


def write_summary(records: list[dict[str, object]]) -> None:
    serializable = [
        {key: value for key, value in record.items() if key != "table"} for record in records
    ]
    (OUTPUT / "current_results.json").write_text(
        json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    set_style()
    records = current_evaluations()
    plot_training_dynamics()
    plot_fixed_evaluation(records)
    plot_episode_efficiency(records)
    plot_archived_stability()
    plot_first_confirmatory_seed()
    write_summary(records)
    print(f"Wrote {len(records)} current evaluation records and five figures to {OUTPUT}")


if __name__ == "__main__":
    main()
