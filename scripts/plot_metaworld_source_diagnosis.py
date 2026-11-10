"""Plot descriptive saved-source diagnostics; no model or environment imports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> tuple[Any, str]:
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    manifest, manifest_hash = read_json(args.manifest)
    amendment = manifest["plan"]["extension"]
    roots = [Path(amendment[key]).resolve() for key in ("failed_root", "parent_root", "output")]
    if any(args.output_prefix.resolve().is_relative_to(root) for root in roots):
        raise ValueError("figures must remain outside immutable source-run directories")
    outputs = [args.output_prefix.with_suffix(suffix) for suffix in (".png", ".svg", ".json")]
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise FileExistsError("existing diagnosis figures are preserved")
    tasks = ("reach-v3", "push-v3", "pick-place-v3")
    data: dict[str, Any] = {
        "scope": (
            "one developmental library; descriptive, no independent replication or causal claim"
        ),
        "manifest_sha256": manifest_hash,
        "progress_file_sha256": {},
        "source_blocks": {},
        "development": {},
        "entropy": {},
    }
    progress = []
    for root in roots[1:]:
        rows, digest = read_json(root / "progress.json")
        data["progress_file_sha256"][str(root / "progress.json")] = digest
        progress.extend(rows)
    for task in tasks:
        record = manifest["tasks"][task]
        history = record["checkpoints"][-1]["artifacts"]["environment"]["episode_history"]
        complete = [row for row in history if row["complete_episode"] and row["length"] == 500]
        data["source_blocks"][task] = [
            {
                "end_steps": (start + len(complete[start : start + 200])) * 500,
                "episodes": len(complete[start : start + 200]),
                "successes": sum(row["success"] for row in complete[start : start + 200]),
            }
            for start in range(0, len(complete), 200)
        ]
        data["development"][task] = [
            {
                "steps": row["actual_steps"],
                "episodes": len(row["evaluation"]),
                "successes": sum(episode["success"] for episode in row["evaluation"]),
            }
            for row in record["checkpoints"]
            if row.get("evaluation_complete")
        ]
        data["entropy"][task] = [
            {"steps": row["training_steps"], "alpha": row["diagnostics"]["train/ent_coef"]}
            for row in progress
            if row["task"] == task and "train/ent_coef" in row.get("diagnostics", {})
        ]
    # Optional plotting dependency lives in the pinned measurement environment.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"reach-v3": "#2465A6", "push-v3": "#D66A1F", "pick-place-v3": "#398343"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for task in tasks:
        color, label = colors[task], task.removesuffix("-v3")
        blocks = data["source_blocks"][task]
        axes[0].plot(
            [row["end_steps"] / 1e6 for row in blocks],
            [100 * row["successes"] / row["episodes"] for row in blocks],
            marker="o",
            color=color,
            label=label,
        )
        checkpoints = data["development"][task]
        axes[1].plot(
            [row["steps"] / 1e6 for row in checkpoints],
            [100 * row["successes"] / row["episodes"] for row in checkpoints],
            marker="o",
            color=color,
            label=label,
        )
        alpha = data["entropy"][task]
        axes[2].plot(
            [row["steps"] / 1e6 for row in alpha],
            [row["alpha"] for row in alpha],
            color=color,
            linewidth=1,
            label=label,
        )
    axes[0].set(title="Success on source configurations", ylabel="Successful episodes (%)")
    axes[1].set(title="Fixed development evaluation", ylabel="Successful episodes (%)")
    axes[1].axhline(80, color="#555555", linewidth=1, linestyle="--")
    axes[1].text(0.97, 82, "passing gate", ha="right", fontsize=9, color="#555555")
    axes[2].set(title="Recorded SAC entropy coefficient", ylabel="Alpha (log scale)", yscale="log")
    for axis in axes[:2]:
        axis.set_ylim(-3, 103)
    for axis in axes:
        axis.set(xlabel="Cumulative source steps (millions)", xlim=(0, 1.02))
        axis.grid(alpha=0.2)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle(
        "Source-competence diagnosis: one development library\n"
        "Source blocks: 200 complete episodes; development: 20 complete episodes per checkpoint",
        fontsize=12,
    )
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[0], dpi=180)
    fig.savefig(outputs[1], metadata={"Date": None})
    plt.close(fig)
    outputs[2].write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    print("\n".join(str(path) for path in outputs))


if __name__ == "__main__":
    main()
