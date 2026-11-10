"""Descriptive fixed-goal comparison of saved scripted traces and source replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.metaworld_source_diagnosis import assert_no_native_imports, load_replay


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(obs: Any, nxt: Any, rewards: Any) -> dict[str, Any] | None:
    if len(obs) == 0:
        return None
    return {
        "rows": len(obs),
        "median_hand_object_distance_proxy": float(
            np.median(np.linalg.norm(obs[:, :3] - obs[:, 4:7], axis=1))
        ),
        "median_object_goal_distance": float(
            np.median(np.linalg.norm(obs[:, 4:7] - obs[:, 36:39], axis=1))
        ),
        "minimum_object_goal_distance": float(
            np.min(np.linalg.norm(nxt[:, 4:7] - nxt[:, 36:39], axis=1))
        ),
        "maximum_object_height": float(nxt[:, 6].max()),
        "median_gripper_opening": float(np.median(obs[:, 3])),
        "mean_reward": float(rewards.mean()),
        "maximum_reward": float(rewards.max()),
        "rewards_at_least_5": int((rewards >= 5).sum()),
    }


def main() -> None:
    assert_no_native_imports()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--sentinel-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.source_manifest.read_text())
    sentinel = json.loads(args.sentinel_manifest.read_text())
    if sentinel.get("status") != "COMPLETE" or not sentinel.get("validation_complete"):
        raise ValueError("require the completed scripted diagnostic")
    for path in (args.source_manifest, args.sentinel_manifest):
        if args.output.resolve().is_relative_to(path.resolve().parent):
            raise ValueError("comparison output must remain outside both input run directories")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("existing comparison is preserved")
    rows = []
    for episode in sentinel["episodes"]:
        checkpoint = source["tasks"][episode["task"]]["checkpoints"][-1]
        record = checkpoint["artifacts"][".replay.pkl"]
        root = Path(checkpoint.get("artifact_root", str(args.source_manifest.parent)))
        replay_path = root / record["path"]
        trace_path = args.sentinel_manifest.parent / episode["trace"]
        if digest(replay_path) != record["sha256"] or digest(trace_path) != episode["trace_sha256"]:
            raise ValueError("input replay or trace hash mismatch")
        with replay_path.open("rb") as handle:
            replay = load_replay(handle)
        count = replay.buffer_size if replay.full else replay.pos
        obs, nxt = replay.observations[:count, 0], replay.next_observations[:count, 0]
        goal = np.asarray(episode["initial"]["observation"][36:39])
        mask = np.all(np.isclose(obs[:, 36:39], goal, atol=1e-8, rtol=0), axis=1)
        trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
        if len(trace) != 500:
            raise ValueError("scripted comparison requires a complete 500-step trace")
        rows.append(
            {
                "task": episode["task"],
                "source_checkpoint_steps": checkpoint["actual_steps"],
                "source_replay_sha256": record["sha256"],
                "scripted_trace_sha256": episode["trace_sha256"],
                "goal_id": episode["initial"]["episode"]["goal_id"],
                "goal_xyz": goal.tolist(),
                "scripted_first_success_step": next(
                    (row["step"] + 1 for row in trace if row["components"]["success"]), None
                ),
                "scripted_last_step_success": bool(trace[-1]["components"]["success"]),
                "scripted": summarize(
                    np.asarray([row["observation"] for row in trace]),
                    np.asarray([row["next_observation"] for row in trace]),
                    np.asarray([row["reward"] for row in trace]),
                ),
                "retained_source": summarize(obs[mask], nxt[mask], replay.rewards[:count, 0][mask]),
            }
        )
    report = {
        "scope": (
            "descriptive within-goal comparison; source replay contains stochastic policies while "
            "learning with different episode seeds, "
            "not matched frozen-policy rollouts or a causal test"
        ),
        "source_manifest_sha256": digest(args.source_manifest),
        "sentinel_manifest_sha256": digest(args.sentinel_manifest),
        "comparisons": rows,
    }
    assert_no_native_imports()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
