"""Read-only, stdlib/NumPy diagnosis of a completed source-development library.

No native learner/environment is imported or constructed. Restricted unpickling
is for these hash-verified local artifacts, not a general untrusted-file sandbox.
All summaries are descriptive; checkpoints and episodes are not independent
policy-library replicates, and these statistics do not establish causality.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

NATIVE_MODULES = ("torch", "stable_baselines3", "gymnasium", "metaworld", "mujoco")
NUMPY_GLOBALS = frozenset(
    {
        ("numpy", "dtype"),
        ("numpy", "ndarray"),
        ("numpy.core.numeric", "_frombuffer"),
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy.core.multiarray", "scalar"),
        ("numpy.random._pickle", "__generator_ctor"),
        ("numpy.random._pickle", "__bit_generator_ctor"),
    }
)


class Record:
    """Data-only replacement; no native ReplayBuffer/Box constructor runs."""


class ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in {
            ("stable_baselines3.common.buffers", "ReplayBuffer"),
            ("gymnasium.spaces.box", "Box"),
        }:
            return Record
        if (module, name) == ("torch", "device"):
            return str
        if (module, name) in NUMPY_GLOBALS:
            return getattr(importlib.import_module(module), name)
        raise ValueError(f"unsupported pickle global {module}.{name}")

    def persistent_load(self, pid: Any) -> Any:
        raise ValueError("persistent/native pickle references are forbidden")


def load_replay(handle: BinaryIO) -> Any:
    return ArrayOnlyUnpickler(handle).load()


def assert_no_native_imports() -> None:
    loaded = [name for name in sys.modules if name.split(".", 1)[0] in NATIVE_MODULES]
    if loaded:
        raise ValueError("offline diagnosis requires a fresh process without the native RL stack")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def quantiles(values: Any) -> dict[str, float] | None:
    array = np.asarray(values)
    if not array.size:
        return None
    if not np.isfinite(array).all():
        raise ValueError("nonfinite values cannot be summarized as valid diagnostics")
    return {
        "min": float(array.min()),
        "q10": float(np.quantile(array, 0.1)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.9)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def _correlation(left: Any, right: Any) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def replay_summary(replay: Any) -> dict[str, Any]:
    size, position = replay.buffer_size, replay.pos
    if (
        type(size) is not int
        or size <= 0
        or type(position) is not int
        or not 0 <= position < size
        or type(replay.full) is not bool
        or replay.n_envs != 1
        or replay.optimize_memory_usage
        or replay.handle_timeout_termination is not True
    ):
        raise ValueError("require the saved one-env, separate-next-observation replay ring")
    count = size if replay.full else position
    if count == 0:
        raise ValueError("source replay contains no committed transitions")
    indices = (
        np.concatenate((np.arange(position, size), np.arange(position)))
        if replay.full
        else np.arange(count)
    )
    arrays = {}
    shapes = {
        "observations": (size, 1, 39),
        "next_observations": (size, 1, 39),
        "actions": (size, 1, 4),
        "rewards": (size, 1),
        "dones": (size, 1),
        "timeouts": (size, 1),
    }
    for name, shape in shapes.items():
        array = getattr(replay, name)
        if not isinstance(array, np.ndarray) or array.shape != shape or array.dtype.hasobject:
            raise ValueError(f"invalid replay array {name}")
        arrays[name] = array[indices, 0]
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"nonfinite replay array {name}")
    obs, nxt, actions, rewards = (
        arrays[name] for name in ("observations", "next_observations", "actions", "rewards")
    )
    dones, timeouts = arrays["dones"], arrays["timeouts"]
    if (
        not np.isin(dones, [0, 1]).all()
        or not np.isin(timeouts, [0, 1]).all()
        or np.any(timeouts > dones)
    ):
        raise ValueError("invalid native done/timeout masks")
    adjacent = dones[:-1] == 0
    reset_rows = obs[1:][dones[:-1] == 1]
    return {
        "scope": "retained replay window only; overlapping windows are not summed as costs",
        "samples": count,
        "buffer_size": size,
        "pos": position,
        "full": replay.full,
        "chronological_order": "ring oldest to newest",
        "finite": True,
        "rewards": quantiles(rewards),
        "reward_at_least_1": int(np.sum(rewards >= 1)),
        "reward_at_least_5": int(np.sum(rewards >= 5)),
        "reward_at_least_9_999": int(np.sum(rewards >= 9.999)),
        "raw_dones": int(dones.sum()),
        "timeouts": int(timeouts.sum()),
        "true_dones": int(np.sum(dones * (1 - timeouts))),
        "unique_goal_values": len(np.unique(obs[:, 36:39], axis=0)),
        "next_previous_frame_matches_current": bool(np.array_equal(nxt[:, 18:36], obs[:, :18])),
        "goal_unchanged_within_transitions": bool(np.array_equal(nxt[:, 36:39], obs[:, 36:39])),
        "adjacent_nonterminal_pairs": int(adjacent.sum()),
        "adjacent_nonterminal_next_observation_matches": bool(
            np.array_equal(nxt[:-1][adjacent], obs[1:][adjacent])
        ),
        "reset_observations_checked": len(reset_rows),
        "reset_previous_frame_matches_current": bool(
            np.array_equal(reset_rows[:, 18:36], reset_rows[:, :18])
        ),
        "observation_coordinate_min": obs.min(axis=0).tolist(),
        "observation_coordinate_max": obs.max(axis=0).tolist(),
        "hand_z": quantiles(obs[:, 2]),
        "object_z": quantiles(obs[:, 6]),
        "gripper_separation": quantiles(obs[:, 3]),
        "hand_object_distance_proxy_not_tcp": quantiles(
            np.linalg.norm(obs[:, :3] - obs[:, 4:7], axis=1)
        ),
        "object_goal_distance": quantiles(np.linalg.norm(obs[:, 4:7] - obs[:, 36:39], axis=1)),
        "action_mean": actions.mean(axis=0).tolist(),
        "action_std": actions.std(axis=0).tolist(),
        "action_min": actions.min(axis=0).tolist(),
        "action_max": actions.max(axis=0).tolist(),
        "action_saturated_fraction": (np.abs(actions) > 0.99).mean(axis=0).tolist(),
        "action_coordinates": (
            "saved normalized replay actions; not assumed bit-identical to physical actions"
        ),
        "hand_xyz_delta_action_correlations": [
            _correlation(actions[:, i], nxt[:, i] - obs[:, i]) for i in range(3)
        ],
        "gripper_delta_action_correlation": _correlation(actions[:, 3], nxt[:, 3] - obs[:, 3]),
    }


def source_history_summary(history: list[dict[str, Any]], goal_ids: set[str]) -> dict[str, Any]:
    full, partial, zero = [], [], []
    blocks: dict[int, dict[str, int]] = {}
    steps = 0
    for index, row in enumerate(history):
        length = row.get("length")
        if (
            row.get("episode") != index
            or type(length) is not int
            or not 0 <= length <= 500
            or row.get("goal_id") not in goal_ids
        ):
            raise ValueError("source history has invalid episode/goal/length metadata")
        complete = row.get("complete_episode") is True
        if complete and (
            length != 500 or row.get("terminated") is not False or row.get("truncated") is not True
        ):
            raise ValueError("complete source episodes must retain the 500-step timeout semantics")
        if complete:
            full.append(row)
            block = steps // 100000
            entry = blocks.setdefault(block, {"complete_episodes": 0, "successes": 0})
            entry["complete_episodes"] += 1
            entry["successes"] += int(bool(row.get("success")))
        elif length:
            partial.append(row)
        else:
            zero.append(row)
        steps += length
    return {
        "scope": (
            "full saved source history; partial/zero-length rows excluded from success denominator"
        ),
        "physical_steps": steps,
        "complete_episodes": len(full),
        "successful_complete_episodes": sum(bool(row.get("success")) for row in full),
        "partial_episodes": partial,
        "zero_length_rows": len(zero),
        "complete_episode_goal_coverage": len({row["goal_id"] for row in full}),
        "declared_source_goals": len(goal_ids),
        "blocks_100k": [
            {"block_start_step": key * 100000, **value} for key, value in sorted(blocks.items())
        ],
    }


def _artifact(root: Path, record: dict[str, Any]) -> Path:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts or root.is_symlink():
        raise ValueError("artifact path escapes its declared root")
    path = root / relative
    if (
        path.is_symlink()
        or not path.resolve().is_relative_to(root.resolve())
        or digest(path) != record["sha256"]
    ):
        raise ValueError("artifact path/hash mismatch")
    return path


def input_roots(manifest_path: Path, manifest: dict[str, Any]) -> tuple[Path, list[Path]]:
    parent = Path(manifest["plan"]["extension"]["parent_root"])
    roots = [Path(row["root"]) for row in manifest["parent_binding"]["inventories"]]
    roots.append(manifest_path.parent)
    if parent not in roots or any(not root.is_absolute() or root.is_symlink() for root in roots):
        raise ValueError("manifest does not bind the inherited artifact roots")
    return parent, roots


def progress_summary(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text())
    tasks = {}
    for task in sorted({row["task"] for row in rows}):
        selected = [row for row in rows if row["task"] == task]
        keys = sorted({key for row in selected for key in row.get("diagnostics", {})})
        tasks[task] = {
            "rows": len(selected),
            "first_steps": selected[0]["training_steps"],
            "last_steps": selected[-1]["training_steps"],
            "diagnostics": {
                key: quantiles(
                    [
                        row["diagnostics"][key]
                        for row in selected
                        if key in row.get("diagnostics", {})
                    ]
                )
                for key in keys
            },
        }
    return {
        "path": str(path),
        "sha256": digest(path),
        "tasks": tasks,
        "scope": "logged snapshots, not independent training observations",
    }


def diagnose(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("phase") != "FINISHED" or manifest.get("status") not in {
        "COMPLETE_PASS",
        "COMPLETE_FAIL",
        "INCOMPLETE",
        "ERROR",
    }:
        raise ValueError("diagnosis requires a terminal FINISHED source extension manifest")
    parent, roots = input_roots(manifest_path, manifest)
    output: dict[str, Any] = {
        "purpose": (
            "descriptive offline source diagnosis; no causal conclusion or independent replication"
        ),
        "manifest": {"path": str(manifest_path), "sha256": digest(manifest_path)},
        "input_artifact_roots": [str(root) for root in roots],
        "ledger": manifest["ledger"],
        "source_histories": {},
        "development_evaluations": [],
        "checkpoints": [],
        "partial_checkpoints": [],
        "progress": [],
        "actor_forward_performed": False,
    }
    for task, state in sorted(manifest["tasks"].items()):
        banks = {bank["split"]: bank for bank in manifest["goal_manifests"][task]["banks"]}
        goal_ids = {goal["goal_id"] for goal in banks["source"]["goals"]}
        checkpoints = state["checkpoints"]
        output["source_histories"][task] = source_history_summary(
            checkpoints[-1]["artifacts"]["environment"]["episode_history"], goal_ids
        )
        for checkpoint in checkpoints:
            root = Path(checkpoint.get("artifact_root", str(parent)))
            if root not in roots:
                raise ValueError("checkpoint names an unbound artifact root")
            artifacts = checkpoint["artifacts"]
            replay_path = _artifact(root, artifacts[".replay.pkl"])
            actor_path = _artifact(root, artifacts["actor"])
            with replay_path.open("rb") as handle:
                replay = load_replay(handle)
            row = {
                "task": task,
                "steps": checkpoint["actual_steps"],
                "target_steps": checkpoint["target_steps"],
                "partial": bool(checkpoint.get("partial")),
                "artifact_root": str(root),
                "replay_sha256": artifacts[".replay.pkl"]["sha256"],
                "actor_sha256": artifacts["actor"]["sha256"],
                "actor_path": str(actor_path),
                **replay_summary(replay),
            }
            output["checkpoints"].append(row)
            if row["partial"]:
                output["partial_checkpoints"].append(
                    {key: row[key] for key in ("task", "steps", "target_steps", "artifact_root")}
                )
            if checkpoint.get("evaluation_complete"):
                episodes = checkpoint["evaluation"]
                if len(episodes) != 20 or any(
                    not episode.get("complete_episode")
                    or episode.get("length") != 500
                    or episode.get("terminated") is not False
                    or episode.get("truncated") is not True
                    for episode in episodes
                ):
                    raise ValueError("complete development evaluation lacks twenty full episodes")
                output["development_evaluations"].append(
                    {
                        "task": task,
                        "steps": checkpoint["actual_steps"],
                        "episodes": len(episodes),
                        "successes": sum(bool(episode["success"]) for episode in episodes),
                        "returns": quantiles([episode["return"] for episode in episodes]),
                        "scope": "same frozen development goals reused across checkpoints",
                    }
                )
    for root in (parent, manifest_path.parent):
        path = root / "progress.json"
        if path.is_file():
            output["progress"].append(progress_summary(path))
    return output


def main(argv: list[str] | None = None) -> int:
    assert_no_native_imports()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.manifest.is_symlink():
        raise ValueError("input manifest must not be a symlink")
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    _, roots = input_roots(manifest_path, manifest)
    destination = args.output.resolve()
    if destination.exists() or any(
        destination.is_relative_to(root.resolve()) or root.resolve().is_relative_to(destination)
        for root in roots
    ):
        raise ValueError("output must be a new directory outside all input artifact roots")
    report = diagnose(manifest_path)
    assert_no_native_imports()
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "source_diagnosis.json").open("x") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "report": str(destination / "source_diagnosis.json"),
                "checkpoints": len(report["checkpoints"]),
                "native_modules_imported": [],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
