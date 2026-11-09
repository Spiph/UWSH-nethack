"""Run a small, time-bounded Meta-World task competence screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
import traceback
from collections.abc import Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, cast

import numpy as np

from scripts.metaworld_deadlines import BudgetExpired, RunDeadline, guard_model, parse_deadline

TASKS = ("reach-v3", "push-v3", "pick-place-v3")
BASE_SEED = 20261015


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-utc", type=parse_deadline, required=True)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--steps-per-policy", type=int, default=25_000)
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--reserve-seconds", type=int, default=300)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=BASE_SEED)
    args = parser.parse_args(argv)
    if not 1 <= args.seconds <= 3600:
        parser.error("--seconds must be between 1 and 3600")
    if args.steps_per_policy <= 0 or args.evaluation_episodes <= 0:
        parser.error("steps and evaluation episodes must be positive")
    if not 0 < args.reserve_seconds < args.seconds:
        parser.error("--reserve-seconds must be positive and less than --seconds")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory; existing runs are preserved")
    return args


def finite_real(value: Any) -> bool:
    """Return whether a non-boolean real scalar is finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return False
    return bool(np.isfinite(value))


def summarize_competence(manifest: dict[str, Any]) -> dict[str, Any]:
    """Validate exact training/evaluation coverage and apply the 80% criterion."""
    tasks = manifest.get("tasks", list(TASKS))
    requested_steps = manifest.get("steps_per_policy_requested")
    requested_episodes = manifest.get("evaluation_episodes_requested")

    def positive_int(value: Any) -> bool:
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    metadata_valid = (
        tasks == list(TASKS) and positive_int(requested_steps) and positive_int(requested_episodes)
    )
    threshold_episodes = cast(int, requested_episodes) if positive_int(requested_episodes) else 0
    threshold = int(np.ceil(0.8 * threshold_episodes)) if threshold_episodes else None
    runs = manifest.get("task_runs", [])
    eval_rows = manifest.get("evaluations", [])
    rows_valid = isinstance(runs, list) and isinstance(eval_rows, list)
    if not rows_valid:
        runs = runs if isinstance(runs, list) else []
        eval_rows = eval_rows if isinstance(eval_rows, list) else []
    rows_valid = rows_valid and all(
        isinstance(row, dict) and row.get("task") in TASKS for row in [*runs, *eval_rows]
    )
    clean_runs = [row for row in runs if isinstance(row, dict)]
    clean_evals = [row for row in eval_rows if isinstance(row, dict)]
    episode_ids_valid = all(
        isinstance(row.get("episode"), int) and not isinstance(row.get("episode"), bool)
        for row in clean_evals
    )
    identities = [
        (row.get("task"), row.get("episode"))
        for row in clean_evals
        if isinstance(row.get("episode"), int) and not isinstance(row.get("episode"), bool)
    ]
    no_duplicate_rows = len(identities) == len(set(identities))
    results: dict[str, Any] = {}
    for task in TASKS:
        task_runs = [r for r in clean_runs if r.get("task") == task]
        run = task_runs[0] if len(task_runs) == 1 else None
        rows = [r for r in clean_evals if r.get("task") == task]
        episode_ids = [r.get("episode") for r in rows if isinstance(r.get("episode"), int)]
        unique_ids = set(episode_ids)
        complete_rows = [r for r in rows if r.get("complete_episode") is True]
        success_count = sum(
            r.get("success") is True or r.get("success") == 1 for r in complete_rows
        )
        valid_rows = all(
            finite_real(r.get("return"))
            and (isinstance(r.get("success"), bool) or r.get("success") in (0, 1))
            for r in rows
        )
        exact_eval = (
            metadata_valid
            and rows_valid
            and no_duplicate_rows
            and episode_ids_valid
            and valid_rows
            and requested_episodes is not None
            and len(rows) == requested_episodes
            and len(unique_ids) == requested_episodes
            and unique_ids == set(range(requested_episodes))
            and len(complete_rows) == requested_episodes
        )
        exact_train = (
            run is not None
            and requested_steps is not None
            and positive_int(run.get("steps_requested"))
            and positive_int(run.get("environment_steps_completed"))
            and run.get("steps_requested") == requested_steps
            and run.get("environment_steps_completed") == requested_steps
        )
        passed = bool(
            exact_train and exact_eval and threshold is not None and success_count >= threshold
        )
        results[task] = {
            "training_complete": bool(exact_train),
            "training_runs_found": len(task_runs),
            "evaluation_rows": len(rows),
            "complete_evaluation_episodes": len(complete_rows),
            "successes": success_count,
            "success_threshold": threshold,
            "evaluation_complete": bool(exact_eval),
            "pass": passed,
        }
    hashes = [r.get("initial_actor_sha256") for r in clean_runs]
    shared_hash = bool(
        len(runs) == len(TASKS)
        and all(isinstance(value, str) and value for value in hashes)
        and len(set(hashes)) == 1
    )
    complete = (
        metadata_valid
        and rows_valid
        and no_duplicate_rows
        and episode_ids_valid
        and len(runs) == len(TASKS)
        and all(results[task]["training_complete"] for task in TASKS)
        and all(results[task]["evaluation_complete"] for task in TASKS)
    )
    return {
        "metadata_valid": bool(metadata_valid and rows_valid and no_duplicate_rows),
        "per_task": results,
        "shared_initial_actor_hash_valid": shared_hash,
        "all_tasks_complete": bool(complete),
        "all_tasks_pass": bool(complete and shared_hash and all(results[t]["pass"] for t in TASKS)),
    }


def _atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def _save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _runtime_package_hashes() -> dict[str, str | None]:
    """Hash the loaded package entry-point files, alongside their versions."""
    result: dict[str, str | None] = {}
    for package, module_name in (
        ("gymnasium", "gymnasium"),
        ("mujoco", "mujoco"),
        ("stable_baselines3", "stable_baselines3"),
        ("metaworld", "metaworld"),
    ):
        try:
            module = importlib.import_module(module_name)
            source = Path(module.__file__) if module.__file__ else None
            result[package] = _sha256(source) if source is not None and source.is_file() else None
        except (ImportError, OSError):
            result[package] = None
    return result


def _run(args: argparse.Namespace, *, budget: RunDeadline | None = None) -> int:
    # Start and persist clocks before importing or initializing the RL runtime.
    budget = budget or RunDeadline(args.seconds, args.deadline_utc)
    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "tasks": list(TASKS),
        "steps_per_policy_requested": args.steps_per_policy,
        "evaluation_episodes_requested": args.evaluation_episodes,
        "runner_seconds": args.seconds,
        "task_runs": [],
        "evaluations": [],
        "errors": [],
    }
    path = output / "competence_manifest.json"
    budget.record(manifest)
    _atomic_json(path, manifest)
    try:
        budget.require("runtime_initialization")
        _execute(args, budget, manifest)
    except BudgetExpired:
        manifest["status"] = "STOPPED_AT_BUDGET"
    except Exception as error:
        manifest["status"] = "ERROR"
        manifest["errors"].append(f"{type(error).__name__}: {error}")
        manifest["traceback"] = traceback.format_exc()
        raise
    finally:
        budget.record(manifest)
        summary = summarize_competence(manifest)
        manifest["competence"] = summary
        manifest["incomplete_tasks"] = [
            task
            for task, row in summary["per_task"].items()
            if not row["training_complete"] or not row["evaluation_complete"]
        ]
        manifest["missing_evaluation_episodes"] = {
            task: [
                episode
                for episode in range(args.evaluation_episodes)
                if not any(
                    row.get("task") == task
                    and row.get("episode") == episode
                    and row.get("complete_episode") is True
                    for row in manifest["evaluations"]
                )
            ]
            for task in TASKS
        }
        budget.record(manifest)
        manifest["validation_complete"] = bool(
            summary["all_tasks_complete"]
            and summary["shared_initial_actor_hash_valid"]
            and manifest["budget_compliant"]
            and manifest["status"] in ("COMPLETE_PASS", "COMPLETE_FAIL")
        )
        _atomic_json(path, manifest)
    print(json.dumps(manifest["competence"], indent=2))
    return 0 if manifest.get("status") == "COMPLETE_PASS" else 2


def _execute(args: argparse.Namespace, budget: RunDeadline, manifest: dict[str, Any]) -> None:
    import torch

    sac_module = importlib.import_module("stable_baselines3")
    SAC = sac_module.SAC

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    torch.set_num_threads(4)

    # Runtime-only dependencies: importing this module for tests needs NumPy only.
    from scripts.metaworld_geometry import actor_vector, hash_actor

    pilot = importlib.import_module("scripts.metaworld_hour_pilot")
    DeadlineCallback = pilot.DeadlineCallback
    make_env = pilot.make_env

    output: Path = args.output
    budget.require("runtime_ready")
    manifest.update(
        {
            "status": "RUNNING",
            "purpose": "small-run Meta-World competence screen",
            "mode": "smoke"
            if args.steps_per_policy == 128 and args.evaluation_episodes == 1
            else "competence_screen",
            "runner_sha256": _sha256(Path(__file__)),
            "measurement_code_sha256": {
                name: _sha256(Path(__file__).with_name(name))
                for name in (
                    "metaworld_geometry.py",
                    "metaworld_hour_pilot.py",
                    "metaworld_deadlines.py",
                )
            },
            "software": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": torch.__version__,
                "gymnasium": _package_version("gymnasium"),
                "mujoco": _package_version("mujoco"),
                "stable_baselines3": _package_version("stable-baselines3"),
                "metaworld": _package_version("metaworld"),
                "package_entrypoint_sha256": _runtime_package_hashes(),
            },
            "hardware": {
                "platform": platform.platform(),
                "device": device,
                "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            },
            "seed": args.seed,
            "seed_convention": {
                "reference_actor_torch": "seed",
                "training_environment_and_sac": "seed + 1000 + task_index*100",
                "evaluation_episode_reset": "seed + 100000 + task_index*1000 + episode",
                "critic_and_environment_randomness": (
                    "task-specific; actor state copied after SAC construction"
                ),
            },
            "tasks": list(TASKS),
            "algorithm": {
                "policy": "MlpPolicy",
                "net_arch": [256, 256],
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "tau": 0.005,
                "learning_starts": min(1000, args.steps_per_policy),
                "buffer_size": 50_000,
                "batch_size": 256,
                "train_freq": 1,
                "gradient_steps": 1,
                "ent_coef": "auto",
            },
            "environment": {
                "observation_dimensions": 39,
                "action_dimensions": 4,
                "max_episode_steps": 500,
                "visible_goal_indices": [36, 37, 38],
                "task_reward": "Meta-World v3 task reward",
                "observation_adapter": "goal bounds metadata corrected; observations unchanged",
            },
            "steps_per_policy_requested": args.steps_per_policy,
            "evaluation_episodes_requested": args.evaluation_episodes,
            "success_threshold": int(np.ceil(0.8 * args.evaluation_episodes)),
            "reserve_seconds": args.reserve_seconds,
            "runner_seconds": args.seconds,
            "task_runs": [],
            "evaluations": [],
            "errors": [],
        }
    )
    manifest_path = output / "competence_manifest.json"
    _atomic_json(manifest_path, manifest)
    reference_env = env = evaluation_env = None
    try:
        budget.require("reference_environment_initialization")
        reference_env = make_env(TASKS[0], args.seed + 1, budget=budget)
        budget.require("reference_model_initialization")
        torch.manual_seed(args.seed)
        reference = SAC(
            "MlpPolicy",
            reference_env,
            seed=args.seed,
            device=device,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=0,
            buffer_size=50_000,
            learning_starts=min(1000, args.steps_per_policy),
            learning_rate=3e-4,
            gamma=0.99,
            tau=0.005,
            ent_coef="auto",
            batch_size=256,
            train_freq=1,
            gradient_steps=1,
        )
        budget.require("reference_model_ready")
        shared_actor_state = {
            k: v.detach().clone() for k, v in reference.policy.actor.state_dict().items()
        }
        initial_vector, initial_layout = actor_vector(reference.policy.actor)
        initial_actor_sha256 = hash_actor(reference.policy.actor)
        initial_actor_path = output / "shared_initial_actor.npz"
        np.savez_compressed(
            initial_actor_path,
            vector=initial_vector,
            layout_json=np.asarray(json.dumps(initial_layout)),
        )
        manifest["shared_initial_actor"] = {
            "sha256": initial_actor_sha256,
            "artifact": str(initial_actor_path),
            "layout": initial_layout,
        }
        _atomic_json(manifest_path, manifest)
        del reference
        reference_env.close()
        reference_env = None

        policy_dir = output / "policies"
        policy_dir.mkdir(exist_ok=True)
        for task_index, task in enumerate(TASKS):
            remaining = len(TASKS) - task_index
            reserve = args.reserve_seconds * remaining / len(TASKS)
            if not budget.allowed("training_initialization", reserve_seconds=reserve):
                manifest["task_runs"].append({"task": task, "status": "SKIPPED_TRAINING_BUDGET"})
                continue
            env_seed = args.seed + 1000 + task_index * 100
            torch.manual_seed(env_seed)
            np.random.seed(env_seed)
            env = make_env(task, env_seed, budget=budget)
            budget.require("source_model_initialization")
            model = SAC(
                "MlpPolicy",
                env,
                seed=env_seed,
                device=device,
                policy_kwargs={"net_arch": [256, 256]},
                verbose=0,
                buffer_size=50_000,
                learning_starts=min(1000, args.steps_per_policy),
                learning_rate=3e-4,
                gamma=0.99,
                tau=0.005,
                ent_coef="auto",
                batch_size=256,
                train_freq=1,
                gradient_steps=1,
            )
            model.policy.actor.load_state_dict(shared_actor_state, strict=True)
            initial_hash = hash_actor(model.policy.actor)
            callback = DeadlineCallback(budget, reserve_seconds=reserve)
            before_steps = int(model.num_timesteps)
            train_started = budget.monotonic()
            try:
                with guard_model(model, budget):
                    if budget.allowed("training_start", reserve_seconds=reserve):
                        model.learn(args.steps_per_policy, callback=callback, progress_bar=False)
            except BudgetExpired:
                pass  # Retain partial training before handling the global cutoff.
            budget.allowed("training_return")
            steps = int(model.num_timesteps - before_steps)
            train_duration = budget.monotonic() - train_started
            model_path = policy_dir / task
            model.save(str(model_path))
            vector, layout = actor_vector(model.policy.actor)
            actor_path = policy_dir / f"{task}.actor.npz"
            np.savez_compressed(
                actor_path, vector=vector, layout_json=np.asarray(json.dumps(layout))
            )
            record = {
                "task": task,
                "status": "TRAINED" if steps == args.steps_per_policy else "TRAINING_INCOMPLETE",
                "environment_seed": env_seed,
                "training_seed": env_seed,
                "initialization_seed": args.seed,
                "initial_actor_sha256": initial_hash,
                "final_actor_sha256": hash_actor(model.policy.actor),
                "steps_requested": args.steps_per_policy,
                "environment_steps_completed": steps,
                "duration_seconds": train_duration,
                "steps_per_second": steps / max(train_duration, 1e-9),
                "model_file": str(model_path.with_suffix(".zip")),
                "actor_file": str(actor_path),
                "actor_layout": layout,
                "stopped_for_budget": callback.stopped_for_budget
                or budget.cutoff_reason is not None,
                "training_cutoff_reason": callback.cutoff_reason or budget.cutoff_reason,
            }
            manifest["task_runs"].append(record)
            _atomic_json(manifest_path, manifest)
            print(
                f"trained {task}: {steps}/{args.steps_per_policy} steps, "
                f"{record['steps_per_second']:.1f} steps/s",
                flush=True,
            )

            rows: list[dict[str, Any]] = []
            budget.require("evaluation_initialization")
            evaluation_env = make_env(task, args.seed + 90000 + task_index, budget=budget)
            try:
                for episode in range(args.evaluation_episodes):
                    if not budget.allowed("evaluation_episode"):
                        break
                    budget.require("evaluation_reset")
                    try:
                        obs, _ = evaluation_env.reset(
                            seed=args.seed + 100000 + task_index * 1000 + episode
                        )
                    except BudgetExpired:
                        break
                    total, max_success, length = 0.0, 0.0, 0
                    terminated = truncated = False
                    while not (terminated or truncated):
                        if not budget.allowed("evaluation_episode"):
                            break
                        action, _ = model.predict(obs, deterministic=True)
                        if not budget.allowed("evaluation_step"):
                            break
                        try:
                            obs, reward, terminated, truncated, info = evaluation_env.step(action)
                        except BudgetExpired:
                            break
                        total += float(reward)
                        max_success = max(max_success, float(info.get("success", 0.0)))
                        length += 1
                    budget.allowed("evaluation_return")
                    rows.append(
                        {
                            "task": task,
                            "episode": episode,
                            "return": total,
                            "success": bool(max_success > 0),
                            "success_value": max_success,
                            "length": length,
                            "complete_episode": bool(terminated or truncated),
                            "cutoff_reason": budget.cutoff_reason,
                        }
                    )
                    manifest["evaluations"].extend(rows[-1:])
                    _save_rows(output / "evaluations.csv", manifest["evaluations"])
                    _atomic_json(manifest_path, manifest)
                    print(
                        f"evaluated {task}: episode {episode + 1}/{args.evaluation_episodes}, "
                        f"return={total:.3f}, success={rows[-1]['success']}",
                        flush=True,
                    )
                    if not budget.allowed("evaluation_episode"):
                        break
            finally:
                evaluation_env.close()
                env.close()
                evaluation_env = env = None
            del model
        summary = summarize_competence(manifest)
        manifest["competence"] = summary
        budget.record(manifest)
        if not budget.allowed("completion"):
            manifest["status"] = "STOPPED_AT_BUDGET"
        else:
            manifest["status"] = (
                "COMPLETE_PASS"
                if summary["all_tasks_pass"]
                else ("COMPLETE_FAIL" if summary["all_tasks_complete"] else "INCOMPLETE")
            )
    finally:
        for resource in (reference_env, evaluation_env, env):
            if resource is not None:
                resource.close()
        budget.record(manifest)
        _atomic_json(manifest_path, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    return _run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
