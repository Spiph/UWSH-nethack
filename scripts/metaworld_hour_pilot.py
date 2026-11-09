"""Run a strictly time bounded, exploratory Meta World policy population pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium as gym
import metaworld
import numpy as np
import torch
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

if __package__:
    from .metaworld_deadlines import (
        BudgetExpired,
        RunDeadline,
        TrainingDeadline,
        guard_env,
        guard_model,
        parse_deadline,
    )
    from .metaworld_geometry import (
        actor_vector,
        fit_centered_basis,
        hash_actor,
        load_actor_vector,
        project_oracle,
        random_orthonormal_basis,
        verify_policy_symmetry,
    )
    from .metaworld_pilot_report import summarize_manifest
else:
    from metaworld_deadlines import (
        BudgetExpired,
        RunDeadline,
        TrainingDeadline,
        guard_env,
        guard_model,
        parse_deadline,
    )
    from metaworld_geometry import (
        actor_vector,
        fit_centered_basis,
        hash_actor,
        load_actor_vector,
        project_oracle,
        random_orthonormal_basis,
        verify_policy_symmetry,
    )
    from metaworld_pilot_report import summarize_manifest

TASKS = ("reach-v3", "push-v3", "pick-place-v3")
REPLICATES = 3
CONDITIONS = ("shared_init", "independent_init")
BASE_SEED = 20261001


class CorrectedGoalBounds(ObservationWrapper):
    """Preserve Meta-World observations while declaring its visible goal bounds."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        original = env.observation_space
        if not isinstance(original, Box) or original.shape != (39,):
            raise ValueError(f"expected Meta-World 39-vector observation, got {original}")
        low = original.low.copy()
        high = original.high.copy()
        # Meta-World 3.1.1 currently declares the observed goal coordinates [36:39]
        # as [0, 0]. Widen only this metadata; observations are not transformed.
        low[36:39] = -np.inf
        high[36:39] = np.inf
        self.observation_space = Box(low=low, high=high, dtype=original.dtype)
        self.original_goal_bounds = {
            "low": original.low[36:39].astype(float).tolist(),
            "high": original.high[36:39].astype(float).tolist(),
        }

    def observation(self, observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(observation, dtype=self.observation_space.dtype)
        if obs.shape != (39,) or not np.isfinite(obs).all():
            raise ValueError("Meta-World returned a malformed or nonfinite observation")
        if np.any(obs[:36] < self.observation_space.low[:36]) or np.any(
            obs[:36] > self.observation_space.high[:36]
        ):
            raise ValueError("observation violates a non-goal Meta-World space bound")
        return obs


def make_env(task: str, seed: int, *, budget: RunDeadline | None = None) -> gym.Env:
    if task not in TASKS:
        raise ValueError(f"unsupported task: {task}")
    env_id = "Meta-World/MT1"
    if env_id not in gym.registry:
        metaworld.register_mw_envs()
    if budget is not None:
        budget.require("environment_initialization")
    env = gym.make(env_id, env_name=task, disable_env_checker=True)
    wrapped = CorrectedGoalBounds(env)
    if budget is not None:
        guard_env(wrapped, budget)
    try:
        wrapped.reset(seed=seed)
    except BaseException:
        wrapped.close()
        raise
    return wrapped


class DeadlineCallback(BaseCallback, TrainingDeadline):
    def __init__(self, budget: RunDeadline, *, reserve_seconds: float = 0):
        BaseCallback.__init__(self, verbose=0)
        TrainingDeadline.__init__(self, budget, reserve_seconds=reserve_seconds)

    def _on_step(self) -> bool:
        return TrainingDeadline.on_step(self)


def atomic_json(path: Path, payload: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp.replace(path)


def save_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def run_id(condition: str, task: str, replicate: int) -> str:
    return f"{condition}__{task}__lib{replicate}"


def eval_model(
    model: SAC,
    task: str,
    replicate: int,
    condition: str,
    policy_id: str,
    kind: str,
    episode_count: int,
    budget: RunDeadline,
) -> list[dict[str, Any]]:
    budget.require("evaluation_initialization")
    env = make_env(
        task,
        BASE_SEED + 700_000 + replicate * 10_000 + TASKS.index(task) * 100,
        budget=budget,
    )
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(episode_count):
            if not budget.allowed("evaluation_episode"):
                break
            try:
                observation, _ = env.reset(seed=BASE_SEED + 800_000 + replicate * 10_000 + episode)
            except BudgetExpired:
                break
            total_return = 0.0
            max_success = 0.0
            length = 0
            terminated = truncated = False
            while not (terminated or truncated):
                if not budget.allowed("evaluation_episode"):
                    break
                action, _ = model.predict(observation, deterministic=True)
                if not budget.allowed("evaluation_step"):
                    break
                try:
                    observation, reward, terminated, truncated, info = env.step(action)
                except BudgetExpired:
                    break
                total_return += float(reward)
                max_success = max(max_success, float(info.get("success", 0.0)))
                length += 1
            budget.allowed("evaluation_return")
            if length:
                rows.append(
                    {
                        "policy_id": policy_id,
                        "condition": condition,
                        "replicate": replicate,
                        "training_task": next((t for t in TASKS if f"__{t}__" in policy_id), ""),
                        "evaluation_task": task,
                        "policy_kind": kind,
                        "episode": episode,
                        "return": total_return,
                        "success": max_success,
                        "length": length,
                        "complete_episode": bool(terminated or truncated),
                        "cutoff_reason": budget.cutoff_reason,
                    }
                )
            if not budget.allowed("evaluation_episode"):
                break
    finally:
        env.close()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deadline-utc", type=parse_deadline, required=True)
    parser.add_argument("--seconds", type=int, default=2700)
    parser.add_argument("--steps-per-policy", type=int, default=25_000)
    parser.add_argument("--evaluation-episodes", type=int, default=3)
    parser.add_argument("--reserve-seconds", type=int, default=600)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.seconds > 3600 or args.seconds <= 0:
        parser.error("--seconds must be between 1 and the user's 3600 second limit")
    if not 0 < args.reserve_seconds < args.seconds or args.steps_per_policy <= 0:
        parser.error("need positive training steps and a nonzero evaluation reserve")
    if args.evaluation_episodes <= 0:
        parser.error("--evaluation-episodes must be positive")
    if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
        parser.error("--output must be a new or empty directory; existing runs are preserved")
    budget = RunDeadline(args.seconds, args.deadline_utc)
    args.output.mkdir(parents=True, exist_ok=True)
    device = args.device
    cuda_available = False

    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "purpose": "one-hour exploratory Meta-World feasibility pilot",
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "measurement_code_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "metaworld_geometry.py",
                "metaworld_pilot_report.py",
                "metaworld_deadlines.py",
            )
        },
        "started_at_utc": budget.started_utc.isoformat(),
        "absolute_deadline_utc": budget.deadline_utc.isoformat(),
        "user_wall_budget_seconds": 3600,
        "runner_budget_seconds": args.seconds,
        "evaluation_reserve_seconds": args.reserve_seconds,
        "steps_per_source_policy_requested": args.steps_per_policy,
        "evaluation_episodes_requested": args.evaluation_episodes,
        "tasks": list(TASKS),
        "replicates": REPLICATES,
        "conditions": list(CONDITIONS),
        "planned_source_runs": len(TASKS) * REPLICATES * len(CONDITIONS),
        "hardware": {},
        "software": {},
        "timing_scope": (
            "Runner duration starts after legacy module imports. Absolute UTC is checked "
            "before thread/CUDA/environment/model initialization; imports precede that check."
        ),
        "observation_adapter": {
            "shape": [39],
            "action_shape": [4],
            "unchanged_observation_values": True,
            "corrected_goal_bounds_indices": [36, 37, 38],
            "original_goal_bounds": {"low": [0.0, 0.0, 0.0], "high": [0.0, 0.0, 0.0]},
        },
        "seed_convention": {
            "base": BASE_SEED,
            "training_environment": (
                "base + 100000 + replicate*1000 + task_index*10 + condition_index"
            ),
            "initial_actor_shared": "base + 10000 + replicate*100 + condition_index",
            "initial_actor_independent": (
                "base + 10000 + replicate*100 + task_index*10 + condition_index"
            ),
        },
        "permutation_control_max_abs_error": None,
        "source_runs": [],
        "cross_task_evaluations": [],
        "oracle_projection_diagnostics": [],
        "errors": [],
    }
    budget.record(manifest)
    atomic_json(args.output / "pilot_manifest.json", manifest)
    source_models: dict[tuple[str, int, str], dict[str, Any]] = {}
    all_eval: list[dict[str, Any]] = []
    active_env = None
    try:
        budget.require("runtime_initialization")
        torch.set_num_threads(4)
        budget.require("cuda_availability")
        cuda_available = torch.cuda.is_available()
        if device == "auto":
            device = "cuda" if cuda_available else "cpu"
        if device == "cuda" and not cuda_available:
            raise ValueError("CUDA requested but unavailable")
        budget.require("hardware_metadata")
        manifest["hardware"] = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "torch_cuda": cuda_available,
            "device": device,
            "gpu": torch.cuda.get_device_name(0) if cuda_available else None,
        }
        budget.require("runtime_metadata")
        manifest["software"] = {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "gymnasium": gym.__version__,
            "mujoco": __import__("mujoco").__version__,
            "stable_baselines3": __import__("stable_baselines3").__version__,
            "metaworld_release": version("metaworld"),
        }
        budget.require("preflight_initialization")
        # Reset/step preflight and actual architecture symmetry control.
        active_env = preflight_env = make_env(TASKS[0], BASE_SEED, budget=budget)
        budget.require("preflight_reset")
        observation, _ = preflight_env.reset(seed=BASE_SEED)
        if observation.shape != (39,) or preflight_env.action_space.shape != (4,):
            raise ValueError("unexpected task interface")
        budget.require("preflight_step")
        step = preflight_env.step(preflight_env.action_space.sample())
        if not np.isfinite(step[0]).all():
            raise ValueError("nonfinite preflight observation")
        budget.require("preflight_model_initialization")
        probe = SAC(
            "MlpPolicy",
            preflight_env,
            seed=BASE_SEED,
            device=device,
            policy_kwargs={"net_arch": [256, 256]},
            verbose=0,
            buffer_size=1000,
            batch_size=64,
            learning_starts=64,
        )
        budget.require("preflight_symmetry")
        manifest["permutation_control_max_abs_error"] = verify_policy_symmetry(
            probe.policy.actor, BASE_SEED + 1
        )
        del probe
        preflight_env.close()
        active_env = None
        atomic_json(args.output / "pilot_manifest.json", manifest)

        total_index = 0
        for replicate in range(REPLICATES):
            shared_actor_state: dict[str, torch.Tensor] | None = None
            for condition_index, condition in enumerate(CONDITIONS):
                for task_index, task in enumerate(TASKS):
                    if not budget.allowed(
                        "training_initialization", reserve_seconds=args.reserve_seconds
                    ):
                        break
                    env_seed = (
                        BASE_SEED + 100_000 + replicate * 1_000 + task_index * 10 + condition_index
                    )
                    init_seed = BASE_SEED + 10_000 + replicate * 100 + condition_index
                    if condition == "independent_init":
                        init_seed += task_index * 10
                    torch.manual_seed(init_seed)
                    np.random.seed(env_seed)
                    active_env = env = make_env(task, env_seed, budget=budget)
                    budget.require("source_model_initialization")
                    model = SAC(
                        "MlpPolicy",
                        env,
                        seed=env_seed,
                        device=device,
                        policy_kwargs={"net_arch": [256, 256]},
                        learning_rate=3e-4,
                        buffer_size=50_000,
                        learning_starts=1_000,
                        batch_size=256,
                        train_freq=1,
                        gradient_steps=1,
                        verbose=0,
                    )
                    # SAC's constructor seeds its actor before learning. Reseed
                    # environment/training randomness only after initialization,
                    # and retain the distinct environment seed for vector resets.
                    model.set_random_seed(env_seed)
                    model.seed = env_seed
                    if condition == "shared_init":
                        if shared_actor_state is None:
                            shared_actor_state = {
                                key: value.detach().clone()
                                for key, value in model.policy.actor.state_dict().items()
                            }
                        else:
                            model.policy.actor.load_state_dict(shared_actor_state)
                    initial_hash = hash_actor(model.policy.actor)
                    if (
                        condition == "shared_init"
                        and len(
                            [
                                r
                                for r in manifest["source_runs"]
                                if r["condition"] == condition and r["replicate"] == replicate
                            ]
                        )
                        > 0
                    ):
                        expected = next(
                            r["initial_actor_sha256"]
                            for r in manifest["source_runs"]
                            if r["condition"] == condition and r["replicate"] == replicate
                        )
                        if initial_hash != expected:
                            raise ValueError("shared actor initialization hash mismatch")

                    callback = DeadlineCallback(budget, reserve_seconds=args.reserve_seconds)
                    job_started = budget.monotonic()
                    guard_model(model, budget)
                    try:
                        if budget.allowed("training_start", reserve_seconds=args.reserve_seconds):
                            model.learn(
                                args.steps_per_policy, callback=callback, progress_bar=False
                            )
                    except BudgetExpired:
                        pass  # Save the actual partial checkpoint before stopping.
                    budget.allowed("training_return")
                    elapsed = budget.monotonic() - job_started
                    ident = run_id(condition, task, replicate)
                    job_dir = args.output / "policies"
                    job_dir.mkdir(exist_ok=True)
                    model_path = job_dir / ident
                    model.save(str(model_path))
                    final_vector, layout = actor_vector(model.policy.actor)
                    np.savez_compressed(
                        job_dir / f"{ident}.actor.npz",
                        vector=final_vector,
                        layout_json=np.asarray(json.dumps(layout)),
                    )
                    record = {
                        "policy_id": ident,
                        "condition": condition,
                        "task": task,
                        "replicate": replicate,
                        "environment_seed": env_seed,
                        "initialization_seed": init_seed,
                        "initial_actor_sha256": initial_hash,
                        "final_actor_sha256": hash_actor(model.policy.actor),
                        "environment_steps_completed": int(model.num_timesteps),
                        "steps_requested": args.steps_per_policy,
                        "duration_seconds": elapsed,
                        "steps_per_second": model.num_timesteps / max(elapsed, 1e-9),
                        "stopped_for_wall_budget": (
                            callback.stopped_for_budget or budget.cutoff_reason is not None
                        ),
                        "training_cutoff_reason": callback.cutoff_reason or budget.cutoff_reason,
                        "model_file": str(model_path.with_suffix(".zip")),
                        "actor_file": str(job_dir / f"{ident}.actor.npz"),
                        "actor_layout": layout,
                    }
                    source_models[(condition, replicate, task)] = {
                        "path": str(model_path.with_suffix(".zip")),
                        "vector": final_vector,
                        "layout": layout,
                    }
                    manifest["source_runs"].append(record)
                    manifest["status"] = "TRAINING"
                    atomic_json(args.output / "pilot_manifest.json", manifest)
                    env.close()
                    active_env = None
                    del model
                    total_index += 1
                    steps_completed = record["environment_steps_completed"]
                    steps_per_second = record["steps_per_second"]
                    print(
                        f"trained {total_index}/18 {ident}: {steps_completed} steps, "
                        f"{steps_per_second:.1f} steps/s",
                        flush=True,
                    )
                    if callback.stopped_for_budget:
                        break
                if not budget.allowed(
                    "training_initialization", reserve_seconds=args.reserve_seconds
                ):
                    break

        manifest["training_complete"] = len(manifest["source_runs"]) == manifest[
            "planned_source_runs"
        ] and all(
            r["environment_steps_completed"] == args.steps_per_policy
            for r in manifest["source_runs"]
        )
        manifest["training_cut_short"] = not manifest["training_complete"]
        manifest["training_stopped_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json(args.output / "pilot_manifest.json", manifest)

        # Every completed source policy is evaluated on each task with new reset seeds.
        for (condition, replicate, training_task), item in source_models.items():
            for evaluation_task in TASKS:
                if not budget.allowed("evaluation_episode"):
                    break
                active_env = env = make_env(evaluation_task, BASE_SEED + 900_000, budget=budget)
                budget.require("evaluation_model_load")
                model = SAC.load(item["path"], env=env, device=device)
                rows = eval_model(
                    model,
                    evaluation_task,
                    replicate,
                    condition,
                    run_id(condition, training_task, replicate),
                    "trained_policy",
                    args.evaluation_episodes,
                    budget,
                )
                all_eval.extend(rows)
                env.close()
                active_env = None
                del model
                manifest["cross_task_evaluations"] = all_eval
                save_rows(args.output / "cross_task_evaluations.csv", all_eval)
                atomic_json(args.output / "pilot_manifest.json", manifest)

        # Fold-local rank-1 oracle projection and equal-rank Gaussian random controls.
        for condition in CONDITIONS:
            for replicate in range(REPLICATES):
                for target_task in TASKS:
                    if not budget.allowed("evaluation_episode"):
                        break
                    sources = [
                        source_models.get((condition, replicate, task))
                        for task in TASKS
                        if task != target_task
                    ]
                    if any(item is None for item in sources):
                        continue
                    source_vectors = np.stack(
                        [item["vector"] for item in sources if item is not None]
                    )
                    center, learned, actual_rank = fit_centered_basis(source_vectors, rank=1)
                    target = source_models.get((condition, replicate, target_task))
                    if target is None:
                        continue
                    target_vector = target["vector"]
                    random_basis = random_orthonormal_basis(
                        target_vector.size,
                        rank=1,
                        seed=BASE_SEED + 500_000 + replicate * 100 + TASKS.index(target_task),
                    )
                    learned_vector = project_oracle(target_vector, center, learned)
                    random_vector = project_oracle(target_vector, center, random_basis)
                    base_error = float(np.linalg.norm(target_vector - center))
                    learned_error = float(np.linalg.norm(target_vector - learned_vector))
                    random_error = float(np.linalg.norm(target_vector - random_vector))
                    diag = {
                        "condition": condition,
                        "replicate": replicate,
                        "held_out_task": target_task,
                        "source_tasks": [task for task in TASKS if task != target_task],
                        "n_source_policies": int(source_vectors.shape[0]),
                        "max_centered_rank": int(source_vectors.shape[0] - 1),
                        "actual_rank": actual_rank,
                        "target_weights_used_for_coefficients": True,
                        "interpretation": (
                            "oracle reconstruction diagnostic only; not target adaptation"
                        ),
                        "parameter_count": int(target_vector.size),
                        "center_error": base_error,
                        "learned_rank_1_error": learned_error,
                        "random_rank_1_error": random_error,
                        "learned_relative_residual": learned_error / max(base_error, 1e-12),
                        "random_relative_residual": random_error / max(base_error, 1e-12),
                    }
                    for kind, vector in (
                        ("oracle_learned_rank1", learned_vector),
                        ("oracle_random_rank1", random_vector),
                    ):
                        if not budget.allowed("evaluation_episode"):
                            break
                        active_env = env = make_env(target_task, BASE_SEED + 950_000, budget=budget)
                        budget.require("oracle_model_load")
                        model = SAC.load(target["path"], env=env, device=device)
                        load_actor_vector(model.policy.actor, vector, target["layout"])
                        rows = eval_model(
                            model,
                            target_task,
                            replicate,
                            condition,
                            run_id(condition, target_task, replicate),
                            kind,
                            args.evaluation_episodes,
                            budget,
                        )
                        all_eval.extend(rows)
                        env.close()
                        active_env = None
                        del model
                    manifest["oracle_projection_diagnostics"].append(diag)
                    manifest["cross_task_evaluations"] = all_eval
                    save_rows(args.output / "cross_task_evaluations.csv", all_eval)
                    atomic_json(args.output / "pilot_manifest.json", manifest)

        budget.record(manifest)
        report = summarize_manifest(manifest)
        manifest["coverage_validation"] = report["coverage"]
        manifest["initialization_validation"] = report["initialization"]
        manifest["evaluation_complete"] = report["coverage"]["complete"]
        manifest["validation_complete"] = (
            manifest["training_complete"]
            and manifest["evaluation_complete"]
            and report["initialization"]["valid"]
        )
        if not budget.allowed("completion"):
            manifest["status"] = "STOPPED_AT_BUDGET"
        elif manifest["validation_complete"]:
            manifest["status"] = "COMPLETE_EXPLORATORY"
        else:
            manifest["status"] = "INCOMPLETE_EXPLORATORY"
    except BudgetExpired:
        manifest["status"] = "STOPPED_AT_BUDGET"
    except Exception as error:
        manifest["status"] = "ERROR"
        manifest["error"] = f"{type(error).__name__}: {error}"
        manifest["traceback"] = traceback.format_exc()
        budget.record(manifest)
        atomic_json(args.output / "pilot_manifest.json", manifest)
        raise
    finally:
        if active_env is not None:
            active_env.close()
        budget.record(manifest)
        report = summarize_manifest(manifest)
        manifest["coverage_validation"] = report["coverage"]
        manifest["initialization_validation"] = report["initialization"]
        manifest["training_complete"] = len(manifest["source_runs"]) == manifest[
            "planned_source_runs"
        ] and all(
            r["environment_steps_completed"] == args.steps_per_policy
            for r in manifest["source_runs"]
        )
        manifest["training_cut_short"] = not manifest["training_complete"]
        manifest["evaluation_complete"] = report["coverage"]["complete"]
        manifest["validation_complete"] = bool(
            manifest["training_complete"]
            and manifest["evaluation_complete"]
            and report["initialization"]["valid"]
            and manifest["budget_compliant"]
            and manifest["status"] == "COMPLETE_EXPLORATORY"
        )
        # Preserve all actual and missing source/evaluation slots, even on late start.
        actual = {r["policy_id"] for r in manifest["source_runs"]}
        manifest["incomplete_source_runs"] = [
            run_id(condition, task, replicate)
            for condition in CONDITIONS
            for replicate in range(REPLICATES)
            for task in TASKS
            if run_id(condition, task, replicate) not in actual
            or any(
                r["policy_id"] == run_id(condition, task, replicate)
                and r["environment_steps_completed"] != args.steps_per_policy
                for r in manifest["source_runs"]
            )
        ]
        budget.record(manifest)
        atomic_json(args.output / "pilot_manifest.json", manifest)
        if cuda_available:
            torch.cuda.empty_cache()
        budget.record(manifest)
        atomic_json(args.output / "pilot_manifest.json", manifest)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "status",
                    "elapsed_seconds",
                    "planned_source_runs",
                    "training_complete",
                    "evaluation_complete",
                    "permutation_control_max_abs_error",
                )
            },
            indent=2,
        )
    )
    return 0 if manifest["status"] == "COMPLETE_EXPLORATORY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
