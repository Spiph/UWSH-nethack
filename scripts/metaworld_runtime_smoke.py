"""Verify the locked Meta-World training/checkpoint/evaluation path on this machine."""

from __future__ import annotations

import argparse
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC

from .metaworld_geometry import hash_actor, verify_policy_symmetry
from .metaworld_hour_pilot import TASKS, atomic_json, make_env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output directory must be new or empty")
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    torch.set_num_threads(4)
    seed = 20261015
    checks = []
    for index, task in enumerate(TASKS):
        env = make_env(task, seed + index)
        try:
            observation, _ = env.reset(seed=seed + index)
            step = env.step(env.action_space.sample())
            if observation.shape != (39,) or env.action_space.shape != (4,):
                raise ValueError("unexpected observation/action interface")
            if not np.isfinite(step[0]).all() or not np.isfinite(step[1]):
                raise ValueError("nonfinite reset/step result")
            checks.append({"task": task, "reset_step_passed": True})
        finally:
            env.close()
    env = make_env(TASKS[0], seed)
    try:
        model = SAC(
            "MlpPolicy",
            env,
            seed=seed,
            device=args.device,
            policy_kwargs={"net_arch": [256, 256]},
            buffer_size=1000,
            learning_starts=64,
            batch_size=64,
            verbose=0,
        )
        symmetry_error = verify_policy_symmetry(model.policy.actor, seed + 1)
        initial_hash = hash_actor(model.policy.actor)
        model.learn(total_timesteps=256)
        final_hash = hash_actor(model.policy.actor)
        if initial_hash == final_hash or model.num_timesteps != 256:
            raise ValueError("training did not complete actor updates")
        checkpoint = args.output / "smoke_policy.zip"
        model.save(checkpoint)
        observation, _ = env.reset(seed=seed + 1000)
        expected, _ = model.predict(observation, deterministic=True)
        restored = SAC.load(checkpoint, env=env, device=args.device)
        actual, _ = restored.predict(observation, deterministic=True)
        if not np.array_equal(expected, actual):
            raise ValueError("checkpoint restored different deterministic actions")
        total_return = 0.0
        success = 0.0
        length = 0
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = restored.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            success = max(success, float(info.get("success", 0.0)))
            length += 1
        report = {
            "validation_passed": True,
            "purpose": "runtime smoke only; excluded from competence cohorts",
            "versions": {
                name: version(name)
                for name in (
                    "torch",
                    "numpy",
                    "gymnasium",
                    "metaworld",
                    "mujoco",
                    "stable-baselines3",
                )
            },
            "python": platform.python_version(),
            "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "task_checks": checks,
            "training_steps": model.num_timesteps,
            "initial_actor_sha256": initial_hash,
            "final_actor_sha256": final_hash,
            "permutation_max_abs_error": symmetry_error,
            "checkpoint_actions_exact": True,
            "fresh_episode": {"return": total_return, "success": success, "length": length},
            "elapsed_seconds": time.monotonic() - started,
        }
        atomic_json(args.output / "runtime_smoke.json", report)
        print(json.dumps(report, indent=2))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
