# mypy: ignore-errors
"""Fixed-seed MiniHack evaluator for one pinned SOL Sample Factory checkpoint."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from ups.artifacts import atomic_json, sha256_file
from ups.sol import make_phase0_env, register_phase0_components


def success_from_info(info: dict[str, Any]) -> bool:
    """Use MiniHack's explicit task-success status; positive reward is not enough."""
    status = info.get("end_status")
    return getattr(status, "name", str(status)) == "TASK_SUCCESSFUL"


def _jsonable(value: Any) -> Any:
    """Convert Gym/Numpy observations into a deterministic JSON value."""
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _policy_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """Retain the exact observation fields consumed by the registered encoder."""
    required = ("glyphs_crop", "blstats")
    missing = [key for key in required if key not in observation]
    if missing:
        raise ValueError(f"policy observation is missing fields: {missing}")
    return {key: _jsonable(observation[key]) for key in required}


def _nethack_markers(observation: dict[str, Any]) -> tuple[int, int]:
    """Return native NLE score and dungeon depth from a NetHack observation."""
    from nle import nethack

    blstats = observation["blstats"]
    return int(blstats[nethack.NLE_BL_SCORE]), int(blstats[nethack.NLE_BL_DEPTH])


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--episodes", type=int, required=True)
    command.add_argument("--eval-seed", type=int, required=True)
    command.add_argument("--artifact-root", type=Path, required=True)
    command.add_argument("--environment", required=True)
    command.add_argument("--experiment", required=True)
    command.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional diagnostic cap; omit for preregistered scientific evaluation.",
    )
    return command


def checkpoint_config_path(checkpoint: Path) -> Path:
    """Return the Sample Factory config beside a checkpoint directory."""
    return checkpoint.parent.parent / "config.json"


def _load_policy(checkpoint: Path, environment: str) -> tuple[Any, Any, Any]:
    from sample_factory.model.actor_critic import create_actor_critic
    from sf_examples.nethack.train_nethack import parse_nethack_args

    # Sample Factory stores config.json beside checkpoint_p0, i.e. two parents
    # above the checkpoint file (checkpoint_p0 -> experiment).
    config_path = checkpoint_config_path(checkpoint)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    sf_config = parse_nethack_args(shlex.split(config["command_line"]))
    env = make_phase0_env(environment, sf_config, {})
    model = create_actor_critic(sf_config, env.observation_space, env.action_space)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(payload["model"])
    model.eval()
    return model, env, sf_config


def evaluate(args: argparse.Namespace) -> Path:
    register_phase0_components()
    model, env, _ = _load_policy(args.checkpoint, args.environment)
    rows: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        seed = args.eval_seed + episode
        observation, _ = env.reset(seed=seed)
        nethack_score, nethack_depth = (
            _nethack_markers(observation) if args.environment == "NetHackScore-v0" else (None, None)
        )
        max_nethack_depth = nethack_depth
        state = torch.zeros(1, 256)
        total_return = 0.0
        steps = 0
        trajectory: list[dict[str, Any]] = []
        terminated = truncated = False
        while not terminated and not truncated:
            tensors = {
                key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()
            }
            with torch.no_grad():
                output = model(tensors, state)
            action = int(output["action_logits"].argmax(dim=-1).item())
            state = output["new_rnn_states"]
            current_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)
            if args.environment == "NetHackScore-v0":
                nethack_score, nethack_depth = _nethack_markers(observation)
                max_nethack_depth = max(int(max_nethack_depth), nethack_depth)
            trajectory.append(
                {
                    "observation": _policy_observation(current_observation),
                    "action": action,
                    "reward": float(reward),
                    "next_observation": _policy_observation(observation),
                    "info": _jsonable(info),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                }
            )
            total_return += float(reward)
            steps += 1
            if args.max_episode_steps is not None and steps >= args.max_episode_steps:
                truncated = True
        row: dict[str, Any] = {
            "environment": args.environment,
            "experiment": args.experiment,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "episode": episode,
            "seed": seed,
            "success": success_from_info(info),
            "return": total_return,
            "steps": steps,
            "terminated": terminated,
            "truncated": truncated,
            # Keep every transition, including observations, actions,
            # rewards, and termination flags; this is the scientific raw
            # trajectory, not merely a per-episode summary.
            "trajectory": json.dumps(trajectory, sort_keys=True, separators=(",", ":")),
        }
        if args.environment == "NetHackScore-v0":
            row.update(
                {
                    "final_nethack_score": nethack_score,
                    "max_dungeon_depth": max_nethack_depth,
                }
            )
        rows.append(row)
    env.close()
    output_root = args.artifact_root / "evaluations"
    output_root.mkdir(parents=True, exist_ok=True)
    table_path = output_root / f"{args.experiment}.parquet"
    table = pd.DataFrame(rows)
    table.to_parquet(table_path, index=False)
    summary_path = output_root / f"{args.experiment}.json"
    atomic_json(
        summary_path,
        {
            "environment": args.environment,
            "experiment": args.experiment,
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "episodes": args.episodes,
            "eval_seed": args.eval_seed,
            "action_selection": "greedy_argmax",
            "max_episode_steps": args.max_episode_steps,
            "success_rate": float(table["success"].mean()),
            "mean_return": float(table["return"].mean()),
            "median_return": float(table["return"].median()),
            "mean_length": float(table["steps"].mean()),
            "table": str(table_path),
            **(
                {
                    "mean_final_nethack_score": float(table["final_nethack_score"].mean()),
                    "mean_max_dungeon_depth": float(table["max_dungeon_depth"].mean()),
                }
                if args.environment == "NetHackScore-v0"
                else {}
            ),
        },
    )
    return summary_path


if __name__ == "__main__":  # pragma: no cover
    sys.exit(0 if evaluate(parser().parse_args()) else 1)
