"""Validate the one approved development allocation without importing RL packages."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

TASKS = ("reach-v3", "push-v3", "pick-place-v3")
SEED_FIELDS = (
    "constructor",
    "task_bank",
    "selection",
    "episode",
    "action",
    "actor",
    "critic",
    "training",
)
APPROVAL_URL = "https://github.com/Spiph/UWSH-nethack/issues/37#issuecomment-6744072078"
RESOLUTION_URL = "https://github.com/Spiph/UWSH-nethack/issues/37#issuecomment-6744153267"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "experiments/metaworld/development_2026-11-09.json"
)


def derive_seed(seed: int, label: str) -> int:
    """The stdlib equivalent of metaworld_protocol.derive_seed, without NumPy."""
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:4], "big")


def _task_record(task: str, index: int) -> dict[str, Any]:
    source_bank, development_bank = 2026111201 + index, 2026111401 + index
    training, development_episode = 2026111101 + index, 2026111501 + index
    source = {
        "constructor": derive_seed(source_bank, "constructor"),
        "task_bank": source_bank,
        "selection": 2026111301 + index,
        "episode": derive_seed(source_bank, "episode"),
        "action": derive_seed(training, "action"),
        "actor": 2026110901,
        "critic": 2026111001 + index,
        "training": training,
    }
    development = {
        "constructor": derive_seed(development_bank, "constructor"),
        "task_bank": development_bank,
        "selection": derive_seed(development_episode, "selection"),
        "episode": derive_seed(development_episode, "episode"),
        "action": derive_seed(development_episode, "action"),
        "actor": source["actor"],
        "critic": source["critic"],
        "training": source["training"],
    }
    return {
        "task": task,
        "source_streams": source,
        "development_streams": development,
        "effective_source_bank_seed": derive_seed(source["task_bank"], "source"),
        "effective_development_bank_seed": derive_seed(development["task_bank"], "development"),
    }


def approved_configuration() -> dict[str, Any]:
    """Prospective implementation of the approved six-hour row, not an option sweep."""
    return {
        "schema_version": 1,
        "purpose": "exploratory_source_competence_development",
        "allocation_option": "approved_6h_local",
        "authorization": {
            "issue": 37,
            "approved": True,
            "user_approval_url": APPROVAL_URL,
            "resolution_url": RESOLUTION_URL,
            "hardware": "NVIDIA GeForce RTX 4060 Laptop GPU",
            "concurrency": 1,
            "paid_services": False,
            "hosted_services": False,
        },
        "session": {
            "started_at_utc": "2026-11-09T15:35:00Z",
            "deadline_utc": "2026-11-09T21:35:00Z",
            "training_deadline_utc": "2026-11-09T21:05:00Z",
            "max_wall_seconds": 21600,
            "evaluation_reserve_seconds": 1800,
            "preflight_phase_seconds": 900,
            "preparation_counts_against_session": True,
        },
        "tasks": [_task_record(task, index) for index, task in enumerate(TASKS)],
        "milestones": [100000, 250000],
        "goal_counts": {"source": 50, "development": 20, "final_test": 0},
        "evaluation": {
            "episodes": 20,
            "required_successes": 16,
            "episode_horizon": 500,
            "success_aggregation": "any_time_in_complete_episode",
            "early_stop": "earliest_passing_milestone",
            "goal_order": "bank_order_once_per_checkpoint",
            "partial_episodes_count": False,
        },
        "limits": {
            "source_policy_starts": 3,
            "retries": 0,
            "source_training_steps": 750000,
            "development_evaluation_steps": 60000,
            "preflight_steps": 1000,
            "total_agent_steps": 811000,
            "target_steps": 0,
            "final_test_steps": 0,
            "new_artifact_bytes": 2 * 1024**3,
        },
        "algorithm": {
            "name": "SAC",
            "policy": "MlpPolicy",
            "net_arch": [256, 256],
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "tau": 0.005,
            "buffer_size": 50000,
            "batch_size": 256,
            "learning_starts": 1000,
            "train_freq": 1,
            "gradient_steps": 1,
            "ent_coef": "auto",
            "device": "cuda",
            "torch_threads": 4,
        },
        "environment": {
            "metaworld_version": "3.1.1",
            "gymnasium_version": "1.1.1",
            "mujoco_version": "3.3.0",
            "stable_baselines3_version": "2.6.0",
            "torch_version": "2.3.0",
            "reward_function_version": "v2",
            "observation_dimensions": 39,
            "action_dimensions": 4,
            "visible_goal_indices": [36, 37, 38],
            "task_id_observation": False,
            "max_episode_steps": 500,
            "terminate_on_success": False,
            "observation_normalization": False,
            "reward_normalization": False,
            "time_limit_bootstrap": True,
            "observation_bounds_adapter": "goal_metadata_only",
        },
        "output": "/home/icannon/code/UWSH-nethack/artifacts/metaworld-development-2026-11-09",
        "stopping_rule": (
            "Round-robin cumulative milestones; stop each task at its first complete 16/20 pass. "
            "Otherwise stop at 250000 source steps. Preserve learner/replay state across stages. "
            "No retry, replacement, sweep, target adaptation, final-test access or extension. "
            "Stop preflight after 900 phase seconds, training at 21:05Z and all interaction at "
            "21:35Z. Retain partial/failed outcomes; the frozen session never restarts."
        ),
        "seed_derivation": {
            "algorithm": "uint32 big-endian first4bytes SHA256(UTF8(str(seed)+':'+label))",
            "effective_bank_seed": "derive_seed(task_bank, split)",
            "source_derived_fields": "constructor/episode from task_bank; action from training",
            "development_namespace": "2026111501 + task_index; labels selection/episode/action",
            "development_model_streams": "actor/critic/training copied from source and unused",
        },
    }


def _exact(actual: Any, expected: Any, path: str = "plan") -> None:
    if type(actual) is not type(expected):
        raise ValueError(f"{path}: expected approved {type(expected).__name__} setting")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{path}: missing or unknown settings")
        for key, value in expected.items():
            _exact(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{path}: unapproved finite grid length")
        for index, (value, reference) in enumerate(zip(actual, expected, strict=True)):
            _exact(value, reference, f"{path}[{index}]")
    elif actual != expected:
        raise ValueError(f"{path}: differs from the approved six-hour configuration")


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("session timestamps must have explicit UTC offsets")
    return result


@dataclass(frozen=True)
class DevelopmentPlan:
    _config: dict[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "_config", copy.deepcopy(self._config))
        self.validate()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DevelopmentPlan:
        return cls(data)

    @classmethod
    def load(cls, path: str | Path) -> DevelopmentPlan:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._config)

    @property
    def sha256(self) -> str:
        payload = json.dumps(self._config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(payload.encode()).hexdigest()

    def validate(self) -> None:
        # Validate structure even after expiry; this is a dry-plan tool. The
        # executable runner separately checks live UTC against this fixed window.
        _exact(self._config, approved_configuration())
        if (self.deadline_utc - self.session_started_utc).total_seconds() != self.session_seconds:
            raise ValueError("session duration mismatch")
        if (self.deadline_utc - self.training_deadline_utc).total_seconds() != self.reserve_seconds:
            raise ValueError("training reserve mismatch")
        if self.ledger() != self._config["limits"]:
            raise ValueError("declared caps disagree with the worst-case finite schedule")

    @property
    def tasks(self) -> tuple[str, ...]:
        return tuple(record["task"] for record in self._config["tasks"])

    @property
    def milestones(self) -> tuple[int, ...]:
        return tuple(self._config["milestones"])

    @property
    def stages(self) -> tuple[int, ...]:
        return self.milestones

    @property
    def actor_seed(self) -> int:
        return 2026110901

    def source_streams(self, task: str) -> dict[str, int]:
        return self._streams(task, "source_streams")

    def development_streams(self, task: str) -> dict[str, int]:
        return self._streams(task, "development_streams")

    def _streams(self, task: str, name: str) -> dict[str, int]:
        for record in self._config["tasks"]:
            if record["task"] == task:
                return dict(record[name])
        raise ValueError(f"task is outside the approved finite list: {task}")

    @property
    def source_goal_count(self) -> int:
        return 50

    @property
    def development_goal_count(self) -> int:
        return 20

    @property
    def evaluation_episodes(self) -> int:
        return 20

    @property
    def success_threshold(self) -> int:
        return 16

    @property
    def max_episode_steps(self) -> int:
        return 500

    @property
    def max_training_steps(self) -> int:
        return len(self.tasks) * self.milestones[-1]

    @property
    def max_eval_steps(self) -> int:
        return len(self.schedule()) * self.evaluation_episodes * self.max_episode_steps

    @property
    def max_preflight_steps(self) -> int:
        return 1000

    @property
    def preflight_seconds(self) -> int:
        return 900

    @property
    def session_started_utc(self) -> datetime:
        return _utc(self._config["session"]["started_at_utc"])

    @property
    def deadline_utc(self) -> datetime:
        return _utc(self._config["session"]["deadline_utc"])

    @property
    def training_deadline_utc(self) -> datetime:
        return _utc(self._config["session"]["training_deadline_utc"])

    @property
    def session_seconds(self) -> int:
        return 21600

    @property
    def reserve_seconds(self) -> int:
        return 1800

    @property
    def artifact_cap_bytes(self) -> int:
        return 2 * 1024**3

    @property
    def output_path(self) -> Path:
        return Path(self._config["output"])

    @property
    def algorithm(self) -> dict[str, Any]:
        return copy.deepcopy(self._config["algorithm"])

    @property
    def environment(self) -> dict[str, Any]:
        return copy.deepcopy(self._config["environment"])

    def ledger(self) -> dict[str, int]:
        return {
            "source_policy_starts": len(self.tasks),
            "retries": 0,
            "source_training_steps": self.max_training_steps,
            "development_evaluation_steps": self.max_eval_steps,
            "preflight_steps": self.max_preflight_steps,
            "total_agent_steps": self.max_training_steps
            + self.max_eval_steps
            + self.max_preflight_steps,
            "target_steps": 0,
            "final_test_steps": 0,
            "new_artifact_bytes": self.artifact_cap_bytes,
        }

    def schedule(self) -> tuple[tuple[str, int], ...]:
        """Worst-case queue; the runner omits later visits to already-passing tasks."""
        return tuple((task, milestone) for milestone in self.milestones for task in self.tasks)

    def worst_case(self) -> dict[str, Any]:
        return {
            "ledger": self.ledger(),
            "source_training_steps_per_task": self.milestones[-1],
            "checkpoint_evaluations": len(self.schedule()),
            "development_evaluation_episodes": len(self.schedule()) * self.evaluation_episodes,
            "training_only_seconds_historical_projection": [
                round(self.max_training_steps / 53.16, 2),
                round(self.max_training_steps / 41.70, 2),
            ],
            "projection_limit": "historical throughput only; UTC/interaction/storage caps override",
            "constructor_reset_physics": "included in wall costs, not agent-transition counters",
            "stopping_rule": self._config["stopping_rule"],
        }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    try:
        plan = DevelopmentPlan.load(args.config)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "status": "VALIDATED_PLAN_ONLY",
                "plan_sha256": plan.sha256,
                "plan": plan.to_dict(),
                "schedule": [
                    {"task": task, "cumulative_steps": steps} for task, steps in plan.schedule()
                ],
                "worst_case": plan.worst_case(),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
