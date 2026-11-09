"""Versioned, explicit RNG and goal contracts for the pinned Meta-World runtime.

The upstream 3.1.1 reset(seed=...) ignores its seed, and its task selector shares
the environment RNG. Here constructor, bank, selection and episode randomness
are separate. SB3's forwarded reset seeds are recorded but cannot reseed them.
Create another environment with the same streams to replay a sequence.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import importlib
import json
import random
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import version
from typing import Any

import numpy as np
import numpy.typing as npt

SCHEMA_VERSION = 1
SPLITS = ("source", "development", "final_test")
ENVIRONMENT_CONTRACT = {
    "metaworld_version": "3.1.1",
    "mujoco_version": "3.3.0",
    "reward_function_version": "v2",
    "observation_dimensions": 39,
    "action_dimensions": 4,
    "visible_goal_indices": [36, 37, 38],
    "task_id_observation": False,
    "max_episode_steps": 500,
    "training_terminate_on_success": False,
    "evaluation_terminate_on_success": False,
    "success_aggregation": "any_time_in_complete_episode",
    "time_limit_bootstrap": True,
    "reset_seed_behavior": "recorded_only; explicit protocol streams are authoritative",
}
_LEGACY_RNG_LOCK = threading.RLock()


def stable_hash(value: Any) -> str:
    """Hash portable JSON, rejecting nonfinite numeric metadata."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def derive_seed(seed: int, label: str) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seeds must be unsigned 32-bit integers")
    return int.from_bytes(hashlib.sha256(f"{seed}:{label}".encode()).digest()[:4], "big")


@dataclass(frozen=True)
class SeedStreams:
    constructor: int
    task_bank: int
    selection: int
    episode: int
    action: int
    actor: int
    critic: int
    training: int

    def __post_init__(self) -> None:
        for name, seed in asdict(self).items():
            derive_seed(seed, name)

    @classmethod
    def from_seed(cls, seed: int, *, actor_seed: int | None = None) -> SeedStreams:
        values = {name: derive_seed(seed, name) for name in cls.__dataclass_fields__}
        if actor_seed is not None:
            values["actor"] = actor_seed
        return cls(**values)


class LegacyRNG:
    """Isolate upstream legacy NumPy/Python draws and restore caller RNG states.

    Controlled environments are synchronous; this lock serializes calls through
    this adapter, not arbitrary concurrent code that mutates global RNGs.
    """

    def __init__(self, seed: int):
        self.numpy_state = np.random.RandomState(seed).get_state()
        self.python_state = random.Random(seed).getstate()

    @contextmanager
    def activate(self) -> Iterator[None]:
        with _LEGACY_RNG_LOCK:
            numpy_state = np.random.get_state()
            python_state = random.getstate()
            np.random.set_state(self.numpy_state)
            random.setstate(self.python_state)
            try:
                yield
            finally:
                self.numpy_state = np.random.get_state()
                self.python_state = random.getstate()
                np.random.set_state(numpy_state)
                random.setstate(python_state)


@dataclass(frozen=True)
class Goal:
    task: str
    payload_base64: str
    random_vector: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.task.endswith("-v3"):
            raise ValueError("expected a pinned Meta-World -v3 task name")
        if not self.random_vector or not np.isfinite(self.random_vector).all():
            raise ValueError("goal random vector must be nonempty and finite")
        if not base64.b64decode(self.payload_base64, validate=True):
            raise ValueError("task payload must be nonempty")
        object.__setattr__(
            self, "random_vector", tuple(float(value) for value in self.random_vector)
        )

    @property
    def goal_id(self) -> str:
        # The random vector includes object and goal configuration, not just XYZ.
        return stable_hash({"task": self.task, "random_vector": self.random_vector})

    def to_dict(self) -> dict[str, Any]:
        payload = base64.b64decode(self.payload_base64, validate=True)
        return {
            **asdict(self),
            "random_vector": list(self.random_vector),
            "goal_id": self.goal_id,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> Goal:
        goal = cls(record["task"], record["payload_base64"], tuple(record["random_vector"]))
        if goal.to_dict() != record:
            raise ValueError("goal manifest content/hash mismatch")
        return goal


@dataclass(frozen=True)
class GoalBank:
    split: str
    seed: int
    goals: tuple[Goal, ...]

    def __post_init__(self) -> None:
        if self.split not in SPLITS or not self.goals:
            raise ValueError("need a named split and nonempty goal bank")
        derive_seed(self.seed, self.split)
        if len({goal.goal_id for goal in self.goals}) != len(self.goals):
            raise ValueError("duplicate configurations within a goal bank")
        if len({goal.task for goal in self.goals}) != 1:
            raise ValueError("each goal bank must contain exactly one task class")

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "seed": self.seed,
            "goals": [goal.to_dict() for goal in self.goals],
        }

    @property
    def sha256(self) -> str:
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> GoalBank:
        return cls(
            split=record["split"],
            seed=record["seed"],
            goals=tuple(Goal.from_dict(goal) for goal in record["goals"]),
        )


@dataclass(frozen=True)
class GoalManifest:
    banks: tuple[GoalBank, ...]
    heldout_task_classes: tuple[str, ...] = ()
    transfer_regime: str = "development_only"

    def __post_init__(self) -> None:
        seen: set[str] = set()
        bank_keys: set[tuple[str, str]] = set()
        for bank in self.banks:
            key = (bank.split, bank.goals[0].task)
            if key in bank_keys:
                raise ValueError("duplicate task/split bank")
            bank_keys.add(key)
            ids = {goal.goal_id for goal in bank.goals}
            if seen.intersection(ids):
                raise ValueError("goal configuration leaks between splits")
            seen.update(ids)
            if bank.split != "final_test" and bank.goals[0].task in self.heldout_task_classes:
                raise ValueError("held-out task class occurs in source/development data")
        final_classes = {b.goals[0].task for b in self.banks if b.split == "final_test"}
        if not set(self.heldout_task_classes).issubset(final_classes):
            raise ValueError("held-out classes must have final-test goal banks")
        source_classes = {b.goals[0].task for b in self.banks if b.split == "source"}
        if self.transfer_regime == "development_only":
            if final_classes or self.heldout_task_classes:
                raise ValueError("development-only manifests cannot contain final-test data")
        elif self.transfer_regime == "within_family":
            if self.heldout_task_classes or not final_classes.issubset(source_classes):
                raise ValueError("within-family final goals require source task classes")
        elif self.transfer_regime == "heldout_family":
            if not final_classes or set(self.heldout_task_classes) != final_classes:
                raise ValueError("all final-test task classes must be explicitly held out")
        else:
            raise ValueError("unknown transfer regime")

    def for_fitting(self) -> tuple[GoalBank, ...]:
        """Basis/alignment/normalization fitting receives source data only."""
        return tuple(bank for bank in self.banks if bank.split == "source")

    def for_tuning(self) -> tuple[GoalBank, ...]:
        return tuple(bank for bank in self.banks if bank.split == "development")

    def for_evaluation(self, split: str) -> tuple[GoalBank, ...]:
        if split not in SPLITS:
            raise ValueError("unknown goal split")
        return tuple(bank for bank in self.banks if bank.split == split)

    def validate_source_weights(self, task_classes: Sequence[str]) -> None:
        source = {bank.goals[0].task for bank in self.for_fitting()}
        if not set(task_classes).issubset(source):
            raise ValueError("basis weights must come only from declared source task classes")

    def validate_source_policy_records(self, records: Sequence[dict[str, Any]]) -> None:
        """Check weight provenance, including same-class target-trained actors."""
        source = {(bank.goals[0].task, bank.sha256) for bank in self.for_fitting()}
        for record in records:
            if (
                record.get("training_split") != "source"
                or (record.get("task"), record.get("training_goal_bank_sha256")) not in source
            ):
                raise ValueError("policy weights were not trained on a declared source goal bank")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "environment": copy.deepcopy(ENVIRONMENT_CONTRACT),
            "heldout_task_classes": list(self.heldout_task_classes),
            "transfer_regime": self.transfer_regime,
            "banks": [bank.to_dict() for bank in self.banks],
        }

    @property
    def sha256(self) -> str:
        return stable_hash(self.to_dict())

    @classmethod
    def from_dict(cls, record: dict[str, Any]) -> GoalManifest:
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported goal manifest schema")
        if record.get("environment") != ENVIRONMENT_CONTRACT:
            raise ValueError("goal manifest does not match the pinned environment contract")
        return cls(
            tuple(GoalBank.from_dict(bank) for bank in record["banks"]),
            tuple(record["heldout_task_classes"]),
            record["transfer_regime"],
        )


def _runtime() -> tuple[Any, Any]:
    metaworld = importlib.import_module("metaworld")
    gym = importlib.import_module("gymnasium")
    for name, expected in (("metaworld", "3.1.1"), ("mujoco", "3.3.0")):
        if version(name) != expected:
            raise ValueError(f"protocol requires {name}=={expected}")
    return metaworld, gym


def generate_goal_bank(task: str, streams: SeedStreams, *, split: str = "source") -> GoalBank:
    """Capture the pinned MT1 bank; serialize only locally generated task payloads.

    Loading a task payload calls upstream pickle-based set_task. These manifests
    are trusted experiment artifacts, not a format for accepting untrusted files.
    """
    if split not in SPLITS:
        raise ValueError("unknown goal split")
    metaworld, _ = _runtime()
    bank_seed = derive_seed(streams.task_bank, split)
    with LegacyRNG(streams.constructor).activate():
        benchmark = metaworld.MT1(task, seed=bank_seed)
        probe = benchmark.train_classes[task](reward_function_version="v2")
        try:
            goals = []
            for item in benchmark.train_tasks:
                probe.set_task(item)
                goals.append(
                    Goal(
                        task=task,
                        payload_base64=base64.b64encode(item.data).decode("ascii"),
                        random_vector=tuple(float(value) for value in probe._last_rand_vec),
                    )
                )
        finally:
            probe.close()
    return GoalBank(split=split, seed=bank_seed, goals=tuple(goals))


class TaskController:
    """Own task selection and episode seeds independently of runtime global RNGs."""

    def __init__(
        self,
        env: Any,
        bank: GoalBank,
        streams: SeedStreams,
        task_factory: Any,
        goal_order: Sequence[int] | None = None,
    ):
        self.env = env
        self.bank = bank
        self.streams = streams
        self.task_factory = task_factory
        self.selection_rng = np.random.default_rng(streams.selection)
        self.episode_rng = np.random.default_rng(streams.episode)
        self.action_rng = np.random.default_rng(streams.action)
        self.goal_order = tuple(goal_order) if goal_order is not None else None
        if self.goal_order is not None and (
            not self.goal_order
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(bank.goals)
                for index in self.goal_order
            )
        ):
            raise ValueError("invalid declared evaluation goal order")
        self.episode_history: list[dict[str, Any]] = []
        self.legacy_rng = LegacyRNG(streams.constructor)
        self.active = False
        self.length = 0
        self.success = False

    def reset(self, *, seed: int | None = None, options: Any = None) -> tuple[Any, Any]:
        index = len(self.episode_history)
        if self.goal_order is not None:
            if index >= len(self.goal_order):
                raise ValueError("declared goal order exhausted")
            goal_index = self.goal_order[index]
        else:
            goal_index = int(self.selection_rng.integers(len(self.bank.goals)))
        episode_seed = int(self.episode_rng.integers(0, 2**32, dtype=np.uint64))
        self.legacy_rng = LegacyRNG(episode_seed)
        goal = self.bank.goals[goal_index]
        payload = base64.b64decode(goal.payload_base64, validate=True)
        with self.legacy_rng.activate():
            self.env.unwrapped.seed(episode_seed)
            # Sawyer.seed also seeds the action space. Give action sampling its
            # own persistent stream after each episode seed is installed.
            self.env.action_space._np_random = self.action_rng
            self.env.unwrapped.set_task(self.task_factory(env_name=goal.task, data=payload))
            if self.env.unwrapped._partially_observable or not np.array_equal(
                self.env.unwrapped._last_rand_vec, goal.random_vector
            ):
                raise ValueError(
                    "task payload disagrees with the declared visible-goal configuration"
                )
            observation, info = self.env.reset(options=options)
            if not np.array_equal(self.env.unwrapped._last_rand_vec, goal.random_vector):
                raise ValueError("reset changed the frozen goal configuration")
        if self.active:
            self.episode_history[-1]["complete_episode"] = False
        record = {
            "episode": index,
            "goal_index": goal_index,
            "goal_id": goal.goal_id,
            "goal_bank_sha256": self.bank.sha256,
            "episode_seed": episode_seed,
            "forwarded_reset_seed": seed,
            "length": 0,
            "success": False,
            "complete_episode": False,
        }
        self.episode_history.append(record)
        self.active, self.length, self.success = True, 0, False
        return observation, {**info, "protocol": dict(record)}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if not self.active:
            raise ValueError("reset required before stepping or after a complete episode")
        with self.legacy_rng.activate():
            observation, reward, terminated, truncated, info = self.env.step(action)
        self.length += 1
        success = info.get("success", 0)
        if success not in (0, 1, False, True):
            raise ValueError("success must be a binary task flag")
        self.success = self.success or bool(success)
        record = self.episode_history[-1]
        record.update(
            length=self.length,
            success=self.success,
            terminated=bool(terminated),
            truncated=bool(truncated),
            complete_episode=bool(terminated or truncated),
        )
        if terminated or bool(truncated) != (self.length == 500):
            raise ValueError("runtime violated full 500-step episode semantics")
        self.active = not record["complete_episode"]
        return observation, reward, terminated, truncated, {**info, "protocol": dict(record)}

    def protocol_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "environment": copy.deepcopy(ENVIRONMENT_CONTRACT),
            "rng_streams": asdict(self.streams),
            "goal_bank": self.bank.to_dict(),
            "goal_bank_sha256": self.bank.sha256,
            "declared_goal_order": list(self.goal_order) if self.goal_order is not None else None,
            "episode_history": [dict(row) for row in self.episode_history],
        }


def make_protocol_env(
    task: str,
    streams: SeedStreams,
    *,
    bank: GoalBank | None = None,
    goal_order: Sequence[int] | None = None,
) -> Any:
    """Create a raw visible-goal task with no upstream random-selection wrapper."""
    metaworld, gym = _runtime()
    bank = generate_goal_bank(task, streams) if bank is None else bank
    if any(goal.task != task for goal in bank.goals):
        raise ValueError("goal bank task differs from requested environment")
    with LegacyRNG(streams.constructor).activate():
        raw = metaworld.env_dict.ALL_V3_ENVIRONMENTS[task](reward_function_version="v2")
        try:
            raw.seed(streams.constructor)
            raw.set_task(
                metaworld.Task(
                    env_name=task,
                    data=base64.b64decode(bank.goals[0].payload_base64, validate=True),
                )
            )
            if (
                raw.reward_function_version != "v2"
                or raw._partially_observable
                or raw.max_path_length != 500
                or raw.observation_space.shape != (39,)
                or raw.action_space.shape != (4,)
                or not np.all(raw.action_space.low == -1)
                or not np.all(raw.action_space.high == 1)
            ):
                raise ValueError("runtime task does not satisfy the environment contract")
            env = gym.wrappers.TimeLimit(raw, max_episode_steps=500)
            env.action_space.seed(streams.action)
        except Exception:
            raw.close()
            raise
    controller = TaskController(env, bank, streams, metaworld.Task, goal_order)

    def reset(wrapper: Any, *, seed: int | None = None, options: Any = None) -> Any:
        return controller.reset(seed=seed, options=options)

    def step(wrapper: Any, action: Any) -> Any:
        return controller.step(action)

    def protocol_manifest(wrapper: Any) -> dict[str, Any]:
        return controller.protocol_manifest()

    def protocol_controller(wrapper: Any) -> TaskController:
        """Expose the synchronous controller for explicitly verified recovery only."""
        return controller

    # Construct the runtime-only subclass after importing optional Gymnasium;
    # manifest/seed tools stay importable and type-checkable without that stack.
    wrapper_type = type(
        "ControlledTaskEnv",
        (gym.Wrapper,),
        {
            "reset": reset,
            "step": step,
            "protocol_manifest": protocol_manifest,
            "protocol_controller": protocol_controller,
        },
    )
    return wrapper_type(env)


def evaluation_order(bank: GoalBank, episodes: int) -> tuple[int, ...]:
    """Cycle explicitly through every goal before repeating, with a finite plan."""
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("evaluation episodes must be positive")
    return tuple(index % len(bank.goals) for index in range(episodes))


def evaluation_record(env: Any) -> dict[str, Any]:
    """Identity of the current episode, including the exact serialized task."""
    protocol = env.get_wrapper_attr("protocol_manifest")()
    record = dict(protocol["episode_history"][-1])
    record["goal"] = protocol["goal_bank"]["goals"][record["goal_index"]]
    return record


def validate_evaluation_coverage(
    rows: Sequence[dict[str, Any]], bank: GoalBank, order: Sequence[int]
) -> None:
    if len(rows) != len(order):
        raise ValueError("evaluation does not contain all declared episodes")
    for episode, (row, goal_index) in enumerate(zip(rows, order, strict=True)):
        if (
            row.get("episode") != episode
            or row.get("goal_id") != bank.goals[goal_index].goal_id
            or row.get("goal_bank_sha256") != bank.sha256
            or row.get("complete_episode") is not True
            or row.get("length") != 500
            or row.get("terminated") is not False
            or row.get("truncated") is not True
        ):
            raise ValueError("evaluation goal coverage or full-episode semantics mismatch")


def bootstrap_mask(terminated: Any, truncated: Any) -> npt.NDArray[np.float32]:
    """Time-limit truncations bootstrap; only actual terminal transitions stop it."""
    terminated, truncated = np.broadcast_arrays(terminated, truncated)
    return np.asarray(~terminated.astype(bool), dtype=np.float32)


def _state_hash(module: Any) -> str:
    digest = hashlib.sha256()
    for key, tensor in sorted(module.state_dict().items()):
        array = tensor.detach().cpu().numpy()
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def build_seeded_sac(env: Any, streams: SeedStreams, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
    """Use critic-seeded SAC construction, then explicit actor and training seeds.

    SAC(seed=...) reseeds all model construction, overriding a preceding manual
    torch seed. Regenerate just the actor and copy it into the optimizer-owned
    actor, leaving the constructor's critic/target intact. Reset forwarded env
    seeds cannot modify the independently configured task controller.
    """
    torch = importlib.import_module("torch")
    sac = importlib.import_module("stable_baselines3").SAC
    if "seed" in kwargs:
        raise ValueError("use explicit SeedStreams instead of SAC seed=")
    replay_kwargs = dict(kwargs.pop("replay_buffer_kwargs", {}))
    if replay_kwargs.get("handle_timeout_termination", True) is not True:
        raise ValueError("the protocol requires time-limit bootstrapping")
    replay_kwargs["handle_timeout_termination"] = True
    kwargs["replay_buffer_kwargs"] = replay_kwargs
    model = sac("MlpPolicy", env, seed=streams.critic, **kwargs)
    if model.replay_buffer.handle_timeout_termination is not True:
        raise ValueError("SAC replay buffer does not bootstrap time-limit truncations")
    device = model.device
    cuda_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(streams.actor)
        actor = model.policy.make_actor()
        model.policy.actor.load_state_dict(actor.state_dict(), strict=True)
    metadata = {
        "rng_streams": asdict(streams),
        "sac_constructor_seed": streams.critic,
        "actor_state_sha256": _state_hash(model.policy.actor),
        "critic_state_sha256": _state_hash(model.policy.critic),
        "critic_target_state_sha256": _state_hash(model.policy.critic_target),
        "training_seed_applied_after_initialization": streams.training,
        "replay_buffer_handle_timeout_termination": True,
    }
    model.set_random_seed(streams.training)
    model.seed = streams.training
    model.action_space.seed(streams.action)
    return model, metadata
