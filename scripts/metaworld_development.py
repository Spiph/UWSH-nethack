"""Run an explicitly approved staged source-development plan; planning is runtime-free."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import io
import json
import math
import pickle
import random
import re
import traceback
import zipfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any

from scripts.metaworld_deadlines import BudgetExpired, RunDeadline, parse_deadline, utc_now
from scripts.metaworld_protocol import (
    GoalBank,
    GoalManifest,
    SeedStreams,
    evaluation_order,
    evaluation_record,
    generate_goal_bank,
    validate_evaluation_coverage,
)


class Stopped(RuntimeError):
    """A resource or phase limit permits retention/cleanup only."""


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for piece in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(piece)
    return hasher.hexdigest()


def code_hashes() -> dict[str, str]:
    return {path.name: digest(path) for path in Path(__file__).parent.glob("metaworld_*.py")}


class CappedWriter(io.BufferedWriter):
    def __init__(self, path: Path, store: ArtifactStore, metadata: bool = False) -> None:
        self.store = store
        self.limit = store.limit if metadata else store.limit - store.reserve
        self.others = store.size() - (path.stat().st_size if path.exists() else 0)
        self.high_water = 0
        super().__init__(io.FileIO(path, "wb"))

    def write(self, data: Any) -> int:
        # Protocol 5 may pass PickleBuffer directly for large NumPy arrays.
        # Its raw view also handles Fortran-contiguous exporters; nbytes counts
        # bytes rather than elements or the leading dimension of a typed view.
        view = data.raw() if isinstance(data, pickle.PickleBuffer) else memoryview(data)
        with view:
            projected = max(self.high_water, self.tell() + view.nbytes)
            if self.others + projected > self.limit:
                raise Stopped("artifact_storage_limit")
            count = super().write(view)
            self.high_water = max(self.high_water, self.tell())
            return count


class ArtifactStore:
    def __init__(self, root: Path, limit: int) -> None:
        if root.exists() and any(root.iterdir()):
            raise ValueError("output directory must be new or empty")
        root.mkdir(parents=True, exist_ok=True)
        self.root, self.limit = root, limit
        self.reserve = min(4 * 1024 * 1024, limit // 10)

    def size(self) -> int:
        return sum(path.stat().st_size for path in self.root.rglob("*") if path.is_file())

    def open(self, relative: str, *, metadata: bool = False) -> CappedWriter:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return CappedWriter(path, self, metadata)

    def json(self, relative: str, value: Any) -> None:
        payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        with self.open(relative + ".tmp", metadata=True) as handle:
            handle.write(payload)
        (self.root / (relative + ".tmp")).replace(self.root / relative)


def capture_rng(torch: Any, cuda: bool) -> dict[str, Any]:
    np = importlib.import_module("numpy")
    return {
        "python": random.getstate(),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()] if cuda else [],
    }


def restore_rng(state: dict[str, Any], torch: Any, cuda: bool) -> None:
    random.setstate(state["python"])
    importlib.import_module("numpy").random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if cuda:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


@dataclass
class LearnerRNG:
    state: dict[str, Any]
    torch: Any
    cuda: bool

    @contextmanager
    def activate(self) -> Iterator[None]:
        caller = capture_rng(self.torch, self.cuda)
        restore_rng(self.state, self.torch, self.cuda)
        try:
            yield
        finally:
            self.state = capture_rng(self.torch, self.cuda)
            restore_rng(caller, self.torch, self.cuda)


@dataclass
class TaskState:
    task: str
    model: Any
    env: Any
    goals: GoalManifest
    rng: Any
    metadata: dict[str, Any]
    status: str = "PENDING"
    checkpoints: list[dict[str, Any]] = field(default_factory=list)


class DevelopmentRunner:
    """Single-process continuation engine; runtime injection supports fake-clock tests."""

    def __init__(
        self,
        plan: Any,
        approval: dict[str, Any],
        *,
        budget: RunDeadline | None = None,
        runtime_factory: Callable[[DevelopmentRunner], Any] | None = None,
    ) -> None:
        validate_approval(plan, approval)
        plan.validate()
        remaining = (plan.deadline_utc - utc_now()).total_seconds()
        self.budget = budget or RunDeadline(max(remaining, 0.001), plan.deadline_utc)
        self.plan = plan
        self.store = ArtifactStore(plan.output_path, plan.artifact_cap_bytes)
        self.runtime_factory = runtime_factory or Runtime
        self.runtime: Any = None
        self.progress_rows: list[dict[str, Any]] = []
        self.states: dict[str, TaskState] = {}
        self.preflight: RunDeadline | None = None
        self.phase = "PREPARATION"
        self.training_stopped = False
        self.ledger = {
            "source_training_steps": 0,
            "development_evaluation_steps": 0,
            "preflight_steps": 0,
            "source_policy_starts": 0,
            "target_steps": 0,
            "final_test_steps": 0,
            "failed_step_calls": 0,
            "optimizer_updates": 0,
            "optimizer_update_attempts": 0,
        }
        self.manifest: dict[str, Any] = {
            "status": "RUNNING",
            "purpose": "staged exploratory source competence development",
            "plan_sha256": plan.sha256,
            "plan": plan.to_dict(),
            "approval": approval,
            "session_started_utc": plan.session_started_utc.isoformat(),
            "preparation_utc_seconds_at_runner_start": (
                self.budget.started_utc - plan.session_started_utc
            ).total_seconds(),
            "runner_monotonic_budget_seconds": self.budget.seconds,
            "training_deadline_utc": plan.training_deadline_utc.isoformat(),
            "worst_case_ledger": plan.ledger(),
            "ledger": self.ledger,
            "tasks": {},
            "queue": [],
            "errors": [],
            "resume_supported": False,
            "continuation": (
                "resident models/envs; same optimizers/entropy/replay; no process restart"
            ),
            "code_sha256": code_hashes(),
        }

    def check(self, kind: str) -> None:
        self.budget.require(kind)
        if self.preflight is not None and not self.preflight.allowed("preflight"):
            raise Stopped("preflight_deadline")
        if kind == "training":
            if self.training_stopped:
                raise Stopped("training_reserve")
            if (
                self.budget.seconds <= self.plan.reserve_seconds
                or not self.budget.allowed(
                    "training_reserve", reserve_seconds=self.plan.reserve_seconds
                )
                or self.budget.now() >= self.plan.training_deadline_utc
            ):
                self.training_stopped = True
                raise Stopped("training_reserve")

    def start_policy(self) -> None:
        self.check("training")
        if self.ledger["source_policy_starts"] >= self.plan.ledger()["source_policy_starts"]:
            raise Stopped("source_policy_starts_limit")
        self.ledger["source_policy_starts"] += 1

    def bind_env(self, env: Any, kind: str) -> Any:
        bucket = {
            "training": "source_training_steps",
            "evaluation": "development_evaluation_steps",
            "preflight": "preflight_steps",
        }[kind]
        cap = {
            "training": self.plan.max_training_steps,
            "evaluation": self.plan.max_eval_steps,
            "preflight": self.plan.max_preflight_steps,
        }[kind]
        original_reset, original_step = env.reset, env.step

        def reset(*args: Any, **kwargs: Any) -> Any:
            self.check(kind)
            return original_reset(*args, **kwargs)

        def step(*args: Any, **kwargs: Any) -> Any:
            self.check(kind)
            if self.ledger[bucket] >= cap:
                raise Stopped(bucket + "_limit")
            try:
                result = original_step(*args, **kwargs)
            except Exception:
                self.ledger["failed_step_calls"] += 1
                raise
            self.ledger[bucket] += 1
            return result

        env.reset, env.step = reset, step
        return env

    @contextmanager
    def optimizer_guard(self, model: Any) -> Iterator[None]:
        original = model.train
        missing = object()
        prior = vars(model).get("train", missing)

        def train(*args: Any, **kwargs: Any) -> Any:
            self.check("training")
            self.ledger["optimizer_update_attempts"] += 1
            result = original(*args, **kwargs)
            self.ledger["optimizer_updates"] += 1
            return result

        model.train = train
        try:
            yield
        finally:
            if prior is missing:
                del model.train
            else:
                model.train = prior

    def persist(self) -> None:
        self.ledger["total_agent_steps"] = sum(
            self.ledger[key]
            for key in ("source_training_steps", "development_evaluation_steps", "preflight_steps")
        )
        timing: dict[str, Any] = {}
        self.budget.record(timing)
        self.manifest.update(
            {k: v for k, v in timing.items() if k not in ("status", "validation_complete")}
        )
        if not timing["budget_compliant"] and self.manifest["status"] != "ERROR":
            self.manifest["status"] = "INCOMPLETE"
        self.manifest["session_utc_elapsed_seconds"] = (
            parse_deadline(timing["finished_at_utc"]) - self.plan.session_started_utc
        ).total_seconds()
        self.manifest["training_reserve_stop_latched"] = self.training_stopped
        self.manifest["phase"] = self.phase
        self.manifest["artifact_bytes_before_manifest"] = self.store.size()
        self.manifest["tasks"] = {
            task: {
                "status": state.status,
                "actual_training_steps": int(state.model.num_timesteps),
                "metadata": state.metadata,
                "checkpoints": state.checkpoints,
            }
            for task, state in self.states.items()
        }
        self.manifest["missing_tasks"] = [
            task for task in self.plan.tasks if task not in self.states
        ]
        self.store.json("development_manifest.json", self.manifest)

    def checkpoint(self, state: TaskState, target: int, partial: bool) -> dict[str, Any]:
        name = f"checkpoints/{state.task}/step{int(state.model.num_timesteps)}"
        if partial:
            name += "-partial"
        saved_at = self.budget.now().isoformat()
        save_started = self.budget.monotonic()
        artifacts = self.runtime.save(state, name)
        row = {
            "target_steps": target,
            "actual_steps": int(state.model.num_timesteps),
            "partial": partial,
            "artifacts": artifacts,
            "evaluation": [],
            "passed": False,
            "checkpoint_started_at_utc": saved_at,
            "checkpoint_duration_seconds": self.budget.monotonic() - save_started,
        }
        state.checkpoints.append(row)
        self.persist()
        return row

    def evaluate(self, state: TaskState, row: dict[str, Any]) -> None:
        self.check("evaluation")
        evaluation_started = self.budget.monotonic()
        row["evaluation_started_at_utc"] = self.budget.now().isoformat()
        env = self.runtime.evaluation_env(state)
        try:
            for episode in range(self.plan.evaluation_episodes):
                self.check("evaluation")
                obs, _ = env.reset()
                total, length, success = 0.0, 0, False
                terminated = truncated = False
                try:
                    while not (terminated or truncated):
                        self.check("evaluation")
                        action, _ = state.model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        total += float(reward)
                        length += 1
                        success = success or bool(info.get("success", False))
                finally:
                    row["evaluation"].append(
                        {
                            "episode": episode,
                            "return": total,
                            "length": length,
                            "success": success,
                            "complete_episode": bool(terminated or truncated),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            **self.runtime.episode_record(env),
                        }
                    )
                    self.persist()
            self.check("evaluation")
            bank = state.goals.for_tuning()[0]
            validate_evaluation_coverage(
                row["evaluation"], bank, evaluation_order(bank, self.plan.evaluation_episodes)
            )
            row["evaluation_complete"] = True
            row["passed"] = (
                sum(r["success"] for r in row["evaluation"]) >= self.plan.success_threshold
            )
        finally:
            row["evaluation_finished_at_utc"] = self.budget.now().isoformat()
            row["evaluation_duration_seconds"] = self.budget.monotonic() - evaluation_started
            env.close()

    def run(self) -> dict[str, Any]:
        self.persist()
        try:
            self.check("initialization")
            self.phase = "PREFLIGHT"
            now = self.budget.now()
            self.preflight = RunDeadline(
                self.plan.preflight_seconds,
                min(self.plan.deadline_utc, now + timedelta(seconds=self.plan.preflight_seconds)),
                now=self.budget.now,
                monotonic=self.budget.monotonic,
            )
            self.runtime = self.runtime_factory(self)
            for task in self.plan.tasks:
                self.check("training")
                state = self.runtime.prepare(task)
                self.states[task] = state
                self.runtime.save_initial(state)
                self.persist()
            hashes = {state.metadata["initial_actor_sha256"] for state in self.states.values()}
            if len(hashes) != 1:
                raise ValueError("shared initial actors differ")
            self.manifest["goal_manifests"] = {
                task: state.goals.to_dict() for task, state in self.states.items()
            }
            self.manifest["prospective_frozen_at_utc"] = self.budget.now().isoformat()
            self.store.json("frozen_launch.json", self.manifest)
            self.runtime.preflight(list(self.states.values()))
            self.check("preflight")
            self.manifest["preflight"] = {
                "passed": True,
                "steps": self.ledger["preflight_steps"],
                "finished_at_utc": self.budget.now().isoformat(),
            }
            self.preflight = None
            self.phase = "SOURCE_DEVELOPMENT"
            for task, milestone in self.plan.schedule():
                state = self.states[task]
                if state.status == "PASS":
                    self.manifest["queue"].append(
                        {
                            "task": task,
                            "milestone": milestone,
                            "status": "SKIPPED_PLANNED_EARLY_STOP",
                        }
                    )
                    continue
                self.check("training")
                increment = milestone - int(state.model.num_timesteps)
                if increment <= 0:
                    raise ValueError("cumulative milestone does not advance learner")
                visit = {
                    "task": task,
                    "milestone": milestone,
                    "increment": increment,
                    "status": "TRAINING",
                }
                self.manifest["queue"].append(visit)
                self.persist()
                error: BaseException | None = None
                training_started = self.budget.monotonic()
                visit["training_started_at_utc"] = self.budget.now().isoformat()
                try:
                    with state.rng.activate(), self.optimizer_guard(state.model):
                        state.model.learn(
                            increment,
                            reset_num_timesteps=False,
                            callback=self.runtime.training_callback(state),
                        )
                except BaseException as caught:
                    error = caught
                finally:
                    visit["training_finished_at_utc"] = self.budget.now().isoformat()
                    visit["training_duration_seconds"] = self.budget.monotonic() - training_started
                partial = int(state.model.num_timesteps) != milestone or error is not None
                row = self.checkpoint(state, milestone, partial)
                if error is not None:
                    raise error
                if partial:
                    raise Stopped("incomplete_training_segment")
                with self.runtime.preserve_rng():
                    self.evaluate(state, row)
                if row["passed"]:
                    state.status = "PASS"
                    state.metadata["earliest_passing_steps"] = milestone
                elif milestone == self.plan.milestones[-1]:
                    state.status = "FULL_BUDGET_FAIL"
                visit["status"] = state.status if state.status != "PENDING" else "CONTINUE"
                self.persist()
            self.check("completion")
            self.manifest["status"] = (
                "COMPLETE_PASS"
                if all(state.status == "PASS" for state in self.states.values())
                else "COMPLETE_FAIL"
            )
        except (BudgetExpired, Stopped) as error:
            self.manifest["status"] = (
                "ERROR" if str(error) == "preflight_deadline" else "INCOMPLETE"
            )
            self.manifest["stop_reason"] = str(error)
            if self.phase == "PREFLIGHT":
                self.manifest["preflight"] = {
                    "passed": False,
                    "reason": str(error),
                    "steps": self.ledger["preflight_steps"],
                }
        except BaseException as error:
            self.manifest["status"] = (
                "INCOMPLETE" if isinstance(error, KeyboardInterrupt) else "ERROR"
            )
            self.manifest["errors"].append(f"{type(error).__name__}: {error}")
            self.manifest["traceback"] = traceback.format_exc()
        finally:
            if self.runtime is not None:
                try:
                    self.runtime.close()
                except BaseException as error:
                    self.manifest["errors"].append(f"cleanup: {type(error).__name__}: {error}")
                    self.manifest["status"] = "ERROR"
            self.phase = "FINISHED"
            self.persist()
        return self.manifest


class Runtime:
    """Pinned runtime adapter; construction is the first RL import in this module."""

    def __init__(self, runner: DevelopmentRunner) -> None:
        self.runner = runner
        runner.check("initialization")
        self.torch = importlib.import_module("torch")
        self.np = importlib.import_module("numpy")
        self.pilot = importlib.import_module("scripts.metaworld_hour_pilot")
        self.protocol = importlib.import_module("scripts.metaworld_protocol")
        self.geometry = importlib.import_module("scripts.metaworld_geometry")
        runner.check("initialization")
        self.torch.set_num_threads(runner.plan.algorithm["torch_threads"])
        runner.check("initialization")
        if self.torch.__version__.split("+")[0] != runner.plan.environment["torch_version"]:
            raise ValueError("runtime Torch version differs from frozen plan")
        if not self.torch.cuda.is_available():
            raise RuntimeError("approved CUDA device is unavailable")
        runner.manifest["runtime"] = {
            "torch": self.torch.__version__,
            "gpu": self.torch.cuda.get_device_name(0),
            "packages": {
                name: importlib.import_module("importlib.metadata").version(name)
                for name in ("metaworld", "mujoco", "stable-baselines3", "gymnasium", "numpy")
            },
        }

        actual = runner.manifest["runtime"]
        expected_gpu = runner.plan.to_dict()["authorization"]["hardware"]
        if actual["gpu"] != expected_gpu:
            raise ValueError("GPU differs from approved local hardware")
        for package, key in (
            ("metaworld", "metaworld_version"),
            ("mujoco", "mujoco_version"),
            ("stable-baselines3", "stable_baselines3_version"),
            ("gymnasium", "gymnasium_version"),
        ):
            if actual["packages"][package] != runner.plan.environment[key]:
                raise ValueError(f"runtime {package} version differs from frozen plan")

    @contextmanager
    def preserve_rng(self) -> Iterator[None]:
        caller = capture_rng(self.torch, True)
        try:
            yield
        finally:
            restore_rng(caller, self.torch, True)

    def prepare(self, task: str) -> TaskState:
        plan = self.runner.plan
        streams = SeedStreams(**plan.source_streams(task))
        development_streams = SeedStreams(**plan.development_streams(task))
        with self.preserve_rng():
            self.runner.check("initialization")
            source = generate_goal_bank(task, streams, split="source")
            self.runner.check("initialization")
            generated = generate_goal_bank(task, development_streams, split="development")
            development = GoalBank(
                "development", generated.seed, generated.goals[: plan.development_goal_count]
            )
            goals = GoalManifest((source, development))
            if (
                len(source.goals) != plan.source_goal_count
                or len(development.goals) != plan.development_goal_count
            ):
                raise ValueError("wrong source/development goal-bank sizes")
            self.runner.check("training")
            env = self.pilot.make_env(task, streams.constructor, bank=source, streams=streams)
            self.runner.bind_env(env, "training")
            config = plan.algorithm
            options = {
                key: value
                for key, value in config.items()
                if key not in ("name", "policy", "net_arch", "torch_threads")
            }
            options["policy_kwargs"] = {"net_arch": config["net_arch"]}
            self.runner.check("training")
            try:
                self.runner.start_policy()
                model, metadata = self.protocol.build_seeded_sac(env, streams, **options)
            except BaseException:
                env.close()
                raise
            initial_hash = self.geometry.hash_actor(model.policy.actor)
            capsule = LearnerRNG(capture_rng(self.torch, True), self.torch, True)
            return TaskState(
                task,
                model,
                env,
                goals,
                capsule,
                {
                    "initialization": metadata,
                    "initial_actor_sha256": initial_hash,
                    "source_streams": plan.source_streams(task),
                    "development_streams": plan.development_streams(task),
                    "evaluation_goal_order": list(
                        evaluation_order(development, plan.evaluation_episodes)
                    ),
                    "evaluation_episode_seeds": self.np.random.default_rng(
                        development_streams.episode
                    )
                    .integers(0, 2**32, size=plan.evaluation_episodes, dtype=self.np.uint64)
                    .tolist(),
                },
            )

    def save_initial(self, state: TaskState) -> None:
        vector, layout = self.geometry.actor_vector(state.model.policy.actor)
        with self.runner.store.open(f"initial/{state.task}.actor.npz") as handle:
            self.np.savez_compressed(
                handle, vector=vector, layout_json=self.np.asarray(json.dumps(layout))
            )
        state.metadata["actor_layout"] = layout
        state.metadata["initial_actor_artifact"] = f"initial/{state.task}.actor.npz"

    def save(self, state: TaskState, name: str) -> dict[str, Any]:
        files: dict[str, Any] = {}
        for suffix, save in (
            (".zip", state.model.save),
            (".replay.pkl", state.model.save_replay_buffer),
        ):
            with self.runner.store.open(name + suffix) as handle:
                save(handle)
            files[suffix] = {
                "path": name + suffix,
                "sha256": digest(self.runner.store.root / (name + suffix)),
            }
        with self.runner.store.open(name + ".rng.pkl") as handle:
            pickle.dump(state.rng.state, handle, protocol=pickle.HIGHEST_PROTOCOL)
        files["rng"] = {
            "path": name + ".rng.pkl",
            "sha256": digest(self.runner.store.root / (name + ".rng.pkl")),
        }
        vector, layout = self.geometry.actor_vector(state.model.policy.actor)
        with self.runner.store.open(name + ".actor.npz") as handle:
            self.np.savez_compressed(
                handle, vector=vector, layout_json=self.np.asarray(json.dumps(layout))
            )
        files["actor_named_parameter_sha256"] = self.geometry.hash_actor(state.model.policy.actor)
        files["actor"] = {
            "path": name + ".actor.npz",
            "sha256": digest(self.runner.store.root / (name + ".actor.npz")),
        }
        files["environment"] = state.env.get_wrapper_attr("protocol_manifest")()
        return files

    def training_callback(self, state: TaskState) -> Any:
        base = importlib.import_module("stable_baselines3.common.callbacks").BaseCallback
        runner = self.runner
        started = runner.budget.monotonic()
        initial_steps = int(state.model.num_timesteps)

        def on_step(callback: Any) -> bool:
            runner.check("training")
            steps = int(callback.model.num_timesteps)
            if steps % 1000 == 0:
                diagnostics = {}
                for key, value in callback.model.logger.name_to_value.items():
                    if key.startswith("train/"):
                        try:
                            numeric = float(value)
                        except (TypeError, ValueError):
                            continue
                        if math.isfinite(numeric):
                            diagnostics[key] = numeric
                episodes = list(callback.model.ep_info_buffer or [])
                returns = [float(row["r"]) for row in episodes if "r" in row]
                record = {
                    "task": state.task,
                    "training_steps": steps,
                    "utc": runner.budget.now().isoformat(),
                    "diagnostics": diagnostics,
                    "training_episode_return_mean": sum(returns) / len(returns)
                    if returns
                    else None,
                    "training_episode_samples": len(returns),
                    "ledger": dict(runner.ledger),
                    "visit_steps_per_second": (steps - initial_steps)
                    / max(runner.budget.monotonic() - started, 0.000001),
                }
                runner.progress_rows.append(record)
                runner.store.json("progress.json", runner.progress_rows)
                print(json.dumps(record, sort_keys=True), flush=True)
            # Normal last transitions must reach replay and the optimizer update.
            return True

        callback_type = type("DevelopmentProgress", (base,), {"_on_step": on_step})
        return callback_type(verbose=0)

    def evaluation_env(self, state: TaskState) -> Any:
        self.runner.check("evaluation")
        streams = SeedStreams(**self.runner.plan.development_streams(state.task))
        bank = state.goals.for_tuning()[0]
        env = self.pilot.make_env(
            state.task,
            streams.constructor,
            bank=bank,
            streams=streams,
            goal_order=evaluation_order(bank, self.runner.plan.evaluation_episodes),
        )
        return self.runner.bind_env(env, "evaluation")

    def episode_record(self, env: Any) -> dict[str, Any]:
        return evaluation_record(env)

    def preflight(self, states: list[TaskState]) -> None:
        for state in states:
            self.runner.check("preflight")
            streams = SeedStreams(**self.runner.plan.development_streams(state.task))
            env = self.pilot.make_env(
                state.task, streams.constructor, bank=state.goals.for_tuning()[0], streams=streams
            )
            self.runner.bind_env(env, "preflight")
            try:
                observation, _ = env.reset()
                action, _ = state.model.predict(observation, deterministic=True)
                _, reward, terminated, truncated, info = env.step(action)
                self.runner.manifest.setdefault("preflight_rows", []).append(
                    {
                        "task": state.task,
                        "steps": 1,
                        "return": float(reward),
                        "success": bool(info.get("success", False)),
                        "complete_episode": bool(terminated or truncated),
                    }
                )
            finally:
                env.close()
            # Save and inspect the full zero-step learner, without constructing a fourth SAC.
            files = self.save(state, f"preflight/{state.task}")
            path = self.runner.store.root / files[".zip"]["path"]
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None or "policy.pth" not in archive.namelist():
                    raise ValueError("invalid source checkpoint archive")
        self.runner.manifest["preflight_checks"] = {
            "reset_step_all_tasks": True,
            "checkpoint_zip_integrity": True,
            "source_learning_performed": False,
            "new_sac_instances": len(states),
        }

    def close(self) -> None:
        for state in self.runner.states.values():
            state.env.close()


def validate_approval(plan: Any, approval: dict[str, Any]) -> None:
    if approval.get("approved") is not True or approval.get("plan_sha256") != plan.sha256:
        raise ValueError("explicit approval of this exact plan hash is required")
    if approval.get("authorization_url") != (
        "https://github.com/Spiph/UWSH-nethack/issues/37#issuecomment-6744072078"
    ):
        raise ValueError("approval must cite the actual user allocation decision")
    if approval.get("code_sha256") != code_hashes():
        raise ValueError("approved code hashes differ from the runtime files")
    if (
        not isinstance(approval.get("code_commit"), str)
        or re.fullmatch(r"[0-9a-f]{40}", approval["code_commit"]) is None
    ):
        raise ValueError("approval requires a full code commit reference")
    for key, expected in (
        ("session_started_utc", plan.session_started_utc),
        ("deadline_utc", plan.deadline_utc),
    ):
        if parse_deadline(approval.get(key, "")) != expected:
            raise ValueError("approval session/cutoff mismatch")
    if Path(approval.get("output", "")) != plan.output_path:
        raise ValueError("approval output path mismatch")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args(argv)
    plan_type = importlib.import_module("scripts.metaworld_development_plan").DevelopmentPlan
    plan = plan_type.load(args.config)
    if not args.run:
        print(json.dumps({"plan_sha256": plan.sha256, "worst_case": plan.worst_case()}, indent=2))
        return 0
    if args.approval is None:
        parser.error("--run requires --approval for the frozen plan")
    approval = json.loads(args.approval.read_text())
    result = DevelopmentRunner(plan, approval).run()
    print(json.dumps({"status": result["status"], "ledger": result["ledger"]}, indent=2))
    return 0 if result["status"].startswith("COMPLETE_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
