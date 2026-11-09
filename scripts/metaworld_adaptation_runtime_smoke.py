"""Opt-in issue-49 mechanical validation; planning and refusal import no RL stack.

Preparing this harness does not authorize its execution. An approved attempt is
one fresh, CPU-only process, after source development exits. Synthetic transitions
are mechanical fixtures, never benchmark transfer evidence or library replicates.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib
import json
import os
import pickle
import re
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

ENTERED_MONOTONIC = time.monotonic()
ENTERED_UTC = datetime.now(timezone.utc)
ARMS = ("learned", "random", "full", "scratch")
ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "experiments/metaworld/adaptation_native_smoke_candidate.json"
PLAN: dict[str, Any] = {
    "schema_version": 1,
    "decision_issue": 49,
    "purpose": "synthetic_native_mechanical_validation_only",
    "approved": False,
    "cpu_only": True,
    "torch_threads": 1,
    "window_seconds": 300,
    "outer_timeout_seconds": 300,
    "outer_kill_cleanup_seconds": 5,
    "attempts": 1,
    "automatic_retries": 0,
    "source_process_must_have_exited": True,
    "sac_initializations": 8,
    "standalone_frozen_actors": 2,
    "synthetic_steps": 120,
    "optimizer_steps_maximum": 312,
    "benchmark_source_steps": 0,
    "benchmark_target_steps": 0,
    "benchmark_final_steps": 0,
    "artifact_bytes": 32 * 1024 * 1024,
    "arms": list(ARMS),
    "initial_steps_per_arm": 20,
    "continuation_steps_per_branch": 5,
    "observation_dimensions": 39,
    "action_dimensions": 4,
    "hidden_layers": [8, 8],
    "episode_timeout": 7,
    "activation": "relu",
    "fixture_seed": 49001,
    "learner_stream_seed": 49002,
    "random_control_seed": 49003,
    "synthetic_estimator": "flat_population_pca",
    "synthetic_basis_rank": 2,
    "reference_policy_id": "synthetic-source-0",
    "config": {
        "learning_rate": 0.0003,
        "buffer_size": 64,
        "learning_starts": 4,
        "batch_size": 4,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_update_interval": 1,
    },
    "output": (
        "/home/icannon/code/UWSH-nethack/artifacts/metaworld-adaptation-native-smoke-2026-11-09"
    ),
}


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_hashes() -> dict[str, str]:
    return {
        path.name: file_hash(path) for path in sorted((ROOT / "scripts").glob("metaworld_*.py"))
    }


def load_plan(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text())
    if canonical_hash(data) != canonical_hash(PLAN):
        raise ValueError("the plan differs from the exact issue-49 candidate")
    return data


def _utc(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("approval needs fixed timezone-aware session timestamps")
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("session timestamps must include a timezone")
    return result.astimezone(timezone.utc)


def validate_approval(
    plan: dict[str, Any],
    approval: dict[str, Any],
    *,
    current_hashes: dict[str, str],
    head: str,
    source_alive: Callable[[int], bool],
    now: datetime,
) -> datetime:
    """Pure gate: false approval fails before runtime imports or output creation."""
    if approval.get("approved") is not True:
        raise ValueError("issue 49 is UNAPPROVED; no native validation is authorized")
    expected_keys = {
        "approved",
        "decision_issue",
        "authorization_url",
        "plan_sha256",
        "code_commit",
        "code_sha256",
        "session_started_utc",
        "deadline_utc",
        "source_process_pid",
        "source_process_exited",
        "source_lifecycle_file",
        "source_lifecycle_sha256",
        "output",
        "attempt",
        "automatic_retries",
    }
    if set(approval) != expected_keys:
        raise ValueError("missing or unexpected approval fields")
    if (
        type(approval["decision_issue"]) is not int
        or approval["decision_issue"] != 49
        or not isinstance(approval["authorization_url"], str)
        or re.fullmatch(
            r"https://github\.com/Spiph/UWSH-nethack/issues/49#issuecomment-[0-9]+",
            approval["authorization_url"],
        )
        is None
        or approval["plan_sha256"] != canonical_hash(plan)
        or approval["code_sha256"] != current_hashes
        or not isinstance(approval["code_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", approval["code_commit"]) is None
        or approval["code_commit"] != head
        or approval["output"] != plan["output"]
        or type(approval["attempt"]) is not int
        or approval["attempt"] != 1
        or type(approval["automatic_retries"]) is not int
        or approval["automatic_retries"] != 0
        or approval["source_process_exited"] is not True
        or not isinstance(approval["source_lifecycle_file"], str)
        or not Path(approval["source_lifecycle_file"]).is_absolute()
        or not isinstance(approval["source_lifecycle_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", approval["source_lifecycle_sha256"]) is None
    ):
        raise ValueError("approval does not bind the exact plan, code and one-attempt allowance")
    pid = approval["source_process_pid"]
    if type(pid) is not int or pid <= 0 or source_alive(pid):
        raise ValueError("the designated source-development process must have exited")
    start, cutoff = _utc(approval["session_started_utc"]), _utc(approval["deadline_utc"])
    if cutoff - start != timedelta(seconds=300) or not start <= now < cutoff:
        raise ValueError("approval needs a current, fixed, inclusive five-minute session")
    return cutoff


def source_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def anchored_allowance_seconds(cutoff: datetime, entered_utc: datetime) -> float:
    """An immutable entry-UTC allowance; later UTC rollback cannot renew setup."""
    if entered_utc.tzinfo is None or entered_utc.utcoffset() is None:
        raise ValueError("the process entry UTC anchor must include a timezone")
    seconds = min(300.0, (cutoff - entered_utc).total_seconds())
    if seconds <= 0:
        raise ValueError("the fixed native-validation window expired before process entry")
    return seconds


def claim_attempt_output(output: Path) -> None:
    """The directory itself latches an attempt, even if subsequent imports fail."""
    output.mkdir(parents=True, exist_ok=False)


class Limits:
    def __init__(self, budget: Any) -> None:
        self.budget = budget
        self.ledger = dict.fromkeys(
            (
                "sac_initializations",
                "standalone_frozen_actors",
                "synthetic_step_attempts",
                "synthetic_steps",
                "optimizer_step_attempts",
                "optimizer_steps",
                "reset_calls",
                "benchmark_source_steps",
                "benchmark_target_steps",
                "benchmark_final_steps",
            ),
            0,
        )

    def charge(self, key: str, maximum: int, phase: str) -> None:
        self.budget.require(phase)
        if self.ledger[key] >= maximum:
            raise RuntimeError(f"{key} limit reached before {phase}")
        self.ledger[key] += 1


class SmokeRunner:
    """Control flow injected with a fake runtime in ordinary unit tests."""

    def __init__(
        self,
        plan: dict[str, Any],
        approval: dict[str, Any],
        budget: Any,
        store: Any,
        runtime_factory: Callable[[dict[str, Any], Limits, Any], Any],
    ) -> None:
        self.plan, self.approval, self.budget, self.store = plan, approval, budget, store
        self.limits = Limits(budget)
        self.factory = runtime_factory
        self.manifest: dict[str, Any] = {
            "schema_version": 1,
            "status": "RUNNING",
            "validation_complete": False,
            "plan": copy.deepcopy(plan),
            "plan_sha256": canonical_hash(plan),
            "approval": copy.deepcopy(approval),
            "ledger": self.limits.ledger,
            "arms": {},
            "errors": [],
            "interpretation": "mechanical integration only",
            "process_entered_at_utc": ENTERED_UTC.isoformat(),
            "process_entered_monotonic": ENTERED_MONOTONIC,
        }

    def persist(self) -> None:
        self.budget.record(self.manifest)
        self.manifest["artifact_bytes_before_manifest_write"] = self.store.size()
        self.store.json("smoke_manifest.json", self.manifest)

    def run(self) -> dict[str, Any]:
        runtime: Any = None
        try:
            self.persist()
            self.budget.require("native_imports_and_fixture_setup")
            runtime = self.factory(self.plan, self.limits, self.store)
            self.manifest["runtime"] = runtime.metadata()
            handles, records = {}, {}
            for arm in ARMS:
                self.limits.charge("sac_initializations", 8, f"construct_{arm}")
                handles[arm], records[arm] = runtime.construct(arm)
            self.budget.require("shared_preflight_before_all_learning")
            runtime.preflight(handles, records)
            self.manifest["preflight"] = records
            self.manifest["shared_preflight_passed"] = True
            for name in ("reference", "source"):
                self.limits.charge("standalone_frozen_actors", 2, f"frozen_{name}_constructor")
                runtime.frozen_prediction_check(name, handles["full"])
            self.persist()
            for arm in ARMS:
                row: dict[str, Any] = {"status": "RUNNING"}
                self.manifest["arms"][arm] = row
                handle = handles[arm]
                self._learn(runtime, handle, 20, reset=True)
                runtime.check_trained_actor(handle, arm)
                runtime.check_timeouts(handle)
                capsule = runtime.save_checkpoint(handle, arm)
                before = runtime.snapshot(handle)
                runtime.restore_capsule(handle, capsule)
                self._learn(runtime, handle, 5, reset=False)
                original = runtime.snapshot(handle)
                runtime.persist_branch(handle, arm, "original")
                self.limits.charge("sac_initializations", 8, f"reload_{arm}")
                clone = runtime.reload(arm, capsule)
                runtime.restore_capsule(clone, capsule)
                restored_before = runtime.snapshot(clone)
                if before != restored_before:
                    raise RuntimeError(f"{arm}: checkpoint state differs before continuation")
                # Snapshot and serialization may use RNG in upstream versions;
                # install the saved capsule again immediately before learning.
                runtime.restore_capsule(clone, capsule)
                self._learn(runtime, clone, 5, reset=False)
                restored = runtime.snapshot(clone)
                runtime.persist_branch(clone, arm, "restored")
                if original != restored:
                    raise RuntimeError(f"{arm}: original and restored continuation states differ")
                row.update(
                    status="PASS",
                    checkpoint_state=before,
                    continued_state=original,
                    restored_state=restored,
                )
                self.persist()
            if any(
                self.limits.ledger[key] != value
                for key, value in (
                    ("sac_initializations", 8),
                    ("standalone_frozen_actors", 2),
                    ("synthetic_step_attempts", 120),
                    ("synthetic_steps", 120),
                    ("optimizer_step_attempts", 312),
                    ("optimizer_steps", 312),
                )
            ):
                raise RuntimeError("completed mechanical check has unexpected resource counters")
            self.budget.require("checks_finished")
            self.manifest["status"] = "PASS"
            self.manifest["validation_complete"] = True
        except BaseException as error:
            self.manifest["status"] = (
                "INTERRUPTED" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "ERROR"
            )
            self.manifest["validation_complete"] = False
            self.manifest["stop_exception_type"] = type(error).__name__
            self.manifest["errors"].append(traceback.format_exc())
            if runtime is not None:
                try:
                    # Retention is cleanup, never a new constructor/update/step.
                    runtime.retain_failure()
                except BaseException:
                    self.manifest["errors"].append(traceback.format_exc())
        finally:
            if runtime is not None:
                try:
                    runtime.close()
                except BaseException:
                    self.manifest["status"] = "ERROR"
                    self.manifest["validation_complete"] = False
                    self.manifest["errors"].append(traceback.format_exc())
            self.persist()
            # A write/cleanup crossing cutoff cannot leave a final passing claim.
            self.budget.record(self.manifest)
            if not self.manifest.get("budget_compliant", False):
                self.manifest["validation_complete"] = False
                self.store.json("smoke_manifest.json", self.manifest)
        return self.manifest

    def _learn(self, runtime: Any, handle: Any, steps: int, *, reset: bool) -> None:
        self.budget.require("learn_entry")
        before = self.limits.ledger["synthetic_steps"]
        runtime.learn(handle, steps, reset=reset)
        if self.limits.ledger["synthetic_steps"] - before != steps:
            raise RuntimeError("native learn collected a different number of synthetic transitions")


class NativeRuntime:
    """Never instantiated in preparation/tests: real imports occur only here."""

    def __init__(self, plan: dict[str, Any], limits: Limits, store: Any) -> None:
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("set CUDA_VISIBLE_DEVICES empty before native imports")
        self.plan, self.limits, self.store = plan, limits, store
        limits.budget.require("numpy_import")
        self.np = importlib.import_module("numpy")
        limits.budget.require("torch_import")
        self.torch = importlib.import_module("torch")
        limits.budget.require("torch_thread_configuration")
        self.torch.set_num_threads(1)
        self.torch.set_num_interop_threads(1)
        limits.budget.require("gymnasium_import")
        self.gym = importlib.import_module("gymnasium")
        limits.budget.require("sb3_import")
        self.sb3 = importlib.import_module("stable_baselines3")
        self.vec = importlib.import_module("stable_baselines3.common.vec_env")
        self.adapter = importlib.import_module("scripts.metaworld_sac_adapter")
        self.protocol = importlib.import_module("scripts.metaworld_protocol")
        self.development = importlib.import_module("scripts.metaworld_development")
        self.families = self._families()
        self.handles: list[dict[str, Any]] = []
        self.initial_buffers: dict[str, str] = {}
        self.initial_actors: dict[str, str] = {}
        self.limits.budget.require("native_setup_finished")

    def metadata(self) -> dict[str, Any]:
        versions = {
            name: version(name) for name in ("torch", "numpy", "gymnasium", "stable-baselines3")
        }
        if (
            versions["stable-baselines3"] != "2.6.0"
            or versions["gymnasium"] != "1.1.1"
            or versions["numpy"] != "1.26.4"
            or versions["torch"] != "2.3.0+cu121"
        ):
            raise ValueError("native smoke requires the pinned measurement runtime")
        return {
            "versions": versions,
            "device": "cpu",
            "torch_threads": self.torch.get_num_threads(),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "environment_wrappers": ["explicit_DummyVecEnv", "synthetic_raw_env"],
            "monitor_wrapper": False,
            "benchmark_environment_imports": False,
            "family_sha256": self.families["learned"].family_sha256,
            "control_sha256": self.families["random"].sha256,
        }

    def _families(self) -> dict[str, Any]:
        basis = importlib.import_module("scripts.metaworld_basis")
        controls = importlib.import_module("scripts.metaworld_adaptation_contract")
        shapes = [
            ("latent_pi.0.weight", [8, 39]),
            ("latent_pi.0.bias", [8]),
            ("latent_pi.2.weight", [8, 8]),
            ("latent_pi.2.bias", [8]),
            ("mu.weight", [4, 8]),
            ("mu.bias", [4]),
            ("log_std.weight", [4, 8]),
            ("log_std.bias", [4]),
        ]
        layout: list[dict[str, Any]] = []
        offset = 0
        for name, shape in shapes:
            width = int(self.np.prod(shape))
            layout.append(dict(name=name, shape=shape, start=offset, end=offset + width))
            offset += width
        rng = self.np.random.default_rng(self.plan["fixture_seed"])
        initial = rng.uniform(-0.01, 0.01, offset).astype(self.np.float32).astype(self.np.float64)
        for item in layout:
            if item["name"] in ("latent_pi.0.bias", "latent_pi.2.bias"):
                initial[item["start"] : item["end"]] = 0.25
        # Positive hidden biases avoid an accidental all-dead ReLU fixture. This
        # is a declared mechanical construction, never a competent source actor.
        vectors = [initial + rng.uniform(-0.002, 0.002, offset) for _ in range(3)]
        source_goal = self.protocol.Goal(
            "synthetic-v3", base64.b64encode(b"synthetic-source-only").decode(), (0.0,)
        )
        dev_goal = self.protocol.Goal(
            "synthetic-v3", base64.b64encode(b"synthetic-dev-only").decode(), (1.0,)
        )
        source = self.protocol.GoalBank("source", 49, (source_goal,))
        goals = self.protocol.GoalManifest(
            (source, self.protocol.GoalBank("development", 50, (dev_goal,)))
        )
        rows = []
        for index, vector in enumerate(vectors):
            vector = vector.astype(self.np.float32).astype(self.np.float64)
            rows.append(
                basis.SourcePolicy(
                    policy_id=f"synthetic-source-{index}",
                    training_run_id=f"synthetic-fixture-{index}",
                    checkpoint_step=0,
                    task_class="synthetic-v3",
                    library_id="synthetic-mechanical-fixture",
                    initialization_condition="shared_init",
                    training_goal_bank_sha256=source.sha256,
                    actor_sha256=basis.actor_hash_from_vector(vector, layout, "float32"),
                    initial_actor_sha256=basis.actor_hash_from_vector(initial, layout, "float32"),
                    rng_streams=self.protocol.SeedStreams.from_seed(
                        600 + index, actor_seed=601
                    ).__dict__,
                    vector=vector,
                    initial_vector=initial,
                    layout=layout,
                    parameter_dtype="float32",
                )
            )
        parent = basis.fit_source_basis(
            rows,
            goals,
            basis.BasisFitSpec(
                "flat_population_pca",
                "final_weights",
                "raw",
                "fixed_predeclared",
                rank=2,
                reference_rule="source_policy",
                reference_policy_id="synthetic-source-0",
            ),
            task_split=basis.TaskSplitManifest(
                ("synthetic-v3",), "development_only", "issue49-synthetic"
            ),
            relations=basis.TaskRelations(),
        )
        control = controls.RandomControlFamily.from_basis(
            parent, seed=self.plan["random_control_seed"]
        )
        self.store.json(
            "synthetic_family_provenance.json",
            {
                "source_fixture_only": True,
                "parent_metadata": parent.metadata,
                "parent_sha256": parent.sha256,
                "control_metadata": control.metadata,
                "control_sha256": control.sha256,
                "fixture_generator_seed": self.plan["fixture_seed"],
                "fixture_generator_revision": "bounded_numpy_weights_positive_hidden_bias_v1",
            },
        )
        return {"learned": parent, "random": control, "full": parent, "scratch": parent}

    def _environment(self) -> dict[str, Any]:
        np, gym, limits = self.np, self.gym, self.limits

        def initialize(env: Any) -> None:
            gym.Env.__init__(env)
            env.observation_space = gym.spaces.Box(-10, 10, shape=(39,), dtype=np.float32)
            env.action_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
            env.position, env.episode_step, env.episode_index = 0.0, 0, 0

        def observation(env: Any) -> Any:
            return np.asarray(np.linspace(-0.1, 0.1, 39) + env.position, dtype=np.float32)

        def reset(env: Any, *, seed: int | None = None, options: Any = None) -> tuple[Any, Any]:
            limits.budget.require("synthetic_reset")
            limits.ledger["reset_calls"] += 1
            gym.Env.reset(env, seed=seed)
            env.position = float(env.np_random.normal(0, 0.01))
            env.episode_step = 0
            env.episode_index += 1
            return observation(env), {}

        def step(env: Any, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
            limits.charge("synthetic_step_attempts", 120, "synthetic_step")
            env.position += float(np.mean(action)) * 0.001
            env.episode_step += 1
            reward = -float(np.square(action).mean())
            result: tuple[Any, float, bool, bool, dict[str, Any]] = (
                observation(env),
                reward,
                False,
                env.episode_step == 7,
                {},
            )
            limits.ledger["synthetic_steps"] += 1
            return result

        # The optional Gym class is resolved only after the approval gate. type()
        # keeps module import and static checking independent of that runtime.
        synthetic_env = type(
            "SyntheticEnv",
            (gym.Env,),
            {
                "metadata": {"render_modes": []},
                "__init__": initialize,
                "reset": reset,
                "step": step,
            },
        )

        limits.budget.require("synthetic_environment_constructor")
        raw = synthetic_env()
        vector = self.vec.DummyVecEnv([lambda: raw])
        return {"raw": raw, "env": vector}

    def construct(self, arm: str) -> tuple[dict[str, Any], dict[str, Any]]:
        handle = self._environment()
        self.handles.append(handle)
        self.limits.budget.require("native_sac_constructor")
        handle["model"], record = self.adapter.build_matched_sac(
            handle["env"],
            self.protocol.SeedStreams.from_seed(self.plan["learner_stream_seed"]),
            family=self.families[arm],
            arm=arm,
            activation="relu",
            config=self.adapter.MatchedSACConfig(**self.plan["config"]),
            device="cpu",
        )
        handle["arm"] = arm
        handle["initial_rng"] = self.development.capture_rng(self.torch, False)
        actor = handle["model"].actor
        self.initial_buffers[arm] = self._hash(dict(actor.named_buffers()))
        self.initial_actors[arm] = self._hash(dict(actor.named_parameters()))
        return handle, record

    def preflight(self, handles: dict[str, Any], records: dict[str, Any]) -> None:
        self.adapter.validate_matched_preflight(records)
        distributions = importlib.import_module("stable_baselines3.common.distributions")
        features = self.torch.linspace(-0.5, 0.5, 78).reshape(2, 39)
        reference = handles["full"]["model"].actor
        if len({self._hash(handle["initial_rng"]) for handle in handles.values()}) != 1:
            raise RuntimeError("learning arms do not share the declared initial training RNG state")
        for arm, handle in handles.items():
            actor = handle["model"].actor
            if not isinstance(actor.action_dist, distributions.SquashedDiagGaussianDistribution):
                raise RuntimeError("native squashed diagonal Gaussian was replaced")
            actions, log_prob = actor.action_log_prob(features)
            if not self.torch.isfinite(actions).all() or not self.torch.isfinite(log_prob).all():
                raise RuntimeError("nonfinite native action/log-prob path")
            if arm in ("learned", "random"):
                for actual, expected in zip(
                    actor.get_action_dist_params(features)[:2],
                    reference.get_action_dist_params(features)[:2],
                    strict=True,
                ):
                    self.torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                parameters = list(actor.parameters())
                optimized = [p for group in actor.optimizer.param_groups for p in group["params"]]
                if (
                    len(parameters) != 1
                    or len(optimized) != 1
                    or parameters[0] is not optimized[0]
                    or parameters[0] is not actor.coefficient_actor.coefficients
                ):
                    raise RuntimeError("coefficient actor optimizer owns extra parameters")
            if str(handle["model"].device) != "cpu":
                raise RuntimeError("native learner is not on CPU")
        if self.initial_actors["full"] == self.initial_actors["scratch"]:
            raise RuntimeError("scratch must have different initial actor weights")

    def frozen_prediction_check(self, name: str, full: dict[str, Any]) -> None:
        factory = (
            self.adapter.FrozenReferencePolicy
            if name == "reference"
            else self.adapter.FrozenSourcePolicy
        )
        policy = factory(
            full["raw"].observation_space,
            full["raw"].action_space,
            family=self.families["learned"],
            activation="relu",
            device="cpu",
        )
        observation = self.np.zeros((2, 39), dtype=self.np.float32)
        expected, _ = full["model"].predict(observation, deterministic=True)
        actual, _ = policy.predict(observation, deterministic=True)
        self.np.testing.assert_array_equal(actual, expected)
        if any(parameter.requires_grad for parameter in policy.actor.parameters()):
            raise RuntimeError("frozen prediction actor has trainable parameters")

    @contextmanager
    def _optimizer_guards(self, model: Any) -> Iterator[None]:
        patched = []
        for optimizer in (model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer):
            original = optimizer.step

            def checked(*args: Any, _original: Any = original, **kwargs: Any) -> Any:
                self.limits.charge("optimizer_step_attempts", 312, "optimizer_step")
                result = _original(*args, **kwargs)
                self.limits.ledger["optimizer_steps"] += 1
                return result

            prior = vars(optimizer).get("step")
            patched.append((optimizer, prior))
            optimizer.step = checked
        try:
            yield
        finally:
            for optimizer, prior in patched:
                if prior is None:
                    del optimizer.step
                else:
                    optimizer.step = prior

    def learn(self, handle: dict[str, Any], steps: int, *, reset: bool) -> None:
        deadlines = importlib.import_module("scripts.metaworld_deadlines")
        model = handle["model"]
        if reset:
            # Constructor/diagnostic calls and previously exercised arms cannot
            # advance this learner's declared initial global training stream.
            self.development.restore_rng(handle["initial_rng"], self.torch, False)
        with deadlines.guard_model(model, self.limits.budget), self._optimizer_guards(model):
            model.learn(total_timesteps=steps, reset_num_timesteps=reset, progress_bar=False)

    def check_trained_actor(self, handle: dict[str, Any], arm: str) -> None:
        actor = handle["model"].actor
        if self._hash(dict(actor.named_buffers())) != self.initial_buffers[arm]:
            raise RuntimeError("frozen actor buffers changed during native learning")
        if self._hash(dict(actor.named_parameters())) == self.initial_actors[arm]:
            raise RuntimeError("native actor parameters did not change")
        if arm in ("learned", "random"):
            gradient = actor.coefficient_actor.coefficients.grad
            if (
                gradient is None
                or not self.torch.isfinite(gradient).all()
                or gradient.abs().sum() == 0
            ):
                raise RuntimeError(
                    "native restricted actor lacks a finite nonzero coefficient gradient"
                )
            if actor.coefficient_actor.metadata["coefficient_count"] != 2:
                raise RuntimeError("the frozen map or fixture coordinate count changed")

    def check_timeouts(self, handle: dict[str, Any]) -> None:
        replay = handle["model"].replay_buffer
        indices = self.np.flatnonzero(replay.timeouts[: replay.pos, 0] == 1)
        if len(indices) != 2 or not self.np.all(replay.dones[indices] == 1):
            raise RuntimeError("the 20-step fixture must retain two native timeout transitions")
        samples = replay._get_samples(indices)
        if not self.torch.equal(samples.dones, self.torch.zeros_like(samples.dones)):
            raise RuntimeError("time-limit transitions failed to bootstrap")

    def _environment_state(self, handle: dict[str, Any]) -> dict[str, Any]:
        raw, env = handle["raw"], handle["env"]
        return copy.deepcopy(
            {
                "position": raw.position,
                "episode_step": raw.episode_step,
                "episode_index": raw.episode_index,
                "raw_rng": raw.np_random.bit_generator.state,
                "raw_rng_seed": raw.np_random_seed,
                "action_rng": raw.action_space.np_random.bit_generator.state,
                "vec": {
                    key: getattr(env, key)
                    for key in (
                        "buf_obs",
                        "buf_dones",
                        "buf_rews",
                        "buf_infos",
                        "reset_infos",
                        "_seeds",
                        "_options",
                    )
                },
            }
        )

    def _restore_environment(self, handle: dict[str, Any], state: dict[str, Any]) -> None:
        raw, env = handle["raw"], handle["env"]
        for key in ("position", "episode_step", "episode_index"):
            setattr(raw, key, state[key])
        raw.np_random.bit_generator.state = copy.deepcopy(state["raw_rng"])
        raw._np_random_seed = state["raw_rng_seed"]
        raw.action_space.np_random.bit_generator.state = copy.deepcopy(state["action_rng"])
        for key, value in state["vec"].items():
            setattr(env, key, copy.deepcopy(value))

    def save_checkpoint(self, handle: dict[str, Any], arm: str) -> dict[str, Any]:
        self.limits.budget.require("checkpoint_serialization")
        model = handle["model"]
        with self.store.open(f"{arm}/model.zip") as writer:
            model.save(writer)
        with self.store.open(f"{arm}/replay.pkl") as writer:
            model.save_replay_buffer(writer)
        capsule = {
            "rng": self.development.capture_rng(self.torch, False),
            "environment": self._environment_state(handle),
            "policy_kwargs": dict(model.policy_kwargs),
        }
        with self.store.open(f"{arm}/capsule.pkl") as writer:
            pickle.dump(capsule, writer, protocol=5)
        self.limits.budget.require("checkpoint_serialization_finished")
        # Exercise the persisted RNG/environment capsule, rather than silently
        # restoring only its still-resident pre-serialization Python object.
        with (self.store.root / arm / "capsule.pkl").open("rb") as reader:
            persisted: dict[str, Any] = pickle.load(reader)
        return persisted

    def restore_capsule(self, handle: dict[str, Any], capsule: dict[str, Any]) -> None:
        self._restore_environment(handle, capsule["environment"])
        self.development.restore_rng(capsule["rng"], self.torch, False)

    def reload(self, arm: str, capsule: dict[str, Any]) -> dict[str, Any]:
        handle = self._environment()
        self.handles.append(handle)
        family = self.families[arm]
        if arm != "random":
            family.validate_integrity()
        else:
            family.deployment_arrays()
        kwargs = self.adapter.verified_policy_kwargs(capsule["policy_kwargs"], family=family)
        self.limits.budget.require("native_sac_reload_constructor")
        handle["model"] = self.sb3.SAC.load(
            self.store.root / arm / "model.zip",
            env=handle["env"],
            device="cpu",
            force_reset=False,
            custom_objects={"policy_kwargs": kwargs},
        )
        handle["model"].load_replay_buffer(self.store.root / arm / "replay.pkl")
        handle["arm"] = arm
        return handle

    def _state(self, handle: dict[str, Any]) -> dict[str, Any]:
        model, replay = handle["model"], handle["model"].replay_buffer
        return {
            "policy": model.policy.state_dict(),
            "actor_optimizer": model.actor.optimizer.state_dict(),
            "critic_optimizer": model.critic.optimizer.state_dict(),
            "entropy_optimizer": model.ent_coef_optimizer.state_dict(),
            "log_entropy_coefficient": model.log_ent_coef,
            "replay": {
                key: getattr(replay, key)
                for key in (
                    "observations",
                    "next_observations",
                    "actions",
                    "rewards",
                    "dones",
                    "timeouts",
                    "pos",
                    "full",
                    "buffer_size",
                    "n_envs",
                    "handle_timeout_termination",
                )
            },
            "counters_and_observations": {
                key: getattr(model, key)
                for key in (
                    "num_timesteps",
                    "_n_updates",
                    "_episode_num",
                    "_total_timesteps",
                    "_last_obs",
                    "_last_original_obs",
                    "_last_episode_starts",
                    "_current_progress_remaining",
                )
            },
            "environment": self._environment_state(handle),
            "rng": self.development.capture_rng(self.torch, False),
        }

    def _hash(self, value: Any) -> str:
        digest = hashlib.sha256()

        def visit(item: Any) -> None:
            if isinstance(item, self.torch.Tensor):
                visit(item.detach().cpu().numpy())
            elif isinstance(item, self.np.ndarray):
                digest.update(f"array:{item.dtype}:{item.shape}:".encode())
                digest.update(item.tobytes(order="C"))
            elif isinstance(item, dict):
                digest.update(b"mapping:")
                for key in sorted(item, key=str):
                    visit(key)
                    visit(item[key])
            elif isinstance(item, (list, tuple)):
                digest.update(f"sequence:{len(item)}:".encode())
                for part in item:
                    visit(part)
            else:
                digest.update(json.dumps(item, sort_keys=True, allow_nan=False).encode() + b";")

        visit(value)
        return digest.hexdigest()

    def snapshot(self, handle: dict[str, Any]) -> dict[str, str]:
        return {key: self._hash(value) for key, value in self._state(handle).items()}

    def persist_branch(self, handle: dict[str, Any], arm: str, name: str) -> None:
        self.limits.budget.require("continuation_state_retention")
        with self.store.open(f"{arm}/{name}_continued_state.pt") as writer:
            self.torch.save(self._state(handle), writer)

    def retain_failure(self) -> None:
        for index, handle in enumerate(self.handles):
            model = handle.get("model")
            if model is None:
                continue
            state = (
                self._state(handle)
                if model.num_timesteps
                else {
                    "policy": model.policy.state_dict(),
                    "num_timesteps": model.num_timesteps,
                    "actor_optimizer": model.actor.optimizer.state_dict(),
                    "critic_optimizer": model.critic.optimizer.state_dict(),
                    "entropy_optimizer": model.ent_coef_optimizer.state_dict(),
                    "log_entropy_coefficient": model.log_ent_coef,
                }
            )
            with self.store.open(f"failure_partial_{index}.pt") as writer:
                self.torch.save(state, writer)

    def close(self) -> None:
        for handle in self.handles:
            handle["env"].close()


def approved_launch(plan: dict[str, Any], approval: dict[str, Any]) -> dict[str, Any]:
    # All of this gate uses the standard library. In particular, refusal creates
    # no output and does not import Torch, SB3, Gym, NumPy or experiment modules.
    if approval.get("approved") is not True:
        raise ValueError("issue 49 is UNAPPROVED; native launch refused")
    cutoff = _utc(approval.get("deadline_utc"))
    seconds = anchored_allowance_seconds(cutoff, ENTERED_UTC)
    deadlines = importlib.import_module("scripts.metaworld_deadlines")
    budget = deadlines.RunDeadline(seconds, cutoff)
    budget.started_monotonic = ENTERED_MONOTONIC
    budget.started_utc = _utc(approval.get("session_started_utc"))
    budget.require("code_commit_check")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=3
    ).strip()
    budget.require("code_hash_and_approval_checks")
    validate_approval(
        plan,
        approval,
        current_hashes=code_hashes(),
        head=head,
        source_alive=source_process_alive,
        now=datetime.now(timezone.utc),
    )
    budget.require("frozen_checkout_check")
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "scripts"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=3,
    )
    tracked = subprocess.check_output(
        ["git", "ls-files", "scripts"], cwd=ROOT, text=True, timeout=3
    ).splitlines()
    if any(f"scripts/{name}" not in tracked for name in approval["code_sha256"]):
        raise ValueError("every approved runtime module must be present in the frozen commit")
    if any(name in sys.modules for name in ("torch", "gymnasium", "stable_baselines3")):
        raise ValueError("native validation requires a fresh process before all RL imports")
    budget.require("source_lifecycle_checks")
    lifecycle_gate = importlib.import_module("scripts.metaworld_source_lifecycle")
    lifecycle = lifecycle_gate.load_and_validate_source_lifecycle(
        Path(approval["source_lifecycle_file"]), approval["source_lifecycle_sha256"]
    )
    if lifecycle.get("original_pid") != approval["source_process_pid"]:
        raise ValueError("source lifecycle must identify the same original source process")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    budget.require("exclusive_attempt_claim")
    output = Path(plan["output"])
    # Even an import failure consumes the single attempt. The exclusive new
    # directory is claimed before NumPy/ArtifactStore or any RL runtime import.
    claim_attempt_output(output)
    try:
        budget.require("setup_before_artifact_store_import")
        store_type = importlib.import_module("scripts.metaworld_development").ArtifactStore
        store = store_type(output, plan["artifact_bytes"])
    except Exception:
        # No data work can occur in this startup fallback. Keep a bounded record
        # even if the normal capped writer's dependencies could not import.
        error = {
            "status": "ERROR",
            "validation_complete": False,
            "approval": approval,
            "error": traceback.format_exc()[-32768:],
        }
        payload = json.dumps(error, sort_keys=True).encode()
        if len(payload) <= plan["artifact_bytes"]:
            (output / "startup_failure.json").write_bytes(payload)
        raise
    store.json(
        "attempt_claim.json",
        {"approval": approval, "automatic_retries": 0, "source_lifecycle_verification": lifecycle},
    )
    runner = SmokeRunner(plan, approval, budget, store, NativeRuntime)
    runner.manifest["source_lifecycle_verification"] = lifecycle
    return runner.run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true", help="print the unapproved plan; no runtime")
    parser.add_argument("--config", type=Path, default=PLAN_PATH)
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args(argv)
    try:
        plan = load_plan(args.config)
        if args.plan:
            print(
                json.dumps(
                    {
                        "approved": False,
                        "plan": plan,
                        "plan_sha256": canonical_hash(plan),
                        "code_sha256": code_hashes(),
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0
        if args.approval is None:
            raise ValueError("an explicit issue-49 approval JSON is required")
        approval = json.loads(args.approval.read_text())
        result = approved_launch(plan, approval)
        print(json.dumps({"status": result["status"], "ledger": result["ledger"]}, sort_keys=True))
        return 0 if result["validation_complete"] else 1
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
