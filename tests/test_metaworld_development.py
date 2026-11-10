from __future__ import annotations

import json
import pickle
import random
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Literal

import numpy as np
import pytest
import torch

from scripts.metaworld_deadlines import RunDeadline
from scripts.metaworld_development import (
    ArtifactStore,
    DevelopmentRunner,
    LearnerRNG,
    Runtime,
    Stopped,
    TaskState,
    capture_rng,
    code_hashes,
    restore_rng,
    validate_approval,
)
from scripts.metaworld_protocol import Goal, GoalBank, GoalManifest


class Clock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 11, 9, 16, tzinfo=timezone.utc)
        self.tick = 0.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.tick

    def jump(self, utc: float, monotonic: float = 0.01) -> None:
        self.wall += timedelta(seconds=utc)
        self.tick += monotonic


class Plan:
    tasks = ("reach-v3", "push-v3", "pick-place-v3")
    milestones = (100000, 250000)
    evaluation_episodes = 20
    success_threshold = 16
    max_episode_steps = 500
    max_training_steps = 750000
    max_eval_steps = 60000
    max_preflight_steps = 1000
    preflight_seconds = 900
    reserve_seconds = 1800
    artifact_cap_bytes = 2**31
    sha256 = "frozen-test-plan"

    def __init__(self, output: Path, clock: Clock) -> None:
        self.output_path = output
        self.session_started_utc = clock.wall - timedelta(minutes=25)
        self.deadline_utc = clock.wall + timedelta(hours=5, minutes=35)
        self.training_deadline_utc = self.deadline_utc - timedelta(minutes=30)

    def validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        return {"test_plan": True, "output": str(self.output_path)}

    def ledger(self) -> dict[str, Any]:
        return {
            "source_training_steps": self.max_training_steps,
            "development_evaluation_steps": self.max_eval_steps,
            "preflight_steps": self.max_preflight_steps,
            "source_policy_starts": 3,
        }

    def schedule(self) -> tuple[tuple[str, int], ...]:
        return tuple((task, stage) for stage in self.milestones for task in self.tasks)


def approval(plan: Plan) -> dict[str, Any]:
    return {
        "approved": True,
        "plan_sha256": plan.sha256,
        "code_sha256": code_hashes(),
        "code_commit": "a" * 40,
        "authorization_url": "https://github.com/Spiph/UWSH-nethack/issues/37#issuecomment-6744072078",
        "session_started_utc": plan.session_started_utc.isoformat(),
        "deadline_utc": plan.deadline_utc.isoformat(),
        "output": str(plan.output_path),
    }


def goals(task: str) -> GoalManifest:
    return GoalManifest(
        tuple(
            GoalBank(
                split,
                1,
                tuple(Goal(task, "eA==", (float(index + offset),)) for index in range(count)),
            )
            for split, count, offset in (("source", 50, 0), ("development", 20, 100))
        )
    )


class FakeRuntime:
    def __init__(self, runner: DevelopmentRunner) -> None:
        self.runner = runner
        clock_callback: Any = runner.budget.now
        self.clock = clock_callback.__self__
        self.early_pass: set[str] = set()
        self.interrupt: str | None = None
        self.never_pass = False
        self.calls: list[tuple[str, int, bool]] = []
        self.saved: list[tuple[str, int, int, int]] = []
        self.instances: dict[str, Any] = {}

    def prepare(self, task: str) -> TaskState:
        self.runner.start_policy()
        runtime = self

        class Model:
            def __init__(self) -> None:
                self.num_timesteps = 0
                self.replay: list[int] = []
                self.optimizer = {"updates": 0}
                self.critic = object()
                self.entropy = object()

            def learn(self, increment: int, *, reset_num_timesteps: bool, callback: Any) -> None:
                frozen = json.loads((runtime.runner.store.root / "frozen_launch.json").read_text())
                assert set(frozen["goal_manifests"]) == set(runtime.runner.plan.tasks)
                assert len(list((runtime.runner.store.root / "initial").glob("*.npz"))) == 3
                runtime.calls.append((task, increment, reset_num_timesteps))
                runtime.runner.check("training")
                amount = 1 if runtime.interrupt == "training" else increment
                if (
                    runtime.runner.ledger["source_training_steps"] + amount
                    > runtime.runner.plan.max_training_steps
                ):
                    raise Stopped("source_training_steps_limit")
                self.num_timesteps += amount
                runtime.runner.ledger["source_training_steps"] += amount
                if runtime.interrupt == "training":
                    runtime.clock.jump(30000)
                self.train()
                self.replay.append(self.num_timesteps)

            def train(self) -> None:
                self.optimizer["updates"] += 1

            def predict(self, observation: Any, **kwargs: Any) -> tuple[int, None]:
                return 0, None

        model = Model()
        self.instances[task] = model
        env = SimpleNamespace(close=lambda: None)
        return TaskState(
            task,
            model,
            env,
            goals(task),
            SimpleNamespace(activate=nullcontext),
            {"initial_actor_sha256": "shared"},
        )

    def save_initial(self, state: TaskState) -> None:
        with self.runner.store.open(f"initial/{state.task}.npz") as handle:
            handle.write(b"initial actor")

    def preflight(self, states: list[TaskState]) -> None:
        for _ in states:
            self.runner.check("preflight")
            self.runner.ledger["preflight_steps"] += 1
        if self.interrupt == "preflight":
            self.clock.jump(901)

    def save(self, state: TaskState, name: str) -> dict[str, Any]:
        model = state.model
        assert "train" not in vars(model)
        self.saved.append((state.task, model.num_timesteps, id(model.replay), id(model.optimizer)))
        for suffix in (".zip", ".replay.pkl", ".rng.pkl", ".actor.npz"):
            with self.runner.store.open(name + suffix) as handle:
                handle.write(str(model.num_timesteps).encode())
        return {"checkpoint": name + ".zip"}

    def training_callback(self, state: TaskState) -> None:
        return None

    @contextmanager
    def preserve_rng(self) -> Any:
        caller = capture_rng(torch, False)
        try:
            yield
        finally:
            restore_rng(caller, torch, False)

    def evaluation_env(self, state: TaskState) -> Any:
        runtime = self
        bank = state.goals.for_tuning()[0]

        class Env:
            def __init__(self) -> None:
                self.episode = -1
                self.bank = bank
                self.length = 0

            def reset(self) -> tuple[int, dict[str, Any]]:
                self.episode += 1
                self.length = 0
                return 0, {}

            def step(self, action: Any) -> tuple[int, float, bool, bool, dict[str, Any]]:
                self.length += 1
                if runtime.interrupt == "evaluation":
                    runtime.clock.jump(30000)
                passing = not runtime.never_pass and (
                    state.task in runtime.early_pass or state.model.num_timesteps >= 250000
                )
                return 0, 1.0, False, self.length == 500, {"success": passing and self.episode < 16}

            def close(self) -> None:
                pass

        env = Env()
        return self.runner.bind_env(env, "evaluation")

    def episode_record(self, env: Any) -> dict[str, Any]:
        return {"goal_id": env.bank.goals[env.episode].goal_id, "goal_bank_sha256": env.bank.sha256}

    def close(self) -> None:
        pass


def runner(
    tmp_path: Path, *, early: set[str] | None = None, interrupt: str | None = None
) -> tuple[DevelopmentRunner, Clock]:
    clock = Clock()
    plan = Plan(tmp_path / "run", clock)
    budget = RunDeadline(20100, plan.deadline_utc, now=clock.now, monotonic=clock.monotonic)

    def factory(engine: DevelopmentRunner) -> FakeRuntime:
        runtime = FakeRuntime(engine)
        runtime.early_pass = early or set()
        runtime.interrupt = interrupt
        return runtime

    return DevelopmentRunner(plan, approval(plan), budget=budget, runtime_factory=factory), clock


def test_cumulative_round_robin_keeps_same_replay_optimizer_and_entropy(tmp_path: Path) -> None:
    engine, _ = runner(tmp_path)
    result = engine.run()
    assert result["status"] == "COMPLETE_PASS"
    assert engine.runtime.calls == [
        (task, amount, False) for amount in (100000, 150000) for task in engine.plan.tasks
    ]
    assert result["ledger"]["source_training_steps"] == 750000
    assert result["ledger"]["development_evaluation_steps"] == 60000
    assert result["ledger"]["source_policy_starts"] == 3
    assert result["ledger"]["preflight_steps"] == 3
    for task, model in engine.runtime.instances.items():
        saved = [row for row in engine.runtime.saved if row[0] == task]
        assert len({row[2] for row in saved}) == len({row[3] for row in saved}) == 1
        assert model.replay == [100000, 250000]
        assert model.optimizer["updates"] == 2
    assert (engine.store.root / "frozen_launch.json").is_file()


def test_earliest_passing_checkpoint_is_complete_and_skips_later_visit(tmp_path: Path) -> None:
    engine, _ = runner(tmp_path, early={"reach-v3"})
    result = engine.run()
    assert result["status"] == "COMPLETE_PASS"
    assert result["ledger"]["source_training_steps"] == 600000
    assert result["ledger"]["development_evaluation_steps"] == 50000
    assert engine.states["reach-v3"].metadata["earliest_passing_steps"] == 100000
    assert len(engine.states["reach-v3"].checkpoints) == 1
    assert any(row["status"] == "SKIPPED_PLANNED_EARLY_STOP" for row in result["queue"])


@pytest.mark.parametrize("phase", ["training", "evaluation", "preflight"])
def test_clock_cutoff_preserves_partial_artifacts_and_outcomes(tmp_path: Path, phase: str) -> None:
    engine, _ = runner(tmp_path, interrupt=phase)
    result = engine.run()
    assert result["status"] in ("INCOMPLETE", "ERROR")
    assert (engine.store.root / "development_manifest.json").is_file()
    if phase == "training":
        checkpoint = engine.states["reach-v3"].checkpoints[0]
        assert checkpoint["partial"]
        assert result["ledger"]["source_training_steps"] == 1
        assert (engine.store.root / checkpoint["artifacts"]["checkpoint"]).is_file()
    elif phase == "evaluation":
        row = engine.states["reach-v3"].checkpoints[0]["evaluation"][0]
        assert row["length"] == 1 and row["complete_episode"] is False
        assert result["ledger"]["development_evaluation_steps"] == 1
    else:
        assert result["stop_reason"] == "preflight_deadline"
        assert engine.runtime.calls == []


def test_reserve_latches_across_clock_rollback_but_evaluation_remains_allowed(
    tmp_path: Path,
) -> None:
    engine, clock = runner(tmp_path)
    clock.wall = engine.plan.training_deadline_utc
    with pytest.raises(Stopped, match="training_reserve"):
        engine.check("training")
    clock.jump(-3600)
    with pytest.raises(Stopped, match="training_reserve"):
        engine.check("training")
    engine.check("evaluation")


@pytest.mark.parametrize("bucket", ["training", "evaluation", "preflight"])
def test_step_caps_guard_before_extra_transition(tmp_path: Path, bucket: str) -> None:
    engine, _ = runner(tmp_path)
    key = {
        "training": "source_training_steps",
        "evaluation": "development_evaluation_steps",
        "preflight": "preflight_steps",
    }[bucket]
    engine.ledger[key] = engine.plan.ledger()[key]
    called: list[bool] = []
    env = engine.bind_env(
        SimpleNamespace(reset=lambda: None, step=lambda action: called.append(True)), bucket
    )
    with pytest.raises(Stopped, match=key + "_limit"):
        env.step(0)
    assert called == []


def test_unapproved_or_mismatched_plan_cannot_create_output(tmp_path: Path) -> None:
    plan = Plan(tmp_path / "uncreated", Clock())
    for invalid in (
        {},
        {**approval(plan), "plan_sha256": "different"},
        {**approval(plan), "approved": False},
    ):
        with pytest.raises(ValueError, match="approval"):
            validate_approval(plan, invalid)
    assert not plan.output_path.exists()


def test_capsule_isolates_python_numpy_and_torch_between_visits_and_evaluation() -> None:
    original = capture_rng(torch, False)
    try:

        def draw() -> tuple[float, float, float]:
            return random.random(), float(np.random.random()), float(torch.rand(1).item())

        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        initial = capture_rng(torch, False)
        expected = [draw(), draw()]
        capsule = LearnerRNG(initial, torch, False)
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        caller = capture_rng(torch, False)
        with capsule.activate():
            first = draw()
        for _ in range(10):
            draw()
        with capsule.activate():
            second = draw()
        assert [first, second] == expected
        restore_rng(caller, torch, False)
        expected_caller = draw()
        restore_rng(caller, torch, False)
        with capsule.activate():
            draw()
        assert draw() == expected_caller
    finally:
        restore_rng(original, torch, False)


def test_storage_limit_keeps_metadata_reserve_and_never_exceeds_cap(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path / "files", 1000)
    with store.open("payload") as handle:
        handle.write(b"x" * 900)
        with pytest.raises(Stopped, match="artifact_storage_limit"):
            handle.write(b"x")
    store.json("failure.json", {"status": "INCOMPLETE"})
    assert store.size() <= 1000
    assert json.loads((store.root / "failure.json").read_text())["status"] == "INCOMPLETE"


def test_full_budget_failure_remains_complete_without_promoting_competence(tmp_path: Path) -> None:
    engine, _ = runner(tmp_path)
    original_factory = engine.runtime_factory

    def factory(engine: DevelopmentRunner) -> Any:
        runtime: Any = original_factory(engine)
        runtime.never_pass = True
        return runtime

    engine.runtime_factory = factory
    result = engine.run()
    assert result["status"] == "COMPLETE_FAIL"
    assert all(state.status == "FULL_BUDGET_FAIL" for state in engine.states.values())
    assert result["ledger"]["source_training_steps"] == 750000


def test_monotonic_exhaustion_rejects_runtime_initialization(tmp_path: Path) -> None:
    engine, clock = runner(tmp_path)
    clock.tick = engine.budget.seconds
    result = engine.run()
    assert result["status"] == "INCOMPLETE"
    assert engine.runtime is None
    assert result["ledger"]["source_policy_starts"] == 0


def test_approval_rejects_code_changes_or_missing_commit_reference(tmp_path: Path) -> None:
    plan = Plan(tmp_path / "run", Clock())
    invalid = approval(plan)
    invalid["code_sha256"] = {}
    with pytest.raises(ValueError, match="code hashes"):
        validate_approval(plan, invalid)
    invalid = approval(plan)
    invalid.pop("code_commit")
    with pytest.raises(ValueError, match="commit reference"):
        validate_approval(plan, invalid)


def test_plan_cli_does_not_import_rl_or_initialize_runtime() -> None:
    code = """
import sys
from scripts.metaworld_development import main
assert main(['--config', 'experiments/metaworld/development_2026-11-09.json', '--plan']) == 0
assert not {'torch', 'gymnasium', 'mujoco', 'stable_baselines3'}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "811000" in result.stdout


@pytest.mark.parametrize("order", ["C", "F"])
def test_large_protocol_five_array_roundtrips_through_capped_writer(
    tmp_path: Path, order: Literal["C", "F"]
) -> None:
    array = np.array(np.arange(40000, dtype=np.float64).reshape(200, 200), order=order)
    assert array.nbytes > 65536
    store = ArtifactStore(tmp_path / "array", 1024**2)
    with store.open("array.pkl") as handle:
        pickle.dump({"array": array}, handle, protocol=5)
    with (store.root / "array.pkl").open("rb") as handle:
        loaded = pickle.load(handle)
    np.testing.assert_array_equal(loaded["array"], array)
    assert loaded["array"].dtype == array.dtype
    assert store.size() > array.nbytes
    assert store.size() <= store.limit


def test_large_pickle_buffer_rejects_quota_before_writing_payload(tmp_path: Path) -> None:
    array = np.arange(40000, dtype=np.float64)
    store = ArtifactStore(tmp_path / "overrun", 128 * 1024)
    with (
        store.open("partial.pkl") as handle,
        pytest.raises(Stopped, match="artifact_storage_limit"),
    ):
        pickle.dump(array, handle, protocol=5)
    assert store.size() < array.nbytes
    assert store.size() <= store.limit - store.reserve
    store.json("failure.json", {"status": "INCOMPLETE"})
    assert store.size() <= store.limit


def test_typed_multidimensional_memoryview_is_charged_by_bytes(tmp_path: Path) -> None:
    array = np.arange(32, dtype=np.float64).reshape(4, 8)
    store = ArtifactStore(tmp_path / "typed", 200)
    exporter: Any = array
    view = memoryview(exporter)
    assert len(view) == 4
    assert array.nbytes == 256
    with (
        store.open("typed.bin") as handle,
        pytest.raises(Stopped, match="artifact_storage_limit"),
    ):
        handle.write(view)
    assert store.size() == 0


def test_progress_refreshes_cumulative_total_before_persist_and_keeps_history_immutable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    engine, clock = runner(tmp_path)
    # One preserved preflight step plus three replacement steps are already charged.
    engine.ledger.update(preflight_steps=4, total_agent_steps=4, source_policy_starts=6)
    model = SimpleNamespace(
        num_timesteps=0, logger=SimpleNamespace(name_to_value={}), ep_info_buffer=[]
    )
    protocol: dict[str, Any] = {"episode_history": []}
    state: Any = SimpleNamespace(
        task="reach-v3",
        model=model,
        env=SimpleNamespace(get_wrapper_attr=lambda name: lambda: protocol),
    )

    class FakeBaseCallback:
        def __init__(self, verbose: int = 0) -> None:
            pass

        def _on_step(self) -> bool:
            return True

        def on_step(self) -> bool:
            return self._on_step()

    module = ModuleType("stable_baselines3.common.callbacks")
    module.__dict__["BaseCallback"] = FakeBaseCallback
    monkeypatch.setitem(sys.modules, module.__name__, module)
    runtime = Runtime.__new__(Runtime)
    runtime.runner = engine
    callback = runtime.training_callback(state)
    callback.model = model

    engine.ledger["source_training_steps"] = model.num_timesteps = 2000
    clock.tick = 1
    assert callback.on_step() is True
    first = engine.progress_rows[0]["ledger"]
    assert first["source_training_steps"] == 2000
    assert first["preflight_steps"] == 4
    assert first["total_agent_steps"] == 2004
    assert first["source_policy_starts"] == 6
    # No checkpoint/manifest persistence has refreshed the mutable live ledger.
    assert engine.ledger["total_agent_steps"] == 4

    engine.ledger["source_training_steps"] = model.num_timesteps = 3000
    engine.ledger["development_evaluation_steps"] = 500
    assert callback.on_step() is True
    assert engine.progress_rows[1]["ledger"]["total_agent_steps"] == 3504
    assert first["total_agent_steps"] == 2004
    assert first["development_evaluation_steps"] == 0
    saved = json.loads((engine.store.root / "progress.json").read_text())
    assert [row["ledger"]["total_agent_steps"] for row in saved] == [2004, 3504]
