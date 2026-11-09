from __future__ import annotations

import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from scripts import metaworld_competence as competence
from scripts.metaworld_deadlines import (
    BudgetExpired,
    RunDeadline,
    TrainingDeadline,
    guard_env,
    guard_model,
    parse_deadline,
)


class FakeClock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 11, 9, 12, tzinfo=timezone.utc)
        self.ticks = 0.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.ticks

    def advance(self, *, utc: float = 0, monotonic: float = 0) -> None:
        self.wall += timedelta(seconds=utc)
        self.ticks += monotonic

    def budget(self, *, seconds: float = 100, cutoff: float = 200) -> RunDeadline:
        return RunDeadline(
            seconds,
            self.wall + timedelta(seconds=cutoff),
            now=self.now,
            monotonic=self.monotonic,
        )


def test_utc_jump_stops_training_on_the_very_next_callback_and_latches() -> None:
    clock = FakeClock()
    budget = clock.budget()
    callback = TrainingDeadline(budget)
    assert callback.on_step()
    clock.advance(utc=201, monotonic=0.001)
    assert not callback.on_step()
    assert callback.cutoff_reason == "absolute_utc_deadline"
    assert callback.stopped_for_budget
    clock.advance(utc=-400)
    assert not budget.allowed("evaluation")
    with pytest.raises(BudgetExpired, match="absolute_utc_deadline"):
        budget.require("reinitialization")


def test_monotonic_exhaustion_survives_a_backward_wall_clock_jump() -> None:
    clock = FakeClock()
    budget = clock.budget()
    clock.advance(utc=-1000, monotonic=100)
    assert not budget.allowed("training")
    assert budget.cutoff_reason == "monotonic_run_budget"


def test_training_reserve_stops_training_but_permits_evaluation() -> None:
    clock = FakeClock()
    budget = clock.budget()
    callback = TrainingDeadline(budget, reserve_seconds=20)
    clock.advance(monotonic=80)
    assert not callback.on_step()
    assert callback.cutoff_reason == "training_evaluation_reserve"
    assert budget.cutoff_reason is None
    assert budget.allowed("evaluation")


@pytest.mark.parametrize("stamp", ["nonsense", "2026-11-09", "2026-11-09T20:00:00"])
def test_ambiguous_or_invalid_utc_inputs_are_rejected(stamp: str) -> None:
    with pytest.raises(ValueError, match="timezone"):
        parse_deadline(stamp)


def test_offset_input_is_normalized_to_utc() -> None:
    assert parse_deadline("2026-11-09T07:00:00-05:00") == parse_deadline("2026-11-09T12:00:00Z")


def test_cutoff_at_finalization_prevents_a_false_complete_status() -> None:
    clock = FakeClock()
    budget = clock.budget()
    manifest = {"status": "COMPLETE_PASS", "validation_complete": True}
    clock.advance(utc=200, monotonic=2)
    budget.record(manifest)
    assert manifest["status"] == "STOPPED_AT_BUDGET"
    assert manifest["budget_compliant"] is False
    assert manifest["validation_complete"] is False
    assert manifest["utc_elapsed_seconds"] == 200
    assert manifest["elapsed_seconds"] == 2


def arguments(output: Path) -> Any:
    return competence.parse_args(
        [
            "--output",
            str(output),
            "--deadline-utc",
            "2026-11-09T12:03:20Z",
            "--seconds",
            "100",
            "--reserve-seconds",
            "10",
            "--device",
            "cpu",
            "--steps-per-policy",
            "4",
            "--evaluation-episodes",
            "1",
        ]
    )


def test_late_start_writes_rejection_without_loading_or_initializing_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = FakeClock()
    budget = clock.budget(cutoff=0)
    calls: list[str] = []
    monkeypatch.setattr(competence, "_execute", lambda *args: calls.append("executed"))
    assert competence._run(arguments(tmp_path), budget=budget) == 2
    assert calls == []
    result = json.loads((tmp_path / "competence_manifest.json").read_text())
    assert result["status"] == "STOPPED_AT_BUDGET"
    assert result["cutoff_reason"] == "absolute_utc_deadline"
    assert result["incomplete_tasks"] == list(competence.TASKS)
    assert result["missing_evaluation_episodes"] == {task: [0] for task in competence.TASKS}


def fake_runtime(monkeypatch: pytest.MonkeyPatch, clock: FakeClock, jump_phase: str) -> list[str]:
    """Exercise the real runner using tiny CPU actors and fake environments/SAC."""
    events: list[str] = []

    class FakeEnv:
        def reset(self, **kwargs: Any) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
            events.append("reset")
            return np.zeros(39), {}

        def step(
            self, action: Any
        ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
            events.append("step")
            if jump_phase == "evaluation":
                clock.advance(utc=300, monotonic=0.01)
            return np.zeros(39), 1.0, jump_phase == "completion", False, {"success": 1}

        def close(self) -> None:
            events.append("close")

    def make_env(*args: Any, budget: RunDeadline, **kwargs: Any) -> Any:
        events.append("make_env")
        return guard_env(FakeEnv(), budget)

    class FakeCallback(TrainingDeadline):
        pass

    class FakeSAC:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            events.append("model")
            self.policy = SimpleNamespace(actor=torch.nn.Linear(39, 4))
            self.num_timesteps = 0

        def learn(self, steps: int, *, callback: TrainingDeadline, **kwargs: Any) -> None:
            events.append("learn")
            for _ in range(steps):
                self.num_timesteps += 1
                if jump_phase == "training":
                    clock.advance(utc=300, monotonic=0.01)
                if not callback.on_step():
                    break
                if jump_phase == "optimizer":
                    clock.advance(utc=300, monotonic=0.01)
                self.train()

        def train(self) -> None:
            events.append("optimizer_update")

        def save(self, name: str) -> None:
            assert "train" not in vars(self)
            events.append("save")
            Path(name + ".zip").write_text("partial checkpoint")

        def predict(self, *args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.float64], None]:
            events.append("predict")
            return np.zeros(4), None

    module = ModuleType("stable_baselines3")
    module.__dict__["SAC"] = FakeSAC
    monkeypatch.setitem(sys.modules, "stable_baselines3", module)
    monkeypatch.setitem(
        sys.modules,
        "scripts.metaworld_hour_pilot",
        SimpleNamespace(
            make_env=make_env,
            DeadlineCallback=FakeCallback,
        ),
    )
    monkeypatch.setattr(competence, "_runtime_package_hashes", dict)
    return events


@pytest.mark.parametrize("phase", ["training", "optimizer", "evaluation"])
def test_runner_retains_partial_checkpoint_and_reports_missing_work_after_wall_jump(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, phase: str
) -> None:
    clock = FakeClock()
    budget = clock.budget()
    events = fake_runtime(monkeypatch, clock, phase)
    assert competence._run(arguments(tmp_path), budget=budget) == 2
    result = json.loads((tmp_path / "competence_manifest.json").read_text())
    assert (tmp_path / "policies/reach-v3.zip").read_text() == "partial checkpoint"
    assert (tmp_path / "policies/reach-v3.actor.npz").is_file()
    assert result["status"] == "STOPPED_AT_BUDGET"
    assert result["budget_compliant"] is False
    assert result["cutoff_reason"] == "absolute_utc_deadline"
    assert result["elapsed_seconds"] == pytest.approx(0.01)
    assert result["utc_elapsed_seconds"] == 300
    assert result["validation_complete"] is False
    assert events.count("learn") == 1
    assert events.count("save") == 1
    assert result["missing_evaluation_episodes"] == {task: [0] for task in competence.TASKS}
    if phase in ("training", "optimizer"):
        assert result["task_runs"][0]["environment_steps_completed"] == 1
        assert result["evaluations"] == []
        assert "predict" not in events
    else:
        assert result["task_runs"][0]["environment_steps_completed"] == 4
        assert len(result["evaluations"]) == 1
        assert result["evaluations"][0]["length"] == 1
        assert result["evaluations"][0]["complete_episode"] is False
        assert (tmp_path / "evaluations.csv").is_file()
        assert events.count("step") == 1


def test_training_reserve_uses_the_shorter_absolute_window() -> None:
    clock = FakeClock()
    budget = clock.budget(seconds=3600, cutoff=60)
    callback = TrainingDeadline(budget, reserve_seconds=300)
    assert not callback.on_step()
    assert callback.cutoff_reason == "training_evaluation_reserve"
    assert budget.cutoff_reason is None
    assert budget.allowed("evaluation")


def test_record_uses_the_same_clock_snapshot_for_finish_and_compliance() -> None:
    start = datetime(2026, 11, 9, 12, tzinfo=timezone.utc)
    deadline = start + timedelta(seconds=100)
    readings = iter([start, deadline - timedelta(seconds=0.1), deadline + timedelta(seconds=0.1)])
    budget = RunDeadline(200, deadline, now=lambda: next(readings), monotonic=lambda: 0)
    result: dict[str, Any] = {"status": "COMPLETE_PASS"}
    budget.record(result)
    finish = datetime.fromisoformat(result["finished_at_utc"])
    assert result["budget_compliant"] == (finish < deadline)
    budget.record(result)
    assert result["status"] == "STOPPED_AT_BUDGET"
    assert result["budget_compliant"] is False


@pytest.mark.parametrize("method", ["reset", "step", "train"])
def test_work_boundaries_reject_late_wake_before_next_transition_or_update(method: str) -> None:
    clock = FakeClock()
    budget = clock.budget()
    calls: list[str] = []
    target = SimpleNamespace(
        reset=lambda: calls.append("reset"),
        step=lambda: calls.append("step"),
        train=lambda: calls.append("train"),
    )
    original_train = target.train
    guard_env(target, budget)
    with guard_model(target, budget):
        # The callback was permitted; suspension happens before the work call.
        assert TrainingDeadline(budget).on_step()
        clock.advance(utc=300, monotonic=0.01)
        with pytest.raises(BudgetExpired, match="absolute_utc_deadline"):
            getattr(target, method)()
    assert calls == []
    assert target.train is original_train


def test_initialization_failure_is_retained_as_error_not_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = FakeClock()

    def fail(*args: Any) -> None:
        raise RuntimeError("CUDA initialization failed")

    monkeypatch.setattr(competence, "_execute", fail)
    with pytest.raises(RuntimeError, match="CUDA initialization failed"):
        competence._run(arguments(tmp_path), budget=clock.budget())
    result = json.loads((tmp_path / "competence_manifest.json").read_text())
    assert result["status"] == "ERROR"
    assert result["errors"] == ["RuntimeError: CUDA initialization failed"]
    assert result["incomplete_tasks"] == list(competence.TASKS)


def load_pilot_without_runtime(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Load the actual callback and pilot orchestration against inert modules."""
    gym = ModuleType("gymnasium")
    spaces = ModuleType("gymnasium.spaces")
    callbacks = ModuleType("stable_baselines3.common.callbacks")
    sb3 = ModuleType("stable_baselines3")

    class FakeBaseCallback:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def on_step(self) -> Any:
            return self._on_step()

        def _on_step(self) -> bool:
            return True

    for module, attributes in (
        (gym, {"ObservationWrapper": object, "Env": object, "__version__": "test"}),
        (spaces, {"Box": object}),
        (callbacks, {"BaseCallback": FakeBaseCallback}),
        (sb3, {"SAC": object, "__version__": "test"}),
    ):
        for name, value in attributes.items():
            setattr(module, name, value)
    runtime_modules: dict[str, Any] = dict(
        (
            ("gymnasium", gym),
            ("gymnasium.spaces", spaces),
            ("metaworld", ModuleType("metaworld")),
            ("stable_baselines3", sb3),
            ("stable_baselines3.common.callbacks", callbacks),
            ("mujoco", SimpleNamespace(__version__="test")),
        )
    )
    for name, fake_module in runtime_modules.items():
        monkeypatch.setitem(sys.modules, name, fake_module)
    monkeypatch.delitem(sys.modules, "scripts.metaworld_hour_pilot", raising=False)
    pilot = importlib.import_module("scripts.metaworld_hour_pilot")
    # Ensure this inert import does not outlive this fixture.
    monkeypatch.setitem(sys.modules, "scripts.metaworld_hour_pilot", pilot)
    monkeypatch.setattr(pilot, "version", lambda name: "test")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return pilot


def test_actual_sb3_callback_adapter_stops_without_recursing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = load_pilot_without_runtime(monkeypatch)
    clock = FakeClock()
    callback = pilot.DeadlineCallback(clock.budget())
    assert callback.on_step()
    clock.advance(utc=300)
    assert callback.on_step() is False
    assert callback.cutoff_reason == "absolute_utc_deadline"


def test_pilot_late_start_retains_all_missing_source_and_evaluation_slots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pilot = load_pilot_without_runtime(monkeypatch)
    clock = FakeClock()
    budget = clock.budget(cutoff=0)
    monkeypatch.setattr(pilot, "RunDeadline", lambda *args: budget)
    monkeypatch.setattr(pilot, "make_env", lambda *args, **kwargs: pytest.fail("late env init"))
    monkeypatch.setattr(torch, "set_num_threads", lambda *args: pytest.fail("late thread init"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("late CUDA query"))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: pytest.fail("late GPU init"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "--output",
            str(tmp_path),
            "--deadline-utc",
            "2026-11-09T12:03:20Z",
            "--device",
            "cpu",
        ],
    )
    assert pilot.main() == 2
    result = json.loads((tmp_path / "pilot_manifest.json").read_text())
    assert result["status"] == "STOPPED_AT_BUDGET"
    assert result["budget_compliant"] is False
    assert len(result["incomplete_source_runs"]) == 18
    assert result["validation_complete"] is False
    assert result["coverage_validation"]["complete"] is False
    assert len(result["coverage_validation"]["missing_evaluation_cells"]) == 90


def test_pilot_evaluation_retains_partial_episode_after_forward_clock_jump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pilot = load_pilot_without_runtime(monkeypatch)
    clock = FakeClock()
    budget = clock.budget()
    events: list[str] = []

    class Env:
        def reset(self, **kwargs: Any) -> tuple[npt.NDArray[np.float64], dict[str, Any]]:
            return np.zeros(39), {}

        def step(
            self, action: Any
        ) -> tuple[npt.NDArray[np.float64], float, bool, bool, dict[str, Any]]:
            events.append("step")
            return np.zeros(39), 1, False, False, {"success": 1}

        def close(self) -> None:
            events.append("close")

    class Model:
        def predict(self, *args: Any, **kwargs: Any) -> tuple[npt.NDArray[np.float64], None]:
            if events:
                clock.advance(utc=300, monotonic=0.01)
            return np.zeros(4), None

    monkeypatch.setattr(pilot, "make_env", lambda *args, **kwargs: guard_env(Env(), budget))
    rows = pilot.eval_model(Model(), "reach-v3", 0, "shared_init", "id", "trained", 2, budget)
    assert len(rows) == 1
    assert rows[0]["length"] == 1
    assert rows[0]["complete_episode"] is False
    assert rows[0]["cutoff_reason"] == "absolute_utc_deadline"
    assert events == ["step", "close"]


def test_every_competence_manifest_write_rejects_late_completion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = FakeClock()
    budget = clock.budget()
    fake_runtime(monkeypatch, clock, "completion")
    writes: list[dict[str, Any]] = []
    original_write = competence._atomic_json
    original_summary = competence.summarize_competence
    jumped = False

    def capture(path: Path, payload: Any) -> None:
        writes.append(json.loads(json.dumps(payload)))
        original_write(path, payload)

    def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
        nonlocal jumped
        result = original_summary(manifest)
        if result["all_tasks_complete"] and not jumped:
            clock.advance(utc=300, monotonic=0.01)
            jumped = True
        return result

    monkeypatch.setattr(competence, "_atomic_json", capture)
    monkeypatch.setattr(competence, "summarize_competence", summarize)
    assert competence._run(arguments(tmp_path), budget=budget) == 2
    assert jumped
    assert not any(str(row["status"]).startswith("COMPLETE") for row in writes)
    assert writes[-1]["status"] == "STOPPED_AT_BUDGET"
    assert writes[-1]["competence"]["all_tasks_complete"] is True
    assert writes[-1]["validation_complete"] is False


def test_pilot_finalizer_cannot_persist_completion_if_summary_crosses_cutoff(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pilot = load_pilot_without_runtime(monkeypatch)
    clock = FakeClock()
    budget = clock.budget(seconds=2700)
    writes: list[dict[str, Any]] = []
    summary_calls = 0
    original_write = pilot.atomic_json

    def capture(path: Path, payload: Any) -> None:
        writes.append(json.loads(json.dumps(payload)))
        original_write(path, payload)

    def summarize(manifest: dict[str, Any]) -> dict[str, Any]:
        nonlocal summary_calls
        summary_calls += 1
        if summary_calls == 2:
            clock.advance(utc=300, monotonic=0.01)
        return {"coverage": {"complete": True}, "initialization": {"valid": True}}

    env = SimpleNamespace(
        reset=lambda **kwargs: (np.zeros(39), {}),
        step=lambda action: (np.zeros(39), 0, False, False, {}),
        close=lambda: None,
        action_space=SimpleNamespace(shape=(4,), sample=lambda: np.zeros(4)),
    )
    # An inert empty source plan isolates status/finalization without any learning.
    monkeypatch.setattr(pilot, "REPLICATES", 0)
    monkeypatch.setattr(
        pilot,
        "SAC",
        lambda *args, **kwargs: SimpleNamespace(policy=SimpleNamespace(actor=object())),
    )
    monkeypatch.setattr(pilot, "verify_policy_symmetry", lambda *args: 0)
    monkeypatch.setattr(pilot, "make_env", lambda *args, **kwargs: env)
    monkeypatch.setattr(pilot, "RunDeadline", lambda *args: budget)
    monkeypatch.setattr(pilot, "summarize_manifest", summarize)
    monkeypatch.setattr(pilot, "atomic_json", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "--output",
            str(tmp_path),
            "--deadline-utc",
            "2026-11-09T12:03:20Z",
            "--device",
            "cpu",
        ],
    )
    assert pilot.main() == 2
    assert summary_calls == 2
    assert not any(str(row["status"]).startswith("COMPLETE") for row in writes)
    assert writes[-1]["status"] == "STOPPED_AT_BUDGET"
    assert writes[-1]["validation_complete"] is False
    assert writes[-1]["budget_compliant"] is False
