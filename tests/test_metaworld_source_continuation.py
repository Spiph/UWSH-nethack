"""Only fake learners/environments: no SAC, GPU, physics or policy files."""

from __future__ import annotations

import base64
import copy
import json
import subprocess
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts import metaworld_source_continuation as continuation
from scripts import metaworld_source_supervisor as supervisor
from scripts.metaworld_development import Stopped, TaskState
from scripts.metaworld_development_plan import DevelopmentPlan, approved_configuration
from scripts.metaworld_protocol import Goal, GoalBank, GoalManifest, derive_seed
from scripts.metaworld_source_extension_plan import (
    AUTHORIZATION_URL,
    ExtensionPlan,
    continuation_schedule,
    evaluation_complete,
    extension_configuration,
    stable_hash,
    validate_parent,
)


def evaluation(passed: bool = False) -> list[dict[str, Any]]:
    return [
        dict(
            episode=index,
            length=500,
            complete_episode=True,
            terminated=False,
            truncated=True,
            success=passed and index < 16,
        )
        for index in range(20)
    ]


def parent() -> dict[str, Any]:
    base = DevelopmentPlan(approved_configuration())
    tasks: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for task in base.tasks:
        source_streams, development_streams = (
            base.source_streams(task),
            base.development_streams(task),
        )
        source = GoalBank(
            "source",
            derive_seed(source_streams["task_bank"], "source"),
            tuple(
                Goal(
                    task,
                    base64.b64encode(f"fake-source-{task}-{index}".encode()).decode(),
                    (float(index),),
                )
                for index in range(50)
            ),
        )
        development = GoalBank(
            "development",
            derive_seed(development_streams["task_bank"], "development"),
            tuple(
                Goal(
                    task,
                    base64.b64encode(f"fake-dev-{task}-{index}".encode()).decode(),
                    (float(index + 100),),
                )
                for index in range(20)
            ),
        )
        manifests[task] = GoalManifest((source, development)).to_dict()
        seeds = list(range(20))

        def decorated_evaluation(
            passed: bool, bank: GoalBank = development, values: list[int] = seeds
        ) -> list[dict[str, Any]]:
            return [
                {
                    **row,
                    "goal_id": bank.goals[index].goal_id,
                    "goal_bank_sha256": bank.sha256,
                    "episode_seed": values[index],
                    "goal_index": index,
                }
                for index, row in enumerate(evaluation(passed))
            ]

        rows = []
        for target in (100000,) if task == "reach-v3" else (100000, 250000):
            rows.append(
                dict(
                    target_steps=target,
                    actual_steps=target,
                    partial=False,
                    evaluation=decorated_evaluation(task == "reach-v3"),
                    evaluation_complete=True,
                    passed=task == "reach-v3",
                    artifacts={},
                )
            )
        tasks[task] = dict(
            status="PASS" if task == "reach-v3" else "FULL_BUDGET_FAIL",
            actual_training_steps=rows[-1]["actual_steps"],
            metadata={
                "source_streams": source_streams,
                "development_streams": development_streams,
                "evaluation_goal_order": list(range(20)),
                "evaluation_episode_seeds": seeds,
                "initial_actor_sha256": "a" * 64,
                "initialization": {
                    "rng_streams": source_streams,
                    "sac_constructor_seed": source_streams["critic"],
                    "actor_state_sha256": "b" * 64,
                    "critic_state_sha256": "c" * 64,
                    "critic_target_state_sha256": "c" * 64,
                    "training_seed_applied_after_initialization": source_streams["training"],
                    "replay_buffer_handle_timeout_termination": True,
                },
            },
            checkpoints=rows,
        )
    return dict(
        phase="FINISHED",
        status="COMPLETE_FAIL",
        errors=[],
        original_scientific_plan=base.to_dict(),
        tasks=tasks,
        queue=[],
        goal_manifests=manifests,
        ledger=dict(
            source_training_steps=600000,
            development_evaluation_steps=50000,
            preflight_steps=4,
            source_policy_starts=6,
            target_steps=0,
            final_test_steps=0,
            failed_step_calls=0,
            optimizer_updates=597000,
            optimizer_update_attempts=597000,
            retries=1,
            technical_retries=1,
            total_agent_steps=650004,
        ),
    )


class Budget:
    seconds = 1000000
    started_utc = datetime(2026, 11, 10, tzinfo=timezone.utc)

    def now(self) -> datetime:
        return self.started_utc

    def monotonic(self) -> float:
        return 0.0

    def require(self, phase: str) -> None:
        pass

    def allowed(self, phase: str, *, reserve_seconds: float = 0) -> bool:
        return True

    def record(self, target: dict[str, Any]) -> None:
        target.update(budget_compliant=True, finished_at_utc=self.started_utc.isoformat())


class FakeModel:
    def __init__(self, runner: Any, steps: int) -> None:
        self.runner, self.num_timesteps = runner, steps
        self._n_updates = steps - 1000
        self.learn_calls: list[int] = []

    def learn(self, steps: int, **kwargs: Any) -> None:
        self.runner.events.append("learn")
        self.learn_calls.append(steps)
        assert kwargs["reset_num_timesteps"] is False
        self.num_timesteps += steps
        self.runner.ledger["source_training_steps"] += steps

    def train(self, **kwargs: Any) -> None:
        self.runner.events.append("pending_train")
        self._n_updates += 1


class FakeRuntime:
    def __init__(self, runner: Any) -> None:
        self.runner = runner

    def prepare(self, task: str) -> TaskState:
        self.runner.start_policy()
        self.runner.events.append("prepare_" + task)
        row = self.runner.plan.parent["tasks"][task]
        metadata = dict(
            physical_training_steps=row.get("fake_physical_steps", row["actual_training_steps"]),
            pending_optimizer_updates=row.get("fake_pending", 0),
        )
        return TaskState(
            task,
            FakeModel(self.runner, row["actual_training_steps"]),
            None,
            cast(Any, None),
            SimpleNamespace(activate=nullcontext),
            metadata,
            checkpoints=copy.deepcopy(row["checkpoints"]),
        )

    def preserve_rng(self) -> Any:
        return nullcontext()

    def training_callback(self, state: Any) -> None:
        return None

    def save(self, state: Any, name: str) -> dict[str, bool]:
        return {"fake_only": True}

    def close(self) -> None:
        self.runner.events.append("closed")


class FakeRunner(continuation.ContinuationRunner):
    events: list[str]

    def evaluate(self, state: Any, row: dict[str, Any]) -> None:
        self.ledger["development_evaluation_steps"] += 10000
        row.update(evaluation=evaluation(True), evaluation_complete=True, passed=True)


def make_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data: dict[str, Any] | None = None,
    runtime: Any = FakeRuntime,
) -> FakeRunner:
    data = data or parent()
    binding = {
        "parent_manifest": data,
        "prior_artifact_bytes": 100,
        "inventories": [],
        "old_pid": 999999,
    }
    plan = ExtensionPlan(extension_configuration(), binding)
    plan.output_path = tmp_path / "new-artifacts"
    monkeypatch.setattr(continuation, "validate_extension_approval", lambda *args: None)
    monkeypatch.setattr(continuation, "verify_binding", lambda *args: None)
    runner = FakeRunner(plan, {}, runtime_factory=runtime, budget=Budget())
    runner.events = []
    return runner


def test_round_robin_extension_reuses_same_ids_and_stops_at_first_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = make_runner(tmp_path, monkeypatch)
    result = runner.run()
    assert result["status"] == "COMPLETE_PASS"
    assert runner.events[:2] == ["prepare_push-v3", "prepare_pick-place-v3"]
    assert result["additional_sac_reloads"] == 2 and result["ledger"]["source_policy_starts"] == 8
    assert result["ledger"]["source_training_steps"] == 1100000
    assert result["ledger"]["development_evaluation_steps"] == 70000
    assert result["tasks"]["reach-v3"]["actual_training_steps"] == 100000
    assert [runner.physical_steps(state) for state in runner.states.values()] == [500000, 500000]
    assert all(state.model.learn_calls == [250000] for state in runner.states.values())


def test_pending_only_visit_runs_one_update_after_all_recovery_and_no_extra_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = parent()
    push = data["tasks"]["push-v3"]
    push["fake_pending"] = 1
    push["status"] = "PENDING"
    data.update(status="INCOMPLETE", stop_reason="training_reserve")
    push["checkpoints"][-1].update(evaluation=[], evaluation_complete=False, partial=True)
    data["queue"] = [dict(task="push-v3", milestone=250000, status="TRAINING")]
    runner = make_runner(tmp_path, monkeypatch, data)
    result = runner.run()
    assert result["status"] == "COMPLETE_PASS"
    assert runner.events[:3] == ["prepare_push-v3", "prepare_pick-place-v3", "pending_train"]
    assert runner.states["push-v3"].model.learn_calls == []
    assert result["ledger"]["optimizer_updates"] == 597001
    assert runner.states["push-v3"].checkpoints[-1]["actual_steps"] == 250000


def test_physical_terminal_step_offset_never_grants_a_compensating_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = parent()
    push = data["tasks"]["push-v3"]
    push.update(actual_training_steps=249999, fake_physical_steps=250000)
    push["status"] = "PENDING"
    data.update(status="INCOMPLETE", stop_reason="training_reserve")
    push["checkpoints"][-1].update(
        actual_steps=249999, evaluation=[], evaluation_complete=False, partial=True
    )
    data["queue"] = [dict(task="push-v3", milestone=250000, status="TRAINING")]
    runner = make_runner(tmp_path, monkeypatch, data)
    result = runner.run()
    assert result["status"] == "COMPLETE_PASS"
    assert runner.states["push-v3"].model.learn_calls == []
    assert runner.states["push-v3"].checkpoints[-1]["actual_steps"] == 250000
    assert runner.states["push-v3"].checkpoints[-1]["model_timesteps"] == 249999


def test_second_recovery_failure_blocks_every_update_and_rollout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BadRecovery(FakeRuntime):
        def prepare(self, task: str) -> TaskState:
            if task == "pick-place-v3":
                self.runner.start_policy()
                raise ValueError("fake exact observation mismatch")
            return super().prepare(task)

    runner = make_runner(tmp_path, monkeypatch, runtime=BadRecovery)
    result = runner.run()
    assert result["status"] == "ERROR"
    assert "learn" not in runner.events and "pending_train" not in runner.events
    assert result["ledger"]["source_policy_starts"] == 8


def test_old_errors_or_user_interruption_never_auto_resume() -> None:
    base = ExtensionPlan(
        extension_configuration(), {"parent_manifest": parent(), "prior_artifact_bytes": 0}
    ).base
    for status, errors in (("ERROR", []), ("INCOMPLETE", ["KeyboardInterrupt"])):
        data = parent()
        data.update(status=status, errors=errors)
        with pytest.raises(ValueError):
            validate_parent(data, base)


def test_unfinished_visit_precedes_stage_round_robin_and_missing_old_eval_is_rejected() -> None:
    data = parent()
    data["tasks"]["pick-place-v3"]["checkpoints"][-1].update(evaluation_complete=False)
    data["queue"] = [dict(task="pick-place-v3", milestone=250000, status="TRAINING")]
    assert continuation_schedule(data, ("push-v3", "pick-place-v3"))[0] == ("pick-place-v3", 250000)
    data["tasks"]["push-v3"]["checkpoints"][0]["evaluation_complete"] = False
    with pytest.raises(ValueError, match="without a complete evaluation"):
        continuation_schedule(data, ("push-v3", "pick-place-v3"))
    assert evaluation_complete(parent()["tasks"]["reach-v3"]["checkpoints"][0])


def test_native_setup_hook_refuses_after_a_suspended_load_without_allocating() -> None:
    allocations: list[str] = []

    class FakeSAC:
        def _setup_model(self) -> None:
            allocations.append("allocation")

    def expired() -> None:
        raise Stopped("training_reserve")

    gated = continuation.gated_sac_type(FakeSAC, expired)
    with pytest.raises(Stopped, match="training_reserve"):
        gated()._setup_model()
    assert not allocations


def test_unvisited_source_without_checkpoint_fails_before_any_reload() -> None:
    data = parent()
    for task in ("push-v3", "pick-place-v3"):
        data["tasks"][task].update(status="PENDING", actual_training_steps=0, checkpoints=[])
    data.update(status="INCOMPLETE", stop_reason="training_reserve")
    data["ledger"].update(
        source_training_steps=100000, development_evaluation_steps=10000, total_agent_steps=110004
    )
    config = extension_configuration()
    binding = {
        "parent_manifest": data,
        "prior_artifact_bytes": 0,
        "inventories": [],
    }
    plan = ExtensionPlan(config, binding)
    with pytest.raises(ValueError, match="checkpoint beyond warmup"):
        continuation.selected_checkpoint(plan, "push-v3")


def test_zero_increment_evaluations_remain_available_during_training_reserve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = parent()
    for task in ("push-v3", "pick-place-v3"):
        data["tasks"][task]["checkpoints"][-1].update(evaluation=[], evaluation_complete=False)
        data["tasks"][task]["status"] = "PENDING"
    data.update(status="INCOMPLETE", stop_reason="training_reserve")

    class EnterReserve(FakeRuntime):
        def prepare(self, task: str) -> TaskState:
            state = super().prepare(task)
            if task == "pick-place-v3":
                self.runner.training_stopped = True
            return state

    runner = make_runner(tmp_path, monkeypatch, data, runtime=EnterReserve)
    result = runner.run()
    assert result["status"] == "COMPLETE_PASS"
    assert all(state.model.learn_calls == [] for state in runner.states.values())


def test_inconsistent_cumulative_source_ledger_blocks_all_learning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = parent()
    data["ledger"]["source_training_steps"] += 1
    data["ledger"]["total_agent_steps"] += 1
    runner = make_runner(tmp_path, monkeypatch, data)
    result = runner.run()
    assert result["status"] == "ERROR"
    assert "learn" not in runner.events and "pending_train" not in runner.events
    assert "physical source histories" in result["errors"][-1]


def test_child_monotonic_cutoff_cannot_be_renewed_or_nonfinite() -> None:
    binding: dict[str, Any] = {
        "supervisor_window": {
            "monotonic_cutoff": 200.0,
            "started_monotonic": 100.0,
            "started_at_utc": "2026-11-09T20:00:00+00:00",
            "deadline_utc": "2026-11-09T20:01:40+00:00",
        }
    }
    deadline = binding["supervisor_window"]["deadline_utc"]
    assert continuation.inherited_monotonic_cutoff(binding, 200.0, deadline) == 200.0
    for value in (None, 201.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="frozen supervisor"):
            continuation.inherited_monotonic_cutoff(binding, value, deadline)


def test_supervisor_records_noop_without_any_child_or_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = dict(extension_configuration(), output=str(tmp_path / "fake-only-artifacts"))
    monkeypatch.setattr(supervisor, "extension_configuration", lambda: config)
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            dict(
                approved=True,
                authorization_url=AUTHORIZATION_URL,
                extension_sha256=stable_hash(config),
                parent_process_identity={
                    "pid": 999999,
                    "start_ticks": 1,
                    "boot_id": "fake",
                    "command_sha256": "a" * 64,
                },
            )
        )
    )
    monkeypatch.setattr(supervisor, "process_alive", lambda pid: False)
    monkeypatch.setattr(supervisor, "process_identity", lambda pid: None)
    monkeypatch.setattr(
        supervisor,
        "bind_parent",
        lambda *args, **kwargs: {"unfinished_tasks": [], "bound_at_utc": "synthetic-only"},
    )
    monkeypatch.setattr(
        subprocess, "Popen", lambda *args, **kwargs: pytest.fail("unexpected child")
    )
    assert (
        supervisor.supervise(
            config=config,
            approval_path=approval,
            old_pid=999999,
            control=tmp_path / "control",
            interpreter=Path("unused"),
        )
        == 0
    )
    assert (
        json.loads((tmp_path / "control/status.json").read_text())["status"]
        == "NOOP_ALREADY_COMPLETE"
    )
