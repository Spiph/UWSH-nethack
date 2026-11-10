"""Fake-only resource/retention checks; no native constructor, environment or physics."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts import metaworld_source_sentinel as sentinel
from scripts.metaworld_deadlines import BudgetExpired


class Budget:
    def __init__(self) -> None:
        self.expired = False

    def require(self, phase: str) -> None:
        if self.expired:
            raise BudgetExpired("synthetic cutoff")

    def record(self, manifest: dict[str, Any]) -> None:
        manifest["budget_compliant"] = not self.expired
        if self.expired:
            manifest["validation_complete"] = False
            if manifest["status"] != "ERROR":
                manifest["status"] = "STOPPED_AT_BUDGET"


class Store:
    def __init__(self, root: Path) -> None:
        self.root = root
        root.mkdir()

    def open(self, name: str) -> Any:
        return (self.root / name).open("wb")

    def json(self, name: str, value: Any) -> None:
        (self.root / name).write_text(json.dumps(value))


class Runtime:
    def __init__(self, runner: sentinel.SentinelRunner) -> None:
        self.runner = runner
        self.closed = False

    def metadata(self) -> dict[str, bool]:
        return {"synthetic_fake_only": True}

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        self.runner.charge("environment_constructor_attempts", 2)
        self.runner.ledger["environment_constructors"] += 1
        self.runner.charge("scripted_constructor_attempts", 2)
        self.runner.ledger["scripted_controllers"] += 1
        return {"step": 0}

    def reset(self, handle: dict[str, Any], request: dict[str, Any]) -> tuple[Any, Any]:
        return [0.0] * 39, {"synthetic_goal": request["goal_id"]}

    def action(self, handle: dict[str, Any], observation: Any) -> tuple[Any, Any]:
        return [3, -3, 0, 0.6], [1, -1, 0, 0.6]

    def step(self, handle: dict[str, Any], action: Any) -> tuple[Any, float, bool, bool, Any]:
        handle["step"] += 1
        return (
            [float(handle["step"])] * 39,
            0.5,
            False,
            handle["step"] == 500,
            {"success": handle["step"] == 300},
        )

    def record(
        self,
        handle: Any,
        step: int,
        observation: Any,
        following: Any,
        raw: Any,
        action: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        info: Any,
    ) -> dict[str, Any]:
        return {
            "step": step,
            "observation": observation,
            "next_observation": following,
            "raw_scripted_action": raw,
            "executed_clipped_action": action,
            "success": info["success"],
            "terminated": terminated,
            "truncated": truncated,
        }

    def close(self) -> None:
        self.closed = True


def run_fake(
    tmp_path: Path, runtime_type: Any = Runtime, budget: Budget | None = None
) -> tuple[Any, Any, Any]:
    instances = []

    def factory(runner: sentinel.SentinelRunner) -> Any:
        instance = runtime_type(runner)
        instances.append(instance)
        return instance

    store = Store(tmp_path / "new-fake-run")
    runner = sentinel.SentinelRunner(
        sentinel.candidate_configuration(),
        {"synthetic_only": True},
        {},
        budget or Budget(),
        store,
        factory,
    )
    return runner.run(), instances, store


def test_exact_scripted_scope_and_full_horizon_costs(tmp_path: Path) -> None:
    result, instances, store = run_fake(tmp_path)
    assert result["status"] == "COMPLETE" and result["validation_complete"]
    assert (
        result["ledger"]["source_diagnostic_steps"]
        == result["ledger"]["source_diagnostic_step_attempts"]
        == 1000
    )
    assert (
        result["ledger"]["environment_constructors"]
        == result["ledger"]["scripted_controllers"]
        == 2
    )
    assert all(
        result["ledger"][key] == 0
        for key in (
            "development_steps",
            "target_steps",
            "final_test_steps",
            "actor_array_loads",
            "sac_initializations",
            "optimizer_updates",
        )
    )
    assert all(
        row["length"] == row["trace_rows"] == 500 and row["complete_episode"] and row["success"]
        for row in result["episodes"]
    )
    assert instances[0].closed
    trace = [json.loads(line) for line in (store.root / "push-v3.jsonl").read_text().splitlines()]
    assert trace[0]["raw_scripted_action"][:2] == [3, -3]
    assert trace[0]["executed_clipped_action"][:2] == [1, -1]
    assert trace[-1]["truncated"] is True


@pytest.mark.parametrize("kind", ["error", "interrupt", "cutoff"])
def test_failures_retain_partial_trace_and_attempted_versus_observed_costs(
    kind: str, tmp_path: Path
) -> None:
    class Failure(Runtime):
        def step(self, handle: dict[str, Any], action: Any) -> Any:
            if handle["step"] == 10:
                if kind == "interrupt":
                    raise KeyboardInterrupt
                if kind == "cutoff":
                    self.runner.budget.expired = True
                    raise BudgetExpired("synthetic cutoff")
                raise RuntimeError("synthetic failure")
            return super().step(handle, action)

    result, instances, store = run_fake(tmp_path, Failure)
    assert not result["validation_complete"]
    assert result["ledger"]["source_diagnostic_step_attempts"] == 11
    assert result["ledger"]["source_diagnostic_steps"] == 10
    assert result["ledger"]["environment_constructors"] == 1
    assert len((store.root / "push-v3.jsonl").read_text().splitlines()) == 10
    assert result["episodes"][0]["trace_sha256"]
    assert result["errors"] and instances[0].closed


def test_early_termination_is_protocol_error_with_charged_step(tmp_path: Path) -> None:
    class Early(Runtime):
        def step(self, handle: dict[str, Any], action: Any) -> Any:
            observation, reward, _, _, info = super().step(handle, action)
            return observation, reward, True, False, info

    result, _, _ = run_fake(tmp_path, Early)
    assert result["status"] == "ERROR"
    assert result["ledger"]["source_diagnostic_steps"] == 1
    assert not result["episodes"][0]["complete_episode"]


def test_cutoff_before_import_never_constructs_runtime(tmp_path: Path) -> None:
    budget = Budget()
    budget.expired = True
    result, instances, _ = run_fake(tmp_path, budget=budget)
    assert not instances and not result["validation_complete"]
    assert result["ledger"]["source_diagnostic_steps"] == 0


def test_late_cleanup_cannot_leave_passing_claim(tmp_path: Path) -> None:
    class Late(Runtime):
        def close(self) -> None:
            self.runner.budget.expired = True

    result, _, store = run_fake(tmp_path, Late)
    assert not result["validation_complete"]
    assert not json.loads((store.root / "sentinel_manifest.json").read_text())[
        "validation_complete"
    ]


def test_episode_completion_promotion_checks_the_cutoff_after_full_trace(tmp_path: Path) -> None:
    class LateBudget(Budget):
        def require(self, phase: str) -> None:
            if phase == "episode_completion_promotion":
                self.expired = True
            super().require(phase)

    result, _, store = run_fake(tmp_path, budget=LateBudget())
    assert not result["validation_complete"]
    assert result["ledger"]["source_diagnostic_steps"] == 500
    assert result["ledger"]["environment_constructors"] == 1
    assert result["episodes"][0]["complete_episode"] is True
    assert result["episodes"][0]["status"] != "COMPLETE"
    assert len((store.root / "push-v3.jsonl").read_text().splitlines()) == 500


def test_approval_is_exact_and_false_template_refuses_before_work(tmp_path: Path) -> None:
    plan = sentinel.candidate_configuration()
    with pytest.raises(ValueError, match="UNAPPROVED"):
        sentinel.approved_launch(plan, {"approved": False})
    now = datetime(2026, 11, 10, tzinfo=timezone.utc)
    approval = {
        "approved": True,
        "decision_issue": 56,
        "authorization_url": "https://github.com/Spiph/UWSH-nethack/issues/56#issuecomment-1",
        "plan_sha256": sentinel.stable_hash(plan),
        "code_commit": "a" * 40,
        "code_sha256": {"synthetic.py": "b" * 64},
        "output": plan["output"],
        "attempt": 1,
        "automatic_retries": 0,
        "session_started_utc": now.isoformat(),
        "deadline_utc": (now + timedelta(seconds=300)).isoformat(),
    }
    arguments = {"now": now, "head": "a" * 40, "hashes": approval["code_sha256"]}
    assert sentinel.validate_approval(plan, approval, **arguments) == now + timedelta(seconds=300)
    for key, value in (
        ("plan_sha256", "wrong"),
        ("attempt", 2),
        ("automatic_retries", 1),
        ("code_commit", "c" * 40),
        ("decision_issue", 49),
    ):
        with pytest.raises(ValueError):
            sentinel.validate_approval(plan, approval | {key: value}, **arguments)


def test_lifecycle_hashes_and_active_source_gate_use_only_fake_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = copy.deepcopy(sentinel.candidate_configuration())
    values = {
        "original": {"phase": "FINISHED"},
        "final": {"phase": "FINISHED", "status": "COMPLETE_FAIL"},
        "supervisor": {"status": "FINISHED", "returncode": 0},
    }
    for key, value in values.items():
        path = tmp_path / (key + ".json")
        path.write_text(json.dumps(value))
        plan["evidence"][key] = {"path": str(path), "sha256": sentinel.digest(path)}
    monkeypatch.setattr(sentinel, "alive", lambda pid: False)
    monkeypatch.setattr(sentinel, "source_processes", lambda: [])
    assert sentinel.source_evidence(plan) == values["final"]
    monkeypatch.setattr(sentinel, "source_processes", lambda: [12345])
    with pytest.raises(ValueError, match="active"):
        sentinel.source_evidence(plan)
    monkeypatch.setattr(sentinel, "source_processes", lambda: [])
    Path(plan["evidence"]["final"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="changed"):
        sentinel.source_evidence(plan)


@pytest.mark.parametrize("plan_only", [True, False])
def test_cli_preparation_and_denial_block_native_imports(tmp_path: Path, plan_only: bool) -> None:
    config = tmp_path / "candidate.json"
    config.write_text(json.dumps(sentinel.candidate_configuration()))
    approval = tmp_path / "false.json"
    approval.write_text('{"approved":false}')
    script = str(Path(__file__).parents[1] / "scripts/metaworld_source_sentinel.py")
    args = [script, "--config", str(config)] + (
        ["--plan"] if plan_only else ["--approval", str(approval)]
    )
    code = (
        "import sys,runpy\nclass Block:\n"
        " def find_spec(self,fullname,path=None,target=None):\n"
        "  if fullname.split('.')[0] in "
        "('numpy','torch','metaworld','mujoco','gymnasium','stable_baselines3'):\n"
        "   raise AssertionError('native import forbidden: '+fullname)\n"
        "sys.meta_path.insert(0,Block())\n"
        + f"sys.argv={args!r}\nrunpy.run_path({script!r},run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == (0 if plan_only else 1), result.stderr
    assert "native import forbidden" not in result.stderr


def test_real_action_boundary_helper_clips_without_rescaling_using_fake_controller() -> None:
    runtime = sentinel.NativeRuntime.__new__(sentinel.NativeRuntime)
    runtime.np = np
    handle = {"policy": SimpleNamespace(get_action=lambda obs: np.array([3.0, -4.0, 0.2, 2.0]))}
    raw, executed = runtime.action(handle, np.zeros(39))
    np.testing.assert_array_equal(raw, [3, -4, 0.2, 2])
    np.testing.assert_array_equal(executed, np.array([1, -1, 0.2, 1], dtype=np.float32))
    assert executed.dtype == np.float32


def test_reset_rejects_wrong_goal_or_seed_without_physics(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = sentinel.NativeRuntime.__new__(sentinel.NativeRuntime)
    runtime.np = np
    request = sentinel.candidate_configuration()["tasks"][0]
    record = {
        "goal_index": 0,
        "goal_id": request["goal_id"],
        "goal_bank_sha256": request["bank_sha256"],
        "episode_seed": request["episode_seed"] + 1,
    }
    monkeypatch.setattr(
        runtime, "protocol", SimpleNamespace(evaluation_record=lambda env: record), raising=False
    )
    env = SimpleNamespace(reset=lambda: (np.zeros(39), {}))
    with pytest.raises(ValueError, match="seed changed"):
        runtime.reset({"env": env}, request)
