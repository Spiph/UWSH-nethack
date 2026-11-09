"""Approval/control-flow tests with fake runtimes only; no native RL imports."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.metaworld_adaptation_runtime_smoke import (
    ARMS,
    PLAN,
    PLAN_PATH,
    Limits,
    SmokeRunner,
    anchored_allowance_seconds,
    approved_launch,
    canonical_hash,
    claim_attempt_output,
    validate_approval,
)
from scripts.metaworld_deadlines import RunDeadline


class FakeBudget:
    def __init__(self, *, expired: bool = False) -> None:
        self.expired = expired
        self.phases: list[str] = []

    def require(self, phase: str) -> None:
        self.phases.append(phase)
        if self.expired:
            raise RuntimeError("fake deadline expired")

    def record(self, manifest: dict[str, Any]) -> None:
        manifest["budget_compliant"] = not self.expired
        if self.expired:
            manifest["validation_complete"] = False
            if manifest["status"] != "ERROR":
                manifest["status"] = "STOPPED_AT_BUDGET"


class FakeStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: dict[str, Any] = {}

    def size(self) -> int:
        return len(json.dumps(self.records))

    def json(self, name: str, value: Any) -> None:
        self.records[name] = copy.deepcopy(value)


class FakeRuntime:
    def __init__(self, plan: dict[str, Any], limits: Limits, store: FakeStore) -> None:
        self.limits, self.store = limits, store
        self.rng = 0
        self.events: list[str] = []
        self.closed = False

    def metadata(self) -> dict[str, Any]:
        return {"fake_runtime_only": True}

    def construct(self, arm: str) -> tuple[dict[str, int], dict[str, str]]:
        self.events.append("construct_" + arm)
        self.rng += 1
        return {"model": 0, "env": 0}, {"arm": arm}

    def preflight(self, handles: dict[str, Any], records: dict[str, Any]) -> None:
        assert set(handles) == set(records) == set(ARMS)
        assert self.limits.ledger["synthetic_steps"] == 0
        self.events.append("preflight")

    def frozen_prediction_check(self, name: str, full: dict[str, int]) -> None:
        self.events.append("frozen_" + name)

    def learn(self, handle: dict[str, int], steps: int, *, reset: bool) -> None:
        self.events.append(f"learn_{steps}_{reset}")
        for _ in range(steps):
            self.limits.charge("synthetic_step_attempts", 120, "fake_step")
            handle["model"] += 1
            handle["env"] += 1
            self.rng += 1
            self.limits.ledger["synthetic_steps"] += 1
            if handle["model"] > 4:
                for _ in range(3):
                    self.limits.charge("optimizer_step_attempts", 312, "fake_optimizer")
                    self.limits.ledger["optimizer_steps"] += 1

    def check_trained_actor(self, handle: dict[str, int], arm: str) -> None:
        assert handle["model"] == 20

    def check_timeouts(self, handle: dict[str, int]) -> None:
        assert handle["env"] == 20

    def save_checkpoint(self, handle: dict[str, int], arm: str) -> dict[str, Any]:
        return {"model": handle["model"], "env": handle["env"], "rng": self.rng}

    def restore_capsule(self, handle: dict[str, int], capsule: dict[str, Any]) -> None:
        handle["env"] = capsule["env"]
        self.rng = capsule["rng"]

    def snapshot(self, handle: dict[str, int]) -> dict[str, str]:
        return {"model": str(handle["model"]), "env": str(handle["env"]), "rng": str(self.rng)}

    def persist_branch(self, handle: dict[str, int], arm: str, name: str) -> None:
        self.store.json(f"{arm}_{name}", self.snapshot(handle))

    def reload(self, arm: str, capsule: dict[str, Any]) -> dict[str, int]:
        self.events.append("reload_" + arm)
        self.rng += 100  # Restoring the capsule must erase constructor RNG use.
        return {"model": capsule["model"], "env": 0}

    def close(self) -> None:
        self.closed = True

    def retain_failure(self) -> None:
        self.events.append("retain_failure")


def fake_run(
    tmp_path: Path, runtime_type: Any = FakeRuntime, budget: FakeBudget | None = None
) -> tuple[Any, Any, Any]:
    store = FakeStore(tmp_path)
    instances = []

    def factory(plan: dict[str, Any], limits: Limits, target: Any) -> Any:
        runtime = runtime_type(plan, limits, target)
        instances.append(runtime)
        return runtime

    runner = SmokeRunner(
        PLAN, {"synthetic_test_fixture_only": True}, budget or FakeBudget(), store, factory
    )
    result = runner.run()
    return result, instances, store


def test_exact_fake_control_flow_and_all_resource_counts(tmp_path: Path) -> None:
    result, instances, store = fake_run(tmp_path)
    assert result["status"] == "PASS" and result["validation_complete"] is True
    assert result["ledger"]["sac_initializations"] == 8
    assert result["ledger"]["standalone_frozen_actors"] == 2
    assert result["ledger"]["synthetic_steps"] == 120
    assert result["ledger"]["optimizer_steps"] == 312
    assert all(
        result["ledger"][name] == 0
        for name in ("benchmark_source_steps", "benchmark_target_steps", "benchmark_final_steps")
    )
    runtime = instances[0]
    assert runtime.events[:7] == [
        *("construct_" + arm for arm in ARMS),
        "preflight",
        "frozen_reference",
        "frozen_source",
    ]
    assert runtime.closed
    assert store.records["smoke_manifest.json"]["validation_complete"] is True
    for row in result["arms"].values():
        assert row["continued_state"] == row["restored_state"]


def test_expired_setup_never_initializes_runtime(tmp_path: Path) -> None:
    result, instances, _ = fake_run(tmp_path, budget=FakeBudget(expired=True))
    assert result["validation_complete"] is False
    assert not instances
    assert result["ledger"]["sac_initializations"] == 0


def test_preflight_failure_stops_before_all_steps_and_retains_failure(tmp_path: Path) -> None:
    class BadPreflight(FakeRuntime):
        def preflight(self, handles: dict[str, Any], records: dict[str, Any]) -> None:
            raise RuntimeError("fake reference mismatch")

    result, instances, store = fake_run(tmp_path, BadPreflight)
    assert result["status"] == "ERROR"
    assert result["ledger"]["sac_initializations"] == 4
    assert result["ledger"]["synthetic_steps"] == 0
    assert "fake reference mismatch" in result["errors"][0]
    assert instances[0].closed
    assert "retain_failure" in instances[0].events
    assert store.records["smoke_manifest.json"]["validation_complete"] is False


def test_bad_restored_state_stops_without_retry(tmp_path: Path) -> None:
    class BadReload(FakeRuntime):
        def reload(self, arm: str, capsule: dict[str, Any]) -> dict[str, int]:
            return {"model": capsule["model"] - 1, "env": 0}

    result, _, _ = fake_run(tmp_path, BadReload)
    assert result["status"] == "ERROR" and result["validation_complete"] is False
    assert result["ledger"]["sac_initializations"] == 5
    assert result["ledger"]["synthetic_steps"] == 25
    assert "before continuation" in result["errors"][0]


def test_failed_step_retains_attempted_cost_and_partial_state(tmp_path: Path) -> None:
    class BadStep(FakeRuntime):
        def learn(self, handle: dict[str, int], steps: int, *, reset: bool) -> None:
            super().learn(handle, 2, reset=reset)
            self.limits.charge("synthetic_step_attempts", 120, "fake_failing_step")
            raise RuntimeError("synthetic injected step failure")

    result, instances, _ = fake_run(tmp_path, BadStep)
    assert result["status"] == "ERROR"
    assert result["ledger"]["synthetic_step_attempts"] == 3
    assert result["ledger"]["synthetic_steps"] == 2
    assert result["ledger"]["optimizer_steps"] == 0
    assert result["ledger"]["sac_initializations"] == 4
    assert "retain_failure" in instances[0].events


def test_interrupt_retains_partial_state_and_charged_cost_without_retry(tmp_path: Path) -> None:
    class Interrupted(FakeRuntime):
        def learn(self, handle: dict[str, int], steps: int, *, reset: bool) -> None:
            super().learn(handle, 2, reset=reset)
            raise KeyboardInterrupt("synthetic test interrupt")

    result, instances, store = fake_run(tmp_path, Interrupted)
    assert result["status"] == "INTERRUPTED"
    assert result["validation_complete"] is False
    assert result["stop_exception_type"] == "KeyboardInterrupt"
    assert result["ledger"]["synthetic_steps"] == 2
    assert result["ledger"]["sac_initializations"] == 4
    assert "retain_failure" in instances[0].events and instances[0].closed
    assert store.records["smoke_manifest.json"]["validation_complete"] is False


def test_entry_utc_and_monotonic_anchors_include_setup_despite_utc_rollback() -> None:
    session = datetime(2026, 11, 9, tzinfo=timezone.utc)
    cutoff = session + timedelta(seconds=300)
    entry = session + timedelta(seconds=30)
    allowance = anchored_allowance_seconds(cutoff, entry)
    assert allowance == 270
    tick = [200.0]  # 100 seconds of hashes/setup since entry tick100.
    wall = [session + timedelta(seconds=40)]  # UTC has rolled back during setup.
    budget = RunDeadline(allowance, cutoff, monotonic=lambda: tick[0], now=lambda: wall[0])
    budget.started_monotonic = 100.0
    assert budget.allowed("after_setup")
    tick[0] = 370.0
    assert not budget.allowed("original_fixed_monotonic_cutoff")
    tick[0], wall[0] = 101.0, session
    assert not budget.allowed("cannot_resume_after_clock_rollback")
    assert anchored_allowance_seconds(cutoff, session - timedelta(seconds=10)) == 300
    with pytest.raises(ValueError, match="expired before process entry"):
        anchored_allowance_seconds(cutoff, cutoff)


def test_cleanup_cutoff_cannot_leave_a_final_passing_record(tmp_path: Path) -> None:
    budget = FakeBudget()

    class LateCleanup(FakeRuntime):
        def close(self) -> None:
            super().close()
            budget.expired = True

    result, _, store = fake_run(tmp_path, LateCleanup, budget=budget)
    assert result["status"] == "STOPPED_AT_BUDGET"
    assert result["validation_complete"] is False
    assert store.records["smoke_manifest.json"]["validation_complete"] is False


def test_exclusive_directory_claim_consumes_attempt_even_when_empty(tmp_path: Path) -> None:
    output = tmp_path / "one_attempt"
    claim_attempt_output(output)
    with pytest.raises(FileExistsError):
        claim_attempt_output(output)
    assert output.is_dir() and not list(output.iterdir())


def test_limits_refuse_before_exceeding_caps() -> None:
    limits = Limits(FakeBudget())
    for _ in range(8):
        limits.charge("sac_initializations", 8, "fake_constructor")
    with pytest.raises(RuntimeError, match="limit reached"):
        limits.charge("sac_initializations", 8, "forbidden_constructor")
    assert limits.ledger["sac_initializations"] == 8


def test_false_approval_refuses_without_runtime_or_output(tmp_path: Path) -> None:
    plan = dict(PLAN, output=str(tmp_path / "must-not-exist"))
    with pytest.raises(ValueError, match="UNAPPROVED"):
        approved_launch(plan, {"approved": False})
    assert not Path(plan["output"]).exists()


def test_pure_approval_binding_and_active_source_rejection() -> None:
    now = datetime(2026, 11, 9, tzinfo=timezone.utc)
    approval = {
        "approved": True,
        "decision_issue": 49,
        "authorization_url": "https://github.com/Spiph/UWSH-nethack/issues/49#issuecomment-1",
        "plan_sha256": canonical_hash(PLAN),
        "code_commit": "a" * 40,
        "code_sha256": {"fake_only.py": "b" * 64},
        "session_started_utc": now.isoformat(),
        "deadline_utc": (now + timedelta(seconds=300)).isoformat(),
        "source_process_pid": 999999,
        "source_process_exited": True,
        "source_lifecycle_file": "/synthetic-fixture-only/source-lifecycle.json",
        "source_lifecycle_sha256": "c" * 64,
        "output": PLAN["output"],
        "attempt": 1,
        "automatic_retries": 0,
    }
    kwargs: dict[str, Any] = {
        "current_hashes": approval["code_sha256"],
        "head": "a" * 40,
        "source_alive": lambda _: False,
        "now": now,
    }
    assert validate_approval(PLAN, approval, **kwargs) == now + timedelta(seconds=300)
    with pytest.raises(ValueError, match="must have exited"):
        validate_approval(PLAN, approval, **{**kwargs, "source_alive": lambda _: True})
    for key, value in (
        ("plan_sha256", "bad"),
        ("attempt", 2),
        ("approved", False),
        ("code_commit", "c" * 40),
        ("decision_issue", 43),
    ):
        with pytest.raises(ValueError):
            validate_approval(PLAN, {**approval, key: value}, **kwargs)


@pytest.mark.parametrize("planning", [True, False])
def test_cli_plan_and_denial_block_all_runtime_imports(tmp_path: Path, planning: bool) -> None:
    approval = tmp_path / "UNAPPROVED.json"
    approval.write_text(json.dumps({"approved": False}))
    script = str(Path(__file__).parents[1] / "scripts/metaworld_adaptation_runtime_smoke.py")
    arguments = [script, "--config", str(PLAN_PATH)]
    arguments += ["--plan"] if planning else ["--approval", str(approval)]
    code = (
        "import sys,runpy\n"
        "class BlockRuntime:\n"
        " def find_spec(self, fullname, path=None, target=None):\n"
        "  if fullname.split('.')[0] in ('numpy','torch','gymnasium','stable_baselines3'):\n"
        "   raise AssertionError('forbidden runtime import: '+fullname)\n"
        "sys.meta_path.insert(0,BlockRuntime())\n"
        f"sys.argv={arguments!r}\n"
        f"runpy.run_path({script!r},run_name='__main__')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert result.returncode == (0 if planning else 2), result.stderr
    assert "forbidden runtime import" not in result.stderr
    if planning:
        payload = json.loads(result.stdout)
        assert payload["approved"] is False
        assert payload["plan"]["synthetic_steps"] == 120
    else:
        assert "UNAPPROVED" in result.stderr
