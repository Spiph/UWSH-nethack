"""Fake clocks/processes only; no subprocess launch, policy, environment or GPU."""

from __future__ import annotations

import fcntl
import io
import json
import os
import select
import signal
import subprocess
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

import pytest

from scripts import metaworld_source_supervisor as supervisor
from scripts.metaworld_source_extension_plan import (
    AUTHORIZATION_URL,
    extension_configuration,
    stable_hash,
)

IDENTITY = {
    "pid": 999999,
    "start_ticks": 12345,
    "boot_id": "synthetic-boot",
    "command_sha256": "a" * 64,
}


class Clock:
    def __init__(self) -> None:
        self.wall = datetime(2026, 11, 10, tzinfo=timezone.utc)
        self.tick = 1000.0

    def now(self) -> datetime:
        return self.wall

    def monotonic(self) -> float:
        return self.tick

    def advance(self, seconds: float) -> None:
        self.wall += timedelta(seconds=seconds)
        self.tick += seconds


class Child:
    pid = 999998

    def __init__(self, output: bytes = b"synthetic output\n") -> None:
        self.stdout = io.BytesIO(output)
        self.returncode: int | None = 0

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, *, timeout: float) -> int:
        assert self.returncode is not None
        return self.returncode


class Harness:
    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.clock = Clock()
        self.config = extension_configuration() | {
            "parent_root": str(tmp_path / "parent"),
            "failed_root": str(tmp_path / "failed"),
            "output": str(tmp_path / "extension"),
            "deadline_utc": (self.clock.wall + timedelta(seconds=100)).isoformat(),
            "supervisor_reserve_bytes": 1024 * 1024,
        }
        self.control = tmp_path / "control"
        self.approval = tmp_path / "approval.json"
        self.approval.write_text(
            json.dumps(
                {
                    "approved": True,
                    "authorization_url": AUTHORIZATION_URL,
                    "extension_sha256": stable_hash(self.config),
                    "parent_process_identity": IDENTITY,
                }
            )
        )
        self.launches: list[list[str]] = []
        self.signals: list[int] = []
        self.child = Child()
        self.binding = {"unfinished_tasks": ["push-v3"], "bound_at_utc": "synthetic-only"}
        monkeypatch.setattr(supervisor, "extension_configuration", lambda: self.config)
        monkeypatch.setattr(supervisor, "process_identity", lambda pid: None)
        monkeypatch.setattr(supervisor, "process_alive", lambda pid: False)
        monkeypatch.setattr(supervisor, "bind_parent", lambda *args, **kwargs: dict(self.binding))

        def popen(command: list[str], **kwargs: Any) -> Child:
            self.launches.append(command)
            return self.child

        def killpg(pid: int, kind: int) -> None:
            assert pid == self.child.pid
            self.signals.append(kind)
            self.child.returncode = -kind

        monkeypatch.setattr(subprocess, "Popen", popen)
        monkeypatch.setattr(os, "killpg", killpg)
        monkeypatch.setattr(select, "select", lambda *args: ([], [], []))

    def run(self, **overrides: Any) -> int:
        arguments: dict[str, Any] = {
            "config": self.config,
            "approval_path": self.approval,
            "old_pid": IDENTITY["pid"],
            "control": self.control,
            "interpreter": Path("synthetic-unused-python"),
            "now": self.clock.now,
            "monotonic": self.clock.monotonic,
            "sleep": self.clock.advance,
        }
        return supervisor.supervise(**(arguments | overrides))

    def status(self) -> dict[str, Any]:
        return cast(dict[str, Any], json.loads((self.control / "status.json").read_text()))


def test_global_one_launch_identity_survives_different_control_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)
    assert fixture.run() == 0
    assert fixture.run() == 1
    assert fixture.status()["status"] == "FINISHED"
    assert fixture.run(control=tmp_path / "different-control") == 1
    assert len(fixture.launches) == 1
    command = fixture.launches[0]
    assert float(command[command.index("--monotonic-cutoff") + 1]) == 1100
    binding = json.loads((fixture.control / "parent_binding.json").read_text())
    assert binding["old_process_identity"] == IDENTITY
    assert binding["supervisor_window"]["monotonic_cutoff"] == 1100


def test_global_lock_is_held_during_child_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)

    def popen(command: list[str], **kwargs: Any) -> Child:
        with (
            Path(fixture.config["output"] + ".supervisor.lock").open("a+") as lock,
            pytest.raises(BlockingIOError),
        ):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fixture.launches.append(command)
        return fixture.child

    monkeypatch.setattr(subprocess, "Popen", popen)
    assert fixture.run() == 0


@pytest.mark.parametrize("root", ["parent_root", "failed_root", "output"])
def test_control_cannot_write_inside_any_protected_run(
    root: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)
    protected = Path(fixture.config[root])
    with pytest.raises(ValueError, match="artifact root"):
        fixture.run(control=protected / "control")
    assert not protected.exists()
    assert fixture.launches == []


def test_quiet_complete_gate_launches_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)
    fixture.binding["unfinished_tasks"] = []
    assert fixture.run() == 0
    assert fixture.status()["status"] == "NOOP_ALREADY_COMPLETE"
    assert fixture.launches == []
    assert not Path(fixture.config["output"] + ".launch_once.json").exists()


def test_parent_pid_reuse_and_unverifiable_dead_identity_cannot_launch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)
    altered = IDENTITY | {"start_ticks": 98765}
    monkeypatch.setattr(supervisor, "process_identity", lambda pid: altered)
    with pytest.raises(ValueError, match="identity"):
        fixture.run()
    monkeypatch.setattr(supervisor, "process_identity", lambda pid: None)
    approval = json.loads(fixture.approval.read_text())
    approval.pop("parent_process_identity")
    fixture.approval.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="identity"):
        fixture.run()
    assert fixture.launches == []


def test_waiting_consumes_original_monotonic_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)
    monkeypatch.setattr(
        supervisor, "process_identity", lambda pid: IDENTITY if fixture.clock.tick < 1020 else None
    )
    assert fixture.run(poll_seconds=20) == 0
    assert fixture.clock.tick == 1020
    assert fixture.status()["monotonic_cutoff"] == 1100
    assert float(fixture.launches[0][-1]) == 1100


@pytest.mark.parametrize("stop", ["utc", "monotonic", "stop_file"])
def test_binding_can_consume_window_or_receive_stop_without_launch(
    stop: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)

    def bind(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if stop == "utc":
            fixture.clock.wall += timedelta(seconds=101)
        elif stop == "monotonic":
            fixture.clock.tick += 101
        else:
            (fixture.control / "STOP").touch()
        return dict(fixture.binding)

    monkeypatch.setattr(supervisor, "bind_parent", bind)
    assert fixture.run() == 1
    assert fixture.status()["status"] == "STOPPED_BEFORE_HANDOFF"
    assert fixture.launches == []


def test_cutoff_latches_across_utc_rollback_and_monotonic_waiting() -> None:
    clock = Clock()
    budget = supervisor.SupervisorWindow(
        clock.wall + timedelta(seconds=100), clock.now, clock.monotonic
    )
    clock.wall += timedelta(seconds=101)
    assert not budget.allowed()
    clock.wall -= timedelta(seconds=500)
    assert not budget.allowed()
    assert budget.reason == "absolute_utc_deadline"
    budget = supervisor.SupervisorWindow(
        clock.wall + timedelta(seconds=100), clock.now, clock.monotonic
    )
    clock.tick += 101
    assert not budget.allowed()
    assert budget.reason == "monotonic_supervisor_budget"


def test_direct_prelaunch_guard_checks_after_marker_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)
    marker = Path(fixture.config["output"] + ".launch_once.json")
    original = Path.open

    @contextmanager
    def opening(path: Path, *args: Any, **kwargs: Any) -> Any:
        with original(path, *args, **kwargs) as handle:
            yield handle
        if path == marker:
            (fixture.control / "STOP").touch()

    monkeypatch.setattr(Path, "open", opening)
    assert fixture.run() == 1
    assert marker.exists()
    assert fixture.launches == []


def test_child_is_interrupted_at_latched_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)
    fixture.child.returncode = None

    def select(*args: Any) -> tuple[list[Any], list[Any], list[Any]]:
        fixture.clock.advance(101)
        return [], [], []

    monkeypatch.setattr("select.select", select)
    assert fixture.run() == 1
    assert fixture.signals == [signal.SIGINT]
    assert (fixture.control / "STOP").exists()
    assert fixture.status()["cutoff_reason"] == "absolute_utc_deadline"


def test_launch_error_consumes_marker_and_never_retries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)
    calls = []

    def popen(*args: Any, **kwargs: Any) -> None:
        calls.append(1)
        raise OSError("synthetic launch failure")

    monkeypatch.setattr(subprocess, "Popen", popen)
    assert fixture.run() == 1
    assert fixture.status()["status"] == "HANDOFF_FAILED_NO_RETRY"
    assert fixture.run(control=tmp_path / "other-control") == 1
    assert calls == [1]


def test_tail_output_cannot_exceed_reserved_storage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)
    fixture.child.stdout = io.BytesIO(b"x" * 2**20)
    assert fixture.run() == 1
    assert fixture.status()["status"] == "HANDOFF_FAILED_NO_RETRY"
    used = sum(path.stat().st_size for path in fixture.control.rglob("*") if path.is_file())
    assert used <= fixture.config["supervisor_reserve_bytes"]


def test_user_interrupt_before_binding_does_not_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Harness(tmp_path, monkeypatch)

    def bind(*args: Any, **kwargs: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(supervisor, "bind_parent", bind)
    assert fixture.run() == 130
    assert fixture.status()["status"] == "SUPERVISOR_INTERRUPTED_NO_RETRY"
    assert fixture.launches == []
