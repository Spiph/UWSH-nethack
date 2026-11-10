"""Observer lifecycle checks using synthetic JSON and stdlib-only child processes."""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from scripts import metaworld_reporting_sidecar as sidecar


def status(path: Path) -> dict[str, Any]:
    return dict(json.loads((path.parent / "live_reports" / "sidecar_status.json").read_text()))


def write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


class FakeProcess:
    def __init__(self, output: Path, *, ready: bool = True, stubborn: bool = False) -> None:
        self.output, self.stubborn = output, stubborn
        self.pid = 543210
        self.returncode: int | None = None
        self.stderr = io.BytesIO()
        self.waits: list[float] = []
        self.terminated, self.killed = False, False
        if ready:
            self.publish("WAITING_FOR_MANIFEST", final=False)

    def publish(self, state: str, *, final: bool) -> None:
        write(
            self.output / "monitor_status.json",
            {"pid": self.pid, "ready": True, "state": state, "final": final},
        )

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float) -> int:
        self.waits.append(timeout)
        if self.returncode is None:
            if self.stubborn and not self.killed:
                raise subprocess.TimeoutExpired("fake observer", timeout)
            self.returncode = -9 if self.killed else 0
            if not self.killed:
                self.publish("STOPPED", final=True)
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True


def install_fake(
    monkeypatch: pytest.MonkeyPatch,
    manifest: Path,
    *,
    ready: bool = True,
    stubborn: bool = False,
) -> tuple[list[FakeProcess], list[tuple[list[str], dict[str, Any]]]]:
    children: list[FakeProcess] = []
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def spawn(command: list[str], **kwargs: Any) -> FakeProcess:
        calls.append((command, kwargs))
        child = FakeProcess(manifest.parent / "live_reports", ready=ready, stubborn=stubborn)
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(sidecar, "HEALTH_SECONDS", 0.005)
    return children, calls


def test_ready_process_uses_exact_command_and_stops_with_final_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "development_manifest.json"
    children, calls = install_fake(monkeypatch, manifest)
    with sidecar.reporting_sidecar(manifest, 2 * 1024**2) as metadata:
        assert metadata["ready"] and metadata["scientific_body_started"]
        assert status(manifest)["observer_state"] == "READY"
        assert not manifest.exists()
    command, options = calls[0]
    assert command == [
        sys.executable,
        "-m",
        "scripts.metaworld_run_monitor",
        "--manifest",
        str(manifest),
        "--output",
        str(tmp_path / "live_reports"),
        "--parent-pid",
        str(os.getpid()),
        "--interval",
        "60",
        "--stale-after",
        "300",
        "--max-bytes",
        str(2 * 1024**2 // 10),
        "--stop-file",
        str(tmp_path / "live_reports" / "STOP"),
    ]
    assert options["start_new_session"] is True
    assert options["stdout"] == subprocess.DEVNULL and options["stderr"] == subprocess.PIPE
    assert children[0].waits == [sidecar.STOP_SECONDS]
    assert status(manifest)["observer_state"] == "FINISHED"
    assert status(manifest)["monitor_status"]["state"] == "STOPPED"
    assert not manifest.exists()


def test_failed_startup_prevents_body_and_retains_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: Any, **kwargs: Any) -> None:
        raise OSError("observer executable unavailable")

    monkeypatch.setattr(subprocess, "Popen", unavailable)
    manifest = tmp_path / "manifest.json"
    entered = False
    with (
        pytest.raises(sidecar.ReportingSidecarError, match="unavailable"),
        sidecar.reporting_sidecar(manifest, 1024**2),
    ):
        entered = True
    assert not entered
    saved = status(manifest)
    assert saved["observer_state"] == "STARTUP_FAILED" and not saved["ready"]
    assert not saved["scientific_body_started"] and not manifest.exists()


@pytest.mark.parametrize("stale_pid", [False, True])
def test_startup_timeout_never_accepts_stale_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stale_pid: bool
) -> None:
    manifest = tmp_path / "manifest.json"
    children, _ = install_fake(monkeypatch, manifest, ready=False)
    if stale_pid:
        output = tmp_path / "live_reports"
        output.mkdir()
        write(output / "monitor_status.json", {"pid": 999, "ready": True, "final": False})
    monkeypatch.setattr(sidecar, "STARTUP_SECONDS", 0)
    with (
        pytest.raises(sidecar.ReportingSidecarError, match="timed out"),
        sidecar.reporting_sidecar(manifest, 1024**2),
    ):
        pytest.fail("unready observer cannot authorize the body")
    assert children[0].returncode == 0
    assert status(manifest)["observer_state"] == "STARTUP_FAILED"


def test_insufficient_report_allowance_is_explicit_and_never_spawns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    children, _ = install_fake(monkeypatch, manifest)
    with (
        pytest.raises(sidecar.ReportingSidecarError, match="32 KiB"),
        sidecar.reporting_sidecar(manifest, 100000),
    ):
        pytest.fail("insufficient report allowance")
    assert not children
    assert status(manifest)["observer_state"] == "STARTUP_FAILED"
    assert sum(path.stat().st_size for path in (tmp_path / "live_reports").iterdir()) < 10000


def test_scientific_exception_identity_survives_unhealthy_observer_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    children, _ = install_fake(monkeypatch, manifest, stubborn=True)
    scientific_error = RuntimeError("synthetic scientific failure")
    with pytest.raises(RuntimeError) as raised, sidecar.reporting_sidecar(manifest, 1024**2):
        raise scientific_error
    assert raised.value is scientific_error
    assert children[0].terminated and children[0].killed
    assert children[0].waits == [
        sidecar.STOP_SECONDS,
        sidecar.TERMINATE_SECONDS,
        sidecar.KILL_SECONDS,
    ]
    saved = status(manifest)
    assert saved["observer_state"] == "UNHEALTHY"
    assert saved["scientific_body_exception"] == "RuntimeError"
    assert saved["returncode"] == -9


def test_worker_crash_is_reported_while_body_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    children, _ = install_fake(monkeypatch, manifest)
    with sidecar.reporting_sidecar(manifest, 1024**2):
        write(manifest, {"status": "RUNNING", "ledger": {"steps": 7}})
        before = manifest.read_bytes()
        children[0].returncode = 6
        wait_for(lambda: status(manifest)["observer_state"] == "UNHEALTHY")
        assert manifest.read_bytes() == before
        write(manifest, {"status": "COMPLETE_FAIL", "phase": "FINISHED"})
    assert status(manifest)["observer_state"] == "UNHEALTHY"
    assert json.loads(manifest.read_text())["status"] == "COMPLETE_FAIL"


def test_duplicate_context_cannot_replace_active_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    children, _ = install_fake(monkeypatch, manifest)
    with sidecar.reporting_sidecar(manifest, 1024**2):
        before = status(manifest)
        with (
            pytest.raises(sidecar.ReportingSidecarError, match="already owns"),
            sidecar.reporting_sidecar(manifest, 1024**2),
        ):
            pytest.fail("duplicate observer")
        assert status(manifest) == before
    assert len(children) == 1


def test_terminal_scientific_failure_is_a_healthy_observer_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "manifest.json"
    children, _ = install_fake(monkeypatch, manifest)
    with sidecar.reporting_sidecar(manifest, 1024**2):
        write(manifest, {"status": "COMPLETE_FAIL", "phase": "FINISHED"})
        children[0].publish("TERMINAL", final=True)
        children[0].returncode = 0
        wait_for(lambda: status(manifest)["observer_state"] == "FINISHED")
    assert status(manifest)["observer_state"] == "FINISHED"
    assert not status(manifest)["stop_requested"]


@pytest.mark.parametrize("wait_for_watchdog", [False, True])
def test_early_observer_stop_is_unhealthy_without_interrupting_science(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, wait_for_watchdog: bool
) -> None:
    manifest = tmp_path / "manifest.json"
    children, _ = install_fake(monkeypatch, manifest)
    with sidecar.reporting_sidecar(manifest, 1024**2):
        children[0].publish("STOPPED", final=True)
        children[0].returncode = 0
        if wait_for_watchdog:
            wait_for(lambda: status(manifest)["observer_state"] == "UNHEALTHY")
        write(manifest, {"status": "COMPLETE_FAIL", "phase": "FINISHED"})
    saved = status(manifest)
    assert saved["observer_state"] == "UNHEALTHY"
    assert "coverage ended" in saved["error"]
    assert json.loads(manifest.read_text())["status"] == "COMPLETE_FAIL"


def wait_for(predicate: Any, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("timed out waiting for synthetic observer")
        time.sleep(0.01)


OBSERVER_CODE = """
import argparse
from pathlib import Path
from scripts.metaworld_run_monitor import RunMonitor
parser=argparse.ArgumentParser()
parser.add_argument('--manifest',type=Path);parser.add_argument('--output',type=Path)
parser.add_argument('--parent-pid',type=int);parser.add_argument('--interval',type=float)
parser.add_argument('--stale-after',type=float);parser.add_argument('--max-bytes',type=int)
parser.add_argument('--stop-file',type=Path)
args=parser.parse_args()
def summary(data,progress=None,**kwargs):
 return {'raw_status':data.get('status'),'terminal':data.get('phase')=='FINISHED',
         'ledger':data.get('ledger',{}),'kind':'synthetic_writer'}
RunMonitor(args.manifest,args.output,parent_pid=args.parent_pid,interval=.02,
 stale_after=args.stale_after,max_bytes=args.max_bytes,stop_file=args.stop_file,
 summarizer=summary,renderer=lambda value: str(value)).run()
"""


def test_real_stdlib_observer_reports_parallel_writer_and_refreshes_on_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = subprocess.Popen

    def spawn(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        return original([sys.executable, "-c", OBSERVER_CODE, *command[3:]], **kwargs)

    monkeypatch.setattr(subprocess, "Popen", spawn)
    manifest, report = tmp_path / "manifest.json", tmp_path / "live_reports" / "report.json"
    with sidecar.reporting_sidecar(manifest, 1024**2):
        for count in range(1, 4):
            write(manifest, {"status": "RUNNING", "ledger": {"steps": count}})
            wait_for(
                lambda expected=count: (
                    report.exists()
                    and json.loads(report.read_text()).get("ledger", {}).get("steps") == expected
                )
            )
        before = manifest.read_bytes()
    assert manifest.read_bytes() == before
    saved = status(manifest)
    assert saved["observer_state"] == "FINISHED"
    assert saved["monitor_status"]["state"] == "STOPPED"
    assert json.loads(report.read_text())["ledger"] == {"steps": 3}
    assert sum(path.stat().st_size for path in report.parent.iterdir()) <= saved["max_bytes"]


def test_parent_hard_exit_preserves_final_incomplete_observation(tmp_path: Path) -> None:
    observer_script = tmp_path / "observer.py"
    observer_script.write_text(OBSERVER_CODE)
    manifest = tmp_path / "manifest.json"
    parent_code = """
import os,subprocess,sys
from pathlib import Path
from scripts import metaworld_reporting_sidecar as sidecar
original=subprocess.Popen
def spawn(command,**kwargs):
 return original([sys.executable,sys.argv[2],*command[3:]],**kwargs)
sidecar.subprocess.Popen=spawn
with sidecar.reporting_sidecar(Path(sys.argv[1]),1024**2):
 Path(sys.argv[1]).write_text('{"status":"RUNNING","ledger":{"steps":2}}')
 os._exit(0)
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(Path(sidecar.__file__).resolve().parents[1])
    subprocess.run(
        [sys.executable, "-c", parent_code, str(manifest), str(observer_script)],
        cwd=Path(sidecar.__file__).resolve().parents[1],
        env=environment,
        timeout=8,
        check=True,
    )
    monitor_status = tmp_path / "live_reports" / "monitor_status.json"
    wait_for(lambda: json.loads(monitor_status.read_text()).get("state") == "PARENT_EXITED")
    observed = json.loads(monitor_status.read_text())
    assert observed["final"]
    assert "parent_exited_without_terminal_manifest" in observed["alerts"]
    report = json.loads((monitor_status.parent / "report.json").read_text())
    assert report["raw_status"] == "RUNNING" and not report["terminal"]
    assert json.loads(manifest.read_text())["ledger"] == {"steps": 2}


def test_sidecar_import_is_stdlib_only() -> None:
    code = """
import sys
from scripts import metaworld_reporting_sidecar
blocked={'torch','numpy','gymnasium','metaworld','stable_baselines3'}
assert not blocked.intersection(sys.modules)
assert callable(metaworld_reporting_sidecar.reporting_sidecar)
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=5)
