"""Launch one independent, read-only observer around an actual research CLI run.

Startup readiness is required before yielding. Once the scientific body starts,
observer failures are retained separately and never change its control flow.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STARTUP_SECONDS = 5.0
STOP_SECONDS = 10.0
TERMINATE_SECONDS = 1.0
KILL_SECONDS = 1.0
POLL_SECONDS = 0.05
HEALTH_SECONDS = 0.5
MIN_REPORT_BYTES = 32768
MAX_REPORT_BYTES = 1024 * 1024
STATUS_BYTES = 8192


class ReportingSidecarError(RuntimeError):
    """Live reporting could not become ready before scientific work started."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _monitor_status(output: Path, pid: int) -> dict[str, Any] | None:
    path = output / "monitor_status.json"
    if path.is_symlink():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) and value.get("pid") == pid else None


def _stderr(child: subprocess.Popen[bytes]) -> str:
    if child.poll() is None or child.stderr is None:
        return ""
    try:
        return child.stderr.read(2048).decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return ""


def _stop(child: subprocess.Popen[bytes], output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"stop_requested": False, "terminated": False, "killed": False}
    if child.poll() is None:
        (output / "STOP").touch(exist_ok=True)
        result["stop_requested"] = True
        try:
            child.wait(timeout=STOP_SECONDS)
        except subprocess.TimeoutExpired:
            result["terminated"] = True
            child.terminate()
            try:
                child.wait(timeout=TERMINATE_SECONDS)
            except subprocess.TimeoutExpired:
                result["killed"] = True
                child.kill()
                try:
                    child.wait(timeout=KILL_SECONDS)
                except subprocess.TimeoutExpired:
                    result["still_alive_after_kill"] = True
    result["returncode"] = child.poll()
    result["stderr"] = _stderr(child)
    return result


@contextmanager
def reporting_sidecar(manifest_path: Path, artifact_limit: int) -> Iterator[dict[str, Any]]:
    """Require observer startup, then preserve scientific results independently.

    ArtifactStore must already own/create the parent directory. All lifecycle
    writes stay in its live_reports subtree and count toward the same allowance.
    The worker reserves 16 KiB for this module's bounded atomic status files.
    """
    if type(artifact_limit) is not int or artifact_limit <= 0:
        raise ValueError("artifact_limit must be a positive integer")
    manifest_path = Path(manifest_path).absolute()
    if manifest_path.is_symlink() or not manifest_path.parent.is_dir():
        raise ValueError("the scientific artifact directory must already exist")
    output = manifest_path.parent / "live_reports"
    if output.is_symlink():
        raise ReportingSidecarError("live_reports cannot be a symlink")
    output.mkdir(exist_ok=True)
    max_bytes = min(MAX_REPORT_BYTES, artifact_limit // 10)
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "observer_state": "STARTING",
        "ready": False,
        "manifest": str(manifest_path),
        "output": str(output),
        "parent_pid": os.getpid(),
        "started_at_utc": _utc(),
        "max_bytes": max_bytes,
        "interval_seconds": 60,
        "stale_after_seconds": 300,
        "scientific_body_started": False,
        "coverage_note": "observer readiness is separate from scientific outcome/completeness",
    }
    metadata_lock = threading.Lock()
    status_path = output / "sidecar_status.json"

    def persist() -> None:
        if status_path.is_symlink() or status_path.with_suffix(".json.tmp").is_symlink():
            raise ReportingSidecarError("sidecar status paths cannot be symlinks")
        payload = (json.dumps(metadata, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
        ceiling = min(STATUS_BYTES, max_bytes // 2)
        if len(payload) > ceiling:
            brief = {
                key: metadata.get(key)
                for key in (
                    "schema_version",
                    "observer_state",
                    "ready",
                    "pid",
                    "returncode",
                    "error",
                    "finished_at_utc",
                )
            }
            error_message = brief.get("error")
            if isinstance(error_message, str):
                brief["error"] = error_message[:256]
            payload = (json.dumps(brief, sort_keys=True) + "\n").encode()
        if len(payload) > ceiling:
            raise ReportingSidecarError("artifact allowance cannot retain observer metadata")
        temporary = status_path.with_suffix(".json.tmp")
        temporary.write_bytes(payload)
        temporary.replace(status_path)

    lock_path = output / "sidecar.lock"
    if lock_path.is_symlink():
        raise ReportingSidecarError("sidecar lock cannot be a symlink")
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ReportingSidecarError("a reporting sidecar already owns this run") from error
        child: subprocess.Popen[bytes] | None = None
        watchdog: threading.Thread | None = None
        watchdog_stop = threading.Event()
        startup_started = time.monotonic()
        try:
            try:
                persist()
                if max_bytes < MIN_REPORT_BYTES:
                    raise ReportingSidecarError(
                        "artifact allowance leaves less than 32 KiB for reports"
                    )
                if (output / "STOP").exists():
                    raise ReportingSidecarError(
                        "reporter STOP already exists; no automatic restart"
                    )
                command = [
                    sys.executable,
                    "-m",
                    "scripts.metaworld_run_monitor",
                    "--manifest",
                    str(manifest_path),
                    "--output",
                    str(output),
                    "--parent-pid",
                    str(os.getpid()),
                    "--interval",
                    "60",
                    "--stale-after",
                    "300",
                    "--max-bytes",
                    str(max_bytes),
                    "--stop-file",
                    str(output / "STOP"),
                ]
                metadata["command"] = command
                child = subprocess.Popen(
                    command,
                    cwd=Path(__file__).resolve().parents[1],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )
                metadata["pid"] = child.pid
                while True:
                    observed = _monitor_status(output, child.pid)
                    if observed is not None and observed.get("ready") is True:
                        if observed.get("state") in ("STOPPED", "PARENT_EXITED"):
                            raise ReportingSidecarError(
                                "observer stopped before scientific startup"
                            )
                        metadata["monitor_status_at_ready"] = observed
                        break
                    if child.poll() is not None:
                        raise ReportingSidecarError(
                            f"observer exited before readiness (code {child.returncode})"
                        )
                    if time.monotonic() - startup_started >= STARTUP_SECONDS:
                        raise ReportingSidecarError("observer readiness timed out")
                    time.sleep(POLL_SECONDS)
                metadata.update(
                    observer_state="READY",
                    ready=True,
                    ready_at_utc=_utc(),
                    startup_seconds=time.monotonic() - startup_started,
                )
                persist()
            except BaseException as error:
                metadata.update(
                    observer_state="STARTUP_FAILED",
                    ready=False,
                    error=f"{type(error).__name__}: {error}",
                    finished_at_utc=_utc(),
                )
                if child is not None:
                    try:
                        metadata.update(_stop(child, output))
                    except BaseException as stop_error:
                        metadata["shutdown_error"] = f"{type(stop_error).__name__}: {stop_error}"
                persist()
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ReportingSidecarError(metadata["error"]) from error

            def watch() -> None:
                assert child is not None
                while not watchdog_stop.wait(HEALTH_SECONDS):
                    if child.poll() is None:
                        continue
                    observed = _monitor_status(output, child.pid)
                    healthy_finish = (
                        child.returncode == 0
                        and observed is not None
                        and observed.get("final") is True
                        and observed.get("state") == "TERMINAL"
                    )
                    with metadata_lock:
                        metadata.update(
                            observer_state="FINISHED" if healthy_finish else "UNHEALTHY",
                            returncode=child.returncode,
                            monitor_status=observed,
                            observer_exit_at_utc=_utc(),
                        )
                        if not healthy_finish:
                            metadata["error"] = "observer exited while scientific body was active"
                        try:
                            persist()
                        except BaseException as error:
                            metadata["status_write_error"] = f"{type(error).__name__}: {error}"
                    return

            try:
                metadata["scientific_body_started"] = True
                persist()
                watchdog = threading.Thread(
                    target=watch, name="research-report-health", daemon=True
                )
                watchdog.start()
            except BaseException as error:
                metadata.update(
                    observer_state="STARTUP_FAILED",
                    scientific_body_started=False,
                    error=f"{type(error).__name__}: {error}",
                    finished_at_utc=_utc(),
                )
                metadata.update(_stop(child, output))
                persist()
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise ReportingSidecarError(metadata["error"]) from error
            try:
                yield metadata
            except BaseException as error:
                metadata["scientific_body_exception"] = type(error).__name__
                raise
            finally:
                watchdog_stop.set()
                if watchdog is not None:
                    watchdog.join(timeout=HEALTH_SECONDS + 0.1)
                with metadata_lock:
                    try:
                        assert child is not None
                        before_stop = _monitor_status(output, child.pid)
                        premature_final = (
                            before_stop is not None
                            and before_stop.get("final") is True
                            and before_stop.get("state") != "TERMINAL"
                        )
                        metadata.update(_stop(child, output))
                        observed = _monitor_status(output, child.pid)
                        metadata["monitor_status"] = observed
                        unhealthy = (
                            metadata.get("observer_state") == "UNHEALTHY"
                            or child.returncode != 0
                            or observed is None
                            or observed.get("final") is not True
                            or premature_final
                            or metadata.get("terminated")
                            or metadata.get("killed")
                        )
                        metadata.update(
                            observer_state="UNHEALTHY" if unhealthy else "FINISHED",
                            finished_at_utc=_utc(),
                        )
                        if premature_final:
                            metadata["error"] = (
                                "observer coverage ended before scientific body exit"
                            )
                    except BaseException as error:
                        metadata.update(
                            observer_state="UNHEALTHY",
                            error=f"{type(error).__name__}: {error}",
                            finished_at_utc=_utc(),
                        )
                    try:
                        persist()
                    except BaseException as error:
                        metadata["status_write_error"] = f"{type(error).__name__}: {error}"
        finally:
            if child is not None and child.stderr is not None:
                with suppress(OSError, ValueError):
                    child.stderr.close()
            with suppress(OSError):
                fcntl.flock(lock, fcntl.LOCK_UN)
