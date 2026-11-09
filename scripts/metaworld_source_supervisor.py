"""One-shot local handoff supervisor; waiting/binding imports no RL runtime."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import select
import signal
import subprocess
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.metaworld_source_extension_plan import (
    AUTHORIZATION_URL,
    DEFAULT_CONFIG,
    bind_parent,
    extension_configuration,
    process_alive,
    stable_hash,
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SupervisorStopped(RuntimeError):
    pass


class SupervisorWindow:
    """One latched UTC/monotonic window, including all setup and waiting."""

    def __init__(
        self, deadline: datetime, now: Callable[[], datetime], monotonic: Callable[[], float]
    ) -> None:
        if deadline.tzinfo is None or deadline.utcoffset() is None:
            raise ValueError("supervisor deadline must include a timezone")
        self.deadline = deadline.astimezone(timezone.utc)
        self.now, self.monotonic = now, monotonic
        self.started_utc, self.started_monotonic = now(), monotonic()
        self.monotonic_cutoff = (
            self.started_monotonic + (self.deadline - self.started_utc).total_seconds()
        )
        self.last_monotonic = self.started_monotonic
        self.reason: str | None = None

    def allowed(self) -> bool:
        moment, tick = self.now(), self.monotonic()
        if not math.isfinite(tick) or tick < self.last_monotonic:
            self.reason = self.reason or "monotonic_clock_regressed"
        elif moment >= self.deadline:
            self.reason = self.reason or "absolute_utc_deadline"
        elif tick >= self.monotonic_cutoff:
            self.reason = self.reason or "monotonic_supervisor_budget"
        self.last_monotonic = max(self.last_monotonic, tick)
        return self.reason is None

    def require(self, phase: str) -> None:
        if not self.allowed():
            raise SupervisorStopped(f"{self.reason} at {phase}")

    def remaining(self) -> float:
        if not self.allowed():
            return 0.0
        return max(
            0.0,
            min(
                self.monotonic_cutoff - self.monotonic(),
                (self.deadline - self.now()).total_seconds(),
            ),
        )


def process_identity(pid: int) -> dict[str, Any] | None:
    """Linux process start identity, without exposing command arguments."""
    try:
        root = Path("/proc") / str(pid)
        stat = (root / "stat").read_text()
        fields = stat[stat.rindex(")") + 1 :].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "command_sha256": hashlib.sha256((root / "cmdline").read_bytes()).hexdigest(),
        }
    except (OSError, ValueError, IndexError):
        return None


def _valid_identity(value: Any, pid: int) -> bool:
    return (
        isinstance(value, dict)
        and value.get("pid") == pid
        and type(value.get("start_ticks")) is int
        and value["start_ticks"] > 0
        and isinstance(value.get("boot_id"), str)
        and bool(value["boot_id"])
        and isinstance(value.get("command_sha256"), str)
        and len(value["command_sha256"]) == 64
        and all(character in "0123456789abcdef" for character in value["command_sha256"])
    )


def _check_parent_identity(pid: int, expected: dict[str, Any]) -> bool:
    current = process_identity(pid)
    if current is not None and current != expected:
        raise ValueError("parent PID was reused or changed its frozen process identity")
    if current is None and process_alive(pid):
        raise ValueError("live parent PID identity cannot be verified")
    return current is not None


def supervise(
    *,
    config: dict[str, Any],
    approval_path: Path,
    old_pid: int,
    control: Path,
    interpreter: Path,
    poll_seconds: float = 5,
    now: Callable[[], datetime] = utc_now,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    # Capture before approval reads, directory creation or binding.
    window = SupervisorWindow(datetime.fromisoformat(config["deadline_utc"]), now, monotonic)
    window.require("supervisor_arming")
    if not math.isfinite(poll_seconds) or not 0 < poll_seconds <= 60:
        raise ValueError("poll interval must be positive and no longer than 60 seconds")
    if stable_hash(config) != stable_hash(extension_configuration()):
        raise ValueError("supervisor configuration differs from approved amendment")
    approval = json.loads(approval_path.read_text())
    if (
        approval.get("approved") is not True
        or approval.get("authorization_url") != AUTHORIZATION_URL
        or approval.get("extension_sha256") != stable_hash(config)
    ):
        raise ValueError("actual issue-51 approval is required before arming a supervisor")
    if type(old_pid) is not int or old_pid <= 0 or old_pid == os.getpid():
        raise ValueError("identify the actual old source process")
    captured = process_identity(old_pid)
    expected = approval.get("parent_process_identity", captured)
    if not _valid_identity(expected, old_pid) or (captured is not None and captured != expected):
        raise ValueError("freeze the original parent PID/start identity before handoff")
    _check_parent_identity(old_pid, expected)
    protected = [Path(config[key]).resolve() for key in ("parent_root", "failed_root", "output")]
    if control.is_symlink():
        raise ValueError("control path must be a real directory")
    control = control.resolve()
    if any(control.is_relative_to(root) for root in protected):
        raise ValueError("supervisor cannot alter an experiment artifact root")
    if control.exists() and (control.is_symlink() or not control.is_dir()):
        raise ValueError("control path must be a real directory")
    window.require("before_control_writes")
    control.mkdir(parents=True, exist_ok=True)
    lock_path = Path(config["output"] + ".supervisor.lock")
    marker = Path(config["output"] + ".launch_once.json")
    if lock_path.is_symlink() or marker.is_symlink():
        raise ValueError("global supervisor records cannot be symlinks")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    status_file, stop_file = control / "status.json", control / "STOP"
    limit = config["supervisor_reserve_bytes"]

    def stored_bytes() -> int:
        paths = [*control.rglob("*"), lock_path, marker]
        return sum(path.stat().st_size for path in set(paths) if path.is_file())

    def write_json(path: Path, data: Any, *, exclusive: bool = False) -> None:
        payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        if path.is_symlink() or stored_bytes() + len(payload) > limit:
            raise RuntimeError("supervisor metadata storage reserve exhausted")
        if exclusive:
            with path.open("xb") as handle:
                handle.write(payload)
        else:
            temporary = path.with_suffix(path.suffix + ".tmp")
            if temporary.is_symlink():
                raise ValueError("supervisor temporary files cannot be symlinks")
            temporary.write_bytes(payload)
            temporary.replace(path)

    def check(phase: str) -> None:
        if stop_file.exists():
            raise SupervisorStopped(f"supervisor_stop_file at {phase}")
        window.require(phase)

    status: dict[str, Any] = {
        "status": "WAITING_FOR_OLD_PROCESS",
        "old_pid": old_pid,
        "old_process_identity": expected,
        "started_at_utc": window.started_utc.isoformat(),
        "started_monotonic": window.started_monotonic,
        "monotonic_cutoff": window.monotonic_cutoff,
        "deadline_utc": config["deadline_utc"],
        "automatic_retries": 0,
    }
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if marker.exists():
                if status_file.exists():
                    return 1  # Preserve the original supervisor's terminal evidence.
                raise ValueError("this fixed extension output has consumed its one launch attempt")
            write_json(status_file, status)
            while _check_parent_identity(old_pid, expected):
                check("waiting_for_parent")
                sleep(min(poll_seconds, window.remaining()))
            check("before_parent_binding")
            binding = bind_parent(config, old_pid=old_pid)
            check("after_parent_binding")
            if _check_parent_identity(old_pid, expected):
                raise ValueError("parent process reappeared after binding")
            binding["old_process_identity"] = expected
            binding["supervisor_window"] = {
                "started_at_utc": window.started_utc.isoformat(),
                "started_monotonic": window.started_monotonic,
                "monotonic_cutoff": window.monotonic_cutoff,
                "deadline_utc": config["deadline_utc"],
            }
            write_json(control / "parent_binding.json", binding)
            check("after_binding_persistence")
            if not binding["unfinished_tasks"]:
                status.update(
                    status="NOOP_ALREADY_COMPLETE", parent_binding_sha256=stable_hash(binding)
                )
                write_json(status_file, status)
                return 0
            write_json(
                marker,
                {
                    "bound_at_utc": binding["bound_at_utc"],
                    "parent_binding_sha256": stable_hash(binding),
                    "old_process_identity": expected,
                    "monotonic_cutoff": window.monotonic_cutoff,
                },
                exclusive=True,
            )
            command = [
                str(interpreter),
                "-u",
                "-m",
                "scripts.metaworld_source_continuation",
                "--config",
                str(DEFAULT_CONFIG),
                "--binding",
                str(control / "parent_binding.json"),
                "--approval",
                str(approval_path),
                "--stop-file",
                str(stop_file),
                "--monotonic-cutoff",
                str(window.monotonic_cutoff),
            ]
            check("directly_before_child_launch")
            child = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                status.update(
                    status="RUNNING",
                    child_pid=child.pid,
                    command=command,
                    launched_at_utc=now().isoformat(),
                )
                write_json(status_file, status)
                if child.stdout is None:
                    raise RuntimeError("child stdout pipe is unavailable")
                log = control / "continuation.log"
                if log.is_symlink():
                    raise ValueError("supervisor log cannot be a symlink")
                with log.open("ab", buffering=0) as output:

                    def consume(block: bytes) -> None:
                        if stored_bytes() + len(block) > limit - 65536:
                            stop_file.touch(exist_ok=True)
                            raise RuntimeError("supervisor log storage reserve exhausted")
                        output.write(block)

                    stop_sent: float | None = None
                    while child.poll() is None:
                        if (stop_file.exists() or not window.allowed()) and stop_sent is None:
                            stop_file.touch(exist_ok=True)
                            os.killpg(child.pid, signal.SIGINT)
                            stop_sent = monotonic()
                        if stop_sent is not None and monotonic() - stop_sent >= 30:
                            os.killpg(child.pid, signal.SIGKILL)
                        ready, _, _ = select.select([child.stdout], [], [], min(poll_seconds, 1))
                        if ready:
                            consume(os.read(child.stdout.fileno(), 65536))
                    while block := child.stdout.read(65536):
                        consume(block)
            finally:
                if child.poll() is None:
                    stop_file.touch(exist_ok=True)
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait(timeout=5)
            status.update(
                status="FINISHED",
                returncode=child.returncode,
                finished_at_utc=now().isoformat(),
                cutoff_reason=window.reason,
            )
            write_json(status_file, status)
            return 0 if child.returncode == 0 else 1
        except SupervisorStopped as error:
            status.update(
                status="STOPPED_BEFORE_HANDOFF", error=str(error), cutoff_reason=window.reason
            )
            write_json(status_file, status)
            return 1
        except KeyboardInterrupt:
            stop_file.touch(exist_ok=True)
            status.update(status="SUPERVISOR_INTERRUPTED_NO_RETRY")
            write_json(status_file, status)
            return 130
        except Exception as error:
            status.update(
                status="HANDOFF_FAILED_NO_RETRY", error=f"{type(error).__name__}: {error}"
            )
            write_json(status_file, status)
            return 1
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--old-pid", type=int, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    args = parser.parse_args(argv)
    return supervise(
        config=json.loads(args.config.read_text()),
        approval_path=args.approval,
        old_pid=args.old_pid,
        control=args.control,
        interpreter=args.python,
    )


if __name__ == "__main__":
    raise SystemExit(main())
