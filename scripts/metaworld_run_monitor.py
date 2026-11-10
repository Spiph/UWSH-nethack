"""Bounded stdlib-only observer; never controls or modifies a scientific run."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import math
import os
import re
import stat
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SIDECAR_FILES = {"sidecar.lock", "sidecar_status.json", "sidecar_status.json.tmp", "STOP"}
OWNED_FILES = {
    ".run_monitor_owner.json",
    ".monitor.lock",
    "report.json",
    "report.md",
    "monitor_status.json",
    "events.json",
}
TEMP_FILES = {f".{name}.tmp" for name in OWNED_FILES if not name.startswith(".")}
SIDECAR_RESERVE = 16384


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parent_alive(pid: int) -> bool:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def parent_start_ticks(pid: int) -> int | None:
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def assert_no_ml_imports() -> None:
    forbidden = {"numpy", "torch", "stable_baselines3", "gymnasium", "metaworld", "mujoco"}
    if any(name.split(".", 1)[0] in forbidden for name in sys.modules):
        raise RuntimeError("fresh observer CLI must remain stdlib-only; native/ML imports found")


def stable_json(path: Path, attempts: int = 3) -> tuple[Any, str]:
    """Reject in-flight/malformed snapshots; the caller retains its last good one."""
    problem = "unavailable"
    for _ in range(attempts):
        try:
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                problem = "changed_during_read"
                continue
            data = json.loads(raw)
            return data, hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError) as error:
            problem = type(error).__name__
    raise ValueError(f"snapshot_unavailable:{path.name}:{problem}")


class StorageLimit(RuntimeError):
    pass


class PublishedArtifactVerifier:
    """Stream only published path/hash commitments; cache unchanged sealed bytes."""

    def __init__(self) -> None:
        self.cache: dict[tuple[Any, ...], str] = {}

    def verify(self, references: Any) -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        checked, hits = 0, 0
        if not isinstance(references, list):
            references = []
            results.append({"status": "unsupported", "error": "artifact references are not a list"})
        for reference in references:
            path_text = reference.get("path") if isinstance(reference, dict) else None
            expected = reference.get("sha256") if isinstance(reference, dict) else None
            if (
                not isinstance(path_text, str)
                or not Path(path_text).is_absolute()
                or re.fullmatch(r"[0-9a-f]{64}", str(expected)) is None
            ):
                results.append({"status": "unsupported", "path": path_text})
                continue
            path = Path(path_text)
            result: dict[str, Any] = {"path": str(path), "expected_sha256": expected}
            for _ in range(3):
                try:
                    before = path.stat()
                    if not stat.S_ISREG(before.st_mode) or path.is_symlink():
                        result.update(status="unsupported", error="not a regular sealed file")
                        break
                    stamp = before.st_size, before.st_mtime_ns, before.st_ino
                    key = str(path), expected, *stamp
                    if key in self.cache:
                        actual = self.cache[key]
                        hits += 1
                    else:
                        digest = hashlib.sha256()
                        with path.open("rb") as handle:
                            for block in iter(lambda: handle.read(1024 * 1024), b""):
                                digest.update(block)
                        after = path.stat()
                        if stamp != (after.st_size, after.st_mtime_ns, after.st_ino):
                            result.update(status="unstable")
                            continue
                        actual = digest.hexdigest()
                        self.cache[key] = actual
                        checked += 1
                        if len(self.cache) > 4096:
                            self.cache.pop(next(iter(self.cache)))
                    result.update(
                        status="verified" if actual == expected else "mismatch",
                        actual_sha256=actual,
                    )
                    break
                except FileNotFoundError:
                    result.update(status="missing")
                    break
                except OSError as error:
                    result.update(status="unavailable", error=type(error).__name__)
                    break
            results.append(result)
        return {
            "status": "checked",
            "references": len(references),
            "new_hashes_checked": checked,
            "cache_hits": hits,
            "verified": sum(row.get("status") == "verified" for row in results),
            "errors": [row for row in results if row.get("status") != "verified"],
            "missing": sum(row.get("status") == "missing" for row in results),
            "mismatched": sum(row.get("status") == "mismatch" for row in results),
            "unstable": sum(row.get("status") == "unstable" for row in results),
            "unsupported": sum(row.get("status") == "unsupported" for row in results),
            "results": results,
            "scope": (
                "published commitments only; unchanged stat/expected-hash cache, no deserialization"
            ),
        }


class ReportStore:
    """Own only fixed report files, with an aggregate cap including atomic staging."""

    def __init__(self, output: Path, manifest: Path, max_bytes: int) -> None:
        if type(max_bytes) is not int or max_bytes < 32768:
            raise ValueError("max-bytes must be an integer of at least32768")
        if output.is_symlink():
            raise ValueError("report output must not be a symlink")
        self.root, self.limit = output.resolve(), max_bytes
        source_root = manifest.resolve().parent
        if (
            self.root == source_root
            or (self.root.is_relative_to(source_root) and self.root != source_root / "live_reports")
            or source_root.is_relative_to(self.root)
        ):
            raise ValueError(
                "use an outside report directory or the dedicated inputroot/live_reports"
            )
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError("report output must be a directory")
        marker = self.root / ".run_monitor_owner.json"
        expected = {
            "kind": "metaworld_run_monitor",
            "schema_version": 1,
            "manifest": str(manifest.resolve()),
        }
        for path in self.root.iterdir():
            if (
                path.is_symlink()
                or not path.is_file()
                or path.name not in OWNED_FILES | TEMP_FILES | SIDECAR_FILES
            ):
                raise ValueError("output contains unowned files; raw artifacts are protected")
        if marker.exists():
            if json.loads(marker.read_text()) != expected:
                raise ValueError("report directory belongs to another input")
        elif any((self.root / name).exists() for name in OWNED_FILES - {".monitor.lock"}):
            raise ValueError("report files lack an ownership marker")
        if (self.root / ".monitor.lock").is_symlink():
            raise ValueError("monitor lock cannot be a symlink")
        self.lock = (self.root / ".monitor.lock").open("a+")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for path in self.root.iterdir():
                if (
                    path.is_symlink()
                    or not path.is_file()
                    or path.name not in OWNED_FILES | TEMP_FILES | SIDECAR_FILES
                ):
                    raise ValueError("output contains unowned files; raw artifacts are protected")
            if marker.exists():
                if json.loads(marker.read_text()) != expected:
                    raise ValueError("report directory belongs to another input")
            elif any((self.root / name).exists() for name in OWNED_FILES - {".monitor.lock"}):
                raise ValueError("report files lack an ownership marker")
            for name in TEMP_FILES:
                (self.root / name).unlink(missing_ok=True)
            if not marker.exists():
                self.write(".run_monitor_owner.json", _json(expected))
        except BaseException:
            self.lock.close()
            raise

    def size(self) -> int:
        total = 0
        for path in self.root.iterdir():
            try:
                info = path.lstat()
            except FileNotFoundError:
                continue  # The sidecar may atomically replace its own metadata.
            if (
                not stat.S_ISREG(info.st_mode)
                or path.name not in OWNED_FILES | TEMP_FILES | SIDECAR_FILES
            ):
                raise ValueError("report directory acquired unowned content")
            total += info.st_size
        return total

    def write(self, name: str, raw: bytes) -> None:
        if name not in OWNED_FILES:
            raise ValueError("observer may only write its fixed report files")
        target, temporary = self.root / name, self.root / f".{name}.tmp"
        if target.is_symlink() or temporary.is_symlink():
            raise ValueError("report destinations cannot be symlinks")
        if self.size() + len(raw) > self.limit:
            raise StorageLimit("report storage cap would be exceeded during atomic replacement")
        try:
            with temporary.open("xb") as handle:
                handle.write(raw)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def close(self) -> None:
        fcntl.flock(self.lock, fcntl.LOCK_UN)
        self.lock.close()


def _json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def progress_token(manifest: dict[str, Any], progress: Any) -> str:
    tasks = {
        task: [row.get("actual_training_steps"), row.get("status"), len(row.get("checkpoints", []))]
        for task, row in manifest.get("tasks", {}).items()
        if isinstance(row, dict)
    }
    latest = progress[-1] if isinstance(progress, list) and progress else None
    if isinstance(latest, dict):
        latest = {key: latest.get(key) for key in ("task", "training_steps", "ledger")}
    value = {
        "ledger": manifest.get("ledger"),
        "phase": manifest.get("phase"),
        "status": manifest.get("status"),
        "tasks": tasks,
        "latest_progress": latest,
    }
    return hashlib.sha256(_json(value)).hexdigest()


class RunMonitor:
    def __init__(
        self,
        manifest: Path,
        output: Path,
        *,
        interval: float = 60,
        stale_after: float = 300,
        max_bytes: int = 1024**2,
        parent_pid: int | None = None,
        stop_file: Path | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], str] = utc_now,
        sleep: Callable[[float], None] = time.sleep,
        alive: Callable[[int], bool] = parent_alive,
        summarizer: Callable[..., dict[str, Any]] | None = None,
        renderer: Callable[[dict[str, Any]], str] | None = None,
        verify_artifacts: bool = True,
        identity_probe: Callable[[int], int | None] = parent_start_ticks,
    ) -> None:
        if (
            not math.isfinite(interval)
            or not 0 < interval <= 3600
            or not math.isfinite(stale_after)
            or stale_after <= 0
        ):
            raise ValueError("interval/stale-after must be positive finite durations")
        if parent_pid is not None and (type(parent_pid) is not int or parent_pid <= 0):
            raise ValueError("parent-pid must be a positive integer")
        self.manifest_path = manifest.resolve()
        self.store = ReportStore(output, self.manifest_path, max_bytes)
        self.interval, self.stale_after = interval, stale_after
        self.parent_pid, self.stop_file = parent_pid, stop_file
        self.identity_probe = identity_probe
        self.parent_start = identity_probe(parent_pid) if parent_pid is not None else None
        self.monotonic, self.now, self.sleep, self.alive = monotonic, now, sleep, alive
        self.summarizer, self.renderer = summarizer, renderer
        self.verifier = PublishedArtifactVerifier() if verify_artifacts else None
        self.last_manifest: dict[str, Any] | None = None
        self.last_progress: Any = None
        self.last_manifest_hash: str | None = None
        self.events: list[dict[str, Any]] = []
        self.last_event: Any = None
        self.last_token: str | None = None
        self.last_changed = self.monotonic()
        self.last_tick = self.last_changed
        self.status: dict[str, Any] = {
            "ready": True,
            "pid": os.getpid(),
            "final": False,
            "state": "WAITING_FOR_MANIFEST",
            "manifest": str(self.manifest_path),
            "observed_at_utc": self.now(),
            "alerts": [],
            "max_bytes": max_bytes,
            "sidecar_reserved_bytes": SIDECAR_RESERVE,
            "parent_pid": parent_pid,
            "parent_start_ticks": self.parent_start,
        }
        self.store.write("monitor_status.json", _json(self.status))

    def _snapshot(self) -> tuple[dict[str, Any], list[str]]:
        alerts = []
        try:
            data, sha = stable_json(self.manifest_path)
            if not isinstance(data, dict):
                raise ValueError("manifest_not_object")
            self.last_manifest, self.last_manifest_hash = data, sha
        except ValueError as error:
            alerts.append(str(error))
        progress_path = self.manifest_path.parent / "progress.json"
        if progress_path.exists():
            try:
                self.last_progress, _ = stable_json(progress_path)
            except ValueError as error:
                alerts.append(str(error))
        if self.summarizer is None or self.renderer is None:
            module = importlib.import_module("scripts.metaworld_live_report")
            self.summarizer, self.renderer = module.summarize_run, module.render_markdown
        report: dict[str, Any]
        if self.last_manifest is None:
            report = {
                "terminal": False,
                "raw_status": None,
                "errors": [],
                "warnings": [],
                "ledger": {},
                "checkpoint_summaries": [],
                "latest_progress": None,
                "artifact_references": [],
                "kind": "unknown",
            }
        else:
            report = self.summarizer(
                self.last_manifest,
                self.last_progress,
                source_path=self.manifest_path,
                observed_utc=self.now(),
            )
        return report, alerts

    def _publish(self, report: dict[str, Any]) -> None:
        available = self.store.limit - SIDECAR_RESERVE
        full = {**report, "observer": dict(self.status)}
        raw = _json(full)
        if len(raw) > available // 4:
            full = {
                key: full.get(key)
                for key in (
                    "kind",
                    "terminal",
                    "raw_status",
                    "ledger",
                    "errors",
                    "warnings",
                    "observer",
                    "artifact_verification",
                )
            }
            full["report_truncated"] = True
            full["truncation_note"] = (
                "checkpoint/progress/artifact detail omitted to retain bounded current report"
            )
            raw = _json(full)
        if len(raw) > available // 4:
            full = {
                "terminal": report.get("terminal", False),
                "raw_status": report.get("raw_status"),
                "report_truncated": True,
                "observer": dict(self.status),
                "artifact_verification": {
                    key: value
                    for key, value in report.get("artifact_verification", {}).items()
                    if key not in ("results", "errors")
                },
            }
            raw = _json(full)
        assert self.renderer is not None
        unchanged = self.status.get("unchanged_progress_observed_seconds", 0)
        markdown = (
            self.renderer(report)
            + f"\n\nObserver: {self.status['state']}. "
            + f"Unchanged progress observed for {unchanged:.1f}s.\n"
            + "\n".join(f"- Observer alert: {alert}" for alert in self.status["alerts"])
            + "\n"
        ).encode()
        if len(markdown) > available // 5:
            markdown = (
                f"# Run observation\n\nSource status: {report.get('raw_status')!s}\n\n"
                "Detailed rendering exceeded the reporting cap; see bounded report.json.\n"
            ).encode()
        while self.events and len(_json(self.events)) > available // 20:
            self.events.pop(0)
        self.store.write("report.json", raw)
        self.store.write("report.md", markdown)
        self.store.write("events.json", _json(self.events))
        self.store.write("monitor_status.json", _json(self.status))

    def run(self, *, once: bool = False) -> dict[str, Any]:
        report: dict[str, Any] = {}
        try:
            while True:
                tick = self.monotonic()
                report, alerts = self._snapshot()
                if tick < self.last_tick:
                    alerts.append("observer_monotonic_clock_regressed")
                self.last_tick = max(tick, self.last_tick)
                token = (
                    progress_token(self.last_manifest, self.last_progress)
                    if self.last_manifest is not None
                    else None
                )
                if token is not None and token != self.last_token:
                    self.last_token, self.last_changed = token, tick
                unchanged = max(0.0, tick - self.last_changed)
                if not report.get("terminal") and unchanged >= self.stale_after:
                    alerts.append("no_observed_source_progress")
                state, final = (
                    ("TERMINAL", True) if report.get("terminal") else ("OBSERVING", False)
                )
                if self.last_manifest is None:
                    state = "WAITING_FOR_MANIFEST"
                changed = False
                if self.parent_pid is not None:
                    current = self.identity_probe(self.parent_pid)
                    changed = (
                        self.parent_start is not None
                        and current is not None
                        and current != self.parent_start
                    )
                    if changed:
                        alerts.append("parent_identity_changed")
                if (
                    self.parent_pid is not None
                    and (changed or not self.alive(self.parent_pid))
                    and not final
                ):
                    # Parent may have exited just after the first read. Refresh
                    # once before declaring that its terminal manifest is absent.
                    report, refreshed_alerts = self._snapshot()
                    alerts.extend(refreshed_alerts)
                    if report.get("terminal"):
                        state, final = "TERMINAL", True
                    else:
                        state, final = "PARENT_EXITED", True
                        alerts.append("parent_exited_without_terminal_manifest")
                if self.stop_file is not None and self.stop_file.exists() and not final:
                    state, final = "STOPPED", True
                if once and not final:
                    state, final = "ONCE", True
                verification = (
                    self.verifier.verify(report.get("artifact_references", []))
                    if self.verifier is not None
                    else {"status": "disabled"}
                )
                report["artifact_verification"] = verification
                if verification.get("errors"):
                    alerts.append("published_artifact_verification_alert")
                if report.get("errors"):
                    alerts.append("reported_evidence_error")
                self.status.update(
                    state=state,
                    final=final,
                    observed_at_utc=self.now(),
                    alerts=sorted(set(alerts)),
                    source_manifest_sha256=self.last_manifest_hash,
                    unchanged_progress_observed_seconds=unchanged,
                    freshness_note=(
                        "monotonic lower bound of observed unchanged progress; earlier age unknown"
                    ),
                    scientific_outcome_changed_by_observer=False,
                    artifact_verification={
                        key: value
                        for key, value in verification.items()
                        if key not in ("results", "errors")
                    },
                    artifact_verification_alerts=len(verification.get("errors", [])),
                    reported_evidence_errors=len(report.get("errors", [])),
                )
                event = (state, tuple(self.status["alerts"]), self.last_token)
                if event != self.last_event:
                    self.events.append(
                        {
                            "observed_at_utc": self.now(),
                            "state": state,
                            "alerts": self.status["alerts"],
                        }
                    )
                    self.events = self.events[-128:]
                    self.last_event = event
                self._publish(report)
                if final:
                    return dict(self.status)
                remaining = self.interval
                while remaining > 0:
                    delay = min(remaining, 1.0)
                    self.sleep(delay)
                    remaining -= delay
                    if self.stop_file is not None and self.stop_file.exists():
                        break
        except BaseException as error:
            self.status.update(
                final=True, state="OBSERVER_ERROR", alerts=[f"{type(error).__name__}:{error}"]
            )
            # Retain a small final report even if the detailed summary/rendering
            # failed. Never reinterpret this as a scientific failure.
            self.renderer = lambda value: "# Observer stopped\n\nSee monitor_status.json.\n"
            for name in ("events.json", "report.md"):
                (self.store.root / name).unlink(missing_ok=True)
            minimal = {
                "terminal": report.get("terminal", False),
                "raw_status": report.get("raw_status"),
                "observer_error": True,
            }
            self._publish(minimal)
            raise
        finally:
            self.store.close()


def main(argv: list[str] | None = None) -> int:
    assert_no_ml_imports()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parent-pid", type=int)
    parser.add_argument("--interval", type=float, default=60)
    parser.add_argument("--stale-after", type=float, default=300)
    parser.add_argument("--max-bytes", type=int, default=1024**2)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-verify-artifacts", action="store_true")
    args = parser.parse_args(argv)
    RunMonitor(
        args.manifest,
        args.output,
        parent_pid=args.parent_pid,
        interval=args.interval,
        stale_after=args.stale_after,
        max_bytes=args.max_bytes,
        stop_file=args.stop_file,
        verify_artifacts=not args.no_verify_artifacts,
    ).run(once=args.once)
    assert_no_ml_imports()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
