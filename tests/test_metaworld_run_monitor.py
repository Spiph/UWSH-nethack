"""Fake writer/clock checks; no learner, environment or scientific interaction."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.metaworld_run_monitor import (
    PublishedArtifactVerifier,
    ReportStore,
    RunMonitor,
    parent_alive,
    stable_json,
)


def summary(manifest: Any, progress: Any = None, **kwargs: Any) -> dict[str, Any]:
    return {
        "terminal": manifest.get("phase") == "FINISHED",
        "raw_status": manifest.get("status"),
        "kind": "source_development",
        "ledger": manifest.get("ledger", {}),
        "errors": manifest.get("errors", []),
        "warnings": [],
        "checkpoint_summaries": manifest.get("details", []),
        "artifact_references": manifest.get("artifact_references", []),
        "latest_progress": progress,
    }


def render(report: dict[str, Any]) -> str:
    return "# Source run\n\n" + str(report.get("raw_status")) + "\n"


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value))
    temporary.replace(path)


def configured(manifest: Path, output: Path, **kwargs: Any) -> RunMonitor:
    return RunMonitor(manifest, output, summarizer=summary, renderer=render, **kwargs)


def test_readiness_precedes_missing_manifest_and_once_retains_unknown_outcome(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "manifest.json"
    output = tmp_path / "report"
    observer = configured(source, output)
    ready = json.loads((output / "monitor_status.json").read_text())
    assert ready["ready"] and ready["pid"] > 0 and not ready["final"]
    assert ready["state"] == "WAITING_FOR_MANIFEST"
    status = observer.run(once=True)
    report = json.loads((output / "report.json").read_text())
    assert status["state"] == "ONCE" and status["final"]
    assert report["raw_status"] is None and not report["terminal"]
    assert status["alerts"] and not source.exists()


def test_complete_fail_is_scientific_status_and_raw_files_are_unchanged(tmp_path: Path) -> None:
    source = tmp_path / "source" / "manifest.json"
    write(source, {"status": "COMPLETE_FAIL", "phase": "FINISHED", "ledger": {"steps": 5}})
    progress = source.parent / "progress.json"
    write(progress, [{"task": "push", "training_steps": 5}])
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in (source, progress)}
    output = tmp_path / "report"
    status = configured(source, output).run()
    report = json.loads((output / "report.json").read_text())
    assert status["state"] == "TERMINAL" and report["raw_status"] == "COMPLETE_FAIL"
    assert not status["alerts"] and report["terminal"]
    assert before == {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in before}


def test_malformed_snapshot_retains_last_good_status_and_parent_death_alert(tmp_path: Path) -> None:
    source, output = tmp_path / "input" / "manifest.json", tmp_path / "report"
    write(source, {"status": "RUNNING", "phase": "TRAINING", "ledger": {"steps": 3}})
    ticks = [0.0]

    def sleep(seconds: float) -> None:
        ticks[0] += seconds
        source.write_text('{"inflight":')

    observer = configured(
        source,
        output,
        interval=1,
        sleep=sleep,
        monotonic=lambda: ticks[0],
        parent_pid=123,
        alive=lambda pid: ticks[0] == 0,
    )
    status = observer.run()
    report = json.loads((output / "report.json").read_text())
    assert report["raw_status"] == "RUNNING" and report["ledger"] == {"steps": 3}
    assert status["state"] == "PARENT_EXITED"
    assert "parent_exited_without_terminal_manifest" in status["alerts"]
    assert any("snapshot_unavailable" in alert for alert in status["alerts"])


def test_staleness_uses_monotonic_progress_and_ignores_utc_heartbeat_rollback(
    tmp_path: Path,
) -> None:
    source, output, stop = (
        tmp_path / "input" / "manifest.json",
        tmp_path / "report",
        tmp_path / "STOP",
    )
    write(source, {"status": "RUNNING", "ledger": {"steps": 10}})
    ticks = [0.0]

    def sleep(seconds: float) -> None:
        ticks[0] += seconds
        write(
            source, {"status": "RUNNING", "ledger": {"steps": 10}, "finished_at_utc": str(ticks[0])}
        )
        if ticks[0] >= 3:
            stop.touch()

    status = configured(
        source,
        output,
        interval=1,
        stale_after=2,
        stop_file=stop,
        monotonic=lambda: ticks[0],
        now=lambda: f"2026-11-10T{20 - int(ticks[0]):02}:00:00Z",
        sleep=sleep,
    ).run()
    assert status["state"] == "STOPPED" and status["final"]
    assert status["unchanged_progress_observed_seconds"] == 3
    assert "no_observed_source_progress" in status["alerts"]
    assert "lower bound" in status["freshness_note"]
    assert "Observer alert" in (output / "report.md").read_text()


def test_progress_counter_change_refreshes_monotonic_lower_bound(tmp_path: Path) -> None:
    source, output, stop = (
        tmp_path / "input" / "manifest.json",
        tmp_path / "report",
        tmp_path / "STOP",
    )
    write(source, {"status": "RUNNING"})
    progress = source.parent / "progress.json"
    ticks = [0.0]

    def sleep(seconds: float) -> None:
        ticks[0] += seconds
        write(progress, [{"task": "push", "training_steps": int(ticks[0])}])
        if ticks[0] >= 4:
            stop.touch()

    status = configured(
        source,
        output,
        interval=1,
        stale_after=2,
        stop_file=stop,
        monotonic=lambda: ticks[0],
        sleep=sleep,
    ).run()
    assert status["unchanged_progress_observed_seconds"] == 0
    assert "no_observed_source_progress" not in status["alerts"]


def test_output_ownership_rejects_raw_or_foreign_files_without_touching_them(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source" / "manifest.json"
    write(source, {})
    output = tmp_path / "unowned"
    output.mkdir()
    (output / "raw_checkpoint.bin").write_bytes(b"preserve")
    with pytest.raises(ValueError, match="unowned"):
        configured(source, output)
    assert sorted(path.name for path in output.iterdir()) == ["raw_checkpoint.bin"]
    with pytest.raises(ValueError, match="outside"):
        configured(source, source.parent)
    with pytest.raises(ValueError, match="outside"):
        configured(source, source.parent / "other_reports")


def test_dedicated_live_subtree_preserves_sidecar_metadata_and_storage_cap(tmp_path: Path) -> None:
    source = tmp_path / "source" / "manifest.json"
    write(source, {"status": "COMPLETE_FAIL", "phase": "FINISHED", "details": ["x" * 10000] * 10})
    output = source.parent / "live_reports"
    output.mkdir()
    sidecar = output / "sidecar_status.json"
    sidecar.write_bytes(b" " * 8192)
    status = configured(source, output, max_bytes=32768).run()
    assert status["state"] == "TERMINAL"
    assert sidecar.read_bytes() == b" " * 8192
    assert sum(path.stat().st_size for path in output.iterdir()) <= 32768
    report = json.loads((output / "report.json").read_text())
    assert report["raw_status"] == "COMPLETE_FAIL" and report["report_truncated"]
    assert (output / "report.md").exists() and (output / "monitor_status.json").exists()


def test_event_history_and_replacements_stay_bounded(tmp_path: Path) -> None:
    source, output = tmp_path / "source" / "manifest.json", tmp_path / "report"
    write(source, {"status": "RUNNING", "ledger": {"steps": 0}})
    ticks = [0.0]

    def sleep(seconds: float) -> None:
        ticks[0] += seconds
        write(source, {"status": "RUNNING", "ledger": {"steps": int(ticks[0])}})

    configured(
        source,
        output,
        interval=1,
        max_bytes=32768,
        parent_pid=123,
        monotonic=lambda: ticks[0],
        sleep=sleep,
        alive=lambda pid: ticks[0] < 100,
    ).run()
    assert sum(path.stat().st_size for path in output.iterdir()) <= 32768
    assert len(json.loads((output / "events.json").read_text())) < 128
    assert not list(output.glob("*.tmp"))


def test_stable_read_retries_transient_json_instead_of_reporting_scientific_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "manifest.json"
    write(path, {"status": "RUNNING"})
    original = Path.read_bytes
    calls = [0]

    def inflight(candidate: Path) -> bytes:
        calls[0] += 1
        return b"{" if calls[0] < 3 else original(candidate)

    monkeypatch.setattr(Path, "read_bytes", inflight)
    assert stable_json(path)[0] == {"status": "RUNNING"}
    assert calls[0] == 3


def test_parallel_stdlib_fake_writer_finishes_and_observer_emits_final_report(
    tmp_path: Path,
) -> None:
    source, output = tmp_path / "source" / "manifest.json", tmp_path / "report"
    code = """
import json,sys,threading,time
from pathlib import Path
from scripts.metaworld_run_monitor import RunMonitor
source,output=map(Path,sys.argv[1:])
source.parent.mkdir()
def producer():
 for value in range(3):
  data={'status':'COMPLETE_FAIL' if value==2 else 'RUNNING',
        'phase':'FINISHED' if value==2 else 'TRAINING','ledger':{'steps':value}}
  temporary=source.with_suffix('.tmp');temporary.write_text(json.dumps(data));temporary.replace(source);time.sleep(.04)
def summary(data,progress=None,**kwargs):
 return {'raw_status':data['status'],'terminal':data.get('phase')=='FINISHED',
         'ledger':data['ledger']}
worker=threading.Thread(target=producer);worker.start()
result=RunMonitor(source,output,interval=.02,summarizer=summary,
                  renderer=lambda data: str(data)).run()
worker.join();print(json.dumps(result))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(source), str(output)],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    status = json.loads(result.stdout)
    assert status["ready"] and status["state"] == "TERMINAL" and status["final"]
    assert json.loads((output / "report.json").read_text())["raw_status"] == "COMPLETE_FAIL"


def test_second_observer_cannot_reuse_live_owned_directory(tmp_path: Path) -> None:
    source, output = tmp_path / "source" / "manifest.json", tmp_path / "report"
    write(source, {})
    observer = configured(source, output)
    with pytest.raises(BlockingIOError):
        configured(source, output)
    observer.run(once=True)
    with pytest.raises(ValueError, match="another input"):
        ReportStore(output, tmp_path / "different" / "manifest.json", 32768)


def test_published_hash_verification_cache_and_corruption(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "sealed.bin"
    checkpoint.write_bytes(b"abc")
    reference = {"path": str(checkpoint), "sha256": hashlib.sha256(b"abc").hexdigest()}
    verifier = PublishedArtifactVerifier()
    first = verifier.verify([reference])
    assert first["new_hashes_checked"] == first["verified"] == 1 and first["cache_hits"] == 0
    original = Path.open

    def no_rehash(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path == checkpoint:
            pytest.fail("unchanged sealed checkpoint was reread")
        return original(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(Path, "open", no_rehash)
        cached = verifier.verify([reference])
    assert cached["cache_hits"] == 1 and cached["new_hashes_checked"] == 0
    stamp = checkpoint.stat()
    checkpoint.write_bytes(b"abd")
    os.utime(checkpoint, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 100))
    corrupted = verifier.verify([reference])
    assert corrupted["mismatched"] == 1 and corrupted["new_hashes_checked"] == 1
    assert corrupted["errors"][0]["status"] == "mismatch"
    assert verifier.verify([reference])["cache_hits"] == 1


def test_missing_unsupported_and_unstable_refs_are_observer_alerts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "sealed.bin"
    checkpoint.write_bytes(b"abc")
    verifier = PublishedArtifactVerifier()
    missing = verifier.verify([{"path": str(tmp_path / "missing"), "sha256": "f" * 64}])
    assert missing["missing"] == 1
    assert verifier.verify([{"path": "relative", "sha256": "f" * 64}])["unsupported"] == 1
    original = Path.stat
    calls = [0]

    def changing(path: Path, *args: Any, **kwargs: Any) -> Any:
        result = original(path, *args, **kwargs)
        if path != checkpoint:
            return result
        calls[0] += 1
        return SimpleNamespace(
            st_mode=result.st_mode,
            st_size=result.st_size,
            st_mtime_ns=result.st_mtime_ns + calls[0],
            st_ino=result.st_ino,
        )

    monkeypatch.setattr(Path, "stat", changing)
    result = verifier.verify(
        [{"path": str(checkpoint), "sha256": hashlib.sha256(b"abc").hexdigest()}]
    )
    assert result["unstable"] == 1 and result["mismatched"] == 0
    assert result["new_hashes_checked"] == 0


def test_final_once_hash_alert_does_not_relabel_scientific_failure(tmp_path: Path) -> None:
    source, output = tmp_path / "source" / "manifest.json", tmp_path / "report"
    write(
        source,
        {
            "status": "COMPLETE_FAIL",
            "phase": "FINISHED",
            "artifact_references": [{"path": str(tmp_path / "missing.bin"), "sha256": "a" * 64}],
        },
    )
    status = configured(source, output).run(once=True)
    report = json.loads((output / "report.json").read_text())
    assert report["raw_status"] == "COMPLETE_FAIL" and report["terminal"]
    assert report["artifact_verification"]["missing"] == 1
    assert "published_artifact_verification_alert" in status["alerts"]
    assert status["artifact_verification_alerts"] == 1


def test_parent_pid_reuse_is_an_observer_alert_not_a_live_parent(tmp_path: Path) -> None:
    source, output = tmp_path / "source" / "manifest.json", tmp_path / "report"
    write(source, {"status": "RUNNING"})
    identities = iter([11, 22])
    status = configured(
        source,
        output,
        parent_pid=123,
        alive=lambda pid: True,
        identity_probe=lambda pid: next(identities),
    ).run()
    assert status["state"] == "PARENT_EXITED"
    assert "parent_identity_changed" in status["alerts"]
    assert json.loads((output / "report.json").read_text())["raw_status"] == "RUNNING"


def test_zombie_parent_is_dead_without_signal_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: "123 (fake) Z 0 0 0")
    monkeypatch.setattr(os, "kill", lambda *args: pytest.fail("zombie treated as live"))
    assert not parent_alive(123)


def test_fresh_once_cli_runs_without_site_or_ml_packages(tmp_path: Path) -> None:
    source, output = tmp_path / "source" / "manifest.json", tmp_path / "report"
    write(source, {"status": "RUNNING", "phase": "TRAINING"})
    completed = subprocess.run(
        [
            sys.executable,
            "-S",
            "-m",
            "scripts.metaworld_run_monitor",
            "--manifest",
            str(source),
            "--output",
            str(output),
            "--once",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    status = json.loads((output / "monitor_status.json").read_text())
    assert status["ready"] and status["final"]


def test_reported_evidence_errors_become_events_without_relabeling_competence(
    tmp_path: Path,
) -> None:
    source, output = tmp_path / "source" / "manifest.json", tmp_path / "report"
    write(source, {"status": "RUNNING", "errors": ["inconsistent cumulative counter"]})
    status = configured(source, output).run(once=True)
    assert "reported_evidence_error" in status["alerts"]
    assert status["reported_evidence_errors"] == 1
    assert (
        "reported_evidence_error" in json.loads((output / "events.json").read_text())[-1]["alerts"]
    )
    assert json.loads((output / "report.json").read_text())["raw_status"] == "RUNNING"
    write(source, {"status": "COMPLETE_FAIL", "phase": "FINISHED", "errors": []})
    status = configured(source, output).run()
    assert "reported_evidence_error" not in status["alerts"]
    assert json.loads((output / "report.json").read_text())["raw_status"] == "COMPLETE_FAIL"
