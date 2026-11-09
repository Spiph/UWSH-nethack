"""Only synthetic JSON/process probes; no source artifacts, native imports or jobs."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import metaworld_source_lifecycle as lifecycle


def identity(pid: int) -> dict[str, Any]:
    return {
        "pid": pid,
        "start_ticks": pid * 10,
        "boot_id": "synthetic-boot",
        "command_sha256": "a" * 64,
    }


def write(path: Path, value: Any) -> dict[str, str]:
    data = json.dumps(value, sort_keys=True).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def evidence_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    original_path = tmp_path / "original" / "development_manifest.json"
    output_path = tmp_path / "extension" / "development_manifest.json"
    monkeypatch.setattr(lifecycle, "ORIGINAL_MANIFEST", original_path)
    monkeypatch.setattr(lifecycle, "EXTENSION_MANIFEST", output_path)
    original = {"phase": "FINISHED", "status": "INCOMPLETE"}
    original_record = write(original_path, original)
    binding = {
        "parent_manifest_sha256": original_record["sha256"],
        "parent_manifest": original,
        "old_pid": 101,
        "old_process_identity": identity(101),
        "unfinished_tasks": ["push-v3"],
    }
    launch = {
        "parent_binding_sha256": lifecycle._canonical_hash(binding),
        "old_process_identity": identity(101),
    }
    supervisor = {
        "status": "FINISHED",
        "old_pid": 101,
        "old_process_identity": identity(101),
        "child_pid": 103,
        "returncode": 0,
    }
    return {
        "schema_version": 1,
        "final_idle": True,
        "pending_source_jobs": [],
        "automatic_source_restarts": 0,
        "original": {"manifest": original_record, "process_identity": identity(101)},
        "extension": {
            "supervisor_status": write(tmp_path / "control" / "status.json", supervisor),
            "supervisor_identity": identity(102),
            "parent_binding": write(tmp_path / "control" / "binding.json", binding),
            "launch_record": write(tmp_path / "extension.launch_once.json", launch),
            "output_manifest": write(
                output_path,
                {"phase": "FINISHED", "status": "COMPLETE_FAIL", "parent_binding": binding},
            ),
            "child_identity": None,
            "reviewed_disposition": "finished",
        },
    }


def validate(evidence: dict[str, Any], **probes: Any) -> dict[str, Any]:
    data = json.dumps(evidence).encode()
    return lifecycle.validate_source_lifecycle(
        evidence,
        expected_sha256=hashlib.sha256(data).hexdigest(),
        evidence_bytes=data,
        identity_probe=probes.get("identity_probe", lambda pid: None),
        alive_probe=probes.get("alive_probe", lambda pid: False),
        active_jobs_probe=probes.get("active_jobs_probe", lambda: []),
    )


def change_record(evidence: dict[str, Any], section: str, key: str, **updates: Any) -> None:
    record = evidence[section][key]
    value = json.loads(Path(record["path"]).read_text())
    value.update(updates)
    evidence[section][key] = write(Path(record["path"]), value)


def test_finished_original_and_extension_require_full_handoff_not_single_dead_pid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = evidence_fixture(tmp_path, monkeypatch)
    report = validate(evidence)
    assert report["verified_final_idle"]
    assert report["original_pid"] == 101
    assert report["all_known_pids_checked_gone"] == [101, 102, 103]
    assert not report["child_identity_available"]
    assert "independently required" in report["approval_note"]
    evidence["extension"]["child_identity"] = identity(103)
    assert validate(evidence)["child_identity_available"]


@pytest.mark.parametrize("pid", [101, 102, 103])
def test_any_relevant_live_or_reused_pid_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid: int
) -> None:
    evidence = evidence_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="still alive"):
        validate(evidence, alive_probe=lambda value: value == pid)
    with pytest.raises(ValueError, match="PID was reused"):
        validate(evidence, identity_probe=lambda value: identity(value) if value == pid else None)


@pytest.mark.parametrize("status", ["WAITING_FOR_OLD_PROCESS", "RUNNING", "unknown"])
def test_pending_supervisor_cannot_hide_behind_exited_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    evidence = evidence_fixture(tmp_path, monkeypatch)
    change_record(evidence, "extension", "supervisor_status", status=status)
    with pytest.raises(ValueError, match="active, pending"):
        validate(evidence)


def test_active_source_scan_blocks_even_if_all_recorded_pids_exited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = evidence_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="active or waiting"):
        validate(
            evidence, active_jobs_probe=lambda: [{"pid": 999, "entrypoint": "source_supervisor"}]
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("final_idle", False),
        ("pending_source_jobs", ["extension"]),
        ("automatic_source_restarts", 1),
    ],
)
def test_source_work_must_be_reviewed_final_and_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any
) -> None:
    evidence = evidence_fixture(tmp_path, monkeypatch)
    evidence[field] = value
    with pytest.raises(ValueError, match="final/idle"):
        validate(evidence)


def test_reference_hash_tamper_and_foreign_handoff_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = evidence_fixture(tmp_path, monkeypatch)
    path = Path(evidence["extension"]["output_manifest"]["path"])
    saved = path.read_bytes()
    path.write_text("{}")
    with pytest.raises(ValueError, match="metadata changed"):
        validate(evidence)
    path.write_bytes(saved)
    change_record(evidence, "extension", "launch_record", parent_binding_sha256="b" * 64)
    with pytest.raises(ValueError, match="one-shot source launch"):
        validate(evidence)


def test_live_or_unfinished_source_output_fails_without_explicit_reviewed_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = evidence_fixture(tmp_path, monkeypatch)
    change_record(
        evidence, "extension", "output_manifest", phase="SOURCE_DEVELOPMENT", status="RUNNING"
    )
    with pytest.raises(ValueError, match="still active or incomplete"):
        validate(evidence)
    evidence["extension"]["reviewed_disposition"] = "failure_stop"
    # The future approval must bind this actual failed disposition; it cannot
    # turn a successful/active supervisor record into reviewed failure evidence.
    with pytest.raises(ValueError, match="terminal failed"):
        validate(evidence)
    change_record(evidence, "extension", "supervisor_status", returncode=1)
    assert validate(evidence)["extension_disposition"] == "failure_stop"


def test_noop_requires_empty_pending_tasks_and_no_child_or_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = evidence_fixture(tmp_path, monkeypatch)
    extension = evidence["extension"]
    lifecycle.EXTENSION_MANIFEST.unlink()
    extension.update(
        reviewed_disposition="noop", child_identity=None, launch_record=None, output_manifest=None
    )
    status = json.loads(Path(extension["supervisor_status"]["path"]).read_text())
    status.update(status="NOOP_ALREADY_COMPLETE")
    status.pop("child_pid")
    extension["supervisor_status"] = write(Path(extension["supervisor_status"]["path"]), status)
    with pytest.raises(ValueError, match="pending unfinished"):
        validate(evidence)
    binding = json.loads(Path(extension["parent_binding"]["path"]).read_text())
    binding["unfinished_tasks"] = []
    extension["parent_binding"] = write(Path(extension["parent_binding"]["path"]), binding)
    assert validate(evidence)["extension_disposition"] == "noop"


def test_reviewed_failure_before_child_needs_no_invented_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = evidence_fixture(tmp_path, monkeypatch)
    extension = evidence["extension"]
    lifecycle.EXTENSION_MANIFEST.unlink()
    extension.update(
        reviewed_disposition="failure_stop",
        child_identity=None,
        parent_binding=None,
        launch_record=None,
        output_manifest=None,
    )
    status = json.loads(Path(extension["supervisor_status"]["path"]).read_text())
    status.update(status="STOPPED_BEFORE_HANDOFF")
    status.pop("child_pid")
    extension["supervisor_status"] = write(Path(extension["supervisor_status"]["path"]), status)
    assert validate(evidence)["all_known_pids_checked_gone"] == [101, 102]


def test_evidence_file_bytes_and_complete_schema_are_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = evidence_fixture(tmp_path, monkeypatch)
    path = tmp_path / "source_lifecycle.json"
    record = write(path, evidence)
    report = lifecycle.load_and_validate_source_lifecycle(
        path,
        record["sha256"],
        identity_probe=lambda pid: None,
        alive_probe=lambda pid: False,
        active_jobs_probe=lambda: [],
    )
    assert report["evidence_file_sha256"] == record["sha256"]
    with pytest.raises(ValueError, match="final/idle"):
        lifecycle.load_and_validate_source_lifecycle(path, "b" * 64)
    incomplete = copy.deepcopy(evidence)
    incomplete.pop("extension")
    with pytest.raises(ValueError, match="final/idle"):
        validate(incomplete)


def test_proc_scan_includes_pending_supervisor_but_excludes_native_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for pid, module in (
        (101, "scripts.metaworld_source_supervisor"),
        (102, "scripts.metaworld_adaptation_runtime_smoke"),
        (103, "scripts.metaworld_technical_retry"),
    ):
        directory = tmp_path / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(b"python\0-m\0" + module.encode() + b"\0")
    original_path = Path
    monkeypatch.setattr(
        lifecycle, "Path", lambda value: tmp_path if value == "/proc" else original_path(value)
    )
    assert sorted(lifecycle.active_source_jobs(), key=lambda row: row["pid"]) == [
        {"pid": 101, "entrypoint": "scripts.metaworld_source_supervisor"},
        {"pid": 103, "entrypoint": "scripts.metaworld_technical_retry"},
    ]
