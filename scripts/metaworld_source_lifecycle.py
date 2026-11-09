"""Stdlib-only terminal source-lifecycle evidence gate for the issue-49 proposal.

Evidence is not approval. The caller must independently require actual issue-49
user approval binding this evidence file's byte hash. Nothing here authorizes,
launches, repairs or cancels source work or a native mechanical check.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

ORIGINAL_MANIFEST = Path(
    "/home/icannon/code/UWSH-nethack/artifacts/metaworld-development-2026-11-09-r2/"
    "development_manifest.json"
)
EXTENSION_MANIFEST = Path(
    "/home/icannon/code/UWSH-nethack/artifacts/metaworld-development-2026-11-09-r2-extension/"
    "development_manifest.json"
)
SOURCE_MODULES = frozenset(
    {
        "scripts.metaworld_development",
        "scripts.metaworld_technical_retry",
        "scripts.metaworld_source_continuation",
        "scripts.metaworld_source_supervisor",
        "scripts.metaworld_competence",
        "scripts.metaworld_hour_pilot",
        "scripts.metaworld_runtime_smoke",
    }
)
TERMINAL_MANIFEST_STATUSES = frozenset({"COMPLETE_PASS", "COMPLETE_FAIL", "INCOMPLETE", "ERROR"})
FAILURE_SUPERVISOR_STATUSES = frozenset(
    {
        "HANDOFF_FAILED_NO_RETRY",
        "STOPPED_BEFORE_HANDOFF",
        "SUPERVISOR_INTERRUPTED_NO_RETRY",
    }
)


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_hash(value: Any) -> str:
    return _hash(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_identity(pid: int) -> dict[str, Any] | None:
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text()
        fields = stat[stat.rindex(")") + 1 :].split()
        return {
            "pid": pid,
            "start_ticks": int(fields[19]),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
            "command_sha256": _hash((root / "cmdline").read_bytes()),
        }
    except FileNotFoundError:
        return None
    except (OSError, ValueError, IndexError) as error:
        raise ValueError("cannot verify source process identity") from error


def active_source_jobs() -> list[dict[str, Any]]:
    """Find known local source entrypoints, including a waiting supervisor.

    This is a check of this project's explicit local entrypoints, not a generic
    scheduler inventory. Pending external work must also be absent from the
    reviewed, approval-bound evidence.
    """
    found = []
    filenames = {name.rsplit(".", 1)[-1] + ".py" for name in SOURCE_MODULES}
    for root in Path("/proc").iterdir():
        if not root.name.isdigit():
            continue
        try:
            if root.stat().st_uid != os.getuid():
                continue
            arguments = (root / "cmdline").read_bytes().split(b"\0")
            argv = [argument.decode(errors="replace") for argument in arguments if argument]
            module = next(
                (argv[index + 1] for index, item in enumerate(argv[:-1]) if item == "-m"),
                None,
            )
            if module in SOURCE_MODULES or any(Path(arg).name in filenames for arg in argv):
                found.append({"pid": int(root.name), "entrypoint": module or "source_script"})
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError("cannot inspect local source-job state") from error
    return found


def _identity(value: Any) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {"pid", "start_ticks", "boot_id", "command_sha256"}
        or type(value.get("pid")) is not int
        or value["pid"] <= 0
        or type(value.get("start_ticks")) is not int
        or value["start_ticks"] <= 0
        or not isinstance(value.get("boot_id"), str)
        or not value["boot_id"]
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("command_sha256", ""))) is None
    ):
        raise ValueError("complete frozen PID/start/boot/command identity is required")
    return value


def _record(value: Any, *, expected_path: Path | None = None) -> tuple[dict[str, Any], str]:
    if (
        not isinstance(value, dict)
        or set(value) != {"path", "sha256"}
        or not isinstance(value.get("path"), str)
        or re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256", ""))) is None
    ):
        raise ValueError("each lifecycle metadata file needs its exact path and byte hash")
    path = Path(value["path"])
    if (
        not path.is_absolute()
        or path.suffix != ".json"
        or path.is_symlink()
        or (expected_path is not None and path != expected_path)
    ):
        raise ValueError("lifecycle evidence must bind the expected absolute JSON metadata path")
    data = path.read_bytes()
    if _hash(data) != value["sha256"]:
        raise ValueError("source lifecycle metadata changed after review")
    parsed = json.loads(data)
    if not isinstance(parsed, dict):
        raise ValueError("source lifecycle metadata must be a JSON object")
    return parsed, value["sha256"]


def validate_source_lifecycle(
    evidence: dict[str, Any],
    *,
    expected_sha256: str,
    evidence_bytes: bytes,
    identity_probe: Callable[[int], dict[str, Any] | None] = process_identity,
    alive_probe: Callable[[int], bool] = process_alive,
    active_jobs_probe: Callable[[], list[dict[str, Any]]] = active_source_jobs,
) -> dict[str, Any]:
    """Require reviewed final/idle source work before native imports or outputs.

    ``expected_sha256`` is SHA256 of the *evidence file bytes*, independently
    bound in actual future issue-49 user approval. A reviewed failure-stop may
    retain incomplete output, but never a live/pending process or an automatic
    restart. All checks are read-only; incomplete evidence is refused.
    """
    if (
        _hash(evidence_bytes) != expected_sha256
        or json.loads(evidence_bytes) != evidence
        or set(evidence)
        != {
            "schema_version",
            "final_idle",
            "pending_source_jobs",
            "automatic_source_restarts",
            "original",
            "extension",
        }
        or evidence.get("schema_version") != 1
        or evidence.get("final_idle") is not True
        or evidence.get("pending_source_jobs") != []
        or type(evidence.get("automatic_source_restarts")) is not int
        or evidence["automatic_source_restarts"] != 0
    ):
        raise ValueError("future approval must bind complete final/idle source lifecycle evidence")
    original, extension = evidence["original"], evidence["extension"]
    if (
        not isinstance(original, dict)
        or set(original) != {"manifest", "process_identity"}
        or not isinstance(extension, dict)
        or set(extension)
        != {
            "supervisor_status",
            "supervisor_identity",
            "parent_binding",
            "launch_record",
            "output_manifest",
            "child_identity",
            "reviewed_disposition",
        }
    ):
        raise ValueError(
            "source lifecycle must include original and full extension handoff records"
        )
    identities = [
        _identity(original["process_identity"]),
        _identity(extension["supervisor_identity"]),
    ]
    exit_pids = [identity["pid"] for identity in identities]
    parent, parent_sha = _record(original["manifest"], expected_path=ORIGINAL_MANIFEST)
    if parent.get("phase") != "FINISHED" or parent.get("status") not in TERMINAL_MANIFEST_STATUSES:
        raise ValueError("original source process has no terminal FINISHED manifest")
    supervisor, supervisor_sha = _record(extension["supervisor_status"])
    if (
        supervisor.get("old_pid") != identities[0]["pid"]
        or supervisor.get("old_process_identity") != identities[0]
    ):
        raise ValueError("supervisor does not bind this original source process identity")
    status, disposition = supervisor.get("status"), extension["reviewed_disposition"]
    if status == "FINISHED" and type(supervisor.get("returncode")) is not int:
        raise ValueError("finished source supervisor must retain the actual child return code")
    if disposition == "finished":
        if status != "FINISHED":
            raise ValueError("extension supervisor is active, pending or not finished")
    elif disposition == "noop":
        if status != "NOOP_ALREADY_COMPLETE":
            raise ValueError("no-op disposition lacks its terminal supervisor evidence")
    elif disposition == "failure_stop":
        if status not in FAILURE_SUPERVISOR_STATUSES and not (
            status == "FINISHED"
            and type(supervisor.get("returncode")) is int
            and supervisor["returncode"] != 0
        ):
            raise ValueError("reviewed failure-stop lacks terminal failed supervisor evidence")
    else:
        raise ValueError("future approval must review the actual terminal source disposition")
    binding = None
    if extension["parent_binding"] is not None:
        binding, _ = _record(extension["parent_binding"])
        if (
            binding.get("parent_manifest_sha256") != parent_sha
            or binding.get("parent_manifest") != parent
            or binding.get("old_pid") != identities[0]["pid"]
            or binding.get("old_process_identity") != identities[0]
        ):
            raise ValueError("extension handoff does not bind the final original source evidence")
    elif disposition != "failure_stop":
        raise ValueError("finished/no-op source lifecycle requires its frozen parent binding")
    child_pid = supervisor.get("child_pid")
    if child_pid is not None:
        if type(child_pid) is not int or child_pid <= 0 or binding is None:
            raise ValueError("launched extension child requires its known positive PID and binding")
        if extension["child_identity"] is not None:
            child = _identity(extension["child_identity"])
            if child["pid"] != child_pid:
                raise ValueError("recorded child identity differs from the supervisor PID")
            identities.append(child)
        exit_pids.append(child_pid)
        launch, _ = _record(extension["launch_record"])
        if (
            launch.get("parent_binding_sha256") != _canonical_hash(binding)
            or launch.get("old_process_identity") != identities[0]
        ):
            raise ValueError("one-shot source launch does not match the frozen handoff")
        if extension["output_manifest"] is None:
            if disposition != "failure_stop" or EXTENSION_MANIFEST.exists():
                raise ValueError("launched source child is missing its bound output evidence")
        else:
            output, _ = _record(extension["output_manifest"], expected_path=EXTENSION_MANIFEST)
            if output.get("parent_binding") != binding:
                raise ValueError("extension output belongs to another source handoff")
            if disposition != "failure_stop" and (
                output.get("phase") != "FINISHED"
                or output.get("status") not in TERMINAL_MANIFEST_STATUSES
            ):
                raise ValueError("extension source output is still active or incomplete")
    else:
        if any(
            extension[key] is not None
            for key in ("child_identity", "launch_record", "output_manifest")
        ):
            raise ValueError(
                "unlaunched lifecycle cannot omit an existing child from its supervisor"
            )
        if disposition == "finished" or EXTENSION_MANIFEST.exists():
            raise ValueError(
                "finished extension needs child evidence; existing output cannot be omitted"
            )
        if disposition == "noop" and binding is not None and binding.get("unfinished_tasks") != []:
            raise ValueError("source no-op still contains pending unfinished policies")
    if len(set(exit_pids)) != len(exit_pids):
        raise ValueError("source lifecycle process identities must be distinct")
    for pid in exit_pids:
        if identity_probe(pid) is not None or alive_probe(pid):
            raise ValueError(
                "a relevant source/supervisor process is still alive or its PID was reused"
            )
    jobs = active_jobs_probe()
    if jobs:
        raise ValueError("active or waiting source jobs remain; native validation must wait")
    return {
        "verified_final_idle": True,
        "evidence_file_sha256": expected_sha256,
        "original_pid": identities[0]["pid"],
        "original_manifest_sha256": parent_sha,
        "supervisor_status_sha256": supervisor_sha,
        "extension_disposition": disposition,
        "dead_process_identities": identities,
        "all_known_pids_checked_gone": exit_pids,
        "child_identity_available": extension["child_identity"] is not None,
        "active_source_jobs": [],
        "approval_note": (
            "lifecycle evidence only; actual issue-49 approval is independently required"
        ),
    }


def load_and_validate_source_lifecycle(
    path: Path, expected_sha256: str, **probes: Any
) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or path.suffix != ".json":
        raise ValueError("approval must bind an absolute nonsymlink source lifecycle JSON file")
    data = path.read_bytes()
    evidence = json.loads(data)
    if not isinstance(evidence, dict):
        raise ValueError("source lifecycle evidence must be a JSON object")
    return validate_source_lifecycle(
        evidence, expected_sha256=expected_sha256, evidence_bytes=data, **probes
    )
