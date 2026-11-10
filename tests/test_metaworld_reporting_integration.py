"""A real observer process reads a synthetic writer; no environments or learning."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from scripts.metaworld_reporting_sidecar import reporting_sidecar


def test_real_sidecar_reports_parallel_writer_and_fresh_terminal_snapshot(tmp_path: Path) -> None:
    manifest = tmp_path / "sentinel_manifest.json"
    requests: list[dict[str, Any]] = [
        {"task": task, "goal_id": task, "bank_sha256": "b" * 64, "episode_seed": index}
        for index, task in enumerate(("push-v3", "pick-place-v3"))
    ]
    plan = {
        "tasks": requests,
        "source_diagnostic_steps": 1000,
        "episode_horizon": 500,
        "goal_index": 0,
        "output": str(tmp_path),
    }
    data: dict[str, Any] = {
        "plan": plan,
        "plan_sha256": hashlib.sha256(
            json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "status": "RUNNING",
        "validation_complete": False,
        "errors": [],
        "ledger": {"source_diagnostic_steps": 0},
        "episodes": [],
    }
    manifest.write_text(json.dumps(data))
    output = tmp_path / "live_reports"
    with reporting_sidecar(manifest, artifact_limit=8 * 1024**2) as metadata:
        assert metadata["pid"] != os.getpid()
        until = time.monotonic() + 5
        while not (output / "report.json").exists():
            assert time.monotonic() < until, "ready observer did not publish its initial report"
            time.sleep(0.01)
        running = json.loads((output / "report.json").read_text())
        assert running["raw_status"] == "RUNNING" and running["terminal"] is False
        for request in requests:
            trace = tmp_path / (request["task"] + ".jsonl")
            trace.write_text('{"synthetic_fixture":true}\n')
            data["episodes"].append(
                {
                    "task": request["task"],
                    "length": 500,
                    "success": True,
                    "status": "COMPLETE",
                    "complete_episode": True,
                    "initial": {
                        "episode": {
                            "goal_index": 0,
                            "goal_id": request["goal_id"],
                            "goal_bank_sha256": request["bank_sha256"],
                            "episode_seed": request["episode_seed"],
                        }
                    },
                    "trace": trace.name,
                    "trace_sha256": hashlib.sha256(trace.read_bytes()).hexdigest(),
                }
            )
        data.update(status="COMPLETE", validation_complete=True)
        data["ledger"]["source_diagnostic_steps"] = 1000
        # Publish between polls; context shutdown must force a fresh final read.
        pending = tmp_path / "pending.json"
        pending.write_text(json.dumps(data))
        pending.replace(manifest)
        writer_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    final = json.loads((output / "report.json").read_text())
    assert final["terminal"] and final["raw_status"] == "COMPLETE"
    assert final["errors"] == []
    assert final["artifact_verification"]["verified"] == 2
    assert final["observer"]["state"] == "TERMINAL"
    assert metadata["observer_state"] == "FINISHED" and metadata["returncode"] == 0
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == writer_hash
