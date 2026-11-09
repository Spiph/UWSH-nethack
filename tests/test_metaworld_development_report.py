from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.metaworld_development_report import main, render_markdown, summarize_manifest

TASKS = ("reach-v3", "push-v3", "pick-place-v3")


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bank(task: str, split: str, count: int, offset: int) -> dict[str, Any]:
    goals = []
    for index in range(count):
        payload = f"synthetic-only-{split}-{index}".encode()
        vector = [float(index + offset)]
        goals.append(
            {
                "task": task,
                "payload_base64": base64.b64encode(payload).decode(),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "random_vector": vector,
                "goal_id": digest({"task": task, "random_vector": vector}),
            }
        )
    return {"split": split, "seed": 73 + offset, "goals": goals}


def checkpoint(target: int, successes: int, goals: dict[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "episode": index,
            "return": float(index),
            "length": 500,
            "success": index < successes,
            "complete_episode": True,
            "terminated": False,
            "truncated": True,
            "goal_index": index,
            "episode_seed": index + 400,
            "goal_bank_sha256": digest(goals),
            "goal_id": goals["goals"][index]["goal_id"],
            "goal": copy.deepcopy(goals["goals"][index]),
        }
        for index in range(20)
    ]
    return {
        "target_steps": target,
        "actual_steps": target,
        "partial": False,
        "evaluation": rows,
        "evaluation_complete": True,
        "passed": successes >= 16,
        "evaluation_duration_seconds": 5.0,
        "checkpoint_duration_seconds": 1.0,
        "artifacts": {},
    }


def fixture(*, passed: bool = True) -> dict[str, Any]:
    plan = {
        "tasks": [{"task": task} for task in TASKS],
        "milestones": [100000, 250000],
        "evaluation": {"episodes": 20, "episode_horizon": 500, "required_successes": 16},
        "goal_counts": {"source": 50, "development": 20},
        "output": "/synthetic-only/development-attempt",
    }
    manifest: dict[str, Any] = {
        "status": "COMPLETE_PASS" if passed else "COMPLETE_FAIL",
        "purpose": "synthetic unit fixture only",
        "plan": plan,
        "plan_sha256": digest(plan),
        "session_started_utc": "2026-11-09T15:35:00Z",
        "started_at_utc": "2026-11-09T17:56:00Z",
        "finished_at_utc": "2026-11-09T18:06:00Z",
        "absolute_deadline_utc": "2026-11-09T21:35:00Z",
        "training_deadline_utc": "2026-11-09T21:05:00Z",
        "elapsed_seconds": 600.0,
        "utc_elapsed_seconds": 600.0,
        "budget_compliant": True,
        "tasks": {},
        "queue": [],
        "errors": [],
        "goal_manifests": {},
        "ledger": {
            "source_training_steps": 300000 if passed else 750000,
            "development_evaluation_steps": 30000 if passed else 60000,
            "preflight_steps": 3,
            "source_policy_starts": 3,
            "failed_step_calls": 0,
            "optimizer_update_attempts": 297000 if passed else 747000,
            "optimizer_updates": 297000 if passed else 747000,
            "target_steps": 0,
            "final_test_steps": 0,
        },
    }
    manifest["ledger"]["total_agent_steps"] = (
        manifest["ledger"]["source_training_steps"]
        + manifest["ledger"]["development_evaluation_steps"]
        + 3
    )
    for task in TASKS:
        development = bank(task, "development", 20, 100)
        manifest["goal_manifests"][task] = {
            "banks": [bank(task, "source", 50, 0), development],
        }
        checkpoints = [checkpoint(100000, 16 if passed else 0, development)]
        if not passed:
            checkpoints.append(checkpoint(250000, 15, development))
        manifest["tasks"][task] = {
            "status": "PASS" if passed else "FULL_BUDGET_FAIL",
            "actual_training_steps": 100000 if passed else 250000,
            "metadata": {
                "evaluation_goal_order": list(range(20)),
                "evaluation_episode_seeds": list(range(400, 420)),
            },
            "checkpoints": checkpoints,
        }
        for row in checkpoints:
            manifest["queue"].append(
                {
                    "task": task,
                    "milestone": row["target_steps"],
                    "training_duration_seconds": 50.0,
                }
            )
    return manifest


def test_earliest_pass_has_all_planned_cells_and_no_independent_episode_claims() -> None:
    manifest = fixture()
    original = copy.deepcopy(manifest)
    report = summarize_manifest(manifest, strict_coverage=True)
    assert manifest == original
    assert report["status"] == "COMPLETE_PASS"
    assert len(report["scheduled_checkpoints"]) == 6
    assert (
        sum(
            row["status"] == "SKIPPED_PLANNED_EARLY_STOP" for row in report["scheduled_checkpoints"]
        )
        == 3
    )
    assert all(row["earliest_passing_milestone"] == 100000 for row in report["task_outcomes"])
    assert report["validation"]["errors"] == []
    markdown = render_markdown(report)
    assert "SKIPPED_PLANNED_EARLY_STOP" in markdown
    assert "not library replications" in markdown
    assert "No episode confidence intervals" in markdown


def test_completed_maximum_budget_failure_is_distinct_from_missing_later_visit() -> None:
    manifest = fixture(passed=False)
    full = summarize_manifest(manifest, strict_coverage=True)
    assert full["status"] == "COMPLETE_FAIL"
    assert all(row["outcome"] == "FULL_BUDGET_FAIL" for row in full["task_outcomes"])
    manifest["status"] = "INCOMPLETE"
    manifest["tasks"][TASKS[-1]]["checkpoints"].pop()
    manifest["tasks"][TASKS[-1]]["actual_training_steps"] = 100000
    report = summarize_manifest(manifest, strict_coverage=True)
    assert report["status"] == "INCOMPLETE"
    assert report["task_outcomes"][-1]["outcome"] == "INCOMPLETE"
    assert report["scheduled_checkpoints"][-1]["status"] == "MISSING"


@pytest.mark.parametrize("partial_training", [False, True])
def test_cutoff_or_partial_episode_does_not_become_an_unobserved_budget_failure(
    partial_training: bool,
) -> None:
    manifest = fixture(passed=False)
    manifest["status"] = "INCOMPLETE"
    manifest["stop_reason"] = "training_reserve" if partial_training else "absolute_utc_deadline"
    row = manifest["tasks"][TASKS[-1]]["checkpoints"][-1]
    row["evaluation_complete"] = False
    row["passed"] = False
    if partial_training:
        row.update(partial=True, actual_steps=140000, evaluation=[])
    else:
        row["evaluation"] = row["evaluation"][:3]
        row["evaluation"][-1].update(length=73, complete_episode=False, truncated=False)
    report = summarize_manifest(manifest, strict_coverage=True)
    assert report["status"] == "INCOMPLETE"
    assert report["task_outcomes"][-1]["outcome"] == "INCOMPLETE"
    final = report["scheduled_checkpoints"][-1]
    assert final["partial_episodes"] == (0 if partial_training else 1)
    assert final["missing_episodes"] == (20 if partial_training else 17)


def test_all_twenty_rows_without_completion_marker_or_running_finish_never_pass() -> None:
    manifest = fixture()
    manifest["status"] = "INCOMPLETE"
    row = manifest["tasks"][TASKS[-1]]["checkpoints"][0]
    row.update(evaluation_complete=False, passed=False)
    assert summarize_manifest(manifest)["task_outcomes"][-1]["outcome"] == "INCOMPLETE"
    manifest = fixture()
    manifest["status"] = "RUNNING"
    report = summarize_manifest(manifest)
    assert report["status"] == "INCOMPLETE"
    assert report["derived_operational_gate_status"] == "COMPLETE_PASS"
    assert report["evidence_mode"] == "AS_OF_LOWER_BOUND"
    assert report["timing"]["snapshot_timestamp_is_final_finish"] is False


@pytest.mark.parametrize("damage", ["length", "seed", "goal", "duplicate", "complete_flag"])
def test_protocol_errors_are_not_competence_failures(damage: str) -> None:
    manifest = fixture()
    checkpoint_row = manifest["tasks"][TASKS[0]]["checkpoints"][0]
    row = checkpoint_row["evaluation"][0]
    if damage == "length":
        row["length"] = 499
    elif damage == "seed":
        row["episode_seed"] += 1
    elif damage == "goal":
        row["goal_id"] = "wrong"
    elif damage == "duplicate":
        checkpoint_row["evaluation"][1]["episode"] = 0
    else:
        checkpoint_row["evaluation"] = []
    report = summarize_manifest(manifest, strict_coverage=True)
    assert report["status"] == "ERROR"
    assert report["task_outcomes"][0]["outcome"] == "ERROR"
    assert report["validation"]["errors"]


def retry_fixture() -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = fixture()
    prior = copy.deepcopy(manifest)
    prior.update(
        status="ERROR",
        started_at_utc="2026-11-09T16:10:59Z",
        finished_at_utc="2026-11-09T16:11:10Z",
        elapsed_seconds=11.0,
    )
    prior["ledger"] = dict.fromkeys(manifest["ledger"], 0)
    prior["ledger"].update(preflight_steps=1, source_policy_starts=3, total_agent_steps=1)
    manifest["ledger"].update(
        preflight_steps=4,
        source_policy_starts=6,
        total_agent_steps=330004,
        retries=1,
        technical_retries=1,
    )
    manifest["technical_retry_accounting"] = {
        "prior_ledger": copy.deepcopy(prior["ledger"]),
        "cumulative_ledger": copy.deepcopy(manifest["ledger"]),
        "prior_artifact_bytes": 100,
    }
    return manifest, prior


def test_failed_prior_is_charged_exactly_once_and_timing_separates_wait_from_compute() -> None:
    manifest, prior = retry_fixture()
    report = summarize_manifest(manifest, prior_manifest=prior)
    costs, timing = report["costs"], report["timing"]
    assert report["status"] == "COMPLETE_PASS"
    assert costs["component_sum_at_manifest"] == 330004
    assert costs["cumulative_components_at_manifest"]["source_policy_starts"] == 6
    assert costs["attempt_only_components"]["source_policy_starts"] == 3
    assert costs["retained_prior_components"]["preflight_steps"] == 1
    assert costs["attempt_only_components"]["preflight_steps"] == 3
    assert costs["cumulative_components_at_manifest"]["technical_retries"] == 1
    assert timing["combined_measured_runtime_monotonic_seconds"] == 611
    assert timing["pre_run_setup_repair_approval_wait_wall_seconds"] == 8460
    assert timing["measured_training_seconds"] == 150
    assert timing["measured_evaluation_seconds"] == 15
    assert timing["inclusive_session_wall_seconds_as_of"] == 9060


def test_raw_stale_aggregate_is_preserved_and_live_progress_is_separate_lower_bound() -> None:
    manifest, prior = retry_fixture()
    manifest["status"] = "RUNNING"
    manifest["ledger"]["total_agent_steps"] = 4
    progress = [
        {
            "task": TASKS[0],
            "utc": "2026-11-09T18:07:00Z",
            "training_steps": 101000,
            "ledger": manifest["ledger"]
            | {"source_training_steps": 301000, "total_agent_steps": 4},
        }
    ]
    report = summarize_manifest(manifest, prior_manifest=prior, progress=progress)
    assert report["status"] == "INCOMPLETE"
    costs = report["costs"]
    assert costs["raw_total_agent_steps_at_manifest"] == 4
    assert costs["component_sum_at_manifest"] == 330004
    assert costs["monitoring_components_lower_bound"]["total_agent_steps"] == 331004
    assert report["as_of_observed_utc"] == "2026-11-09T18:07:00+00:00"
    assert report["validation"]["warnings"]
    manifest["status"] = "COMPLETE_PASS"
    report = summarize_manifest(manifest, prior_manifest=prior, progress=progress)
    assert report["status"] == "ERROR"
    assert report["costs"]["monitoring_components_lower_bound"] is None


@pytest.mark.parametrize(
    "task,time", [("foreign-v3", "2026-11-09T18:00:00Z"), (TASKS[0], "2026-11-08T18:00:00Z")]
)
def test_progress_from_another_cohort_is_rejected(task: str, time: str) -> None:
    manifest = fixture()
    manifest["status"] = "RUNNING"
    report = summarize_manifest(
        manifest, progress=[{"task": task, "utc": time, "ledger": manifest["ledger"]}]
    )
    assert report["status"] == "ERROR"


def test_runtime_failure_and_utc_monotonic_discrepancy_are_reported_without_causal_guess() -> None:
    manifest = fixture()
    manifest["status"] = "ERROR"
    manifest["errors"] = ["synthetic serialization failure"]
    manifest["elapsed_seconds"] = 100
    report = summarize_manifest(manifest)
    assert report["status"] == "ERROR"
    assert report["timing"]["utc_minus_monotonic_seconds"] == 500
    assert report["stopping"]["errors"] == manifest["errors"]
    assert "unverified causes" in report["timing"]["interpretation"]


def test_failed_preflight_does_not_fabricate_source_competence_or_optimizer_work() -> None:
    _, manifest = retry_fixture()
    manifest["errors"] = ["synthetic preflight serialization failure"]
    for task in manifest["tasks"].values():
        task.update(status="PENDING", actual_training_steps=0, checkpoints=[])
    manifest["queue"] = []
    report = summarize_manifest(manifest, strict_coverage=True)
    assert report["status"] == "ERROR"
    assert all(row["outcome"] == "INCOMPLETE" for row in report["task_outcomes"])
    assert report["costs"]["component_sum_at_manifest"] == 1
    assert report["costs"]["cumulative_components_at_manifest"]["optimizer_updates"] == 0
    assert report["costs"]["cumulative_components_at_manifest"]["source_policy_starts"] == 3


@pytest.mark.parametrize("damage", ["plan_hash", "repeated_goal_order", "cost_cap", "prior"])
def test_evidence_commitment_and_cost_errors_fail_closed(damage: str) -> None:
    manifest, prior = retry_fixture()
    if damage == "plan_hash":
        manifest["plan_sha256"] = "0" * 64
    elif damage == "repeated_goal_order":
        manifest["tasks"][TASKS[0]]["metadata"]["evaluation_goal_order"] = [0] * 20
    elif damage == "cost_cap":
        manifest["technical_retry_contract"] = {
            "cumulative_interaction_caps": {"source_training_steps": 1}
        }
    else:
        prior["ledger"]["preflight_steps"] = 2
    report = summarize_manifest(manifest, prior_manifest=prior, strict_coverage=True)
    assert report["status"] == "ERROR"
    assert report["validation"]["errors"]


def test_artifact_inventory_hashes_bytes_without_loading_files_and_preserves_them(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    root.mkdir()
    blob = root / "synthetic.actor.npz"
    content = b"synthetic bytes; not a NumPy archive or policy"
    blob.write_bytes(content)
    manifest = fixture()
    manifest["tasks"][TASKS[0]]["checkpoints"][0]["artifacts"] = {
        "actor": {"path": blob.name, "sha256": hashlib.sha256(content).hexdigest()},
    }
    report = summarize_manifest(manifest, artifact_root=root)
    assert report["status"] == "COMPLETE_PASS"
    inventory = report["storage"]["actual_inventory"]
    assert inventory["bytes"] == len(content)
    assert inventory["files"][0]["sha256"] == hashlib.sha256(content).hexdigest()
    assert blob.read_bytes() == content
    blob.write_bytes(b"changed synthetic bytes")
    assert summarize_manifest(manifest, artifact_root=root)["status"] == "ERROR"


def test_cli_outputs_outside_run_exclusively_and_imports_no_rl_runtime(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = tmp_path / "run"
    run.mkdir()
    source = run / "manifest.json"
    source.write_text(json.dumps(fixture()))
    output = tmp_path / "report.json"
    assert main(["--manifest", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "COMPLETE_PASS"
    with pytest.raises(FileExistsError):
        main(["--manifest", str(source), "--output", str(output)])
    with pytest.raises(ValueError, match="outside"):
        main(["--manifest", str(source), "--output", str(run / "unsafe.json")])
    link = tmp_path / "dangling-report.json"
    link.symlink_to(tmp_path / "new-target.json")
    with pytest.raises(FileExistsError):
        main(["--manifest", str(source), "--output", str(link)])
    main(["--manifest", str(source), "--format", "markdown"])
    assert "Developmental competence" in capsys.readouterr().out
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import scripts.metaworld_development_report; "
            "assert not any(name in sys.modules for name in "
            "('numpy','torch','gymnasium','metaworld','scripts.metaworld_development',"
            "'scripts.metaworld_development_plan','scripts.metaworld_technical_retry'))",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
