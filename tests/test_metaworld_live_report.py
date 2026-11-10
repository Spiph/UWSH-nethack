"""Synthetic JSON evidence only; no learner, environment or artifact loads."""

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

from scripts.metaworld_live_report import render_markdown, summarize_run

OBSERVED = "2026-11-10T05:00:00+00:00"
TASKS = ("reach-v3", "push-v3", "pick-place-v3")


def sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def bank(task: str, split: str, size: int, offset: int) -> dict[str, Any]:
    goals = []
    for index in range(size):
        payload = f"fake-{task}-{split}-{index}".encode()
        vector = [float(index + offset)]
        goals.append(
            {
                "task": task,
                "payload_base64": base64.b64encode(payload).decode(),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
                "random_vector": vector,
                "goal_id": sha({"task": task, "random_vector": vector}),
            }
        )
    return {"split": split, "seed": offset, "goals": goals}


def checkpoint(task: str, milestone: int, successes: int, goals: dict[str, Any]) -> dict[str, Any]:
    return {
        "target_steps": milestone,
        "actual_steps": milestone,
        "partial": False,
        "evaluation_complete": True,
        "passed": successes >= 16,
        "evaluation": [
            {
                "episode": index,
                "goal_index": index,
                "episode_seed": index + 400,
                "goal_bank_sha256": sha(goals),
                "goal_id": goals["goals"][index]["goal_id"],
                "goal": copy.deepcopy(goals["goals"][index]),
                "return": float(index),
                "length": 500,
                "success": index < successes,
                "complete_episode": True,
                "terminated": False,
                "truncated": True,
            }
            for index in range(20)
        ],
        "artifacts": {
            ".zip": {"path": f"checkpoints/{task}/step{milestone}.zip", "sha256": "a" * 64}
        },
    }


def original(*, scientific_failure: bool = False) -> dict[str, Any]:
    plan = {
        "tasks": [{"task": task} for task in TASKS],
        "milestones": [100000, 250000],
        "evaluation": {"episodes": 20, "episode_horizon": 500, "required_successes": 16},
        "goal_counts": {"source": 50, "development": 20},
        "output": "/failed-first-attempt",
        "limits": {
            "source_training_steps": 750000,
            "development_evaluation_steps": 60000,
            "preflight_steps": 1000,
            "source_policy_starts": 3,
        },
    }
    result: dict[str, Any] = {
        "plan": plan,
        "plan_sha256": sha(plan),
        "status": "COMPLETE_FAIL" if scientific_failure else "COMPLETE_PASS",
        "phase": "FINISHED",
        "errors": [],
        "goal_manifests": {},
        "tasks": {},
        "finished_at_utc": "2026-11-10T04:00:00+00:00",
        "attempt_artifact_root": "/original",
        "ledger": {
            "source_training_steps": 600000 if scientific_failure else 300000,
            "development_evaluation_steps": 50000 if scientific_failure else 30000,
            "preflight_steps": 4,
            "source_policy_starts": 6,
            "optimizer_updates": 597000 if scientific_failure else 297000,
            "optimizer_update_attempts": 597000 if scientific_failure else 297000,
            "target_steps": 0,
            "final_test_steps": 0,
            "retries": 1,
            "technical_retries": 1,
        },
        "technical_retry_contract": {"cumulative_policy_starts": 6, "technical_retries": 1},
    }
    result["technical_retry_contract_sha256"] = sha(result["technical_retry_contract"])
    result["approval"] = {
        "approved": True,
        "technical_retry_contract_sha256": result["technical_retry_contract_sha256"],
    }
    for task in TASKS:
        goals = bank(task, "development", 20, 100)
        result["goal_manifests"][task] = {"banks": [bank(task, "source", 50, 0), goals]}
        fails = scientific_failure and task != "reach-v3"
        rows = [checkpoint(task, 100000, 0 if fails else 16, goals)]
        if fails:
            rows.append(checkpoint(task, 250000, 1, goals))
        result["tasks"][task] = {
            "status": "FULL_BUDGET_FAIL" if fails else "PASS",
            "actual_training_steps": 250000 if fails else 100000,
            "metadata": {
                "evaluation_goal_order": list(range(20)),
                "evaluation_episode_seeds": list(range(400, 420)),
            },
            "checkpoints": rows,
        }
    result["ledger"]["total_agent_steps"] = sum(
        result["ledger"][key]
        for key in ("source_training_steps", "development_evaluation_steps", "preflight_steps")
    )
    result["technical_retry_accounting"] = {
        "cumulative_ledger": copy.deepcopy(result["ledger"]),
        "prior_ledger": {"source_policy_starts": 3, "preflight_steps": 1},
    }
    return result


def continuation() -> dict[str, Any]:
    parent = original(scientific_failure=True)
    result = copy.deepcopy(parent)
    result.pop("technical_retry_contract")
    result.pop("technical_retry_accounting")
    result.pop("technical_retry_contract_sha256")
    result.pop("attempt_artifact_root")
    result["parent_binding"] = {"parent_manifest": parent}
    result["plan"] = {
        **copy.deepcopy(parent["plan"]),
        "original_scientific_plan": copy.deepcopy(parent["plan"]),
        "parent_binding_sha256": sha(result["parent_binding"]),
        "extension": {
            "parent_root": "/original",
            "output": "/extension",
            "milestones": [100000, 250000, 500000, 1000000],
            "limits": {
                "source_training_steps": 2100000,
                "development_evaluation_steps": 90000,
                "preflight_steps": 1000,
                "source_policy_starts": 8,
            },
        },
    }
    result["plan_sha256"] = sha(result["plan"])
    result["ledger"].update(
        source_training_steps=2100000,
        development_evaluation_steps=90000,
        preflight_steps=219,
        source_policy_starts=8,
        total_agent_steps=2190219,
        optimizer_updates=2097000,
        optimizer_update_attempts=2097000,
    )
    for task in TASKS[1:]:
        state = result["tasks"][task]
        for row in state["checkpoints"]:
            row["artifact_root"] = "/original"
        goals = result["goal_manifests"][task]["banks"][1]
        state["checkpoints"].extend(checkpoint(task, step, 0, goals) for step in (500000, 1000000))
        state["actual_training_steps"] = 1000000
    partial = {
        "target_steps": 250000,
        "actual_steps": 200215,
        "partial": True,
        "passed": False,
        "evaluation": [],
        "artifacts": {},
        "artifact_root": "/original",
    }
    result["tasks"]["pick-place-v3"]["checkpoints"].insert(1, partial)
    return result


def sentinel() -> dict[str, Any]:
    requests: list[dict[str, Any]] = [
        {
            "task": task,
            "goal_id": task + "-goal",
            "bank_sha256": "b" * 64,
            "episode_seed": index + 100,
        }
        for index, task in enumerate(TASKS[1:])
    ]
    plan = {
        "tasks": requests,
        "source_diagnostic_steps": 1000,
        "episode_horizon": 500,
        "goal_index": 0,
        "output": "/sentinel",
        "environment_constructors": 2,
        "scripted_controller_instances": 2,
        "source_actor_array_loads": 0,
        "sac_initializations": 0,
        "optimizer_updates": 0,
    }
    return {
        "plan": plan,
        "plan_sha256": sha(plan),
        "status": "COMPLETE",
        "validation_complete": True,
        "errors": [],
        "ledger": {
            "source_diagnostic_steps": 1000,
            "source_diagnostic_step_attempts": 1000,
            "environment_constructor_attempts": 2,
            "environment_constructors": 2,
            "scripted_constructor_attempts": 2,
            "scripted_controllers": 2,
            "development_steps": 0,
            "target_steps": 0,
            "final_test_steps": 0,
            "actor_array_loads": 0,
            "sac_initializations": 0,
            "optimizer_updates": 0,
        },
        "episodes": [
            {
                "task": row["task"],
                "length": 500,
                "success": False,
                "status": "COMPLETE",
                "complete_episode": True,
                "initial": {
                    "episode": {
                        "goal_index": 0,
                        "goal_id": row["goal_id"],
                        "goal_bank_sha256": row["bank_sha256"],
                        "episode_seed": row["episode_seed"],
                    }
                },
                "trace": row["task"] + ".jsonl",
                "trace_sha256": "c" * 64,
            }
            for row in requests
        ],
    }


def test_scientific_failure_is_terminal_outcome_without_report_errors() -> None:
    data = original(scientific_failure=True)
    report = summarize_run(data, observed_utc=OBSERVED)
    assert report["raw_status"] == "COMPLETE_FAIL" and report["terminal"] is True
    assert not report["errors"]
    assert any("competence gate not met" in warning for warning in report["warnings"])
    assert report["ledger"]["source_policy_starts"] == 6


def test_live_persistence_timestamp_does_not_imply_terminal_and_input_stays_unchanged() -> None:
    data = original()
    data.update(status="RUNNING", phase="SOURCE_DEVELOPMENT")
    before = copy.deepcopy(data)
    report = summarize_run(data, observed_utc=OBSERVED)
    assert report["terminal"] is False
    assert data == before
    assert report["latest_progress"]["status"] == "absent"
    assert report["source_history_summaries"]["reach-v3"]["scope"] == "absent"


def test_continuation_counts_parent_once_and_keeps_complete_stage_over_partial() -> None:
    report = summarize_run(
        continuation(),
        source_path=Path("/extension/development_manifest.json"),
        observed_utc=OBSERVED,
    )
    assert report["kind"] == "source_continuation" and not report["errors"]
    assert report["ledger"]["source_training_steps"] == 2100000
    assert report["ledger"]["total_agent_steps"] == 2190219
    selected = next(
        row
        for row in report["checkpoint_summaries"]
        if row["task"] == "pick-place-v3" and row["milestone"] == 250000
    )
    assert (
        selected["status"] == "COMPLETE_CHECKPOINT_FAIL"
        and selected["actual_training_steps"] == 250000
    )
    assert selected["retained_partial_checkpoints"] == [
        {"actual_training_steps": 200215, "partial": True}
    ]
    paths = {row["path"] for row in report["artifact_references"]}
    assert "/original/checkpoints/reach-v3/step100000.zip" in paths
    assert "/extension/checkpoints/push-v3/step1000000.zip" in paths
    assert "payload_base64" not in json.dumps(report)


def test_damage_to_seed_or_nonfinite_episode_is_reported_without_throwing() -> None:
    for damage in ("seed", "nan"):
        data = original()
        row = data["tasks"]["reach-v3"]["checkpoints"][0]["evaluation"][0]
        row["episode_seed" if damage == "seed" else "return"] = (
            -1 if damage == "seed" else float("nan")
        )
        report = summarize_run(data, observed_utc=OBSERVED)
        assert report["errors"]
        json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("value", [None, -1, True, float("nan")])
def test_invalid_ledger_is_explicit_and_never_emits_nonfinite_values(value: Any) -> None:
    data = original()
    data["ledger"]["source_training_steps"] = value
    report = summarize_run(data, observed_utc=OBSERVED)
    assert report["errors"] and report["ledger"]["source_training_steps"] == 0
    json.dumps(report, allow_nan=False)


def test_live_progress_raises_only_lower_bound_and_exposes_telemetry() -> None:
    data = original()
    data.update(status="RUNNING", phase="SOURCE_DEVELOPMENT")
    ledger = copy.deepcopy(data["ledger"])
    ledger["source_training_steps"] += 1000
    progress = [
        {
            "task": "push-v3",
            "training_steps": 101000,
            "utc": "2026-11-10T04:59:00Z",
            "ledger": ledger,
            "diagnostics": {"train/actor_loss": float("nan")},
            "nonfinite_diagnostic_keys": ["train/critic_loss"],
            "source_training_episode_summary": {
                "complete_episodes": 202,
                "successful_complete_episodes": 4,
                "recent_complete_episodes": 100,
                "recent_successes": 1,
                "scope": "completed_training_episodes",
            },
        }
    ]
    report = summarize_run(data, progress, observed_utc=OBSERVED)
    assert report["ledger"]["source_training_steps"] == 301000
    assert report["ledger"]["total_agent_steps"] == 331004
    assert report["latest_progress"]["age_seconds"] == 60
    assert report["source_history_summaries"]["push-v3"]["scope"] == "live_progress"
    assert any("nonfinite" in warning for warning in report["warnings"])
    json.dumps(report, allow_nan=False)


def test_checkpoint_source_history_excludes_partial_success_and_uses_last100() -> None:
    data = original()
    history = [
        {
            "episode": index,
            "length": 500,
            "success": index >= 90,
            "complete_episode": True,
            "terminated": False,
            "truncated": True,
        }
        for index in range(200)
    ]
    history.append({"episode": 200, "length": 100, "success": True, "complete_episode": False})
    data["tasks"]["reach-v3"]["checkpoints"][0]["artifacts"]["environment"] = {
        "episode_history": history
    }
    report = summarize_run(data, observed_utc=OBSERVED)
    summary = report["source_history_summaries"]["reach-v3"]
    assert summary["complete_episodes"] == 200 and summary["successful_complete_episodes"] == 110
    assert summary["recent_successes"] == 100 and summary["partial_episodes_excluded"] == 1
    assert summary["scope"] == "known_checkpoint_only"


def test_sentinel_failure_to_succeed_is_diagnostic_not_reporting_error() -> None:
    report = summarize_run(
        sentinel(), source_path=Path("/sentinel/sentinel_manifest.json"), observed_utc=OBSERVED
    )
    assert report["kind"] == "scripted_source_sentinel" and report["terminal"]
    assert report["ledger"]["total_agent_steps"] == 1000 and not report["errors"]
    assert len(report["checkpoint_summaries"]) == 2
    assert len(report["artifact_references"]) == 2
    assert "diagnostic only" in " ".join(report["warnings"])


def test_sentinel_damaged_seed_and_resource_overrun_are_explicit() -> None:
    data = sentinel()
    data["episodes"][0]["initial"]["episode"]["episode_seed"] += 1
    data["ledger"]["source_diagnostic_steps"] = 1001
    report = summarize_run(data, observed_utc=OBSERVED)
    assert any("seed" in error for error in report["errors"])
    assert any("cap" in error for error in report["errors"])


def test_live_progress_cannot_hide_a_saved_cap_overrun() -> None:
    data = original()
    data.update(status="RUNNING", phase="SOURCE_DEVELOPMENT")
    ledger = {**data["ledger"], "source_training_steps": 750001}
    progress = [
        {
            "task": "push-v3",
            "utc": "2026-11-10T04:59:00+00:00",
            "training_steps": 750001,
            "ledger": ledger,
            "diagnostics": {},
        }
    ]
    report = summarize_run(data, progress, observed_utc=OBSERVED)
    assert any("source_training_steps exceeds" in error for error in report["errors"])


def test_missing_evaluation_goal_bank_and_malformed_sentinel_are_reported() -> None:
    data = original()
    data["goal_manifests"].pop("push-v3")
    report = summarize_run(data, observed_utc=OBSERVED)
    assert any("evaluation lacks" in error for error in report["errors"])
    bad = sentinel()
    bad["plan"].update(tasks=None, episode_horizon=None)
    malformed = summarize_run(bad, observed_utc=OBSERVED)
    assert any("requests" in error for error in malformed["errors"])
    assert any("horizon" in error for error in malformed["errors"])


def test_approved_retry_and_recorded_stage_durations_override_base_defaults() -> None:
    data = original()
    data["plan"]["limits"]["retries"] = 0
    data["plan_sha256"] = sha(data["plan"])
    data["technical_retry_contract"]["technical_retries"] = 1
    data["queue"] = [
        {
            "task": "push-v3",
            "milestone": 100000,
            "training_duration_seconds": 123.5,
        }
    ]
    data["tasks"]["push-v3"]["checkpoints"][0]["continuation_evaluation_seconds"] = 12.5
    report = summarize_run(data, observed_utc=OBSERVED)
    assert not report["errors"]
    push = next(
        row
        for row in report["checkpoint_summaries"]
        if row["task"] == "push-v3" and row["milestone"] == 100000
    )
    assert push["training_seconds"] == 123.5 and push["evaluation_seconds"] == 12.5
    data["technical_retry_contract"]["technical_retries"] = 2
    damaged = summarize_run(data, observed_utc=OBSERVED)
    assert any("technical retry contract" in error for error in damaged["errors"])


def test_resumed_visit_duration_is_a_recorded_lower_bound() -> None:
    data = continuation()
    data["parent_binding"]["parent_manifest"]["queue"] = [
        {
            "task": "pick-place-v3",
            "milestone": 250000,
            "training_duration_seconds": 1834.895,
            "status": "TRAINING",
        }
    ]
    data["queue"] = [
        {
            "task": "pick-place-v3",
            "milestone": 250000,
            "increment": 49785,
        }
    ]
    data["plan"]["parent_binding_sha256"] = sha(data["parent_binding"])
    data["plan_sha256"] = sha(data["plan"])
    report = summarize_run(data, observed_utc=OBSERVED)
    row = next(
        row
        for row in report["checkpoint_summaries"]
        if row["task"] == "pick-place-v3" and row["milestone"] == 250000
    )
    assert row["training_seconds"] is None
    assert row["recorded_training_seconds"] == 1834.895
    assert row["missing_training_duration_visits"] == 1
    assert row["training_timing_scope"] == "partial_recorded_visits"
    later = next(
        row
        for row in report["checkpoint_summaries"]
        if row["task"] == "pick-place-v3" and row["milestone"] == 500000
    )
    assert later["training_seconds"] is None and later["training_timing_scope"] == "unrecorded"


def test_stale_progress_rejected_but_retained_parent_progress_is_valid() -> None:
    data = original()
    data.update(status="RUNNING", phase="SOURCE_DEVELOPMENT", started_at_utc="2026-11-10T03:00:00Z")
    progress = [
        {
            "task": "push-v3",
            "utc": "2026-11-09T18:00:00Z",
            "training_steps": 1000,
            "ledger": copy.deepcopy(data["ledger"]),
            "diagnostics": {},
        }
    ]
    stale = summarize_run(data, progress, observed_utc=OBSERVED)
    assert stale["latest_progress"]["status"] == "absent"
    assert any("predates" in error for error in stale["errors"])
    extended = continuation()
    extended["started_at_utc"] = "2026-11-10T03:00:00Z"
    extended["parent_binding"]["parent_manifest"]["started_at_utc"] = "2026-11-09T17:56:00Z"
    extended["plan"]["parent_binding_sha256"] = sha(extended["parent_binding"])
    extended["plan_sha256"] = sha(extended["plan"])
    retained = summarize_run(extended, progress, observed_utc=OBSERVED)
    assert retained["latest_progress"]["status"] == "available"
    assert not any("predates" in error for error in retained["errors"])


def test_unsupported_source_complete_pass_is_flagged_without_rewriting_status() -> None:
    data = original(scientific_failure=True)
    data["status"] = "COMPLETE_PASS"
    report = summarize_run(data, observed_utc=OBSERVED)
    assert report["raw_status"] == "COMPLETE_PASS" and report["terminal"]
    assert any("COMPLETE_PASS lacks" in error for error in report["errors"])


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "partial", "unvalidated"])
def test_terminal_sentinel_requires_exact_complete_task_coverage(mutation: str) -> None:
    data = sentinel()
    if mutation == "missing":
        data["episodes"].pop()
    elif mutation == "duplicate":
        data["episodes"].append(copy.deepcopy(data["episodes"][0]))
    elif mutation == "partial":
        data["episodes"][0]["complete_episode"] = False
    else:
        data["validation_complete"] = False
    report = summarize_run(data, observed_utc=OBSERVED)
    assert report["raw_status"] == "COMPLETE" and report["terminal"]
    assert any("planned task coverage" in error for error in report["errors"])


def test_no_artifact_reads_and_markdown_keeps_outcomes_and_limits_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "open", lambda *args, **kwargs: pytest.fail("unexpected file read"))
    report = summarize_run(
        continuation(),
        source_path=Path("/extension/development_manifest.json"),
        observed_utc=OBSERVED,
    )
    markdown = render_markdown(report)
    assert "COMPLETE_FAIL" in markdown and "2190219" in markdown
    assert "known_checkpoint_only" not in markdown  # no invented historical telemetry
    assert "absent" in markdown and "competence gate not met" in markdown


def test_import_is_stdlib_only() -> None:
    code = (
        "import sys\nclass Block:\n"
        " def find_spec(self,fullname,path=None,target=None):\n"
        "  forbidden=('numpy','torch','gymnasium','metaworld','mujoco','stable_baselines3')\n"
        "  if fullname.split('.')[0] in forbidden:\n"
        "   raise AssertionError(fullname)\n"
        "sys.meta_path.insert(0,Block())\n"
        "from scripts.metaworld_live_report import summarize_run,render_markdown\n"
        "r=summarize_run({})\nassert r['errors'] and not r['terminal']\n"
        "assert isinstance(render_markdown(r),str)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stderr
