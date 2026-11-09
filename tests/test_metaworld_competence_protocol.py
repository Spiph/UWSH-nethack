from __future__ import annotations

import base64
from dataclasses import asdict
from typing import Any

import numpy as np
import pytest

from scripts.metaworld_competence import TASKS, summarize_competence
from scripts.metaworld_protocol import SCHEMA_VERSION, Goal, GoalBank, GoalManifest, SeedStreams


def complete_protocol_screen() -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "tasks": list(TASKS),
        "steps_per_policy_requested": 4,
        "evaluation_episodes_requested": 1,
        "protocol_schema_version": SCHEMA_VERSION,
        "evaluation_protocol_validation": dict.fromkeys(TASKS, True),
        "goal_manifests": {},
        "evaluation_plans": {},
        "task_runs": [],
        "evaluations": [],
    }
    for index, task in enumerate(TASKS):
        banks = []
        for split, offset in [("source", 0), ("development", 100)]:
            vector = (float(index + offset),)
            goal = Goal(task, base64.b64encode(f"{task}:{split}".encode()).decode(), vector)
            banks.append(GoalBank(split, index + offset, (goal,)))
        source, development = banks
        streams = SeedStreams.from_seed(index + 8000)
        manifest["evaluation_plans"][task] = {
            "goal_bank_sha256": development.sha256,
            "goal_order": [0],
            "goal_ids": [development.goals[0].goal_id],
            "rng_streams": asdict(streams),
        }
        manifest["goal_manifests"][task] = GoalManifest(tuple(banks)).to_dict()
        manifest["task_runs"].append(
            {
                "task": task,
                "steps_requested": 4,
                "environment_steps_completed": 4,
                "initial_actor_sha256": "shared",
                "training_split": "source",
                "training_goal_bank_sha256": source.sha256,
            }
        )
        manifest["evaluations"].append(
            {
                "task": task,
                "episode": 0,
                "episode_seed": int(
                    np.random.default_rng(streams.episode).integers(0, 2**32, dtype=np.uint64)
                ),
                "return": 1.0,
                "success": True,
                "goal_id": development.goals[0].goal_id,
                "goal_bank_sha256": development.sha256,
                "complete_episode": True,
                "length": 500,
                "terminated": False,
                "truncated": True,
            }
        )
    return manifest


def test_saved_goal_provenance_and_full_coverage_are_required_for_completion() -> None:
    summary = summarize_competence(complete_protocol_screen())
    assert summary["protocol_valid"] is True
    assert summary["all_tasks_complete"] is True
    assert summary["all_tasks_pass"] is True


@pytest.mark.parametrize(
    "damage", ["wrong_goal", "wrong_bank", "partial", "terminal", "target_weights", "schema"]
)
def test_report_rechecks_evidence_instead_of_trusting_recorded_validation(damage: str) -> None:
    manifest = complete_protocol_screen()
    if damage == "wrong_goal":
        manifest["evaluations"][0]["goal_id"] = "different"
    elif damage == "wrong_bank":
        manifest["evaluations"][0]["goal_bank_sha256"] = "different"
    elif damage == "partial":
        manifest["evaluations"][0]["length"] = 499
    elif damage == "terminal":
        manifest["evaluations"][0]["terminated"] = True
    elif damage == "target_weights":
        manifest["task_runs"][0]["training_split"] = "development"
    else:
        manifest["protocol_schema_version"] = 99
    assert manifest["evaluation_protocol_validation"] == dict.fromkeys(TASKS, True)
    summary = summarize_competence(manifest)
    assert summary["protocol_valid"] is False
    assert summary["all_tasks_complete"] is False
    assert summary["all_tasks_pass"] is False


def test_missing_new_protocol_evidence_fails_closed() -> None:
    manifest = complete_protocol_screen()
    del manifest["goal_manifests"]
    assert summarize_competence(manifest)["all_tasks_pass"] is False


def test_legacy_results_do_not_claim_goal_protocol_validation() -> None:
    manifest = complete_protocol_screen()
    del manifest["protocol_schema_version"]
    assert summarize_competence(manifest)["protocol_valid"] is None


@pytest.mark.parametrize(
    "damage",
    ["missing_plan", "wrong_order", "wrong_ids", "wrong_bank", "episode_seed", "episode_stream"],
)
def test_saved_plan_and_realized_episode_stream_must_agree(damage: str) -> None:
    manifest = complete_protocol_screen()
    plan = manifest["evaluation_plans"][TASKS[0]]
    if damage == "missing_plan":
        del manifest["evaluation_plans"]
    elif damage == "wrong_order":
        plan["goal_order"] = [99]
    elif damage == "wrong_ids":
        plan["goal_ids"] = ["wrong"]
    elif damage == "wrong_bank":
        plan["goal_bank_sha256"] = "wrong"
    elif damage == "episode_seed":
        manifest["evaluations"][0]["episode_seed"] = 999
    else:
        plan["rng_streams"]["episode"] = 0
    summary = summarize_competence(manifest)
    assert summary["protocol_valid"] is False
    assert summary["all_tasks_pass"] is False


@pytest.mark.parametrize("field, invalid", [("task_runs", 42), ("evaluations", None)])
def test_malformed_rows_report_invalid_protocol_without_raising(field: str, invalid: Any) -> None:
    manifest = complete_protocol_screen()
    manifest[field].append(invalid)
    summary = summarize_competence(manifest)
    assert summary["protocol_valid"] is False
    assert summary["all_tasks_complete"] is False
