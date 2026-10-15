from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.metaworld_competence import TASKS, parse_args, summarize_competence


def complete_manifest(episodes: int = 20, successes: int = 16) -> dict[str, Any]:
    return {
        "tasks": list(TASKS),
        "steps_per_policy_requested": 25000,
        "evaluation_episodes_requested": episodes,
        "task_runs": [
            {
                "task": task,
                "steps_requested": 25000,
                "environment_steps_completed": 25000,
                "initial_actor_sha256": "shared",
            }
            for task in TASKS
        ],
        "evaluations": [
            {
                "task": task,
                "episode": episode,
                "complete_episode": True,
                "success": episode < successes,
                "return": float(episode),
            }
            for task in TASKS
            for episode in range(episodes)
        ],
    }


def test_complete_runs_apply_ceil_eighty_percent_threshold() -> None:
    summary = summarize_competence(complete_manifest())
    assert summary["all_tasks_complete"] is True
    assert summary["all_tasks_pass"] is True
    assert all(row["success_threshold"] == 16 for row in summary["per_task"].values())


def test_twenty_episode_screen_fails_with_fifteen_successes() -> None:
    summary = summarize_competence(complete_manifest(successes=15))
    assert summary["all_tasks_complete"] is True
    assert summary["all_tasks_pass"] is False
    assert all(not row["pass"] for row in summary["per_task"].values())


@pytest.mark.parametrize("damage", ["missing", "truncated", "duplicate"])
def test_missing_truncated_and_duplicate_episode_rows_are_incomplete(damage: str) -> None:
    manifest = complete_manifest()
    if damage == "missing":
        manifest["evaluations"].pop()
    elif damage == "truncated":
        row = next(
            r for r in manifest["evaluations"] if r["task"] == TASKS[0] and r["episode"] == 0
        )
        row["complete_episode"] = False
    else:
        manifest["evaluations"].append(dict(manifest["evaluations"][0]))
    summary = summarize_competence(manifest)
    assert summary["all_tasks_complete"] is False
    assert summary["all_tasks_pass"] is False


def test_duplicate_task_run_and_shared_hash_mismatch_cannot_pass() -> None:
    manifest = complete_manifest()
    manifest["task_runs"][1]["initial_actor_sha256"] = "different"
    manifest["task_runs"].append(dict(manifest["task_runs"][0]))
    summary = summarize_competence(manifest)
    assert summary["shared_initial_actor_hash_valid"] is False
    assert summary["per_task"][TASKS[0]]["training_complete"] is False
    assert summary["all_tasks_pass"] is False


def test_incomplete_training_steps_prevent_overall_completion() -> None:
    manifest = complete_manifest()
    manifest["task_runs"][2]["environment_steps_completed"] = 24999
    summary = summarize_competence(manifest)
    assert summary["per_task"][TASKS[2]]["training_complete"] is False
    assert summary["all_tasks_complete"] is False


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tasks", [*TASKS, "unknown-v3"]),
        ("steps_per_policy_requested", 0),
        ("steps_per_policy_requested", True),
        ("evaluation_episodes_requested", -1),
        ("evaluation_episodes_requested", 1.5),
    ],
)
def test_invalid_summary_metadata_cannot_pass(field: str, value: Any) -> None:
    manifest = complete_manifest()
    manifest[field] = value
    summary = summarize_competence(manifest)
    assert summary["metadata_valid"] is False
    assert summary["all_tasks_pass"] is False


@pytest.mark.parametrize("invalid", ["unknown_task", "nonfinite_return", "nonbinary_success"])
def test_invalid_evaluation_rows_cannot_pass(invalid: str) -> None:
    manifest = complete_manifest()
    row = manifest["evaluations"][0]
    if invalid == "unknown_task":
        row["task"] = "other-v3"
    elif invalid == "nonfinite_return":
        row["return"] = float("nan")
    else:
        row["success"] = 2
    summary = summarize_competence(manifest)
    assert summary["all_tasks_complete"] is False
    assert summary["all_tasks_pass"] is False


def test_unknown_task_run_cannot_pass() -> None:
    manifest = complete_manifest()
    manifest["task_runs"][0]["task"] = "other-v3"
    summary = summarize_competence(manifest)
    assert summary["metadata_valid"] is False
    assert summary["all_tasks_pass"] is False


@pytest.mark.parametrize(
    "options",
    [
        ["--seconds", "3601"],
        ["--seconds", "0"],
        ["--steps-per-policy", "0"],
        ["--evaluation-episodes", "0"],
        ["--reserve-seconds", "3600"],
        ["--reserve-seconds", "-1"],
        ["--reserve-seconds", "0"],
    ],
)
def test_invalid_budgets_are_rejected(options: list[str], tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--output", str(tmp_path / "new-run"), *options])


def test_output_with_existing_contents_is_preserved(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("user data", encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(["--output", str(output)])
    assert sentinel.read_text(encoding="utf-8") == "user data"


def test_smoke_configuration_is_explicitly_small() -> None:
    args = parse_args(
        ["--output", "unused", "--steps-per-policy", "128", "--evaluation-episodes", "1"]
    )
    assert args.steps_per_policy == 128
    assert args.evaluation_episodes == 1
