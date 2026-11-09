from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.metaworld_pilot_report import main, summarize_manifest


def complete_fixture() -> dict[str, Any]:
    tasks = ["reach-v3", "push-v3"]
    conditions = ["shared_init", "independent_init"]
    source_runs = []
    evaluations = []
    diagnostics = []
    for condition in conditions:
        for replicate in range(2):
            for task_index, task in enumerate(tasks):
                policy_id = f"{condition}__{task}__lib{replicate}"
                init_hash = f"{condition}-{replicate}-" + (
                    "same" if condition == "shared_init" else task
                )
                source_runs.append(
                    {
                        "policy_id": policy_id,
                        "condition": condition,
                        "replicate": replicate,
                        "task": task,
                        "initial_actor_sha256": init_hash,
                        "environment_steps_completed": 100,
                        "steps_requested": 100,
                    }
                )
                for eval_task in tasks:
                    for episode in range(2):
                        evaluations.append(
                            {
                                "policy_id": policy_id,
                                "condition": condition,
                                "replicate": replicate,
                                "training_task": task,
                                "evaluation_task": eval_task,
                                "policy_kind": "trained_policy",
                                "episode": episode,
                                "return": 10.0 if eval_task == "reach-v3" else 100.0,
                                "success": float(episode == 0),
                                "complete_episode": True,
                            }
                        )
                for kind in ("oracle_learned_rank1", "oracle_random_rank1"):
                    for episode in range(2):
                        evaluations.append(
                            {
                                "policy_id": policy_id,
                                "condition": condition,
                                "replicate": replicate,
                                "training_task": task,
                                "evaluation_task": task,
                                "policy_kind": kind,
                                "episode": episode,
                                "return": 5.0,
                                "success": 0.0,
                                "complete_episode": True,
                            }
                        )
                diagnostics.append(
                    {
                        "condition": condition,
                        "replicate": replicate,
                        "held_out_task": task,
                        "learned_relative_residual": 0.2 + task_index * 0.2,
                        "random_relative_residual": 0.8 + task_index * 0.2,
                    }
                )
    return {
        "tasks": tasks,
        "conditions": conditions,
        "replicates": 2,
        "planned_source_runs": 8,
        "steps_per_source_policy_requested": 100,
        "evaluation_episodes_requested": 2,
        "source_runs": source_runs,
        "cross_task_evaluations": evaluations,
        "oracle_projection_diagnostics": diagnostics,
        "evaluation_complete": True,
    }


def test_complete_fixture_reports_all_cells_and_replicate_level_geometry() -> None:
    report = summarize_manifest(complete_fixture())
    assert report["coverage"]["complete"] is True
    assert report["coverage"]["expected_evaluation_cells"] == 32
    assert report["initialization"]["valid"] is True
    learned = next(x for x in report["geometry_residuals"] if x["method"] == "learned_rank1")
    assert learned["n_library_replicates"] == 2
    assert abs(learned["replicate_means"][0]["mean_relative_residual"] - 0.3) < 1e-12
    assert report["interpretation"]["oracle_target_weight_access"] is True
    assert report["interpretation"]["novelty_claim"] is None


def test_truncated_episode_and_missing_cells_are_incomplete() -> None:
    manifest = complete_fixture()
    manifest["cross_task_evaluations"] = [
        row
        for row in manifest["cross_task_evaluations"]
        if not (
            row["policy_kind"] == "oracle_random_rank1"
            and row["condition"] == "shared_init"
            and row["replicate"] == 0
            and row["training_task"] == "reach-v3"
        )
    ]
    row = manifest["cross_task_evaluations"][0]
    row["complete_episode"] = False
    report = summarize_manifest(manifest)
    assert report["coverage"]["complete"] is False
    assert report["coverage"]["missing_evaluation_cells"]
    assert report["coverage"]["incomplete_evaluation_cells"]


def test_evaluation_complete_flag_cannot_override_incomplete_data() -> None:
    manifest = complete_fixture()
    manifest["cross_task_evaluations"] = []
    manifest["evaluation_complete"] = True
    report = summarize_manifest(manifest)
    assert report["coverage"]["complete"] is False
    assert report["coverage"]["evaluation_complete_flag_ignored"] is True


def test_duplicate_source_and_episode_rows_are_flagged() -> None:
    manifest = complete_fixture()
    manifest["source_runs"].append(copy.deepcopy(manifest["source_runs"][0]))
    manifest["cross_task_evaluations"].append(copy.deepcopy(manifest["cross_task_evaluations"][0]))
    report = summarize_manifest(manifest)
    assert report["coverage"]["duplicate_source_slots"]
    assert report["coverage"]["duplicate_evaluation_rows"]
    assert report["coverage"]["complete"] is False


def test_initialization_hash_mismatch_and_absence_are_reported() -> None:
    manifest = complete_fixture()
    manifest["source_runs"][1]["initial_actor_sha256"] = "different"
    mismatch = summarize_manifest(manifest)
    assert mismatch["initialization"]["valid"] is False
    assert any(
        issue["issue"] == "shared_hash_mismatch" for issue in mismatch["initialization"]["issues"]
    )
    manifest["source_runs"][1].pop("initial_actor_sha256")
    missing = summarize_manifest(manifest)
    assert missing["initialization"]["hashes_present"] is False
    assert any(issue["issue"] == "missing_hashes" for issue in missing["initialization"]["issues"])


def test_episode_multiplicity_does_not_change_library_replicate_count() -> None:
    manifest = complete_fixture()
    # Unequal episode counts are retained as observed; they are never counted as libraries.
    manifest["cross_task_evaluations"] = [
        row
        for row in manifest["cross_task_evaluations"]
        if not (row["condition"] == "shared_init" and row["replicate"] == 0 and row["episode"] == 1)
    ]
    report = summarize_manifest(manifest)
    assert report["coverage"]["complete"] is False
    assert all(item["n_library_replicates"] == 2 for item in report["geometry_residuals"])


def test_task_reward_scales_are_not_pooled() -> None:
    report = summarize_manifest(complete_fixture())
    trained = [
        item
        for item in report["policy_by_task"]
        if item["policy_kind"] == "trained_policy" and item["condition"] == "shared_init"
    ]
    reach = next(item for item in trained if item["evaluation_task"] == "reach-v3")
    push = next(item for item in trained if item["evaluation_task"] == "push-v3")
    assert reach["mean_return"] == 10.0
    assert push["mean_return"] == 100.0
    assert reach["n_library_replicates"] == 2
    assert len(trained) == 4
    assert "task scales are not pooled" in report["interpretation"]["pooling"]


def test_empty_and_invalid_protocol_metadata_are_not_vacuously_valid() -> None:
    empty = summarize_manifest({})
    assert empty["coverage"]["complete"] is False
    assert empty["initialization"]["valid"] is False
    manifest = complete_fixture()
    manifest["tasks"] = ["reach-v3", "reach-v3"]
    manifest["replicates"] = 0
    manifest["steps_per_source_policy_requested"] = -1
    manifest["evaluation_episodes_requested"] = 0
    report = summarize_manifest(manifest)
    assert report["coverage"]["complete"] is False
    assert {issue["field"] for issue in report["coverage"]["protocol_issues"]} == {
        "tasks",
        "replicates",
        "steps_per_source_policy_requested",
        "evaluation_episodes_requested",
    }


def test_episode_ids_must_be_exact_and_metrics_finite() -> None:
    manifest = complete_fixture()
    target = manifest["cross_task_evaluations"][0]
    target["episode"] = 2  # expected IDs are 0 and 1
    target["return"] = float("nan")
    report = summarize_manifest(manifest)
    assert report["coverage"]["complete"] is False
    issue = next(
        item
        for item in report["coverage"]["incomplete_evaluation_cells"]
        if "trained_policy" in item["cell"]
    )
    assert issue["missing_episode_ids"] == [0]
    assert issue["extra_episode_ids"] == [2]
    assert issue["episodes_with_invalid_metrics"] == [2]


def test_geometry_diagnostics_require_exact_keys_and_finite_values() -> None:
    manifest = complete_fixture()
    manifest["oracle_projection_diagnostics"].pop()
    manifest["oracle_projection_diagnostics"][0]["learned_relative_residual"] = float("inf")
    report = summarize_manifest(manifest)
    audit = report["coverage"]["geometry_diagnostics"]
    assert report["coverage"]["complete"] is False
    assert audit["missing"]
    assert audit["nonfinite_or_missing_values"]

    manifest = complete_fixture()
    manifest["oracle_projection_diagnostics"].append(
        copy.deepcopy(manifest["oracle_projection_diagnostics"][0])
    )
    manifest["oracle_projection_diagnostics"][-1]["held_out_task"] = "extra-task"
    manifest["oracle_projection_diagnostics"].append(
        copy.deepcopy(manifest["oracle_projection_diagnostics"][1])
    )
    report = summarize_manifest(manifest)
    assert report["coverage"]["geometry_diagnostics"]["extra"]
    assert report["coverage"]["geometry_diagnostics"]["duplicates"]


def test_missing_expected_own_task_groups_are_retained() -> None:
    manifest = complete_fixture()
    manifest["cross_task_evaluations"] = [
        row
        for row in manifest["cross_task_evaluations"]
        if not (
            row["condition"] == "independent_init"
            and row["replicate"] == 1
            and row["training_task"] == "push-v3"
            and row["evaluation_task"] == "push-v3"
        )
    ]
    report = summarize_manifest(manifest)
    group = next(
        item
        for item in report["own_task_competence_by_condition_replicate"]
        if item["condition"] == "independent_init"
        and item["replicate"] == 1
        and item["task"] == "push-v3"
        and item["policy_kind"] == "trained_policy"
    )
    assert group["n_complete_episodes"] == 0
    assert group["n_missing_episodes"] == 2


def test_initial_actor_hash_cannot_be_reused_across_library_replicates() -> None:
    manifest = complete_fixture()
    shared_replicate_zero = next(
        row["initial_actor_sha256"]
        for row in manifest["source_runs"]
        if row["condition"] == "shared_init" and row["replicate"] == 0
    )
    for row in manifest["source_runs"]:
        if row["condition"] == "shared_init" and row["replicate"] == 1:
            row["initial_actor_sha256"] = shared_replicate_zero
    report = summarize_manifest(manifest)
    assert report["initialization"]["valid"] is False
    assert any(
        issue["issue"] == "hash_reused_across_library_replicates"
        for issue in report["initialization"]["issues"]
    )


def test_cli_writes_parent_directories_and_returns_failure_for_incomplete_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({}), encoding="utf-8")
    output = tmp_path / "nested" / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        ["metaworld_pilot_report", "--manifest", str(manifest_path), "--output", str(output)],
    )
    assert main() == 1
    assert json.loads(output.read_text(encoding="utf-8"))["coverage"]["complete"] is False
