from __future__ import annotations

import copy
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from scripts.metaworld_development_plan import (
    APPROVAL_URL,
    DEFAULT_CONFIG,
    SEED_FIELDS,
    DevelopmentPlan,
    approved_configuration,
    derive_seed,
)


def test_frozen_configuration_and_worst_case_ledger() -> None:
    plan = DevelopmentPlan.load(DEFAULT_CONFIG)
    assert plan.to_dict() == approved_configuration()
    assert plan.ledger() == {
        "source_policy_starts": 3,
        "retries": 0,
        "source_training_steps": 750000,
        "development_evaluation_steps": 60000,
        "preflight_steps": 1000,
        "total_agent_steps": 811000,
        "target_steps": 0,
        "final_test_steps": 0,
        "new_artifact_bytes": 2147483648,
    }
    assert plan.worst_case()["checkpoint_evaluations"] == 6
    assert plan.worst_case()["development_evaluation_episodes"] == 120
    assert plan.source_goal_count == 50
    assert plan.development_goal_count == plan.evaluation_episodes == 20
    assert plan.success_threshold == 16
    assert plan.max_episode_steps == 500


def test_round_robin_milestones_are_cumulative_not_additive() -> None:
    plan = DevelopmentPlan.load(DEFAULT_CONFIG)
    assert plan.stages == plan.milestones == (100000, 250000)
    assert plan.schedule() == (
        ("reach-v3", 100000),
        ("push-v3", 100000),
        ("pick-place-v3", 100000),
        ("reach-v3", 250000),
        ("push-v3", 250000),
        ("pick-place-v3", 250000),
    )
    previous = dict.fromkeys(plan.tasks, 0)
    increments = []
    for task, milestone in plan.schedule():
        increments.append(milestone - previous[task])
        previous[task] = milestone
    assert increments == [100000] * 3 + [150000] * 3
    assert sum(increments) == plan.max_training_steps


def test_materialized_actor_critic_training_and_goal_streams() -> None:
    plan = DevelopmentPlan.load(DEFAULT_CONFIG)
    source = [plan.source_streams(task) for task in plan.tasks]
    development = [plan.development_streams(task) for task in plan.tasks]
    assert {streams["actor"] for streams in source} == {plan.actor_seed} == {2026110901}
    for index, (training, evaluation) in enumerate(zip(source, development, strict=True)):
        assert set(training) == set(evaluation) == set(SEED_FIELDS)
        assert all(type(seed) is int and 0 <= seed < 2**32 for seed in training.values())
        assert all(type(seed) is int and 0 <= seed < 2**32 for seed in evaluation.values())
        assert training["critic"] == 2026111001 + index
        assert training["training"] == 2026111101 + index
        assert training["task_bank"] == 2026111201 + index
        assert training["selection"] == 2026111301 + index
        assert evaluation["task_bank"] == 2026111401 + index
        for stream in ("constructor", "task_bank", "selection", "episode", "action"):
            assert training[stream] != evaluation[stream]
        record = plan.to_dict()["tasks"][index]
        assert record["effective_source_bank_seed"] == derive_seed(training["task_bank"], "source")
        assert record["effective_development_bank_seed"] == derive_seed(
            evaluation["task_bank"], "development"
        )
    for field in ("critic", "training", "task_bank", "selection", "episode", "action"):
        assert len({streams[field] for streams in source}) == 3
    with pytest.raises(ValueError, match="finite list"):
        plan.source_streams("drawer-open-v3")


def test_fixed_session_and_distinct_preflight_phase_limit() -> None:
    plan = DevelopmentPlan.load(DEFAULT_CONFIG)
    assert plan.session_started_utc == datetime(2026, 11, 9, 15, 35, tzinfo=timezone.utc)
    assert plan.training_deadline_utc == datetime(2026, 11, 9, 21, 5, tzinfo=timezone.utc)
    assert plan.deadline_utc == datetime(2026, 11, 9, 21, 35, tzinfo=timezone.utc)
    assert (plan.deadline_utc - plan.session_started_utc).total_seconds() == plan.session_seconds
    assert (plan.deadline_utc - plan.training_deadline_utc).total_seconds() == plan.reserve_seconds
    assert plan.session_seconds == 21600
    assert plan.preflight_seconds == 900
    assert plan.reserve_seconds == 1800
    assert plan.to_dict()["session"]["preparation_counts_against_session"] is True


@pytest.mark.parametrize(
    ("section", "field", "replacement"),
    [
        ("limits", "source_training_steps", 750001),
        ("limits", "development_evaluation_steps", 30000),
        ("limits", "preflight_steps", 1001),
        ("limits", "total_agent_steps", 750000),
        ("limits", "source_policy_starts", 4),
        ("limits", "retries", 1),
        ("limits", "target_steps", 1),
        ("limits", "final_test_steps", 1),
        ("limits", "new_artifact_bytes", 4 * 1024**3),
        ("goal_counts", "source", 49),
        ("goal_counts", "development", 50),
        ("goal_counts", "final_test", 20),
        ("evaluation", "required_successes", 15),
        ("evaluation", "partial_episodes_count", True),
        ("evaluation", "episodes", 20.0),
        ("algorithm", "learning_rate", 1e-3),
        ("algorithm", "learning_rate", float("nan")),
        ("algorithm", "device", "cpu"),
        ("algorithm", "torch_threads", True),
        ("environment", "reward_function_version", "v1"),
        ("environment", "terminate_on_success", True),
        ("authorization", "approved", False),
        ("authorization", "user_approval_url", "https://example.com/assumed-approval"),
        ("authorization", "paid_services", True),
        ("authorization", "concurrency", 2),
        ("session", "started_at_utc", "2026-11-09T16:00:00Z"),
        ("session", "deadline_utc", "2026-11-10T03:35:00Z"),
        ("session", "training_deadline_utc", "2026-11-09T21:35:00Z"),
        ("session", "evaluation_reserve_seconds", 0),
        ("session", "preflight_phase_seconds", 1800),
        ("session", "preparation_counts_against_session", False),
    ],
)
def test_unapproved_caps_settings_and_deadlines_rejected(
    section: str, field: str, replacement: Any
) -> None:
    data = approved_configuration()
    data[section][field] = replacement
    with pytest.raises(ValueError, match="approved"):
        DevelopmentPlan.from_dict(data)


@pytest.mark.parametrize("field", ["authorization", "milestones", "limits", "seed_derivation"])
def test_missing_required_contract_sections_rejected(field: str) -> None:
    data = approved_configuration()
    del data[field]
    with pytest.raises(ValueError, match="missing or unknown"):
        DevelopmentPlan.from_dict(data)


def test_unknown_fields_options_grid_and_seed_changes_rejected() -> None:
    variants = []
    data = approved_configuration()
    data["algorithm"]["hidden_sweep"] = [128, 512]
    variants.append(data)
    data = approved_configuration()
    data["allocation_option"] = "optional_12h_local"
    variants.append(data)
    data = approved_configuration()
    data["milestones"] = [100000, 250000, 500000]
    variants.append(data)
    data = approved_configuration()
    data["tasks"][0]["source_streams"]["actor"] += 1
    variants.append(data)
    data = approved_configuration()
    data["tasks"][0]["effective_source_bank_seed"] += 1
    variants.append(data)
    data = approved_configuration()
    data["output"] = "/tmp/unapproved-run"
    variants.append(data)
    for data in variants:
        with pytest.raises(ValueError):
            DevelopmentPlan.from_dict(data)


def test_roundtrip_hash_and_returned_data_are_independent(tmp_path: Path) -> None:
    original = approved_configuration()
    plan = DevelopmentPlan.from_dict(original)
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan.to_dict()), encoding="utf-8")
    restored = DevelopmentPlan.load(path)
    assert restored.sha256 == plan.sha256
    assert restored.to_dict()["authorization"]["user_approval_url"] == APPROVAL_URL
    copy_before = copy.deepcopy(plan.to_dict())
    original["algorithm"]["net_arch"][0] = 1
    returned = plan.to_dict()
    returned["tasks"][0]["source_streams"]["actor"] = 1
    plan.algorithm["net_arch"][0] = 1
    assert plan.to_dict() == copy_before
    assert restored.sha256 == plan.sha256


def test_dry_run_has_only_stdlib_dependencies_and_starts_no_runtime() -> None:
    code = (
        "import sys; from scripts.metaworld_development_plan import main; main(); "
        "assert not set(('numpy','torch','gymnasium','metaworld','stable_baselines3'))"
        ".intersection(sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], check=True, capture_output=True, text=True
    )
    report = json.loads(result.stdout)
    assert report["status"] == "VALIDATED_PLAN_ONLY"
    assert report["worst_case"]["ledger"]["total_agent_steps"] == 811000
    assert len(report["schedule"]) == 6
    # Structural validation intentionally works after the fixed allocation
    # expires, too; only the separate execution path consults a live clock.
    assert report["plan"]["session"]["deadline_utc"] == "2026-11-09T21:35:00Z"
