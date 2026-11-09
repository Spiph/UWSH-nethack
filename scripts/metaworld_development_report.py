"""Describe staged developmental evidence offline, without importing the runner.

Only JSON is parsed. Optional artifact verification streams bytes for hashes;
it never deserializes a policy, replay buffer, array or RNG checkpoint.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

COMPONENTS = ("source_training_steps", "development_evaluation_steps", "preflight_steps")
COUNTERS = (
    *COMPONENTS,
    "source_policy_starts",
    "failed_step_calls",
    "optimizer_update_attempts",
    "optimizer_updates",
    "target_steps",
    "final_test_steps",
    "retries",
    "technical_retries",
)
FINAL_STATUSES = ("COMPLETE_PASS", "COMPLETE_FAIL", "INCOMPLETE", "ERROR")


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if result.tzinfo is None or result.utcoffset() is None:
        return None
    return result.astimezone(timezone.utc)


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _steps(value: Any) -> bool:
    return type(value) is int and value >= 0


def _finite(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return float(value)
    return None


def _seconds(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows if _number(row.get(key)) and row[key] >= 0)


def _counters(value: Any, errors: list[str], name: str) -> dict[str, int]:
    if not isinstance(value, dict):
        errors.append(f"{name}: missing ledger")
        value = {}
    result = {}
    if value and any(key not in value for key in COMPONENTS):
        errors.append(f"{name}: missing component counters")
    for key in COUNTERS:
        count = value.get(key, 0)
        if not _steps(count):
            errors.append(f"{name}.{key}: expected a nonnegative integer")
            count = 0
        result[key] = count
    result["total_agent_steps"] = sum(result[key] for key in COMPONENTS)
    return result


def _strict_goals(
    manifest: dict[str, Any], task: str, plan: dict[str, Any], errors: list[str]
) -> dict[str, Any] | None:
    banks = manifest.get("goal_manifests", {}).get(task, {}).get("banks", [])
    source, development = [], []
    seen: set[str] = set()
    for bank in banks:
        split = bank.get("split")
        if split not in ("source", "development"):
            errors.append(f"{task}: unexpected goal split {split!r}")
        for goal in bank.get("goals", []):
            try:
                expected_id = _hash({"task": goal["task"], "random_vector": goal["random_vector"]})
                payload = base64.b64decode(goal["payload_base64"], validate=True)
                valid = (
                    goal["task"] == task
                    and goal["goal_id"] == expected_id
                    and goal["payload_sha256"] == hashlib.sha256(payload).hexdigest()
                    and bool(payload)
                    and bool(goal["random_vector"])
                    and all(_number(number) for number in goal["random_vector"])
                )
            except (KeyError, ValueError, TypeError):
                valid = False
            if not valid:
                errors.append(f"{task}: invalid goal identity/payload commitment")
            goal_id = goal.get("goal_id")
            if goal_id in seen:
                errors.append(f"{task}: duplicate or source/development leaking goal identity")
            seen.add(str(goal_id))
        if split == "source":
            source.append(bank)
        elif split == "development":
            development.append(bank)
    counts = plan.get("goal_counts", {})
    for split, selected in (("source", source), ("development", development)):
        if len(selected) != 1 or len(selected[0].get("goals", [])) != counts.get(split):
            errors.append(f"{task}: missing or incorrect {split} goal-bank coverage")
    return development[0] if len(development) == 1 else None


def _stage(
    task: str,
    milestone: int,
    row: dict[str, Any],
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    bank: dict[str, Any] | None,
    strict: bool,
    errors: list[str],
) -> dict[str, Any]:
    prefix = f"{task}@{milestone}"
    errors_before = len(errors)
    episodes, horizon = evaluation["episodes"], evaluation["episode_horizon"]
    rows = row.get("evaluation", [])
    complete, partial, successes, ids = 0, 0, 0, []
    order, seeds = (
        metadata.get("evaluation_goal_order", []),
        metadata.get("evaluation_episode_seeds", []),
    )
    if len(order) != episodes or len(seeds) != episodes:
        errors.append(f"{prefix}: missing declared evaluation order/seeds")
    elif order != list(range(episodes)) or any(
        type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds
    ):
        errors.append(f"{prefix}: expected every bank goal once and unsigned episode seeds")
    if not _steps(row.get("actual_steps")) or row["actual_steps"] > milestone:
        errors.append(f"{prefix}: invalid cumulative training step count")
    for index, episode in enumerate(rows):
        ordinal = episode.get("episode")
        if type(ordinal) is not int or ordinal != index or not 0 <= ordinal < episodes:
            errors.append(f"{prefix}: duplicated, reordered or invalid episode ordinal")
        ids.append(ordinal)
        if (
            index < len(order)
            and index < len(seeds)
            and (
                episode.get("goal_index") != order[index]
                or episode.get("episode_seed") != seeds[index]
            )
        ):
            errors.append(f"{prefix}: saved episode order/seed does not match declaration")
        length = episode.get("length")
        if not _steps(length) or length > horizon or type(episode.get("success")) is not bool:
            errors.append(f"{prefix}: invalid episode length/success")
        full = episode.get("complete_episode") is True
        if full:
            if (
                length != horizon
                or episode.get("terminated") is not False
                or episode.get("truncated") is not True
            ):
                errors.append(f"{prefix}: complete episode violates full-horizon semantics")
            complete += 1
            successes += episode.get("success") is True
        else:
            partial += 1
        if not _number(episode.get("return")):
            errors.append(f"{prefix}: episode return is missing or nonfinite")
        if strict and bank is not None and index < len(order) and index < len(seeds):
            goal_index = order[index]
            goals = bank.get("goals", [])
            if type(goal_index) is not int or not 0 <= goal_index < len(goals):
                errors.append(f"{prefix}: invalid declared goal order")
            elif (
                episode.get("goal_index") != goal_index
                or episode.get("goal_id") != goals[goal_index]["goal_id"]
                or episode.get("goal_bank_sha256") != _hash(bank)
                or episode.get("goal") != goals[goal_index]
                or episode.get("episode_seed") != seeds[index]
            ):
                errors.append(
                    f"{prefix}: saved goal identity/order/seed does not match declaration"
                )
    training_complete = row.get("partial") is False and row.get("actual_steps") == milestone
    coverage_complete = (
        len(rows) == episodes and complete == episodes and ids == list(range(episodes))
    )
    marked_complete = row.get("evaluation_complete") is True
    if marked_complete and (not training_complete or not coverage_complete):
        errors.append(f"{prefix}: claimed complete evaluation has incomplete training/episodes")
    complete_stage = training_complete and coverage_complete and marked_complete
    passed = complete_stage and successes >= evaluation["required_successes"]
    if row.get("passed") is True and not passed:
        errors.append(f"{prefix}: recorded pass is unsupported by complete evaluation")
    if marked_complete and type(row.get("passed")) is bool and row["passed"] != passed:
        errors.append(f"{prefix}: recorded pass disagrees with frozen threshold")
    status = (
        "ERROR"
        if len(errors) > errors_before
        else "PASS"
        if passed
        else "COMPLETE_CHECKPOINT_FAIL"
        if complete_stage
        else "INCOMPLETE"
    )
    return {
        "task": task,
        "milestone": milestone,
        "status": status,
        "actual_training_steps": row.get("actual_steps"),
        "training_complete": training_complete,
        "evaluation_complete_marker": marked_complete,
        "complete_episodes": complete,
        "partial_episodes": partial,
        "missing_episodes": max(episodes - len(rows), 0),
        "successes_in_complete_episodes": successes,
        "required_successes": evaluation["required_successes"],
        "training_seconds": None,
        "evaluation_seconds": row.get("evaluation_duration_seconds"),
        "checkpoint_seconds": row.get("checkpoint_duration_seconds"),
    }


def _inventory(root: Path, live: bool) -> dict[str, Any]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("artifact root must be a real directory")
    rows, issues = [], []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            issues.append(f"symlink not read: {path.relative_to(root)}")
            continue
        if not path.is_file():
            continue
        try:
            before = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            after = path.stat()
            stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
            rows.append(
                {
                    "path": str(path.relative_to(root)),
                    "bytes": after.st_size,
                    "sha256": digest.hexdigest(),
                    "stable_during_read": stable,
                }
            )
            if not stable:
                issues.append(f"file changed while reading: {path.relative_to(root)}")
        except OSError as error:
            issues.append(f"{path.relative_to(root)}: {type(error).__name__}")
    return {
        "root": str(root.resolve()),
        "local_only": True,
        "mode": "as_of_nonatomic_lower_bound" if live else "completed_run_read_only_inventory",
        "bytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "inventory_sha256": _hash(rows),
        "issues": issues,
    }


def _references(value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):
        own = (
            [{"path": value["path"], "sha256": value["sha256"]}]
            if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str)
            else []
        )
        return own + [row for child in value.values() for row in _references(child)]
    if isinstance(value, list):
        return [row for child in value for row in _references(child)]
    return []


def summarize_manifest(
    manifest: dict[str, Any],
    *,
    prior_manifest: dict[str, Any] | None = None,
    progress: list[dict[str, Any]] | None = None,
    strict_coverage: bool = False,
    artifact_root: Path | None = None,
    as_of_utc: str | None = None,
) -> dict[str, Any]:
    """Describe one developmental cohort; counters never pool historical studies."""
    errors: list[str] = []
    warnings: list[str] = []
    plan = manifest.get("original_scientific_plan", manifest.get("plan", {}))
    tasks = [row["task"] for row in plan.get("tasks", [])]
    milestones = plan.get("milestones", [])
    evaluation = plan.get("evaluation", {})
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or not milestones
        or milestones != sorted(set(milestones))
        or any(type(value) is not int or value <= 0 for value in milestones)
        or any(
            type(evaluation.get(key)) is not int or evaluation[key] <= 0
            for key in ("episodes", "episode_horizon", "required_successes")
        )
        or evaluation.get("required_successes", 1) > evaluation.get("episodes", 0)
    ):
        raise ValueError("manifest lacks a valid saved task/milestone/evaluation plan")
    if manifest.get("plan_sha256") != _hash(plan):
        errors.append("saved scientific plan content does not match its declared hash")
    raw_status = manifest.get("status")
    if raw_status not in (*FINAL_STATUSES, "RUNNING"):
        errors.append("unknown manifest status")
    live = raw_status == "RUNNING"
    snapshot = _time(manifest.get("finished_at_utc"))
    started = _time(manifest.get("started_at_utc"))
    session = _time(
        manifest.get("session_started_utc", plan.get("session", {}).get("started_at_utc"))
    )
    deadline = _time(
        manifest.get("absolute_deadline_utc", plan.get("session", {}).get("deadline_utc"))
    )
    if snapshot is None or started is None or session is None or deadline is None:
        errors.append("missing or invalid timezone-aware runtime/session timestamps")
    if snapshot is not None and started is not None and snapshot < started:
        errors.append("runtime snapshot precedes runtime start")
    requested_as_of = _time(as_of_utc) if as_of_utc is not None else None
    if as_of_utc is not None and requested_as_of is None:
        raise ValueError("as-of timestamp must include a timezone")
    ledgers = manifest.get("technical_retry_accounting", {})
    current = _counters(manifest.get("ledger"), errors, "manifest.ledger")
    prior = _counters(ledgers.get("prior_ledger", {}), errors, "prior.ledger")
    if ledgers:
        embedded = _counters(ledgers.get("cumulative_ledger"), errors, "retry.cumulative_ledger")
        if embedded != current:
            errors.append("retry cumulative components disagree with manifest ledger")
    elif prior_manifest is not None:
        errors.append("prior manifest supplied without explicit cumulative retry accounting")
    if prior_manifest is not None:
        observed_prior = _counters(prior_manifest.get("ledger"), errors, "provided_prior.ledger")
        if observed_prior != prior or prior_manifest.get("plan_sha256") != manifest.get(
            "plan_sha256"
        ):
            errors.append("provided prior manifest does not match cumulative retry provenance")
        if prior_manifest.get("status") != "ERROR":
            errors.append("provided prior attempt is not the retained failed attempt")
    if any(current[key] < prior[key] for key in COUNTERS):
        errors.append("cumulative counters are smaller than retained prior costs")
    if current["optimizer_updates"] > current["optimizer_update_attempts"]:
        errors.append("completed optimizer calls exceed attempted optimizer calls")
    attempt = {key: current[key] - prior[key] for key in current}
    caps = copy.deepcopy(plan.get("limits", {}))
    contract = manifest.get("technical_retry_contract", {})
    caps.update(contract.get("cumulative_interaction_caps", {}))
    if "cumulative_policy_starts" in contract:
        caps["source_policy_starts"] = contract["cumulative_policy_starts"]
        caps["retries"] = contract["technical_retries"]
    if any(key in current and _steps(cap) and current[key] > cap for key, cap in caps.items()):
        errors.append("cumulative costs exceed a recorded resource cap")
    reported_total = manifest.get("ledger", {}).get("total_agent_steps")
    if reported_total != current["total_agent_steps"]:
        warnings.append("raw aggregate total differs from component sum; raw evidence preserved")
    monitored = dict(current)
    progress_tasks: dict[str, int] = {}
    accepted_progress = []
    previous = dict(prior)
    for index, row in enumerate(progress or []):
        moment = _time(row.get("utc"))
        if (
            row.get("task") not in tasks
            or moment is None
            or started is None
            or deadline is None
            or not started <= moment <= deadline
            or (not live and snapshot is not None and moment > snapshot)
            or (requested_as_of is not None and moment > requested_as_of)
        ):
            errors.append(f"progress[{index}]: unknown task or timestamp outside this attempt")
            continue
        counters = _counters(row.get("ledger"), errors, f"progress[{index}].ledger")
        if any(counters[key] < previous[key] for key in COMPONENTS):
            warnings.append(f"progress[{index}]: nonmonotone components; monitoring uses maxima")
        previous = counters
        for key in monitored:
            monitored[key] = max(monitored[key], counters[key])
        if _steps(row.get("training_steps")):
            task = row["task"]
            progress_tasks[task] = max(progress_tasks.get(task, 0), row["training_steps"])
        accepted_progress.append(moment)
    monitored["total_agent_steps"] = sum(monitored[key] for key in COMPONENTS)
    as_of = max(
        [moment for moment in (snapshot, *accepted_progress) if moment is not None], default=None
    )
    if requested_as_of is not None and as_of is not None and requested_as_of < as_of:
        errors.append("requested as-of precedes the input manifest snapshot")
    stages, outcomes = [], []
    checkpoint_rows: list[dict[str, Any]] = []
    recorded_tasks = manifest.get("tasks", {})
    if set(recorded_tasks) - set(tasks):
        errors.append("manifest contains tasks outside its saved plan")
    for task in tasks:
        task_errors_before = len(errors)
        state = recorded_tasks.get(task, {})
        bank = _strict_goals(manifest, task, plan, errors) if strict_coverage else None
        checkpoints = state.get("checkpoints", [])
        checkpoint_rows.extend(checkpoints)
        if any(row.get("target_steps") not in milestones for row in checkpoints):
            errors.append(f"{task}: checkpoint target outside planned milestones")
        passing = None
        final_failed = False
        for milestone in milestones:
            selected = [row for row in checkpoints if row.get("target_steps") == milestone]
            if len(selected) > 1:
                errors.append(f"{task}@{milestone}: duplicate checkpoint records")
            if passing is not None:
                if selected:
                    errors.append(f"{task}@{milestone}: checkpoint exists after planned early stop")
                stage = {
                    "task": task,
                    "milestone": milestone,
                    "status": "SKIPPED_PLANNED_EARLY_STOP",
                    "passing_milestone": passing,
                }
            elif not selected:
                stage = {
                    "task": task,
                    "milestone": milestone,
                    "status": "MISSING",
                    "complete_episodes": 0,
                    "partial_episodes": 0,
                    "missing_episodes": evaluation["episodes"],
                }
            else:
                stage = _stage(
                    task,
                    milestone,
                    selected[0],
                    state.get("metadata", {}),
                    evaluation,
                    bank,
                    strict_coverage,
                    errors,
                )
                visits = [
                    row
                    for row in manifest.get("queue", [])
                    if row.get("task") == task
                    and row.get("milestone") == milestone
                    and "training_duration_seconds" in row
                ]
                stage["training_seconds"] = (
                    _seconds(visits, "training_duration_seconds") if visits else None
                )
                if stage["status"] == "PASS":
                    passing = milestone
                final_failed = (
                    milestone == milestones[-1] and stage["status"] == "COMPLETE_CHECKPOINT_FAIL"
                )
            stages.append(stage)
        outcomes.append(
            {
                "task": task,
                "outcome": "ERROR"
                if len(errors) > task_errors_before
                else "PASS"
                if passing is not None
                else "FULL_BUDGET_FAIL"
                if final_failed
                else "INCOMPLETE",
                "earliest_passing_milestone": passing,
                "manifest_task_status": state.get("status", "MISSING"),
                "training_steps_at_manifest": state.get("actual_training_steps", 0),
                "training_steps_monitoring_lower_bound": max(
                    state.get("actual_training_steps", 0), progress_tasks.get(task, 0)
                )
                if live
                else None,
            }
        )
    derived = (
        "ERROR"
        if any(row["outcome"] == "ERROR" for row in outcomes)
        else "INCOMPLETE"
        if any(row["outcome"] == "INCOMPLETE" for row in outcomes)
        else "COMPLETE_PASS"
        if all(row["outcome"] == "PASS" for row in outcomes)
        else "COMPLETE_FAIL"
    )
    status = (
        "ERROR"
        if errors or raw_status == "ERROR"
        else "INCOMPLETE"
        if live or raw_status == "INCOMPLETE" or manifest.get("budget_compliant") is False
        else derived
    )
    if raw_status in ("COMPLETE_PASS", "COMPLETE_FAIL") and raw_status != derived:
        errors.append("terminal manifest status disagrees with complete planned-stage evidence")
        status = "ERROR"
    training_seconds = _seconds(manifest.get("queue", []), "training_duration_seconds")
    evaluation_seconds = _seconds(checkpoint_rows, "evaluation_duration_seconds")
    checkpoint_seconds = _seconds(checkpoint_rows, "checkpoint_duration_seconds")
    recorded_source_steps = sum(
        row.get("actual_training_steps", 0)
        for row in recorded_tasks.values()
        if _steps(row.get("actual_training_steps", 0))
    )
    recorded_development_steps = sum(
        row["length"]
        for checkpoint in checkpoint_rows
        for row in checkpoint.get("evaluation", [])
        if _steps(row.get("length"))
    )
    if (
        recorded_source_steps != attempt["source_training_steps"]
        or recorded_development_steps != attempt["development_evaluation_steps"]
    ):
        message = "task/episode step records differ from cumulative-minus-prior component counters"
        if raw_status in ("COMPLETE_PASS", "COMPLETE_FAIL"):
            errors.append(message)
            status = "ERROR"
        else:
            warnings.append(message + "; incomplete evidence is not imputed")
    elapsed = _finite(manifest.get("elapsed_seconds"))
    if elapsed is None or elapsed < 0:
        errors.append("invalid or missing monotonic runtime duration")
        status = "ERROR"
    runtime_utc = (snapshot - started).total_seconds() if snapshot and started else None
    inclusive = (as_of - session).total_seconds() if as_of and session else None
    prior_elapsed = (
        _finite(prior_manifest.get("elapsed_seconds")) if prior_manifest is not None else None
    )
    prior_utc = None
    if prior_manifest is not None:
        prior_start, prior_finish = (
            _time(prior_manifest.get("started_at_utc")),
            _time(prior_manifest.get("finished_at_utc")),
        )
        if prior_start and prior_finish:
            prior_utc = (prior_finish - prior_start).total_seconds()
    storage: dict[str, Any] = {
        "manifest_artifact_bytes_before_manifest": manifest.get("artifact_bytes_before_manifest"),
        "prior_bytes_recorded_by_retry": ledgers.get("prior_artifact_bytes"),
        "cumulative_bytes_before_manifest_recorded": ledgers.get(
            "cumulative_artifact_bytes_before_manifest"
        ),
        "note": (
            "recorded counters precede the final manifest write; "
            "actual inventory is optional and local-only"
        ),
        "declared_checkpoint_files": _references(manifest.get("tasks", {})),
    }
    if artifact_root is not None:
        inventory = _inventory(artifact_root, live)
        storage["actual_inventory"] = inventory
        actual = {row["path"]: row for row in inventory["files"]}
        mismatches = []
        for declared in storage["declared_checkpoint_files"]:
            path = Path(declared["path"])
            if path.is_absolute() or ".." in path.parts:
                mismatches.append(f"unsafe declared path: {path}")
            elif str(path) not in actual or actual[str(path)]["sha256"] != declared["sha256"]:
                mismatches.append(f"declared hash mismatch or missing file: {path}")
        storage["checkpoint_verification_issues"] = mismatches
        if mismatches or (inventory["issues"] and not live):
            errors.extend(mismatches + inventory["issues"])
            status = "ERROR"
        prior_bytes = ledgers.get("prior_artifact_bytes", 0)
        if _steps(prior_bytes):
            storage["cumulative_current_inventory_plus_recorded_prior_bytes"] = (
                inventory["bytes"] + prior_bytes
            )
    return {
        "schema_version": 1,
        "purpose": "descriptive developmental competence and cost report; post-outcome software",
        "status": status,
        "manifest_status": raw_status,
        "derived_operational_gate_status": derived,
        "evidence_mode": "AS_OF_LOWER_BOUND" if live else "TERMINAL_MANIFEST",
        "as_of_observed_utc": as_of.isoformat() if as_of else None,
        "requested_report_as_of_utc": requested_as_of.isoformat() if requested_as_of else None,
        "plan_sha256": manifest.get("plan_sha256"),
        "input_content_sha256": _hash(manifest),
        "prior_input_content_sha256": _hash(prior_manifest) if prior_manifest is not None else None,
        "approval": copy.deepcopy(manifest.get("approval")),
        "code_sha256": copy.deepcopy(manifest.get("code_sha256")),
        "saved_protocol": {
            key: copy.deepcopy(plan.get(key))
            for key in (
                "purpose",
                "environment",
                "algorithm",
                "evaluation",
                "limits",
                "stopping_rule",
            )
        },
        "initial_actor_sha256_by_task": {
            task: recorded_tasks.get(task, {}).get("metadata", {}).get("initial_actor_sha256")
            for task in tasks
        },
        "goal_manifest_content_sha256_by_task": {
            task: _hash(record) for task, record in manifest.get("goal_manifests", {}).items()
        },
        "prelearning_verification": copy.deepcopy(manifest.get("prelearning_verification")),
        "task_outcomes": outcomes,
        "scheduled_checkpoints": stages,
        "validation": {
            "strict_saved_goal_order_seed_checks": strict_coverage,
            "errors": errors,
            "warnings": warnings,
        },
        "costs": {
            "cumulative_components_at_manifest": current,
            "retained_prior_components": prior,
            "attempt_only_components": attempt,
            "raw_total_agent_steps_at_manifest": reported_total,
            "component_sum_at_manifest": current["total_agent_steps"],
            "source_steps_from_task_states": recorded_source_steps,
            "development_steps_from_episode_records": recorded_development_steps,
            "monitoring_components_lower_bound": monitored if live else None,
            "monitoring_is_final_evidence": False,
            "source_training_includes_warmup": True,
            "optimizer_counter_semantics": (
                "attempted/completed SAC model.train calls, not individual Adam operations"
            ),
            "counter_rule": (
                "retry ledger already includes prior costs; derive attempt by subtraction, "
                "never add prior twice"
            ),
        },
        "timing": {
            "snapshot_timestamp_is_final_finish": not live,
            "runtime_started_at_utc": manifest.get("started_at_utc"),
            "runtime_snapshot_or_finish_utc": manifest.get("finished_at_utc"),
            "attempt_runtime_monotonic_seconds": elapsed,
            "attempt_runtime_utc_seconds": runtime_utc,
            "utc_minus_monotonic_seconds": runtime_utc - elapsed
            if runtime_utc is not None and elapsed is not None
            else None,
            "recorded_utc_elapsed_seconds": manifest.get("utc_elapsed_seconds"),
            "inclusive_session_wall_seconds_as_of": inclusive,
            "pre_run_setup_repair_approval_wait_wall_seconds": (started - session).total_seconds()
            if started and session
            else None,
            "pre_run_wall_excluding_prior_runtime_utc_seconds": (started - session).total_seconds()
            - prior_utc
            if started and session and prior_utc is not None
            else None,
            "pre_run_wall_includes_prior_attempt_until_separately_subtracted": bool(ledgers),
            "prior_measured_runtime_monotonic_seconds": prior_elapsed,
            "combined_measured_runtime_monotonic_seconds": elapsed + prior_elapsed
            if elapsed is not None and prior_elapsed is not None
            else elapsed
            if not ledgers
            else None,
            "session_wall_outside_measured_attempt_utc_spans_seconds": inclusive
            - runtime_utc
            - prior_utc
            if inclusive is not None and runtime_utc is not None and prior_utc is not None
            else None,
            "measured_training_seconds": training_seconds,
            "measured_evaluation_seconds": evaluation_seconds,
            "measured_checkpoint_seconds": checkpoint_seconds,
            "recorded_stage_durations_are_lower_bounds": live,
            "preflight_seconds_charged_to_prior_phase_cap": ledgers.get(
                "preflight_seconds_charged_to_prior"
            ),
            "additional_preflight_phase_seconds_cap": ledgers.get(
                "additional_preflight_seconds_cap"
            ),
            "other_measured_runtime_seconds": elapsed
            - training_seconds
            - evaluation_seconds
            - checkpoint_seconds
            if elapsed is not None
            else None,
            "unattributed_runtime_note": (
                "difference from recorded stage durations, not a compute-overhead estimate; "
                "a live record can omit timing for the ongoing visit"
            ),
            "interpretation": (
                "pre-run waiting/setup is inclusive session wall cost, not measured algorithm "
                "compute; clock discrepancies have unverified causes"
            ),
        },
        "stopping": {
            key: copy.deepcopy(manifest.get(key))
            for key in (
                "stop_reason",
                "errors",
                "absolute_deadline_utc",
                "training_deadline_utc",
                "budget_compliant",
                "cutoff_reason",
                "cutoff_phase",
                "cutoff_observed_at_utc",
                "training_reserve_stop_latched",
            )
        },
        "storage": storage,
        "limitations": [
            "One developmental source library; episodes/checkpoints are not library replications.",
            "No episode confidence intervals, confirmation or policy-subspace adaptation claim.",
            "Historical 4k/25k cohorts and MiniHack are not pooled with this manifest.",
            "Running snapshots/progress maxima are lower bounds, never final completion evidence.",
        ],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"Developmental competence: **{report['status']}** ({report['evidence_mode']}).",
        "",
        f"Observed as of `{report['as_of_observed_utc']}`; "
        f"input content SHA-256 `{report['input_content_sha256']}`.",
        f"Plan SHA-256 `{report['plan_sha256']}`. "
        f"Frozen executable commit `{(report.get('approval') or {}).get('code_commit')}`.",
        "",
        "| Task | Milestone | Outcome | Complete / partial / missing episodes | Successes |",
        "| --- | ---: | --- | --- | ---: |",
    ]
    for row in report["scheduled_checkpoints"]:
        coverage = "/".join(
            str(row.get(key, "—"))
            for key in ("complete_episodes", "partial_episodes", "missing_episodes")
        )
        lines.append(
            f"| {row['task']} | {row['milestone']} | {row['status']} | {coverage} | "
            f"{row.get('successes_in_complete_episodes', '—')} |"
        )
    costs, timing = report["costs"], report["timing"]
    cumulative = costs["cumulative_components_at_manifest"]
    prior, attempt = costs["retained_prior_components"], costs["attempt_only_components"]
    lines += [
        "",
        f"Cumulative component costs: `{json.dumps(cumulative, sort_keys=True)}`.",
        f"Retained prior / current attempt: {prior['total_agent_steps']} / "
        f"{attempt['total_agent_steps']} interactions; "
        f"{prior['source_policy_starts']} / {attempt['source_policy_starts']} constructors. "
        "The cumulative ledger already includes the prior attempt.",
        f"Measured training {timing['measured_training_seconds']:.3f}s; "
        f"evaluation {timing['measured_evaluation_seconds']:.3f}s. "
        f"Runtime monotonic {timing['attempt_runtime_monotonic_seconds']}s; "
        f"runtime UTC {timing['attempt_runtime_utc_seconds']}s; "
        f"inclusive session wall {timing['inclusive_session_wall_seconds_as_of']}s. "
        "Pre-run setup/repair/wait "
        f"{timing['pre_run_setup_repair_approval_wait_wall_seconds']}s includes any prior attempt "
        "until separately subtracted; it is not an algorithm-compute measurement. "
        f"Combined measured runtime: {timing['combined_measured_runtime_monotonic_seconds']}s.",
        f"Stopping: `{json.dumps(report['stopping'], sort_keys=True)}`.",
        "",
    ]
    if costs["monitoring_components_lower_bound"] is not None:
        lines += [
            "Live component maxima (monitoring lower bound): "
            f"`{json.dumps(costs['monitoring_components_lower_bound'], sort_keys=True)}`.",
            "",
        ]
    storage = report["storage"]
    if "actual_inventory" in storage:
        inventory = storage["actual_inventory"]
        lines += [
            f"Local-only artifact inventory: {inventory['bytes']} bytes; "
            f"SHA-256 `{inventory['inventory_sha256']}` ({inventory['mode']}).",
            f"Checkpoint verification issues: {storage['checkpoint_verification_issues']}.",
            "",
        ]
    else:
        lines += [
            "Actual artifact hashing was not requested; recorded pre-manifest bytes "
            f"{storage['manifest_artifact_bytes_before_manifest']}, "
            f"recorded prior bytes {storage['prior_bytes_recorded_by_retry']}.",
            "",
        ]
    for issue in report["validation"]["errors"] + report["validation"]["warnings"]:
        lines.append(f"- Evidence audit: {issue}")
    lines += ["", *report["limitations"], ""]
    return "\n".join(lines)


def _read_json(path: Path) -> tuple[Any, str]:
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--prior-manifest", type=Path)
    parser.add_argument("--progress", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--strict-coverage", action="store_true")
    parser.add_argument("--as-of-utc")
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    manifest, manifest_hash = _read_json(args.manifest)
    prior, prior_hash = (
        _read_json(args.prior_manifest) if args.prior_manifest is not None else (None, None)
    )
    progress, progress_hash = (
        _read_json(args.progress) if args.progress is not None else (None, None)
    )
    if not isinstance(manifest, dict) or (prior is not None and not isinstance(prior, dict)):
        raise ValueError("manifest inputs must be JSON objects")
    if progress is not None and (
        not isinstance(progress, list) or any(not isinstance(row, dict) for row in progress)
    ):
        raise ValueError("progress must be a JSON array of records")
    report = summarize_manifest(
        manifest,
        prior_manifest=prior,
        progress=progress,
        strict_coverage=args.strict_coverage,
        artifact_root=args.artifact_root,
        as_of_utc=args.as_of_utc,
    )
    report["inputs"] = {
        "manifest": str(args.manifest.resolve()),
        "manifest_file_sha256": manifest_hash,
        "prior_manifest": str(args.prior_manifest.resolve()) if args.prior_manifest else None,
        "progress": str(args.progress.resolve()) if args.progress else None,
        "prior_manifest_file_sha256": prior_hash,
        "progress_file_sha256": progress_hash,
        "local_only": True,
    }
    text = (
        render_markdown(report)
        if args.format == "markdown"
        else json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    if args.output is None:
        print(text, end="")
    else:
        if args.output.exists() or args.output.is_symlink():
            raise FileExistsError("existing report output is preserved")
        roots = [args.manifest.resolve().parent]
        for path in (args.prior_manifest, args.progress, args.artifact_root):
            if path is not None:
                roots.append(path.resolve() if path.is_dir() else path.resolve().parent)
        for record in (manifest, prior or {}):
            for value in (
                record.get("attempt_artifact_root"),
                record.get("plan", {}).get("output"),
                record.get("technical_retry_contract", {}).get("prior_artifact_root"),
                record.get("technical_retry_contract", {}).get("replacement_artifact_root"),
            ):
                if isinstance(value, str):
                    roots.append(Path(value).resolve())
        output = args.output.resolve()
        if any(output.is_relative_to(root) for root in roots):
            raise ValueError(
                "report output must be outside all input and declared artifact run directories"
            )
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
