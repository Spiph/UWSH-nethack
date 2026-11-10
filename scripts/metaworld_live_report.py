"""Compact, read-only scientific summaries of JSON run evidence.

This module uses only the standard library. It never opens artifacts, loads
arrays/models, changes a stopping rule or treats a failed competence gate as a
reporter failure. Paths/hashes below are publisher commitments, not file checks.
"""

from __future__ import annotations

import copy
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.metaworld_development_report import _hash, _stage, _strict_goals, _time

COMPONENTS = (
    "source_training_steps",
    "development_evaluation_steps",
    "preflight_steps",
    "source_diagnostic_steps",
    "target_steps",
    "final_test_steps",
)
SOURCE_FINAL = {"COMPLETE_PASS", "COMPLETE_FAIL", "INCOMPLETE", "ERROR", "NOOP_ALREADY_COMPLETE"}
SENTINEL_FINAL = {"COMPLETE", "INCOMPLETE", "ERROR", "STOPPED_AT_BUDGET", "INTERRUPTED"}


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _messages(value: Any, prefix: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return [f"{prefix}: malformed error list"]
    return [f"{prefix}: {str(item)[:500]}" for item in value]


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _ledger(value: Any, errors: list[str], name: str, *, sentinel: bool = False) -> dict[str, int]:
    if not isinstance(value, dict):
        errors.append(f"{name}: missing ledger")
        value = {}
    result: dict[str, int] = {}
    required = ("source_diagnostic_steps",) if sentinel else COMPONENTS[:3]
    for key in required:
        if key not in value:
            errors.append(f"{name}: missing {key}")
    for key, item in value.items():
        if not isinstance(key, str) or key == "total_agent_steps":
            continue
        if not _count(item):
            errors.append(f"{name}.{key}: expected a nonnegative integer")
            continue
        result[key] = item
    if "development_steps" in result:
        if (
            "development_evaluation_steps" in result
            and result["development_steps"] != result["development_evaluation_steps"]
        ):
            errors.append(f"{name}: conflicting development counter aliases")
        result.setdefault("development_evaluation_steps", result["development_steps"])
    for key in COMPONENTS:
        result.setdefault(key, 0)
    result["total_agent_steps"] = sum(result[key] for key in COMPONENTS)
    if result.get("optimizer_updates", 0) > result.get(
        "optimizer_update_attempts", result.get("optimizer_updates", 0)
    ):
        errors.append(f"{name}: completed optimizer work exceeds attempts")
    return result


def _same_plan_hash(value: Any, expected: Any, errors: list[str], name: str) -> None:
    try:
        if not isinstance(value, dict) or _hash(value) != expected:
            errors.append(f"{name}: content does not match its declared hash")
    except (ValueError, TypeError):
        errors.append(f"{name}: invalid/nonfinite JSON content")


def _summary_counts(value: Any, errors: list[str], name: str) -> dict[str, Any] | None:
    keys = (
        "complete_episodes",
        "successful_complete_episodes",
        "recent_complete_episodes",
        "recent_successes",
    )
    if not isinstance(value, dict) or any(not _count(value.get(key)) for key in keys):
        errors.append(f"{name}: malformed completed-source-episode telemetry")
        return None
    if (
        value[keys[1]] > value[keys[0]]
        or value[keys[2]] > min(100, value[keys[0]])
        or value[keys[3]] > value[keys[2]]
    ):
        errors.append(f"{name}: inconsistent source-episode success counts")
        return None
    return {
        **{key: value[key] for key in keys},
        "scope": str(value.get("scope", "live_progress"))[:150],
    }


def _history_summary(history: Any, errors: list[str], name: str) -> dict[str, Any] | None:
    if not isinstance(history, list):
        return None
    complete: list[dict[str, Any]] = []
    partial = 0
    steps = 0
    for index, row in enumerate(history):
        if (
            not isinstance(row, dict)
            or row.get("episode") != index
            or not _count(row.get("length"))
            or row["length"] > 500
            or type(row.get("success")) is not bool
        ):
            errors.append(f"{name}: malformed source episode history")
            return None
        steps += row["length"]
        if row.get("complete_episode") is True:
            if (
                row["length"] != 500
                or row.get("terminated") is not False
                or row.get("truncated") is not True
            ):
                errors.append(f"{name}: completed source episode violates full500-step semantics")
                return None
            complete.append(row)
        elif row["length"]:
            partial += 1
    recent = complete[-100:]
    return {
        "complete_episodes": len(complete),
        "successful_complete_episodes": sum(row["success"] for row in complete),
        "recent_complete_episodes": len(recent),
        "recent_successes": sum(row["success"] for row in recent),
        "partial_episodes_excluded": partial,
        "recorded_physical_steps": steps,
        "scope": "known_checkpoint_only",
    }


def _progress(
    rows: Any,
    tasks: set[str],
    observed: datetime,
    earliest: datetime | None,
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], dict[str, int]]:
    per_task: dict[str, Any] = {}
    maxima: dict[str, int] = {}
    latest: datetime | None = None
    if rows is not None and not isinstance(rows, list):
        errors.append("progress: expected a list of JSON records")
        rows = []
    for index, row in enumerate(rows or []):
        if not isinstance(row, dict):
            errors.append(f"progress[{index}]: expected an object")
            continue
        task, moment = row.get("task"), _time(row.get("utc"))
        if not isinstance(task, str) or task not in tasks or moment is None:
            errors.append(f"progress[{index}]: unknown task or invalid timestamp")
            continue
        if moment > observed:
            warnings.append(f"progress[{index}]: timestamp is later than report observation")
            continue
        if earliest is not None and moment < earliest:
            errors.append(f"progress[{index}]: timestamp predates retained run provenance")
            continue
        counters = _ledger(row.get("ledger"), errors, f"progress[{index}].ledger")
        for key, value in counters.items():
            maxima[key] = max(maxima.get(key, 0), value)
        diagnostics = row.get("diagnostics", {})
        clean: dict[str, float | None] = {}
        flagged: set[str] = set()
        for key in (
            row.get("nonfinite_diagnostic_keys", [])
            if isinstance(row.get("nonfinite_diagnostic_keys", []), list)
            else []
        ):
            if isinstance(key, str):
                flagged.add(key)
        if isinstance(diagnostics, dict):
            for key, value in diagnostics.items():
                if not isinstance(key, str):
                    continue
                clean[key] = float(value) if _finite(value) else None
                if not _finite(value):
                    flagged.add(key)
        else:
            errors.append(f"progress[{index}]: malformed diagnostics")
        if flagged:
            warnings.append(f"{task}: nonfinite training telemetry: {', '.join(sorted(flagged))}")
        source = row.get("source_training_episode_summary")
        episode_summary = (
            _summary_counts(source, errors, f"progress[{index}]") if source is not None else None
        )
        steps = row.get("training_steps")
        if not _count(steps):
            errors.append(f"progress[{index}]: invalid training_steps")
            steps = None
        previous = per_task.get(task)
        previous_moment = _time(previous["utc"]) if previous is not None else None
        if previous_moment is not None and moment < previous_moment:
            continue
        per_task[task] = {
            "utc": moment.isoformat(),
            "training_steps": steps,
            "diagnostics": clean,
            "nonfinite_diagnostic_keys": sorted(flagged),
            "source_training_episode_summary": episode_summary,
        }
        latest = max(latest, moment) if latest is not None else moment
    maxima["total_agent_steps"] = sum(maxima.get(key, 0) for key in COMPONENTS)
    return {
        "status": "available" if per_task else "absent",
        "per_task": per_task,
        "last_utc": latest.isoformat() if latest is not None else None,
        "age_seconds": (observed - latest).total_seconds() if latest is not None else None,
    }, maxima


def _absolute_root(manifest: dict[str, Any], source_path: Path | None) -> Path | None:
    if source_path is not None:
        return source_path.absolute().parent
    plan = manifest.get("plan", {})
    extension = plan.get("extension", {}) if isinstance(plan, dict) else {}
    if not isinstance(extension, dict):
        extension = {}
    value = (
        manifest.get("attempt_artifact_root")
        or extension.get("output")
        or (plan.get("output") if isinstance(plan, dict) else None)
    )
    return Path(value) if isinstance(value, str) and Path(value).is_absolute() else None


def _artifact_refs(
    value: Any,
    root: Path | None,
    errors: list[str],
    warnings: list[str],
) -> list[dict[str, str]]:
    result = []
    if isinstance(value, dict):
        path, sha = value.get("path"), value.get("sha256")
        if isinstance(path, str) and isinstance(sha, str):
            candidate = Path(path)
            if re.fullmatch(r"[0-9a-f]{64}", sha) is None or ".." in candidate.parts:
                errors.append("artifact reference has an invalid path/hash commitment")
            elif candidate.is_absolute():
                result.append({"path": str(candidate), "sha256": sha})
            elif root is not None:
                result.append({"path": str(root / candidate), "sha256": sha})
            else:
                warnings.append("relative artifact commitment has no known artifact root")
        for child in value.values():
            result.extend(_artifact_refs(child, root, errors, warnings))
    elif isinstance(value, list):
        for child in value:
            result.extend(_artifact_refs(child, root, errors, warnings))
    return result


def _source_stages(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    milestones: list[int],
    tasks: list[str],
    root: Path | None,
    errors: list[str],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, str]]]:
    stages: list[dict[str, Any]] = []
    histories: dict[str, Any] = {}
    references: list[dict[str, str]] = []
    evaluation = plan.get("evaluation", {})
    if (
        not isinstance(evaluation, dict)
        or any(
            type(evaluation.get(key)) is not int or evaluation[key] <= 0
            for key in ("episodes", "episode_horizon", "required_successes")
        )
        or evaluation.get("required_successes", 1) > evaluation.get("episodes", 0)
    ):
        errors.append("saved evaluation plan is malformed")
        return stages, histories, references
    states = manifest.get("tasks", {})
    if not isinstance(states, dict):
        errors.append("manifest tasks must be an object")
        states = {}
    binding = manifest.get("parent_binding", {})
    parent = binding.get("parent_manifest", {}) if isinstance(binding, dict) else {}
    current_plan = manifest.get("plan", {})
    extension = current_plan.get("extension", {}) if isinstance(current_plan, dict) else {}
    if not isinstance(extension, dict):
        extension = {}
    parent_root_value = extension.get("parent_root")
    parent_root = (
        Path(parent_root_value)
        if isinstance(parent_root_value, str) and Path(parent_root_value).is_absolute()
        else None
    )
    parent_states = parent.get("tasks", {}) if isinstance(parent, dict) else {}
    for task in tasks:
        state = states.get(task, {})
        if not isinstance(state, dict):
            errors.append(f"{task}: malformed task state")
            state = {}
        metadata = state.get("metadata", {})
        if not isinstance(metadata, dict):
            errors.append(f"{task}: malformed metadata")
            metadata = {}
        goals = manifest.get("goal_manifests", {})
        if isinstance(goals, dict) and task in goals:
            try:
                bank = _strict_goals(manifest, task, plan, errors)
            except (TypeError, AttributeError, KeyError, ValueError, IndexError):
                bank = None
                errors.append(f"{task}: malformed goal-bank commitments")
        else:
            bank = None
            warnings.append(f"{task}: goal manifest has not been published")
        raw_rows = state.get("checkpoints", [])
        if not isinstance(raw_rows, list):
            raw_rows = []
            errors.append(f"{task}: malformed checkpoint list")
        rows = []
        for row in raw_rows:
            if not isinstance(row, dict):
                errors.append(f"{task}: malformed checkpoint record")
                continue
            rows.append(row)
            owner = root
            explicit = row.get("artifact_root")
            if explicit is not None:
                if not isinstance(explicit, str) or not Path(explicit).is_absolute():
                    errors.append(f"{task}: checkpoint artifact_root must be absolute")
                    owner = None
                else:
                    owner = Path(explicit)
            elif parent_root is not None and isinstance(parent_states, dict):
                old_state = parent_states.get(task, {})
                old_rows = old_state.get("checkpoints", []) if isinstance(old_state, dict) else []
                if any(
                    isinstance(old, dict) and old.get("artifacts") == row.get("artifacts")
                    for old in old_rows
                ):
                    owner = parent_root
            references.extend(_artifact_refs(row.get("artifacts", {}), owner, errors, warnings))
        if rows:
            latest_row = max(
                rows,
                key=lambda row: int(row["actual_steps"]) if _count(row.get("actual_steps")) else -1,
            )
            artifacts = latest_row.get("artifacts", {})
            environment = artifacts.get("environment", {}) if isinstance(artifacts, dict) else {}
            history = environment.get("episode_history") if isinstance(environment, dict) else None
            histories[task] = _history_summary(history, errors, task)
        else:
            histories[task] = None
        passing: int | None = None
        for milestone in milestones:
            selected = [row for row in rows if row.get("target_steps") == milestone]
            if passing is not None:
                stages.append(
                    {
                        "task": task,
                        "milestone": milestone,
                        "status": "SKIPPED_PLANNED_EARLY_STOP",
                        "passing_milestone": passing,
                    }
                )
                continue
            if not selected:
                stages.append(
                    {
                        "task": task,
                        "milestone": milestone,
                        "status": "MISSING",
                        "complete_episodes": 0,
                        "successes_in_complete_episodes": 0,
                        "required_successes": evaluation["required_successes"],
                    }
                )
                continue
            # A retained interrupted visit is not a duplicate completed checkpoint.
            chosen = max(
                selected,
                key=lambda row: (
                    row.get("partial") is False and row.get("actual_steps") == milestone,
                    row.get("evaluation_complete") is True,
                    row.get("actual_steps") if _count(row.get("actual_steps")) else -1,
                    len(row.get("evaluation", []))
                    if isinstance(row.get("evaluation", []), list)
                    else -1,
                ),
            )
            clean = copy.copy(chosen)
            episodes = clean.get("evaluation", [])
            if episodes and bank is None:
                errors.append(f"{task}@{milestone}: evaluation lacks a valid goal-bank commitment")
            if not isinstance(episodes, list) or any(not isinstance(row, dict) for row in episodes):
                errors.append(f"{task}@{milestone}: malformed evaluation records")
                clean["evaluation"] = []
            clean_metadata = copy.copy(metadata)
            for key in ("evaluation_goal_order", "evaluation_episode_seeds"):
                if not isinstance(clean_metadata.get(key), list):
                    clean_metadata[key] = []
            try:
                stage = _stage(
                    task, milestone, clean, clean_metadata, evaluation, bank, True, errors
                )
            except (ValueError, KeyError, TypeError, AttributeError, IndexError):
                stage = {"task": task, "milestone": milestone, "status": "ERROR"}
                errors.append(f"{task}@{milestone}: invalid checkpoint protocol data")
            stage["physical_training_steps"] = chosen.get(
                "physical_actual_steps", chosen.get("actual_steps")
            )
            stage["model_timesteps"] = chosen.get("model_timesteps", chosen.get("actual_steps"))
            if stage.get("evaluation_seconds") is None:
                stage["evaluation_seconds"] = chosen.get("continuation_evaluation_seconds")
            visits: list[dict[str, Any]] = []
            for evidence in (parent, manifest):
                queue = evidence.get("queue", []) if isinstance(evidence, dict) else []
                if isinstance(queue, list):
                    for visit in queue:
                        if (
                            isinstance(visit, dict)
                            and visit.get("task") == task
                            and visit.get("milestone") == milestone
                            and visit not in visits
                        ):
                            visits.append(visit)
            recorded_visits = [
                visit
                for visit in visits
                if _finite(visit.get("training_duration_seconds"))
                and visit["training_duration_seconds"] >= 0
            ]
            missing_durations = len(visits) - len(recorded_visits)
            recorded_seconds = (
                sum(visit["training_duration_seconds"] for visit in recorded_visits)
                if recorded_visits
                else None
            )
            stage["training_seconds"] = recorded_seconds if not missing_durations else None
            stage["recorded_training_seconds"] = recorded_seconds
            stage["missing_training_duration_visits"] = missing_durations
            stage["training_timing_scope"] = (
                "partial_recorded_visits"
                if recorded_visits and missing_durations
                else "all_published_visits_recorded"
                if recorded_visits
                else "unrecorded"
            )
            stage["retained_partial_checkpoints"] = [
                {
                    "actual_training_steps": row.get("actual_steps"),
                    "partial": row.get("partial") is True,
                }
                for row in selected
                if row is not chosen
            ]
            if stage["status"] == "PASS":
                passing = milestone
            elif stage["status"] == "COMPLETE_CHECKPOINT_FAIL":
                warnings.append(
                    f"{task}@{milestone}: {stage['successes_in_complete_episodes']}/"
                    f"{evaluation['episodes']} complete successes; competence gate not met"
                )
            stages.append(stage)
    return stages, histories, references


def _sentinel_stages(
    manifest: dict[str, Any],
    plan: dict[str, Any],
    root: Path | None,
    errors: list[str],
    warnings: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows = manifest.get("episodes", [])
    if not isinstance(rows, list):
        errors.append("sentinel episodes must be a list")
        rows = []
    task_requests = plan.get("tasks", [])
    if not isinstance(task_requests, list):
        errors.append("saved sentinel task requests must be a list")
        task_requests = []
    if type(plan.get("episode_horizon")) is not int or plan.get("episode_horizon") != 500:
        errors.append("saved sentinel episode horizon must be 500")
    horizon = 500
    requests = {
        row["task"]: row
        for row in task_requests
        if isinstance(row, dict) and isinstance(row.get("task"), str)
    }
    stages, references = [], []
    for row in rows:
        if not isinstance(row, dict):
            errors.append("malformed sentinel episode record")
            continue
        task, length = row.get("task"), row.get("length")
        if (
            not isinstance(task, str)
            or task not in requests
            or not isinstance(length, int)
            or isinstance(length, bool)
            or length < 0
            or length > horizon
            or type(row.get("success")) is not bool
        ):
            errors.append("sentinel episode has invalid task/length/success")
            continue
        initial = row.get("initial", {})
        episode = initial.get("episode", {}) if isinstance(initial, dict) else {}
        if not isinstance(episode, dict):
            episode = {}
        if length and not episode:
            errors.append(f"{task}: executed sentinel lacks its declared source episode identity")
        if episode:
            request = requests[task]
            if (
                episode.get("goal_index") != plan.get("goal_index")
                or episode.get("goal_id") != request.get("goal_id")
                or episode.get("goal_bank_sha256") != request.get("bank_sha256")
                or episode.get("episode_seed") != request.get("episode_seed")
            ):
                errors.append(f"{task}: sentinel goal/seed differs from frozen source request")
        complete = row.get("complete_episode") is True and length == horizon
        if row.get("status") == "COMPLETE" and not complete:
            errors.append(f"{task}: sentinel completion lacks a full episode")
        if complete and not row["success"]:
            warnings.append(
                f"{task}: completed scripted source sentinel without success; diagnostic only"
            )
        stages.append(
            {
                "task": task,
                "milestone": None,
                "status": row.get("status", "RUNNING"),
                "actual_training_steps": 0,
                "source_diagnostic_steps": length,
                "complete_episodes": int(complete),
                "partial_episodes": int(not complete and length > 0),
                "successes_in_complete_episodes": int(complete and row["success"]),
                "required_successes": None,
            }
        )
        if isinstance(row.get("trace"), str) and isinstance(row.get("trace_sha256"), str):
            references.extend(
                _artifact_refs(
                    {"path": row["trace"], "sha256": row["trace_sha256"]}, root, errors, warnings
                )
            )
    return stages, references


def summarize_run(
    manifest: dict[str, Any],
    progress: list[Any] | None = None,
    *,
    source_path: Path | None = None,
    observed_utc: str | None = None,
) -> dict[str, Any]:
    """Summarize parsed JSON only; the input objects and sealed files stay untouched."""
    errors = _messages(manifest.get("errors"), "runner")
    warnings: list[str] = []
    observed = _time(observed_utc) if observed_utc is not None else datetime.now(timezone.utc)
    if observed is None:
        observed = datetime.now(timezone.utc)
        errors.append("invalid observed_utc timestamp")
    saved = manifest.get("plan", {})
    if not isinstance(saved, dict):
        saved = {}
        errors.append("missing saved plan")
    _same_plan_hash(saved, manifest.get("plan_sha256"), errors, "saved plan")
    sentinel = "source_diagnostic_steps" in saved or "scripted_source" in str(
        manifest.get("interpretation", "")
    )
    continuation = isinstance(manifest.get("parent_binding"), dict) or isinstance(
        saved.get("extension"), dict
    )
    kind = (
        "scripted_source_sentinel"
        if sentinel
        else "source_continuation"
        if continuation
        else "source_development"
        if saved.get("tasks")
        else "unknown"
    )
    raw_status = manifest.get("status") if isinstance(manifest.get("status"), str) else None
    phase = manifest.get("phase") if isinstance(manifest.get("phase"), str) else None
    terminal = (
        raw_status in SENTINEL_FINAL
        if sentinel
        else phase == "FINISHED" and raw_status in SOURCE_FINAL
    )
    if raw_status not in (
        (SENTINEL_FINAL | {"RUNNING"}) if sentinel else (SOURCE_FINAL | {"RUNNING"})
    ):
        errors.append("unknown raw run status")
    ledger = _ledger(manifest.get("ledger"), errors, "manifest.ledger", sentinel=sentinel)
    raw_ledger = manifest.get("ledger", {})
    if isinstance(raw_ledger, dict) and raw_ledger.get("total_agent_steps") not in (
        None,
        ledger["total_agent_steps"],
    ):
        warnings.append(
            "raw aggregate total differs from component sum; component counters are used"
        )
    accounting = manifest.get("technical_retry_accounting", {})
    if isinstance(accounting, dict) and accounting:
        recorded = _ledger(accounting.get("cumulative_ledger"), errors, "retry.cumulative_ledger")
        if any(recorded.get(key, 0) != ledger.get(key, 0) for key in COMPONENTS):
            errors.append("retry accounting disagrees with manifest cumulative components")
    binding = manifest.get("parent_binding", {})
    parent = binding.get("parent_manifest", {}) if isinstance(binding, dict) else {}
    if continuation and isinstance(parent, dict):
        prior = _ledger(parent.get("ledger"), errors, "parent.ledger")
        if any(ledger.get(key, 0) < prior.get(key, 0) for key in COMPONENTS):
            errors.append("continuation cumulative ledger is smaller than retained parent costs")
        if saved.get("parent_binding_sha256") is not None:
            _same_plan_hash(binding, saved["parent_binding_sha256"], errors, "parent binding")
    extension = saved.get("extension", {})
    plan = manifest.get("original_scientific_plan", saved.get("original_scientific_plan", saved))
    if not isinstance(plan, dict):
        plan = saved
    task_rows = plan.get("tasks", [])
    tasks = (
        [
            row["task"]
            for row in task_rows
            if isinstance(row, dict) and isinstance(row.get("task"), str)
        ]
        if isinstance(task_rows, list)
        else []
    )
    if len(set(tasks)) != len(tasks):
        errors.append("saved task list contains duplicates")
    milestone_value = (
        extension.get("milestones", plan.get("milestones", []))
        if isinstance(extension, dict)
        else plan.get("milestones", [])
    )
    milestones = milestone_value if isinstance(milestone_value, list) else []
    if not sentinel and (
        not tasks
        or not milestones
        or any(not _count(value) or value == 0 for value in milestones)
        or milestones != sorted(set(milestones))
    ):
        errors.append("saved source task/milestone plan is malformed")
        milestones = []
    caps = (
        extension.get("limits", plan.get("limits", {}))
        if isinstance(extension, dict)
        else plan.get("limits", {})
    )
    if sentinel:
        caps = {
            **saved,
            "environment_constructor_attempts": saved.get("environment_constructors"),
            "scripted_constructor_attempts": saved.get("scripted_controller_instances"),
            "scripted_controllers": saved.get("scripted_controller_instances"),
            "source_diagnostic_step_attempts": saved.get("source_diagnostic_steps"),
            "actor_array_loads": saved.get("source_actor_array_loads"),
        }
    elif isinstance(caps, dict):
        caps = dict(caps)
        retry = manifest.get("technical_retry_contract", {})
        if isinstance(retry, dict) and retry:
            before_retry = len(errors)
            declared_retry_hash = manifest.get("technical_retry_contract_sha256")
            _same_plan_hash(retry, declared_retry_hash, errors, "technical retry contract")
            approval = manifest.get("approval", {})
            if (
                not isinstance(approval, dict)
                or approval.get("approved") is not True
                or approval.get("technical_retry_contract_sha256") != declared_retry_hash
            ):
                errors.append("technical retry overlay lacks matching saved approval binding")
            if len(errors) != before_retry:
                retry = {}
            retry_caps = retry.get("cumulative_interaction_caps", {})
            if isinstance(retry_caps, dict):
                caps.update(retry_caps)
            else:
                errors.append("technical retry cumulative caps must be an object")
            if "cumulative_policy_starts" in retry:
                caps["source_policy_starts"] = retry["cumulative_policy_starts"]
            if "technical_retries" in retry:
                caps["technical_retries"] = retry["technical_retries"]
                caps["retries"] = retry["technical_retries"]
    root = _absolute_root(manifest, source_path)
    if sentinel:
        stages, references = _sentinel_stages(manifest, saved, root, errors, warnings)
        histories: dict[str, Any] = {}
    else:
        stages, histories, references = _source_stages(
            manifest, plan, milestones, tasks, root, errors, warnings
        )
    if raw_status == "COMPLETE_PASS" and not sentinel:
        for task in tasks:
            task_stages = [stage for stage in stages if stage.get("task") == task]
            if not any(stage.get("status") == "PASS" for stage in task_stages) or any(
                stage.get("status")
                not in {"PASS", "COMPLETE_CHECKPOINT_FAIL", "SKIPPED_PLANNED_EARLY_STOP"}
                for stage in task_stages
            ):
                errors.append(f"{task}: COMPLETE_PASS lacks supported complete passing stages")
    if raw_status == "COMPLETE" and sentinel:
        coverage = [str(stage.get("task")) for stage in stages]
        if (
            manifest.get("validation_complete") is not True
            or sorted(coverage) != sorted(tasks)
            or any(
                stage.get("status") != "COMPLETE" or stage.get("complete_episodes") != 1
                for stage in stages
            )
        ):
            errors.append("sentinel COMPLETE lacks exact full-episode planned task coverage")
    start_evidence = [manifest]
    if continuation and isinstance(parent, dict):
        start_evidence.append(parent)
    starts = []
    for evidence in start_evidence:
        start = _time(evidence.get("started_at_utc")) or _time(evidence.get("session_started_utc"))
        if start is not None:
            starts.append(start)
    earliest = min(starts) if starts else None
    latest, maxima = _progress(progress, set(tasks), observed, earliest, errors, warnings)
    if not terminal:
        for key, value in maxima.items():
            ledger[key] = max(ledger.get(key, 0), value)
        ledger["total_agent_steps"] = sum(ledger[key] for key in COMPONENTS)
        if latest["status"] == "absent" and not sentinel:
            warnings.append(
                "live training progress is absent; counters reflect saved manifest evidence only"
            )
    elif any(maxima.get(key, 0) > ledger.get(key, 0) for key in COMPONENTS):
        errors.append("terminal cumulative counters are smaller than recorded progress")
    if isinstance(caps, dict):
        for key, cap in caps.items():
            if key in ledger and _count(cap) and ledger[key] > cap:
                errors.append(f"cumulative {key} exceeds its saved cap")
    source_history: dict[str, Any] = {}
    for task in tasks:
        live_summary = latest["per_task"].get(task, {}).get("source_training_episode_summary")
        known = histories.get(task)
        source_history[task] = (
            {**live_summary, "scope": "live_progress"} if live_summary is not None else known
        )
        if source_history[task] is None:
            source_history[task] = {
                "scope": "absent",
                "complete_episodes": None,
                "successful_complete_episodes": None,
                "recent_complete_episodes": None,
                "recent_successes": None,
            }
    unique = {(row["path"], row["sha256"]): row for row in references}
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": kind,
        "raw_status": raw_status,
        "status": raw_status,
        "phase": phase,
        "stop_reason": manifest.get("stop_reason"),
        "cutoff_reason": manifest.get("cutoff_reason"),
        "terminal": bool(terminal),
        "observed_utc": observed.isoformat(),
        "ledger": ledger,
        "ledger_scope": "cumulative components; inherited costs are not added again",
        "checkpoint_summaries": stages,
        "latest_progress": latest,
        "source_history_summaries": source_history,
        "errors": list(dict.fromkeys(errors))[:100],
        "warnings": list(dict.fromkeys(warnings))[:100],
        "artifact_references": sorted(
            unique.values(), key=lambda row: (row["path"], row["sha256"])
        ),
        "artifact_reference_scope": "declared sealed publisher commitments; no artifact file read",
    }
    return dict(_json_safe(result))


def _cell(value: Any) -> str:
    return str(value if value is not None else "unknown").replace("|", "\\|").replace("\n", " ")


def render_markdown(report: dict[str, Any]) -> str:
    """Render the compact summary without reading any other source."""
    ledger = report.get("ledger", {})
    evidence = "terminal" if report.get("terminal") else "live / not finalized"
    lines = [
        f"**{_cell(report.get('kind'))}** — {_cell(report.get('raw_status'))}",
        "",
        f"Evidence: {evidence}; phase {_cell(report.get('phase'))}.",
        "",
        f"Cumulative interactions: {_cell(ledger.get('total_agent_steps'))} "
        f"(source {_cell(ledger.get('source_training_steps'))}; "
        f"development {_cell(ledger.get('development_evaluation_steps'))}; "
        f"preflight {_cell(ledger.get('preflight_steps'))}; "
        f"diagnostic {_cell(ledger.get('source_diagnostic_steps'))}).",
        "",
        "| Task | Milestone | Outcome | Complete successes / episodes | Physical / model steps |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for row in report.get("checkpoint_summaries", []):
        successes = _cell(row.get("successes_in_complete_episodes"))
        episodes = _cell(row.get("complete_episodes"))
        physical = _cell(row.get("physical_training_steps", row.get("actual_training_steps")))
        model = _cell(row.get("model_timesteps"))
        cells = [
            row.get("task"),
            row.get("milestone"),
            row.get("status"),
            f"{successes} / {episodes}",
            f"{physical} / {model}",
        ]
        lines.append("| " + " | ".join(_cell(cell) for cell in cells) + " |")
    lines.extend(["", "Source episode telemetry (completed episodes only):", ""])
    for task, summary in report.get("source_history_summaries", {}).items():
        successes = _cell(summary.get("successful_complete_episodes"))
        episodes = _cell(summary.get("complete_episodes"))
        recent = _cell(summary.get("recent_successes"))
        count = _cell(summary.get("recent_complete_episodes"))
        lines.append(
            f"- {_cell(task)}: {successes}/{episodes}; "
            f"recent {recent}/{count}; {_cell(summary.get('scope'))}."
        )
    latest = report.get("latest_progress", {})
    lines.extend(
        [
            "",
            f"Latest progress: {_cell(latest.get('status'))}; {_cell(latest.get('last_utc'))}; "
            f"age {_cell(latest.get('age_seconds'))} seconds.",
        ]
    )
    for key, label in (("errors", "Evidence errors"), ("warnings", "Observations")):
        if report.get(key):
            lines.extend(["", label + ":", ""])
            lines.extend(f"- {_cell(item)}" for item in report[key])
    return "\n".join(lines) + "\n"
