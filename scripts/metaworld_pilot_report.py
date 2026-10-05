"""Audit and summarize saved Meta-World pilot manifests without changing raw data."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


def _mean(values: list[float]) -> float | None:
    return fmean(values) if values else None


def _key(*parts: Any) -> tuple[Any, ...]:
    return tuple(parts)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _finite_float(value: Any) -> float | None:
    return float(value) if _finite_number(value) else None


def _metric_summary(rows: list[dict[str, Any]], missing: int = 0) -> dict[str, Any]:
    complete = [row for row in rows if row.get("complete_episode") is True]
    returns = [float(row["return"]) for row in complete if _finite_number(row.get("return"))]
    successes = [float(row["success"]) for row in complete if _finite_number(row.get("success"))]
    return {
        "n_complete_episodes": len(complete),
        "n_incomplete_episodes": len(rows) - len(complete),
        "n_missing_episodes": missing,
        "n_failures": sum(
            _finite_number(row.get("success")) and float(row["success"]) <= 0 for row in complete
        ),
        "n_missing_return": sum(not _finite_number(row.get("return")) for row in complete),
        "n_missing_success": sum(not _finite_number(row.get("success")) for row in complete),
        "mean_return": _mean(returns),
        "mean_success": _mean(successes),
    }


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return a bounded completeness audit and summaries of saved pilot records."""
    protocol_issues = []
    tasks_value = manifest.get("tasks")
    conditions_value = manifest.get("conditions")
    tasks = list(tasks_value) if isinstance(tasks_value, (list, tuple)) else []
    conditions = list(conditions_value) if isinstance(conditions_value, (list, tuple)) else []
    for name, value, entries in (
        ("tasks", tasks_value, tasks),
        ("conditions", conditions_value, conditions),
    ):
        if not entries:
            protocol_issues.append({"field": name, "issue": "must_be_nonempty"})
        if not isinstance(value, (list, tuple)):
            protocol_issues.append({"field": name, "issue": "must_be_a_list"})
        if any(not isinstance(item, str) or not item.strip() for item in entries):
            protocol_issues.append({"field": name, "issue": "entries_must_be_nonempty_strings"})
        if len(entries) != len(set(item for item in entries if isinstance(item, str))):
            protocol_issues.append({"field": name, "issue": "duplicate_entries"})
    replicates = manifest.get("replicates")
    if isinstance(replicates, int) and not isinstance(replicates, bool) and replicates > 0:
        replicate_ids = list(range(replicates))
    else:
        replicate_ids = []
        protocol_issues.append({"field": "replicates", "issue": "must_be_a_positive_integer"})
    requested_steps = manifest.get("steps_per_source_policy_requested")
    requested_episodes = manifest.get("evaluation_episodes_requested")
    for name, value in (
        ("steps_per_source_policy_requested", requested_steps),
        ("evaluation_episodes_requested", requested_episodes),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            protocol_issues.append({"field": name, "issue": "must_be_a_positive_integer"})
    source_runs = list(manifest.get("source_runs") or [])
    evaluation_rows = list(manifest.get("cross_task_evaluations") or [])

    expected_sources = {
        _key(condition, replicate, task)
        for condition in conditions
        for replicate in replicate_ids
        for task in tasks
    }
    source_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    source_slots: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in source_runs:
        group = _key(record.get("condition"), record.get("replicate"))
        slot = _key(*group, record.get("task"))
        source_groups[group].append(record)
        source_slots[slot].append(record)

    missing_sources = sorted(expected_sources - source_slots.keys(), key=str)
    extra_sources = sorted(source_slots.keys() - expected_sources, key=str)
    duplicate_sources = {str(key): len(rows) for key, rows in source_slots.items() if len(rows) > 1}
    incomplete_training = []
    zero_step_training = []
    step_mismatches = []
    for slot, records in source_slots.items():
        for record in records:
            steps = record.get("environment_steps_completed")
            if not steps:
                zero_step_training.append(str(slot))
            if steps is not None and requested_steps is not None and steps != requested_steps:
                step_mismatches.append(
                    {"slot": str(slot), "completed": steps, "requested": requested_steps}
                )
            if requested_steps is not None and steps != requested_steps:
                incomplete_training.append(str(slot))
    planned_sources = manifest.get("planned_source_runs", len(expected_sources))
    source_count_mismatch = {
        "recorded": len(source_runs),
        "planned_field": planned_sources,
        "derived_expected": len(expected_sources),
    }
    source_count_mismatch["mismatch"] = len(
        source_runs
    ) != planned_sources or planned_sources != len(expected_sources)

    # Keys identify evaluation cells, including kind and evaluation task. Episode
    # numbers identify repeated rows within a cell and expose accidental duplicates.
    expected_cells: dict[tuple[Any, ...], int | None] = {}
    source_policy = {
        _key(r.get("condition"), r.get("replicate"), r.get("task")): r for r in source_runs
    }
    for condition, replicate, task in expected_sources:
        source = source_policy.get(_key(condition, replicate, task), {})
        policy_id = source.get("policy_id", f"{condition}__{task}__lib{replicate}")
        for evaluated_task in tasks:
            expected_cells[
                _key(policy_id, condition, replicate, task, "trained_policy", evaluated_task)
            ] = requested_episodes
        for kind in ("oracle_learned_rank1", "oracle_random_rank1"):
            expected_cells[_key(policy_id, condition, replicate, task, kind, task)] = (
                requested_episodes
            )

    observed_cells: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        cell = _key(
            row.get("policy_id"),
            row.get("condition"),
            row.get("replicate"),
            row.get("training_task"),
            row.get("policy_kind"),
            row.get("evaluation_task"),
        )
        observed_cells[cell].append(row)
    missing_cells = sorted(expected_cells.keys() - observed_cells.keys(), key=str)
    extra_cells = sorted(observed_cells.keys() - expected_cells.keys(), key=str)
    cell_audit = []
    duplicate_rows = []
    incomplete_cells = []
    for cell in sorted(expected_cells.keys() | observed_cells.keys(), key=str):
        rows = observed_cells.get(cell, [])
        required = expected_cells.get(cell)
        episode_ids = [row.get("episode") for row in rows]
        valid_ids = [
            episode
            for episode in episode_ids
            if isinstance(episode, int) and not isinstance(episode, bool)
        ]
        duplicates = sorted({episode for episode in valid_ids if valid_ids.count(episode) > 1})
        if duplicates:
            duplicate_rows.append({"cell": str(cell), "episode_ids": duplicates})
        expected_ids = set(range(required)) if isinstance(required, int) and required > 0 else set()
        present_ids = set(valid_ids)
        missing_ids = sorted(expected_ids - present_ids)
        extra_ids = sorted(present_ids - expected_ids)
        invalid_id_count = len(episode_ids) - len(valid_ids)
        invalid_metrics = [
            row.get("episode")
            for row in rows
            if not _finite_number(row.get("return")) or not _finite_number(row.get("success"))
        ]
        incomplete_ids = [
            row.get("episode") for row in rows if row.get("complete_episode") is not True
        ]
        n_complete = sum(row.get("complete_episode") is True for row in rows)
        cell_invalid = cell in expected_cells and (
            len(rows) != required
            or missing_ids
            or extra_ids
            or invalid_id_count
            or duplicates
            or incomplete_ids
            or invalid_metrics
        )
        if cell_invalid:
            incomplete_cells.append(
                {
                    "cell": str(cell),
                    "complete": n_complete,
                    "requested": required,
                    "missing_episode_ids": missing_ids,
                    "extra_episode_ids": extra_ids,
                    "invalid_episode_id_count": invalid_id_count,
                    "incomplete_episode_ids": incomplete_ids,
                    "episodes_with_invalid_metrics": invalid_metrics,
                }
            )
        cell_audit.append(
            {
                "cell": list(cell),
                "expected_episodes": required,
                "recorded_episodes": len(rows),
                "complete_episodes": n_complete,
                "incomplete_episodes": len(rows) - n_complete,
                "missing_episode_ids": missing_ids,
                "extra_episode_ids": extra_ids,
            }
        )

    initialization_issues = []
    for group, records in source_groups.items():
        hashes = [record.get("initial_actor_sha256") for record in records]
        absent = sum(not value for value in hashes)
        if absent:
            initialization_issues.append(
                {"group": list(group), "issue": "missing_hashes", "count": absent}
            )
        present = [value for value in hashes if value]
        if group[0] == "shared_init" and len(set(present)) > 1:
            initialization_issues.append({"group": list(group), "issue": "shared_hash_mismatch"})
        if (
            group[0] == "independent_init"
            and len(present) == len(records)
            and len(set(present)) != len(present)
        ):
            initialization_issues.append(
                {"group": list(group), "issue": "independent_hash_collision"}
            )
        if group[0] not in conditions or group[1] not in replicate_ids:
            initialization_issues.append({"group": list(group), "issue": "extra_source_slot"})
    hashes_by_replicate: dict[str, set[Any]] = defaultdict(set)
    for record in source_runs:
        init_hash = record.get("initial_actor_sha256")
        condition = record.get("condition")
        replicate = record.get("replicate")
        if init_hash and replicate in replicate_ids and condition in conditions:
            hashes_by_replicate[init_hash].add(replicate)
    for _init_hash, used_replicates in hashes_by_replicate.items():
        if len(used_replicates) > 1:
            initialization_issues.append(
                {
                    "issue": "hash_reused_across_library_replicates",
                    "replicates": sorted(used_replicates),
                }
            )
    hashes_complete = all(record.get("initial_actor_sha256") for record in source_runs)
    initialization_valid = (
        hashes_complete
        and not initialization_issues
        and not missing_sources
        and not duplicate_sources
        and not extra_sources
        and not protocol_issues
    )

    expected_diagnostics = {
        _key(condition, replicate, task)
        for condition in conditions
        for replicate in replicate_ids
        for task in tasks
    }
    diagnostic_slots: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for diagnostic in manifest.get("oracle_projection_diagnostics") or []:
        diagnostic_slots[
            _key(
                diagnostic.get("condition"),
                diagnostic.get("replicate"),
                diagnostic.get("held_out_task"),
            )
        ].append(diagnostic)
    missing_diagnostics = sorted(expected_diagnostics - diagnostic_slots.keys(), key=str)
    extra_diagnostics = sorted(diagnostic_slots.keys() - expected_diagnostics, key=str)
    duplicate_diagnostics = {
        str(key): len(records) for key, records in diagnostic_slots.items() if len(records) > 1
    }
    invalid_diagnostics = []
    for key, records in diagnostic_slots.items():
        for record in records:
            invalid_fields = [
                field
                for field in ("learned_relative_residual", "random_relative_residual")
                if not _finite_number(record.get(field))
            ]
            if invalid_fields:
                invalid_diagnostics.append({"key": list(key), "fields": invalid_fields})
    diagnostic_audit = {
        "expected": len(expected_diagnostics),
        "recorded": sum(len(records) for records in diagnostic_slots.values()),
        "missing": [list(key) for key in missing_diagnostics],
        "extra": [list(key) for key in extra_diagnostics],
        "duplicates": duplicate_diagnostics,
        "nonfinite_or_missing_values": invalid_diagnostics,
        "complete": not (
            missing_diagnostics or extra_diagnostics or duplicate_diagnostics or invalid_diagnostics
        ),
    }

    coverage_complete = (
        not protocol_issues
        and not missing_sources
        and not extra_sources
        and not duplicate_sources
        and not source_count_mismatch["mismatch"]
        and not incomplete_training
        and not zero_step_training
        and not missing_cells
        and not extra_cells
        and not duplicate_rows
        and not incomplete_cells
        and diagnostic_audit["complete"]
    )

    policy_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    own_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in evaluation_rows:
        key = _key(
            row.get("condition"),
            row.get("policy_kind"),
            row.get("training_task"),
            row.get("evaluation_task"),
        )
        policy_groups[key].append(row)
        if row.get("training_task") == row.get("evaluation_task"):
            own_groups[
                _key(
                    row.get("condition"),
                    row.get("replicate"),
                    row.get("policy_kind"),
                    row.get("training_task"),
                )
            ].append(row)

    for condition, replicate, task in expected_sources:
        for eval_task in tasks:
            policy_groups.setdefault(_key(condition, "trained_policy", task, eval_task), [])
        for kind in ("oracle_learned_rank1", "oracle_random_rank1"):
            policy_groups.setdefault(_key(condition, kind, task, task), [])
        for kind in (
            "trained_policy",
            "oracle_learned_rank1",
            "oracle_random_rank1",
        ):
            own_groups.setdefault(_key(condition, replicate, kind, task), [])

    def missing_for(
        policy_kind: Any,
        training_task: Any,
        evaluation_task: Any,
        condition: Any = None,
        replicate: Any = None,
    ) -> int:
        missing = 0
        for cell, count in expected_cells.items():
            if (
                cell[4:] != (policy_kind, evaluation_task)
                or cell[3] != training_task
                or (condition is not None and cell[1] != condition)
                or (replicate is not None and cell[2] != replicate)
            ):
                continue
            expected_ids = set(range(count)) if isinstance(count, int) and count > 0 else set()
            present_ids = {
                row.get("episode")
                for row in observed_cells.get(cell, [])
                if isinstance(row.get("episode"), int)
                and not isinstance(row.get("episode"), bool)
                and row.get("episode") in expected_ids
            }
            missing += len(expected_ids - present_ids)
        return missing

    policy_by_task = [
        {
            "condition": key[0],
            "policy_kind": key[1],
            "training_task": key[2],
            "evaluation_task": key[3],
            "n_library_replicates": len(
                {row.get("replicate") for row in rows if row.get("complete_episode") is True}
            ),
            **_metric_summary(rows, missing_for(key[1], key[2], key[3], key[0])),
        }
        for key, rows in sorted(policy_groups.items(), key=lambda item: str(item[0]))
    ]
    own_task_competence = [
        {
            "condition": key[0],
            "replicate": key[1],
            "policy_kind": key[2],
            "task": key[3],
            "n_library_replicates": int(any(row.get("complete_episode") is True for row in rows)),
            **_metric_summary(rows, missing_for(key[2], key[3], key[3], key[0], key[1])),
        }
        for key, rows in sorted(own_groups.items(), key=lambda item: str(item[0]))
    ]

    geometry_groups: dict[tuple[Any, ...], dict[Any, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for diagnostic_key, diagnostic_records in diagnostic_slots.items():
        if diagnostic_key not in expected_diagnostics or len(diagnostic_records) != 1:
            continue
        diagnostic = diagnostic_records[0]
        if not all(
            _finite_number(diagnostic.get(field))
            for field in ("learned_relative_residual", "random_relative_residual")
        ):
            continue
        condition = diagnostic.get("condition")
        replicate = diagnostic.get("replicate")
        for method, field in (
            ("learned_rank1", "learned_relative_residual"),
            ("random_rank1", "random_relative_residual"),
        ):
            value = diagnostic.get(field)
            numeric_value = _finite_float(value)
            if numeric_value is not None:
                geometry_groups[_key(condition, method)][replicate].append(numeric_value)
    geometry_residuals = []
    for (condition, method), by_replicate in sorted(
        geometry_groups.items(), key=lambda item: str(item[0])
    ):
        replicate_means = [_mean(values) for values in by_replicate.values()]
        vals = [value for value in replicate_means if value is not None]
        geometry_residuals.append(
            {
                "condition": condition,
                "method": method,
                "mean_relative_residual": _mean(vals),
                "n_library_replicates": len(vals),
                "replicate_means": [
                    {
                        "replicate": replicate,
                        "mean_relative_residual": _mean(values),
                        "n_tasks": len(values),
                    }
                    for replicate, values in sorted(
                        by_replicate.items(), key=lambda item: str(item[0])
                    )
                ],
            }
        )

    return {
        "coverage": {
            "complete": coverage_complete,
            "protocol_issues": protocol_issues,
            "expected_source_slots": len(expected_sources),
            "recorded_source_runs": len(source_runs),
            "missing_source_slots": [list(x) for x in missing_sources],
            "extra_source_slots": [list(x) for x in extra_sources],
            "duplicate_source_slots": duplicate_sources,
            "source_count": source_count_mismatch,
            "zero_step_training_slots": zero_step_training,
            "incomplete_training_slots": incomplete_training,
            "training_step_mismatches": step_mismatches,
            "expected_evaluation_cells": len(expected_cells),
            "missing_evaluation_cells": [list(x) for x in missing_cells],
            "extra_evaluation_cells": [list(x) for x in extra_cells],
            "duplicate_evaluation_rows": duplicate_rows,
            "incomplete_evaluation_cells": incomplete_cells,
            "evaluation_cells": cell_audit,
            "geometry_diagnostics": diagnostic_audit,
            "evaluation_complete_flag_ignored": manifest.get("evaluation_complete"),
        },
        "initialization": {
            "valid": initialization_valid,
            "hashes_present": hashes_complete,
            "issues": initialization_issues,
            "validation_scope": (
                "Shared hashes equal within each (condition, replicate); "
                "independent hashes distinct within each group."
            ),
        },
        "policy_by_task": policy_by_task,
        "own_task_competence_by_condition_replicate": own_task_competence,
        "geometry_residuals": geometry_residuals,
        "interpretation": {
            "oracle_target_weight_access": True,
            "oracle_limitation": (
                "Oracle coefficients use target policy weights; this is reconstruction "
                "diagnostic evidence, not practical held-out adaptation."
            ),
            "replication_unit": (
                "Independent policy-library replicate; episodes and task folds are "
                "not independent replicates."
            ),
            "pooling": (
                "Returns and success are summarized separately by training and "
                "evaluation task; task scales are not pooled."
            ),
            "novelty_claim": None,
            "unavailable_data": [
                name
                for name, present in (
                    ("source training records", bool(source_runs)),
                    ("evaluation episodes", bool(evaluation_rows)),
                    ("geometry diagnostics", bool(manifest.get("oracle_projection_diagnostics"))),
                )
                if not present
            ],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        report = summarize_manifest(json.load(handle))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return int(not report["coverage"]["complete"] or not report["initialization"]["valid"])


if __name__ == "__main__":
    raise SystemExit(main())
