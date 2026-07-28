"""Independent, fail-closed verification of Gate Zero inputs.

The metric producer writes evidence; this module does not trust its claims. It
recomputes the checks that can be established from manifests and artifacts and
returns a structured failure report when anything is absent or inconsistent.
"""

from __future__ import annotations

import json
from math import isfinite
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from ups.artifacts import sha256_file
from ups.config import Phase0Config

REQUIRED_STAGE_PREFIXES = (
    "train",
    "evaluate",
    "collect-states",
    "extract-updates",
    "align",
    "analyze",
    "nulls",
    "reconstruct",
)


def _finite_tree(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return True
    if isinstance(value, (int, float)):
        return isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(key) and _finite_tree(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(item) for item in value)
    return True


def _manifest_checks(config: Phase0Config) -> tuple[bool, list[str]]:
    root = config.artifact_root / "manifests"
    failures: list[str] = []
    if not root.is_dir():
        return False, [f"missing manifest directory: {root}"]
    manifests = sorted(root.glob("*.json"))
    for prefix in REQUIRED_STAGE_PREFIXES:
        if not any(path.name.startswith(f"{prefix}-") for path in manifests):
            failures.append(f"missing manifest for stage {prefix}")
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"invalid manifest {manifest_path}: {error}")
            continue
        if manifest.get("config_hash") != config.digest:
            failures.append(f"stale config hash in {manifest_path}")
        outputs = manifest.get("outputs", [])
        if not isinstance(outputs, list):
            failures.append(f"invalid outputs list in {manifest_path}")
            continue
        for output in outputs:
            if not isinstance(output, dict):
                failures.append(f"invalid output record in {manifest_path}")
                continue
            snapshot = Path(output.get("path", ""))
            expected = output.get("sha256")
            if not snapshot.is_file():
                failures.append(f"missing manifest snapshot {snapshot}")
            elif not isinstance(expected, str) or sha256_file(snapshot) != expected:
                failures.append(f"corrupt manifest snapshot {snapshot}")
    return not failures, failures


def _population_checks(config: Phase0Config) -> tuple[bool, list[str]]:
    training_path = config.artifact_root / "population" / "training_report.json"
    training_failures: list[str] = []
    if training_path.is_file():
        try:
            training = json.loads(training_path.read_text(encoding="utf-8"))
            if training.get("config_hash") != config.digest:
                training_failures.append("stale config hash in training report")
            jobs = training.get("jobs", [])
            expected_pairs = {
                (environment, seed) for environment in config.environments for seed in config.seeds
            }
            observed_pairs = {(job.get("environment"), job.get("seed")) for job in jobs}
            if observed_pairs != expected_pairs or len(jobs) != len(expected_pairs):
                training_failures.append(
                    "training report has incomplete or duplicate task/seed jobs"
                )
            for job in jobs:
                for checkpoint in job.get("checkpoints", []):
                    path = Path(checkpoint.get("checkpoint", ""))
                    expected_hash = checkpoint.get("checkpoint_sha256")
                    if not path.is_file() or not isinstance(expected_hash, str):
                        training_failures.append(f"missing checkpoint record: {path}")
                    elif sha256_file(path) != expected_hash:
                        training_failures.append(f"checkpoint hash mismatch: {path}")
        except (OSError, json.JSONDecodeError, AttributeError):
            training_failures.append("invalid training report")
    report_path = config.artifact_root / "weights" / "extraction.json"
    if not report_path.is_file():
        return False, [*training_failures, f"missing extracted population report: {report_path}"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return False, [*training_failures, f"invalid extracted population report: {error}"]
    expected = len(config.environments) * len(config.seeds)
    failures: list[str] = []
    if report.get("expected_population") != expected:
        failures.append("extraction report has wrong expected population")
    if report.get("extracted_population") != expected:
        failures.append("required task/seed population is incomplete")
    if report.get("qualified_population") is not True:
        failures.append("population has not been independently qualified")
    return not (failures or training_failures), training_failures + failures


def _evaluation_checks(config: Phase0Config) -> tuple[bool, list[str]]:
    """Recompute population coverage and finite success values from raw registry data."""
    report_path = config.artifact_root / "evaluations" / "evaluation_report.json"
    table_path = config.artifact_root / "evaluations" / "policy_registry.parquet"
    failures: list[str] = []
    if not report_path.is_file() or not table_path.is_file():
        return False, ["missing fixed-episode evaluation report or registry"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        table = pd.read_parquet(table_path)
    except (OSError, ValueError, ImportError) as error:
        return False, [f"invalid evaluation artifacts: {error}"]
    if report.get("config_hash") != config.digest:
        failures.append("stale config hash in evaluation report")
    required = {"environment", "seed", "target_environment_steps", "success_rate"}
    if not required.issubset(table.columns):
        failures.append("evaluation registry is missing required columns")
        return False, failures
    expected = {(environment, seed) for environment in config.environments for seed in config.seeds}
    final = table[table["target_environment_steps"] == config.training.max_environment_steps]
    observed = set(zip(final["environment"], final["seed"], strict=False))
    if observed != expected or len(final) != len(expected):
        failures.append("evaluation registry final task/seed pairs are incomplete or duplicated")
    if table[list(required)].isnull().any().any():
        failures.append("evaluation registry contains null required values")
    if (
        not table["success_rate"]
        .map(lambda value: isinstance(value, (int, float)) and isfinite(value))
        .all()
    ):
        failures.append("evaluation registry contains non-finite success rates")
    return not failures, failures


def verify_artifacts(config: Phase0Config, evidence: dict[str, Any]) -> dict[str, Any]:
    """Verify Gate Zero provenance and artifact integrity without trusting evidence flags."""
    checks: dict[str, bool] = {}
    failures: list[str] = []

    checks["evidence_config_hash"] = evidence.get("config_hash") == config.digest
    if not checks["evidence_config_hash"]:
        failures.append("evidence config hash is stale or missing")
    checks["finite_evidence"] = _finite_tree(evidence)
    if not checks["finite_evidence"]:
        failures.append("evidence contains non-finite values")
    checks["manifest_integrity"], manifest_failures = _manifest_checks(config)
    failures.extend(manifest_failures)
    checks["population_integrity"], population_failures = _population_checks(config)
    failures.extend(population_failures)
    checks["evaluation_integrity"], evaluation_failures = _evaluation_checks(config)
    failures.extend(evaluation_failures)

    completeness = evidence.get("completeness", {})
    if not isinstance(completeness, dict):
        completeness = {}
    checks["seed_isolation"] = completeness.get("seed_leakage_detected") is False
    if not checks["seed_isolation"]:
        failures.append("seed isolation was not independently established")
    checks["null_invariants"] = completeness.get("null_invariants_verified") is True
    if not checks["null_invariants"]:
        failures.append("null invariants were not independently established")
    checks["evidence_declares_full"] = evidence.get("evidence_class") == "PREREGISTERED_FULL"
    if not checks["evidence_declares_full"]:
        failures.append("evidence is not classified as preregistered full")

    return {"verified": not failures, "checks": checks, "failures": failures}
