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
from pandas.api.types import is_bool_dtype, is_numeric_dtype  # type: ignore[import-untyped]

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
    training_path = config.artifact_root / "population" / "training_report.json"
    if not training_path.is_file():  # pragma: no cover - guarded artifact failure
        return False, ["missing training report for evaluation linkage"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        table = pd.read_parquet(table_path)
        training = json.loads(training_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, ImportError) as error:  # pragma: no cover - corrupt artifacts
        return False, [f"invalid evaluation artifacts: {error}"]
    if training.get("config_hash") != config.digest:  # pragma: no cover - stale artifact
        failures.append("stale config hash in training report")
    training_map: dict[tuple[str, int, int], tuple[str, str]] = {}
    for job in training.get("jobs", []):
        for checkpoint in job.get("checkpoints", []):
            target = checkpoint.get("target_environment_steps")
            path = checkpoint.get("checkpoint")
            digest = checkpoint.get("checkpoint_sha256")
            if isinstance(target, int) and isinstance(path, str) and isinstance(digest, str):
                training_map[(job.get("environment"), job.get("seed"), target)] = (path, digest)
    if report.get("config_hash") != config.digest:  # pragma: no cover - stale artifact
        failures.append("stale config hash in evaluation report")
    if (
        report.get("episodes_per_checkpoint") != config.training.evaluation_episodes
    ):  # pragma: no cover
        failures.append("evaluation report has wrong fixed episode count")
    required = {
        "environment",
        "seed",
        "target_environment_steps",
        "checkpoint",
        "checkpoint_sha256",
        "success_rate",
        "median_return",
        "episodes",
        "evaluation_table",
        "max_episode_steps",
    }
    if not required.issubset(table.columns):
        failures.append("evaluation registry is missing required columns")
        return False, failures
    expected = {(environment, seed) for environment in config.environments for seed in config.seeds}
    fixed_seeds = dict(zip(config.environments, config.evaluation_buffer.fixed_seeds, strict=True))
    final = table[table["target_environment_steps"] == config.training.max_environment_steps]
    observed = set(zip(final["environment"], final["seed"], strict=False))
    if observed != expected or len(final) != len(expected):
        failures.append("evaluation registry final task/seed pairs are incomplete or duplicated")
    non_nullable = required - {"max_episode_steps"}
    if table[list(non_nullable)].isnull().any().any():
        failures.append("evaluation registry contains null required values")
    if not table["max_episode_steps"].isnull().all():
        failures.append("scientific evaluation registry contains bounded diagnostic episodes")
    for _, row in table.iterrows():
        raw_table = row["evaluation_table"]
        episodes = row["episodes"]
        declared_rate = row["success_rate"]
        declared_median = row["median_return"]
        linked = training_map.get(
            (row["environment"], row["seed"], row["target_environment_steps"])
        )
        if linked is None:  # pragma: no cover - malformed producer linkage
            failures.append("evaluation registry row has no training-report checkpoint")
        else:
            checkpoint_path, checkpoint_hash = linked
            if (  # pragma: no cover - malformed producer linkage
                str(row["checkpoint"]) != checkpoint_path
                or str(row["checkpoint_sha256"]) != checkpoint_hash
            ):
                failures.append("evaluation registry checkpoint linkage mismatch")
            elif (  # pragma: no cover - corrupt producer artifact
                not Path(checkpoint_path).is_file()
                or sha256_file(Path(checkpoint_path)) != checkpoint_hash
            ):
                failures.append(f"training checkpoint is missing or corrupt: {checkpoint_path}")
        raw_path = Path(str(raw_table))
        if not raw_path.is_file():  # pragma: no cover - missing producer artifact
            failures.append(f"missing raw evaluation table: {raw_path}")
            continue
        try:
            raw = pd.read_parquet(raw_path)
        except (OSError, ValueError, ImportError) as error:
            failures.append(f"invalid raw evaluation table {raw_path}: {error}")
            continue
        if not isinstance(episodes, int) or episodes != config.training.evaluation_episodes:
            failures.append("evaluation record has wrong episode count")
        if len(raw) != config.training.evaluation_episodes:
            failures.append(f"raw evaluation table has wrong episode count: {raw_path}")
        required_raw = {
            "environment",
            "checkpoint",
            "checkpoint_sha256",
            "episode",
            "seed",
            "success",
            "return",
            "terminated",
            "truncated",
        }
        if not required_raw.issubset(raw.columns):
            failures.append(f"raw evaluation table is missing required columns: {raw_path}")
            continue
        if raw.empty:
            failures.append(f"raw evaluation table is empty: {raw_path}")
            continue
        expected_episodes = list(range(config.training.evaluation_episodes))
        if raw["episode"].tolist() != expected_episodes:
            failures.append(f"raw evaluation episode IDs are not exactly ordered: {raw_path}")
        environment = str(row["environment"])
        if environment not in fixed_seeds:
            failures.append(f"evaluation registry has unknown environment: {environment}")
            continue
        expected_seeds = [fixed_seeds[environment] + episode for episode in expected_episodes]
        if raw["seed"].tolist() != expected_seeds:
            failures.append(f"raw evaluation seeds do not match fixed seeds: {raw_path}")
        if (  # pragma: no cover - malformed raw linkage
            raw["environment"].nunique() != 1 or str(raw.iloc[0]["environment"]) != environment
        ):
            failures.append(f"raw evaluation environment linkage mismatch: {raw_path}")
        for field in ("checkpoint", "checkpoint_sha256"):
            if (  # pragma: no cover - malformed raw linkage
                raw[field].nunique() != 1 or str(raw.iloc[0][field]) != str(row[field])
            ):
                failures.append(f"raw evaluation {field} linkage mismatch: {raw_path}")
        successes = raw["success"]
        if not is_bool_dtype(successes):  # pragma: no cover - malformed raw schema
            failures.append(f"raw evaluation table has non-boolean success values: {raw_path}")
        returns = raw["return"]
        if (
            not is_numeric_dtype(returns)
            or not returns.map(lambda value: isfinite(float(value))).all()
        ):
            failures.append(  # pragma: no cover - malformed raw schema
                f"raw evaluation table has non-finite returns: {raw_path}"
            )
        if not is_bool_dtype(raw["terminated"]) or not is_bool_dtype(raw["truncated"]):
            failures.append(f"raw evaluation table has invalid termination fields: {raw_path}")
        if is_bool_dtype(successes) and is_numeric_dtype(returns):
            recomputed_rate = float(successes.mean())
            if abs(recomputed_rate - float(declared_rate)) > 1e-12:
                failures.append(f"success-rate summary does not match raw table: {raw_path}")
            if abs(float(returns.median()) - float(declared_median)) > 1e-12:
                failures.append(f"median-return summary does not match raw table: {raw_path}")
            if (
                row["target_environment_steps"] == config.training.max_environment_steps
                and recomputed_rate < config.training.minimum_success
            ):
                failures.append(f"final policy is below success threshold: {raw_path}")
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
