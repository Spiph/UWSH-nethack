"""Immutable Gate Zero evaluation and fail-closed reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ups.config import Phase0Config


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    observed: Any
    requirement: str


REQUIRED_NUMERICAL_CHECKS = (
    "lora_composition",
    "lora_rotation_invariance",
    "merged_adapter_equivalence",
    "alignment_invariance",
    "null_preservation",
    "adapter_svd_hosvd_reproduction",
)

# Hash of configs/phase0.yaml as preregistered before any evidence was generated.
PREREGISTERED_PHASE0_CONFIG_HASH = (
    "c2b9d27a22274e5b59c41ad35a4ab9269aefa130f7c603fd11c9e4015b276bb2"
)
ARTIFACT_VERIFIER_IMPLEMENTED = False


def evaluate_gate(config: Phase0Config, evidence: dict[str, Any]) -> dict[str, Any]:
    """Evaluate preregistered checks. Missing evidence is always a failed check."""
    gate = config.gate
    admissible = (
        ARTIFACT_VERIFIER_IMPLEMENTED
        and evidence.get("evidence_class") == "PREREGISTERED_FULL"
        and evidence.get("artifact_verification") == "VERIFIED"
    )
    metrics = evidence.get("metrics", {}) if admissible else {}
    numerical = evidence.get("numerical_checks", {})
    completeness = evidence.get("completeness", {})
    runtime_requirement = (
        "SOL patched NLE + Sample Factory runtime verified"
        if config.training.runtime == "sol_patched"
        else "Sample Factory 2.1.1 and NLE 1.3.0 resolve in one Python 3.10 environment"
    )
    checks = [
        Check(
            "evidence_class",
            admissible,
            [evidence.get("evidence_class"), evidence.get("artifact_verification")],
            "PREREGISTERED_FULL with independently VERIFIED artifacts",
        ),
        Check(
            "preregistration_hash",
            config.digest == PREREGISTERED_PHASE0_CONFIG_HASH,
            config.digest,
            PREREGISTERED_PHASE0_CONFIG_HASH,
        ),
        Check(
            "locked_runtime_compatible",
            bool(completeness.get("locked_runtime_compatible")),
            completeness.get("locked_runtime_compatible"),
            runtime_requirement,
        ),
        Check(
            "full_population",
            bool(completeness.get("full_population")),
            completeness,
            "4 tasks x 3 seeds",
        ),
        Check(
            "policy_quality",
            bool(completeness.get("all_policies_success_ge_075")),
            completeness.get("minimum_policy_success"),
            "each retained policy success >= 0.75 over 200 fixed episodes",
        ),
        Check(
            "evaluation_buffer",
            completeness.get("evaluation_sequences") == 512
            and completeness.get("sequence_length") == 32,
            [completeness.get("evaluation_sequences"), completeness.get("sequence_length")],
            "512 sequences x 32 steps",
        ),
        Check(
            "null_replicates",
            completeness.get("null_replicates", 0) >= 1000,
            completeness.get("null_replicates"),
            ">= 1000",
        ),
    ]
    for module in ("encoder", "actor"):
        low = metrics.get(f"{module}_learned_minus_null_ci_low")
        checks.append(Check(f"{module}_ci_above_zero", low is not None and low > 0, low, "> 0"))
    comparisons = (
        ("null_normalized_median", gate.null_effect_min_sd, ">="),
        ("retention_median", gate.retention_median_min, ">="),
        ("retention_ci_low", gate.retention_ci_low_min, ">="),
        ("action_kl", gate.action_kl_max, "<="),
        ("action_agreement", gate.action_agreement_min, ">="),
        ("feature_cka", gate.feature_cka_min, ">="),
        ("normalized_value_rmse", gate.normalized_value_rmse_max, "<="),
    )
    for name, threshold, direction in comparisons:
        value = metrics.get(name)
        passed = value is not None and (
            value >= threshold if direction == ">=" else value <= threshold
        )
        checks.append(Check(name, passed, value, f"{direction} {threshold}"))
    for name in REQUIRED_NUMERICAL_CHECKS:
        checks.append(Check(name, numerical.get(name) is True, numerical.get(name), "true"))
    # No current producer can emit VERIFIED. This prevents self-declared evidence
    # from authorizing a scientific PASS until an artifact-level verifier exists.
    passed = all(check.passed for check in checks) and admissible
    return {
        "schema_version": 1,
        "study": "phase0",
        "decision": "PASS" if passed else "NO_GO",
        "phase_one_authorized": False,
        "config_hash": config.digest,
        "thresholds_immutable": gate.immutable,
        "checks": [check.__dict__ for check in checks],
        "statement": (
            "Gate Zero passed; Phase One remains outside this task."
            if passed
            else "Gate Zero did not pass. Phase One is explicitly prohibited."
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    rows = ["| Check | Pass | Observed | Requirement |", "|---|---:|---:|---|"]
    for check in report["checks"]:
        rows.append(
            f"| {check['name']} | {'yes' if check['passed'] else 'no'} | "
            f"`{check['observed']}` | {check['requirement']} |"
        )
    return "\n".join(
        [
            "# Phase Zero Gate Report",
            "",
            f"## {report['decision']}",
            "",
            report["statement"],
            "",
            f"Configuration hash: `{report['config_hash']}`",
            "",
            *rows,
            "",
            "Phase One authorization: **false**.",
            "",
        ]
    )
