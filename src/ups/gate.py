"""Immutable Gate Zero evaluation and fail-closed reporting."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from ups.config import Phase0Config
from ups.verifier import verify_artifacts


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

# These are the Phase Zero negative controls in the governing research plan.
# A merely spectrum-matched comparison is insufficient: the plan requires the
# full set to rule out single-matrix rank, marginal statistics, shared bases, and
# unresolved scratch-network symmetries.
REQUIRED_NULL_ENSEMBLES = (
    "gaussian_norm_matched",
    "spectrum_matched_orientation",
    "independent_low_rank",
    "untrained_networks",
    "element_layer_shuffle",
    "task_label_permutation",
    "shared_vs_independent_base",
    "aligned_vs_unaligned_scratch",
)

REQUIRED_GEOMETRY_METRICS = (
    "cross_validated_effective_rank",
    "explained_variance_curve",
    "principal_angles",
    "projection_distance",
    "subspace_stability",
)

# Hash of revision 2 of the expanded config (6 environments x 5 independent
# training seeds 10--14, disjoint from the inspected pilot). Revision 1 is
# retained as an excluded incident because it did not preserve every checkpoint.
PREREGISTERED_PHASE0_CONFIG_HASH = (
    "e6a24efeb1d001d7a45e41de7c46b4bef51881f316e27e6b5e79618c0c2b428d"
)
ARTIFACT_VERIFIER_IMPLEMENTED = True


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _has_geometry(evidence: dict[str, Any], module: str) -> bool:
    geometry = evidence.get("geometry", {})
    if not isinstance(geometry, dict):
        return False
    module_metrics = geometry.get(module, {})
    if not isinstance(module_metrics, dict):
        return False
    return all(
        name in module_metrics and module_metrics[name] is not None
        for name in REQUIRED_GEOMETRY_METRICS
    )


def _null_ensemble_counts(completeness: dict[str, Any], minimum: int) -> bool:
    ensembles = completeness.get("null_ensembles", {})
    if not isinstance(ensembles, dict):
        return False
    return all(
        isinstance(ensembles.get(name), int) and ensembles[name] >= minimum
        for name in REQUIRED_NULL_ENSEMBLES
    )


def evaluate_gate(config: Phase0Config, evidence: dict[str, Any]) -> dict[str, Any]:
    """Evaluate preregistered checks. Missing evidence is always a failed check."""
    gate = config.gate
    verification = verify_artifacts(config, evidence)
    admissible = (
        ARTIFACT_VERIFIER_IMPLEMENTED
        and evidence.get("evidence_class") == "PREREGISTERED_FULL"
        and verification["verified"]
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
            [evidence.get("evidence_class"), verification],
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
            f"{len(config.environments)} tasks x {len(config.seeds)} seeds",
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
            and completeness.get("sequence_length") == 32
            and completeness.get("common_buffer_replayed") is True,
            [
                completeness.get("evaluation_sequences"),
                completeness.get("sequence_length"),
                completeness.get("common_buffer_replayed"),
            ],
            "512 sequences x 32 steps, replayed identically through original and reconstruction",
        ),
        Check(
            "held_out_construction",
            completeness.get("leave_one_task_out") is True
            and completeness.get("cross_validated_rank_selection") is True,
            [
                completeness.get("leave_one_task_out"),
                completeness.get("cross_validated_rank_selection"),
            ],
            "training-task-only basis; rank selected by cross-validated variance and function",
        ),
        Check(
            "hierarchical_statistics",
            completeness.get("hierarchical_bootstrap_tasks_seeds") is True,
            completeness.get("hierarchical_bootstrap_tasks_seeds"),
            "task is the generalization unit; hierarchical bootstrap over tasks and seeds",
        ),
        Check(
            "null_replicates",
            completeness.get("null_replicates", 0) >= 1000
            and _null_ensemble_counts(completeness, config.analysis.null_replicates),
            completeness.get("null_ensembles"),
            "all mandatory null/control ensembles with >= 1000 replicates each",
        ),
    ]
    for module in ("encoder", "actor"):
        low = metrics.get(f"{module}_learned_minus_spectrum_null_ci_low")
        checks.extend(
            (
                Check(
                    f"{module}_ci_above_zero",
                    _finite_number(low) and isinstance(low, (int, float)) and low > 0,
                    low,
                    "learned basis beats spectrum-matched orientation with 95% CI lower bound > 0",
                ),
                Check(
                    f"{module}_geometry_reported",
                    _has_geometry(evidence, module),
                    evidence.get("geometry", {}).get(module)
                    if isinstance(evidence.get("geometry"), dict)
                    else None,
                    "cross-validated rank, variance curve, angles, projection distance, stability",
                ),
            )
        )
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
        passed = (
            _finite_number(value)
            and isinstance(value, (int, float))
            and (value >= threshold if direction == ">=" else value <= threshold)
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
        # Authorization is a gate decision, not an instruction to launch Phase One.
        # Phase One remains outside this repository task even when Gate Zero passes.
        "phase_one_authorized": passed,
        "config_hash": config.digest,
        "thresholds_immutable": gate.immutable,
        "checks": [check.__dict__ for check in checks],
        "statement": (
            "Gate Zero passed; Phase One is authorized but is not launched by this task."
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
            "Phase One authorization: "
            f"**{'true' if report['phase_one_authorized'] else 'false'}**.",
            "",
        ]
    )
