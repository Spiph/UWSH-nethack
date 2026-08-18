import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load_file

import ups.gate as gate_module
from ups.artifacts import sha256_file
from ups.config import Phase0Config, load_config
from ups.gate import (
    PREREGISTERED_PHASE0_CONFIG_HASH,
    REQUIRED_NUMERICAL_CHECKS,
    _has_geometry,
    _null_ensemble_counts,
    evaluate_gate,
)
from ups.sol_evaluate import checkpoint_config_path
from ups.verifier import _evaluation_checks, _manifest_checks, _population_checks, verify_artifacts
from ups.workflow import (
    _phase0_state_dict,
    _population_stop_reason,
    _retained_checkpoint_records,
    _upsert_population_job,
    checkpoint_environment_steps,
    evaluate,
    extract_updates,
    launch_population,
    population_jobs,
    reproduce,
    train,
    validate_existing_stage,
)

ROOT = Path(__file__).parents[1]


def test_sample_factory_checkpoint_config_path() -> None:
    checkpoint = Path("sample_factory/job/checkpoint_p0/checkpoint_0001_32.pth")
    assert checkpoint_config_path(checkpoint) == Path("sample_factory/job/config.json")


def test_sample_factory_checkpoint_step_parser() -> None:
    assert checkpoint_environment_steps(Path("checkpoint_000000012_384.pth")) == 384
    assert checkpoint_environment_steps(Path("checkpoint_100000.pth")) == 100000
    with pytest.raises(RuntimeError, match="unrecognized"):
        checkpoint_environment_steps(Path("policy.pth"))


def test_config_hash_is_stable_and_schema_is_strict(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    assert config.digest == load_config(ROOT / "configs/smoke.yaml").digest
    raw = (ROOT / "configs/smoke.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(raw + "\nunknown: true\n")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_full_study_declares_five_seeds_and_diverse_environment_cohort() -> None:
    config = load_config(ROOT / "configs/phase0.yaml")
    assert config.digest == PREREGISTERED_PHASE0_CONFIG_HASH
    assert config.seeds == [10, 11, 12, 13, 14]
    assert set(config.seeds).isdisjoint({0, 1})
    assert len(config.environments) == 6
    assert config.experimental_design is not None
    assert config.experimental_design.training_seed_count == len(config.seeds)
    assert set(config.experimental_design.environment_families) == {"room", "maze"}
    assert config.analysis.practical_effect_size == 0.05
    assert config.training.checkpoint_retention == 32
    assert config.training.checkpoint_save_interval_seconds == 3600
    assert config.training.heartbeat_reporting_interval_seconds == 3600

    nethack_baseline = load_config(ROOT / "configs/nethack_baseline.yaml")
    assert nethack_baseline.seeds == [20, 21, 22, 23, 24]
    assert nethack_baseline.environments == ["NetHackScore-v0"]
    assert nethack_baseline.analysis.practical_effect_size == 1.0

    raw = config.model_dump(mode="python")
    raw["seeds"] = [10, 11, 12, 13]
    with pytest.raises(ValidationError, match="five preset"):
        Phase0Config.model_validate(raw)

    raw = config.model_dump(mode="python")
    raw["evaluation_buffer"]["fixed_seeds"] = [0, 1102, 1103, 1104, 1105, 1106]
    with pytest.raises(ValidationError, match="disjoint"):
        Phase0Config.model_validate(raw)

    invalid_cases = (
        ("seeds", [10, 11, 12, 13, 13], "unique"),
        (
            "environments",
            [*config.environments[:-1], config.environments[0]],
            "unique",
        ),
        ("seeds", [10, 11, 12, 13, -1], "non-negative"),
    )
    for field, value, message in invalid_cases:
        raw = config.model_dump(mode="python")
        raw[field] = value
        with pytest.raises(ValidationError, match=message):
            Phase0Config.model_validate(raw)

    raw = config.model_dump(mode="python")
    raw["evaluation_buffer"]["fixed_seeds"][0] = -1
    with pytest.raises(ValidationError, match="non-negative"):
        Phase0Config.model_validate(raw)
    raw = config.model_dump(mode="python")
    raw["experimental_design"] = None
    with pytest.raises(ValidationError, match="experimental_design"):
        Phase0Config.model_validate(raw)
    raw = config.model_dump(mode="python")
    raw["experimental_design"]["training_seed_count"] = 4
    with pytest.raises(ValidationError, match="seed count"):
        Phase0Config.model_validate(raw)

    smoke = load_config(ROOT / "configs/smoke.yaml").model_dump(mode="python")
    smoke["experimental_design"] = {
        "training_seed_count": 2,
        "environment_families": ["room"],
        "evaluation_seed_policy": "fixed_per_environment",
    }
    with pytest.raises(ValidationError, match="seed count"):
        Phase0Config.model_validate(smoke)


def test_gate_rejects_self_declared_complete_evidence() -> None:
    config = load_config(ROOT / "configs/phase0.yaml")
    evidence: dict[str, Any] = {
        "evidence_class": "PREREGISTERED_FULL",
        "artifact_verification": "VERIFIED",
        "completeness": {
            "locked_runtime_compatible": True,
            "full_population": True,
            "all_policies_success_ge_075": True,
            "minimum_policy_success": 0.8,
            "evaluation_sequences": 512,
            "sequence_length": 32,
            "null_replicates": 1000,
        },
        "metrics": {
            "encoder_learned_minus_null_ci_low": 0.1,
            "actor_learned_minus_null_ci_low": 0.1,
            "null_normalized_median": 2.1,
            "retention_median": 0.91,
            "retention_ci_low": 0.81,
            "action_kl": 0.04,
            "action_agreement": 0.96,
            "feature_cka": 0.91,
            "normalized_value_rmse": 0.09,
        },
        "numerical_checks": dict.fromkeys(REQUIRED_NUMERICAL_CHECKS, True),
    }
    # Even internally consistent assertions cannot pass without the exact locked
    # preregistration plus a future artifact verifier. The pure gate evaluator
    # must fail closed rather than treating caller booleans as scientific proof.
    assert evaluate_gate(config, evidence)["decision"] == "NO_GO"


def test_gate_requires_all_phase_zero_plan_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(ROOT / "configs/phase0.yaml")
    evidence: dict[str, Any] = {
        "evidence_class": "PREREGISTERED_FULL",
        "artifact_verification": "VERIFIED",
        "completeness": {
            "locked_runtime_compatible": True,
            "full_population": True,
            "all_policies_success_ge_075": True,
            "minimum_policy_success": 0.8,
            "evaluation_sequences": 512,
            "sequence_length": 32,
            "common_buffer_replayed": True,
            "leave_one_task_out": True,
            "cross_validated_rank_selection": True,
            "hierarchical_bootstrap_tasks_seeds": True,
            "null_replicates": 1000,
            "null_ensembles": dict.fromkeys(gate_module.REQUIRED_NULL_ENSEMBLES, 1000),
        },
        "metrics": {
            "encoder_learned_minus_spectrum_null_ci_low": 0.1,
            "actor_learned_minus_spectrum_null_ci_low": 0.1,
            "null_normalized_median": 2.1,
            "retention_median": 0.91,
            "retention_ci_low": 0.81,
            "action_kl": 0.04,
            "action_agreement": 0.96,
            "feature_cka": 0.91,
            "normalized_value_rmse": 0.09,
        },
        "geometry": {
            module: dict.fromkeys(gate_module.REQUIRED_GEOMETRY_METRICS, 1.0)
            for module in ("encoder", "actor")
        },
        "numerical_checks": dict.fromkeys(REQUIRED_NUMERICAL_CHECKS, True),
    }
    report = evaluate_gate(config, evidence)
    assert report["decision"] == "NO_GO"
    assert report["phase_one_authorized"] is False
    assert "manifest" in str(report["checks"][0]["observed"])
    evidence["completeness"]["null_ensembles"].pop("independent_low_rank")
    assert evaluate_gate(config, evidence)["decision"] == "NO_GO"


def test_gate_contract_rejects_missing_geometry_and_control_manifests() -> None:
    assert _has_geometry({"geometry": None}, "encoder") is False
    assert _has_geometry({"geometry": {"encoder": None}}, "encoder") is False
    assert _null_ensemble_counts({"null_ensembles": None}, 1000) is False


def test_verifier_rejects_stale_nonfinite_and_malformed_evidence(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    result = verify_artifacts(
        config,
        {
            "config_hash": "stale",
            "evidence_class": "SMOKE_ONLY",
            "metrics": {"bad": float("nan")},
            "completeness": [],
        },
    )
    assert result["verified"] is False
    assert "evidence config hash is stale or missing" in result["failures"]
    assert "evidence contains non-finite values" in result["failures"]


def test_verifier_accepts_intact_provenance_shell(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    manifests = config.artifact_root / "manifests"
    manifests.mkdir(parents=True)
    for prefix in (
        "train",
        "evaluate",
        "collect-states",
        "extract-updates",
        "align",
        "analyze",
        "nulls",
        "reconstruct",
    ):
        (manifests / f"{prefix}-fixture.json").write_text(
            json.dumps({"config_hash": config.digest, "outputs": []})
        )
    weights = config.artifact_root / "weights"
    weights.mkdir(parents=True)
    (weights / "extraction.json").write_text(
        json.dumps(
            {
                "expected_population": len(config.environments) * len(config.seeds),
                "extracted_population": len(config.environments) * len(config.seeds),
                "qualified_population": True,
            }
        )
    )
    result = verify_artifacts(
        config,
        {
            "config_hash": config.digest,
            "evidence_class": "PREREGISTERED_FULL",
            "completeness": {
                "seed_leakage_detected": False,
                "null_invariants_verified": True,
            },
        },
    )
    evaluations = config.artifact_root / "evaluations"
    evaluations.mkdir(parents=True)
    raw_table = evaluations / "raw.parquet"
    raw_table_earlier = evaluations / "raw-earlier.parquet"
    checkpoint_path = str(config.artifact_root / "checkpoint_128.pth")
    checkpoint_earlier = config.artifact_root / "checkpoint_64.pth"
    Path(checkpoint_path).write_bytes(b"checkpoint")
    checkpoint_earlier.write_bytes(b"checkpoint-earlier")
    checkpoint_hash = sha256_file(Path(checkpoint_path))
    population = config.artifact_root / "population"
    population.mkdir(parents=True)
    (population / "training_report.json").write_text(
        json.dumps(
            {
                "config_hash": config.digest,
                "jobs": [
                    {
                        "environment": config.environments[0],
                        "seed": config.seeds[0],
                        "checkpoints": [
                            {
                                "target_environment_steps": 64,
                                "actual_environment_steps": 64,
                                "checkpoint": str(checkpoint_earlier),
                                "checkpoint_sha256": sha256_file(checkpoint_earlier),
                            },
                            {
                                "target_environment_steps": config.training.max_environment_steps,
                                "actual_environment_steps": config.training.max_environment_steps,
                                "checkpoint": checkpoint_path,
                                "checkpoint_sha256": checkpoint_hash,
                            },
                        ],
                    }
                ],
            }
        )
    )
    raw_frame = pd.DataFrame(
        {
            "environment": [config.environments[0], config.environments[0]],
            "checkpoint": [checkpoint_path, checkpoint_path],
            "checkpoint_sha256": [checkpoint_hash, checkpoint_hash],
            "episode": [0, 1],
            "seed": [1101, 1102],
            "success": [True, True],
            "return": [1.0, 1.0],
            "steps": [1, 1],
            "terminated": [True, True],
            "truncated": [False, False],
            "trajectory": [
                json.dumps(
                    [
                        {
                            "observation": {"glyphs_crop": [[0]], "blstats": [0, 0]},
                            "action": 0,
                            "reward": 1.0,
                            "next_observation": {
                                "glyphs_crop": [[0]],
                                "blstats": [0, 0],
                            },
                            "info": {"end_status": "TASK_SUCCESSFUL"},
                            "terminated": True,
                            "truncated": False,
                        }
                    ]
                )
            ]
            * 2,
        }
    )
    raw_frame.to_parquet(raw_table, index=False)
    earlier_frame = raw_frame.assign(
        checkpoint=str(checkpoint_earlier),
        checkpoint_sha256=sha256_file(checkpoint_earlier),
    )
    earlier_frame.to_parquet(raw_table_earlier, index=False)
    pd.DataFrame(
        [
            {
                "environment": config.environments[0],
                "seed": config.seeds[0],
                "target_environment_steps": config.training.evaluation_interval,
                "checkpoint": str(checkpoint_earlier),
                "checkpoint_sha256": sha256_file(checkpoint_earlier),
                "success_rate": 1.0,
                "mean_return": 1.0,
                "median_return": 1.0,
                "mean_length": 1.0,
                "action_selection": "greedy_argmax",
                "episodes": config.training.evaluation_episodes,
                "evaluation_table": str(raw_table_earlier),
                "max_episode_steps": None,
            },
            {
                "environment": config.environments[0],
                "seed": config.seeds[0],
                "target_environment_steps": config.training.max_environment_steps,
                "checkpoint": checkpoint_path,
                "checkpoint_sha256": checkpoint_hash,
                "success_rate": 1.0,
                "mean_return": 1.0,
                "median_return": 1.0,
                "mean_length": 1.0,
                "action_selection": "greedy_argmax",
                "episodes": config.training.evaluation_episodes,
                "evaluation_table": str(raw_table),
                "max_episode_steps": None,
            },
        ]
    ).to_parquet(evaluations / "policy_registry.parquet", index=False)
    (evaluations / "evaluation_report.json").write_text(
        json.dumps(
            {
                "config_hash": config.digest,
                "episodes_per_checkpoint": config.training.evaluation_episodes,
            }
        )
    )
    result = verify_artifacts(
        config,
        {
            "config_hash": config.digest,
            "evidence_class": "PREREGISTERED_FULL",
            "completeness": {
                "seed_leakage_detected": False,
                "null_invariants_verified": True,
            },
        },
    )
    assert result["verified"] is True
    registry = pd.read_parquet(evaluations / "policy_registry.parquet")
    final_index = registry.index[
        registry["target_environment_steps"] == config.training.max_environment_steps
    ][0]
    raw_valid = pd.read_parquet(raw_table)
    for column, values in (
        ("episode", [1, 0]),
        ("seed", [1102, 1103]),
        ("checkpoint", ["other", "other"]),
        ("success", [1, 0]),
        ("return", [float("nan"), 1.0]),
        ("terminated", [1, 1]),
    ):
        mutated = raw_valid.copy()
        mutated[column] = values
        mutated.to_parquet(raw_table, index=False)
        assert _evaluation_checks(config)[0] is False
    raw_valid.to_parquet(raw_table, index=False)
    low_quality = raw_valid.copy()
    low_quality["success"] = [True, False]
    low_quality.to_parquet(raw_table, index=False)
    registry.loc[final_index, "success_rate"] = 0.5
    registry.loc[final_index, "median_return"] = 1.0
    registry.to_parquet(evaluations / "policy_registry.parquet", index=False)
    assert _evaluation_checks(config)[0] is False
    raw_valid.to_parquet(raw_table, index=False)
    registry.loc[final_index, "success_rate"] = 1.0
    registry.to_parquet(evaluations / "policy_registry.parquet", index=False)
    pd.DataFrame(columns=raw_valid.columns).to_parquet(raw_table, index=False)
    assert _evaluation_checks(config)[0] is False
    raw_valid.to_parquet(raw_table, index=False)
    registry.loc[final_index, "environment"] = "unknown"
    registry.to_parquet(evaluations / "policy_registry.parquet", index=False)
    assert _evaluation_checks(config)[0] is False
    registry.loc[final_index, "environment"] = config.environments[0]
    registry.to_parquet(evaluations / "policy_registry.parquet", index=False)
    Path(checkpoint_path).write_bytes(b"tampered")
    assert _evaluation_checks(config)[0] is False
    Path(checkpoint_path).write_bytes(b"checkpoint")
    registry = pd.read_parquet(evaluations / "policy_registry.parquet")
    registry.loc[final_index, "checkpoint_sha256"] = sha256_file(Path(checkpoint_path))
    registry.to_parquet(evaluations / "policy_registry.parquet", index=False)
    registry = pd.read_parquet(evaluations / "policy_registry.parquet")
    registry.loc[final_index, "success_rate"] = 0.0
    registry.to_parquet(evaluations / "policy_registry.parquet", index=False)
    assert (
        verify_artifacts(
            config,
            {
                "config_hash": config.digest,
                "evidence_class": "PREREGISTERED_FULL",
                "completeness": {
                    "seed_leakage_detected": False,
                    "null_invariants_verified": True,
                },
            },
        )["verified"]
        is False
    )
    registry.loc[0, "success_rate"] = 1.0
    registry.to_parquet(evaluations / "policy_registry.parquet", index=False)
    pd.DataFrame({"success": [True]}).to_parquet(raw_table, index=False)
    assert _evaluation_checks(config)[0] is False
    pd.DataFrame({"other": [True, True]}).to_parquet(raw_table, index=False)
    assert _evaluation_checks(config)[0] is False
    raw_valid.to_parquet(raw_table, index=False)
    (population / "training_report.json").unlink()
    assert _evaluation_checks(config)[0] is False
    raw_table.unlink()
    assert (
        verify_artifacts(
            config,
            {
                "config_hash": config.digest,
                "evidence_class": "PREREGISTERED_FULL",
                "completeness": {
                    "seed_leakage_detected": False,
                    "null_invariants_verified": True,
                },
            },
        )["verified"]
        is False
    )


def test_verifier_reports_manifest_and_population_corruption(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    manifests = config.artifact_root / "manifests"
    manifests.mkdir(parents=True)
    for prefix in (
        "train",
        "evaluate",
        "collect-states",
        "extract-updates",
        "align",
        "analyze",
        "nulls",
        "reconstruct",
    ):
        (manifests / f"{prefix}-fixture.json").write_text(
            json.dumps({"config_hash": config.digest, "outputs": []})
        )
    (manifests / "invalid.json").write_text("{")
    (manifests / "stale.json").write_text(
        json.dumps({"config_hash": "stale", "outputs": "not-a-list"})
    )
    (manifests / "records.json").write_text(
        json.dumps(
            {
                "config_hash": config.digest,
                "outputs": [None, {"path": str(tmp_path / "missing"), "sha256": "x"}],
            }
        )
    )
    (manifests / "wrong-hash.json").write_text(
        json.dumps(
            {
                "config_hash": config.digest,
                "outputs": [{"path": str(manifests / "train-fixture.json"), "sha256": "wrong"}],
            }
        )
    )
    assert _manifest_checks(config)[0] is False
    weights = config.artifact_root / "weights"
    weights.mkdir(parents=True)
    report = weights / "extraction.json"
    report.write_text("{")
    assert _population_checks(config)[0] is False
    report.write_text(json.dumps({"expected_population": 0, "extracted_population": 0}))
    assert _population_checks(config)[0] is False


def test_verifier_checks_training_checkpoint_hashes(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    checkpoint = config.artifact_root / "sample_factory" / "job" / "checkpoint_64.pth"
    checkpoint_later = checkpoint.with_name("checkpoint_128.pth")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_later.write_bytes(b"checkpoint-later")
    training = config.artifact_root / "population"
    training.mkdir(parents=True)
    (training / "training_report.json").write_text(
        json.dumps(
            {
                "config_hash": config.digest,
                "jobs": [
                    {
                        "environment": config.environments[0],
                        "seed": config.seeds[0],
                        "checkpoints": [
                            {
                                "target_environment_steps": 64,
                                "actual_environment_steps": 64,
                                "checkpoint": str(checkpoint),
                                "checkpoint_sha256": sha256_file(checkpoint),
                            },
                            {
                                "target_environment_steps": 128,
                                "actual_environment_steps": 128,
                                "checkpoint": str(checkpoint_later),
                                "checkpoint_sha256": sha256_file(checkpoint_later),
                            },
                        ],
                    }
                ],
            }
        )
    )
    weights = config.artifact_root / "weights"
    weights.mkdir(parents=True)
    (weights / "extraction.json").write_text(
        json.dumps(
            {
                "expected_population": 1,
                "extracted_population": 1,
                "qualified_population": True,
            }
        )
    )
    assert _population_checks(config)[0] is True
    report_path = training / "training_report.json"
    report = json.loads(report_path.read_text())
    report["jobs"][0]["checkpoints"].pop()
    report_path.write_text(json.dumps(report))
    assert _population_checks(config)[0] is False
    report["jobs"][0]["checkpoints"].append(
        {
            "target_environment_steps": 128,
            "actual_environment_steps": 256,
            "checkpoint": str(checkpoint_later),
            "checkpoint_sha256": sha256_file(checkpoint_later),
        }
    )
    report_path.write_text(json.dumps(report))
    assert _population_checks(config)[0] is False
    report["jobs"][0]["checkpoints"][1]["actual_environment_steps"] = 128
    report_path.write_text(json.dumps(report))
    assert _population_checks(config)[0] is True
    checkpoint.write_bytes(b"tampered")
    assert _population_checks(config)[0] is False


def test_retained_checkpoint_selection_rejects_untrusted_records(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    assert _retained_checkpoint_records(config) is None
    population = config.artifact_root / "population"
    population.mkdir(parents=True)
    report_path = population / "training_report.json"
    report_path.write_text("{")
    with pytest.raises(ValueError, match="invalid training report"):
        _retained_checkpoint_records(config)
    report_path.write_text(json.dumps({"config_hash": "stale", "jobs": []}))
    with pytest.raises(ValueError, match="stale configuration"):
        _retained_checkpoint_records(config)
    base = {"config_hash": config.digest, "jobs": [{"environment": 1, "seed": 0}]}
    report_path.write_text(json.dumps(base))
    with pytest.raises(ValueError, match="invalid task/seed"):
        _retained_checkpoint_records(config)
    job: dict[str, Any] = {
        "environment": config.environments[0],
        "seed": config.seeds[0],
        "checkpoints": [],
    }
    report_path.write_text(json.dumps({"config_hash": config.digest, "jobs": [job]}))
    with pytest.raises(ValueError, match="no retained checkpoint"):
        _retained_checkpoint_records(config)
    missing = config.artifact_root / "missing.pth"
    job["checkpoints"] = [
        {
            "target_environment_steps": config.training.max_environment_steps,
            "checkpoint": str(missing),
            "checkpoint_sha256": "missing",
        }
    ]
    report_path.write_text(json.dumps({"config_hash": config.digest, "jobs": [job]}))
    with pytest.raises(ValueError, match="missing or corrupt"):
        _retained_checkpoint_records(config)
    checkpoint = config.artifact_root / "checkpoint.pth"
    checkpoint.write_bytes(b"valid")
    job["checkpoints"][0].update(
        {"checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint)}
    )
    report_path.write_text(json.dumps({"config_hash": config.digest, "jobs": [job, job]}))
    with pytest.raises(ValueError, match="duplicate final"):
        _retained_checkpoint_records(config)
    later = config.artifact_root / "checkpoint-later.pth"
    later.write_bytes(b"later")
    early_config = config.model_copy(
        update={"training": config.training.model_copy(update={"max_environment_steps": 192})}
    )
    job["checkpoints"] = [
        {
            "target_environment_steps": 64,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "evaluation": {"qualified": True},
        },
        {
            "target_environment_steps": 128,
            "checkpoint": str(later),
            "checkpoint_sha256": sha256_file(later),
            "evaluation": {"qualified": True},
        },
    ]
    report_path.write_text(json.dumps({"config_hash": early_config.digest, "jobs": [job]}))
    with pytest.raises(ValueError, match="maximum budget"):
        _retained_checkpoint_records(early_config)


def test_verifier_rejects_bad_evaluation_registry(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    evaluations = config.artifact_root / "evaluations"
    evaluations.mkdir(parents=True)
    (evaluations / "evaluation_report.json").write_text(json.dumps({"config_hash": "stale"}))
    pd.DataFrame({"wrong": [1]}).to_parquet(evaluations / "policy_registry.parquet", index=False)
    assert _evaluation_checks(config)[0] is False


def test_reduced_reproduction_is_no_go(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    config = config.model_copy(update={"artifact_root": tmp_path / "run"})
    report_path = reproduce(config, reduced=True)
    report = json.loads(report_path.read_text())
    assert report["decision"] == "NO_GO"
    assert report["phase_one_authorized"] is False
    assert (config.artifact_root / "states.zarr").is_dir()
    assert (config.artifact_root / "metrics/functional.parquet").is_file()
    checkpoint = config.artifact_root / "checkpoints/smoke-policy.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="corrupt stage artifact"):
        validate_existing_stage(config, "train")


def test_sol_smoke_launch_records_sample_factory_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    monkeypatch.setenv("SOL_COMMIT", "7c272b66e6ebe72ca008526d33f7e2e40e660af5")

    def fake_run(command: list[str], check: bool) -> None:
        assert check is True
        experiment = command[command.index("--experiment") + 1]
        (config.artifact_root / "sample_factory" / experiment).mkdir(parents=True)

    monkeypatch.setattr("ups.workflow.subprocess", SimpleNamespace(run=fake_run))
    stage = train(config, sol_smoke=True, resume=True)
    payload = json.loads(stage.read_text())
    assert payload["status"] == "SMOKE_ONLY"
    assert "--smoke" in payload["command"]
    assert payload["command"][-1] == "--resume"
    assert payload["resume"] is True


def test_sol_launcher_uses_preregistered_serial_gpu_recovery_settings() -> None:
    launcher = (ROOT / "src/ups/sol_train.py").read_text(encoding="utf-8")
    assert '"--serial_mode=True"' in launcher
    assert (
        '"--heartbeat_reporting_interval={args.heartbeat_reporting_interval_seconds}"' in launcher
    )
    assert '"--save_every_sec={args.checkpoint_save_interval_seconds}"' in launcher
    assert '"--keep_checkpoints={args.checkpoint_retention}"' in launcher


def test_population_plan_is_exact_and_deterministic(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    jobs = population_jobs(config)
    assert [(job["environment"], job["seed"]) for job in jobs] == [
        ("MiniHack-Room-Random-5x5-v0", 0)
    ]
    stage = launch_population(config, plan_only=True)
    payload = json.loads(stage.read_text())
    assert payload["status"] == "PLAN_ONLY"
    assert (
        json.loads((config.artifact_root / "population/population_plan.json").read_text())["jobs"]
        == jobs
    )


def test_population_checkpoint_upsert_does_not_alias_or_replace_peer_jobs() -> None:
    checkpoints = [{"target_environment_steps": 100}]
    records = [
        {"environment": "env-a", "seed": 0, "checkpoints": []},
        {"environment": "env-b", "seed": 1, "checkpoints": [{"target_environment_steps": 50}]},
    ]
    job = {"environment": "env-a", "seed": 0, "experiment": "env-a-seed0"}
    updated = _upsert_population_job(records, job, checkpoints)
    updated_by_key = {(record["environment"], record["seed"]): record for record in updated}
    assert updated_by_key[("env-a", 0)]["checkpoints"] is checkpoints
    assert updated_by_key[("env-b", 1)]["checkpoints"] == [{"target_environment_steps": 50}]
    checkpoints.append({"target_environment_steps": 200})
    assert len(updated_by_key[("env-b", 1)]["checkpoints"]) == 1


def test_population_stops_only_at_preregistered_budget(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    first = {"target_environment_steps": 64, "evaluation": {"qualified": True}}
    second = {"target_environment_steps": 128, "evaluation": {"qualified": True}}
    assert _population_stop_reason(config, []) is None
    assert _population_stop_reason(config, [first]) is None
    assert _population_stop_reason(config, [first, second]) == "MAX_ENVIRONMENT_STEPS"
    config = config.model_copy(
        update={"training": config.training.model_copy(update={"max_environment_steps": 192})}
    )
    assert _population_stop_reason(config, [first, second]) is None
    assert (
        _population_stop_reason(
            config,
            [first, {"target_environment_steps": 128, "evaluation": {"qualified": False}}],
        )
        is None
    )


def test_population_launcher_records_resumable_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run", "seeds": [0, 1]}
    )
    monkeypatch.setenv("SOL_COMMIT", "7c272b66e6ebe72ca008526d33f7e2e40e660af5")

    def fake_run(command: list[str], check: bool) -> None:
        assert check is True
        if "ups.sol_evaluate" in command:
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            experiment = command[command.index("--experiment") + 1]
            evaluations = config.artifact_root / "evaluations"
            evaluations.mkdir(parents=True, exist_ok=True)
            (evaluations / f"{experiment}.json").write_text(
                json.dumps(
                    {
                        "episodes": config.training.evaluation_episodes,
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "eval_seed": 1101,
                        "action_selection": "greedy_argmax",
                        "success_rate": 0.0,
                        "mean_return": 0.0,
                        "median_return": 0.0,
                        "mean_length": 1.0,
                        "table": str(evaluations / f"{experiment}.parquet"),
                    }
                )
            )
            return
        experiment = command[command.index("--experiment") + 1]
        target = command[command.index("--max-steps") + 1]
        checkpoint = (
            config.artifact_root
            / "sample_factory"
            / experiment
            / "checkpoint_p0"
            / f"checkpoint_{target}.pth"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(target)

    monkeypatch.setattr("ups.workflow.subprocess", SimpleNamespace(run=fake_run))
    stage = launch_population(config)
    payload = json.loads(stage.read_text())
    assert payload["status"] == "TRAINED_AND_EVALUATED"
    assert len(payload["jobs"]) == 2
    for job in payload["jobs"]:
        assert len(job["checkpoints"]) == 2
        assert all(isinstance(record["evaluation"], dict) for record in job["checkpoints"])
        assert all(f"seed{job['seed']}" in record["checkpoint"] for record in job["checkpoints"])
    report_path = config.artifact_root / "population" / "training_report.json"
    report = json.loads(report_path.read_text())
    report["jobs"][0]["checkpoints"][0]["evaluation"] = "AWAITING_FIXED_EPISODE_EVALUATION"
    report_path.write_text(json.dumps(report))
    resumed = launch_population(config)
    resumed_payload = json.loads(resumed.read_text())
    assert resumed_payload["status"] == "TRAINED_AND_EVALUATED"
    assert isinstance(resumed_payload["jobs"][0]["checkpoints"][0]["evaluation"], dict)


def test_population_launcher_does_not_stop_on_evaluation_performance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={
            "artifact_root": tmp_path / "run",
            "training": load_config(ROOT / "configs/smoke.yaml").training.model_copy(
                update={"max_environment_steps": 192}
            ),
        }
    )
    monkeypatch.setenv("SOL_COMMIT", "7c272b66e6ebe72ca008526d33f7e2e40e660af5")

    def fake_run(command: list[str], check: bool) -> None:
        assert check is True
        if "ups.sol_evaluate" in command:
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            experiment = command[command.index("--experiment") + 1]
            evaluations = config.artifact_root / "evaluations"
            evaluations.mkdir(parents=True, exist_ok=True)
            (evaluations / f"{experiment}.json").write_text(
                json.dumps(
                    {
                        "episodes": config.training.evaluation_episodes,
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "eval_seed": 1101,
                        "action_selection": "greedy_argmax",
                        "success_rate": 1.0,
                        "mean_return": 1.0,
                        "median_return": 1.0,
                        "mean_length": 1.0,
                        "table": str(evaluations / f"{experiment}.parquet"),
                    }
                )
            )
            return
        experiment = command[command.index("--experiment") + 1]
        target = command[command.index("--max-steps") + 1]
        checkpoint = (
            config.artifact_root
            / "sample_factory"
            / experiment
            / "checkpoint_p0"
            / f"checkpoint_{target}.pth"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(target)

    monkeypatch.setattr("ups.workflow.subprocess", SimpleNamespace(run=fake_run))
    report = json.loads(launch_population(config).read_text())
    job = report["jobs"][0]
    assert [record["target_environment_steps"] for record in job["checkpoints"]] == [64, 128, 192]
    assert job["stop_reason"] == "MAX_ENVIRONMENT_STEPS"


def test_population_launcher_recovers_malformed_progress_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    monkeypatch.setenv("SOL_COMMIT", "7c272b66e6ebe72ca008526d33f7e2e40e660af5")
    population = config.artifact_root / "population"
    population.mkdir(parents=True)
    (population / "training_report.json").write_text("{")

    def fake_run(command: list[str], check: bool) -> None:
        if "ups.sol_evaluate" in command:
            checkpoint = Path(command[command.index("--checkpoint") + 1])
            experiment = command[command.index("--experiment") + 1]
            evaluations = config.artifact_root / "evaluations"
            evaluations.mkdir(parents=True, exist_ok=True)
            (evaluations / f"{experiment}.json").write_text(
                json.dumps(
                    {
                        "episodes": config.training.evaluation_episodes,
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "eval_seed": 1101,
                        "action_selection": "greedy_argmax",
                        "success_rate": 0.0,
                        "mean_return": 0.0,
                        "median_return": 0.0,
                        "mean_length": 1.0,
                        "table": str(evaluations / f"{experiment}.parquet"),
                    }
                )
            )
            return
        experiment = command[command.index("--experiment") + 1]
        target = command[command.index("--max-steps") + 1]
        checkpoint = (
            config.artifact_root
            / "sample_factory"
            / experiment
            / "checkpoint_p0"
            / f"checkpoint_{target}.pth"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(target)

    monkeypatch.setattr("ups.workflow.subprocess", SimpleNamespace(run=fake_run))
    assert json.loads(launch_population(config).read_text())["status"] == "TRAINED_AND_EVALUATED"


def test_evaluator_replays_fixed_episodes_and_writes_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    monkeypatch.setenv("SOL_COMMIT", "7c272b66e6ebe72ca008526d33f7e2e40e660af5")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_text("checkpoint")
    training = {
        "jobs": [
            {
                "environment": config.environments[0],
                "seed": 0,
                "experiment": "phase0-smoke-random-seed0",
                "checkpoints": [
                    {
                        "target_environment_steps": config.training.max_environment_steps,
                        "checkpoint": str(checkpoint),
                    }
                ],
            }
        ]
    }
    population = config.artifact_root / "population"
    population.mkdir(parents=True)
    (population / "training_report.json").write_text(json.dumps(training))

    def fake_run(command: list[str], check: bool) -> None:
        assert check is True
        experiment = command[command.index("--experiment") + 1]
        evaluations = config.artifact_root / "evaluations"
        evaluations.mkdir(parents=True, exist_ok=True)
        (evaluations / f"{experiment}.json").write_text(
            json.dumps(
                {
                    "episodes": config.training.evaluation_episodes,
                    "action_selection": "greedy_argmax",
                    "success_rate": 0.5,
                    "mean_return": 1.0,
                    "median_return": 1.0,
                    "mean_length": 1.0,
                    "table": str(evaluations / f"{experiment}.parquet"),
                    "max_episode_steps": None,
                }
            )
        )

    monkeypatch.setattr("ups.workflow.subprocess", SimpleNamespace(run=fake_run))
    stage = evaluate(config)
    payload = json.loads(stage.read_text())
    assert payload["status"] == "EVALUATED_UNQUALIFIED"
    report = json.loads((config.artifact_root / "evaluations/evaluation_report.json").read_text())
    assert report["records"] == 1
    assert (config.artifact_root / "evaluations/policy_registry.parquet").is_file()


def test_extract_sol_checkpoint_to_safetensors(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    experiment = config.artifact_root / "sample_factory" / "smoke"
    checkpoint = experiment / "checkpoint_p0" / "checkpoint_000000001_32.pth"
    checkpoint.parent.mkdir(parents=True)
    (experiment / "config.json").write_text(
        json.dumps({"algo": "APPO", "env": config.environments[0], "seed": config.seeds[0]})
    )
    state = {
        "encoder.glyph_embedding.weight": torch.zeros(6000, 32),
        "encoder.cnn.0.weight": torch.zeros(64, 32, 3, 3),
        "encoder.cnn.2.weight": torch.zeros(64, 64, 3, 3),
        "encoder.cnn.4.weight": torch.zeros(64, 64, 3, 3),
        "encoder.projection.weight": torch.zeros(256, 5211),
        "core.core.weight_ih_l0": torch.zeros(768, 256),
        "critic_linear.weight": torch.zeros(1, 256),
        "action_parameterization.distribution_linear.weight": torch.zeros(8, 256),
    }
    torch.save({"model": state, "train_step": 1, "env_steps": 32}, checkpoint)
    unrecorded_newer = checkpoint.with_name("checkpoint_000000002_64.pth")
    unrecorded_newer.write_bytes(b"not a compatible checkpoint")
    population = config.artifact_root / "population"
    population.mkdir(parents=True)
    (population / "training_report.json").write_text(
        json.dumps(
            {
                "config_hash": config.digest,
                "jobs": [
                    {
                        "environment": config.environments[0],
                        "seed": config.seeds[0],
                        "checkpoints": [
                            {
                                "target_environment_steps": config.training.max_environment_steps,
                                "actual_environment_steps": config.training.max_environment_steps,
                                "checkpoint": str(checkpoint),
                                "checkpoint_sha256": sha256_file(checkpoint),
                            }
                        ],
                    }
                ],
            }
        )
    )
    evaluations = config.artifact_root / "evaluations"
    evaluations.mkdir(parents=True)
    (evaluations / "evaluation_report.json").write_text(
        json.dumps({"config_hash": config.digest, "qualified_population": True})
    )
    stage = extract_updates(config)
    payload = json.loads(stage.read_text())
    assert payload["status"] == "EXTRACTED_QUALIFIED"
    exported = next((config.artifact_root / "weights").glob("*.safetensors"))
    assert load_file(exported)["encoder.projection.weight"].shape == (256, 5211)
    metadata = json.loads(exported.with_suffix(".json").read_text())
    assert metadata["weights_sha256"] == sha256_file(exported)
    assert metadata["architecture"]["action_count"] == 8
    assert metadata["checkpoint"] == str(checkpoint)
    assert metadata["selected_from_training_report"] is True


def test_extraction_propagates_qualified_evaluation(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    evaluations = config.artifact_root / "evaluations"
    evaluations.mkdir(parents=True)
    report = evaluations / "evaluation_report.json"
    report.write_text(json.dumps({"config_hash": config.digest, "qualified_population": True}))
    stage = extract_updates(config)
    payload = json.loads(stage.read_text())
    extraction = json.loads((config.artifact_root / "weights/extraction.json").read_text())
    assert payload["status"] == "AWAITING_UPSTREAM_ARTIFACTS"
    assert extraction["qualified_population"] is False


def test_extraction_rejects_malformed_qualification_report(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    evaluations = config.artifact_root / "evaluations"
    evaluations.mkdir(parents=True)
    (evaluations / "evaluation_report.json").write_text("{")
    extract_updates(config)
    extraction = json.loads((config.artifact_root / "weights/extraction.json").read_text())
    assert extraction["qualified_population"] is False


def test_checkpoint_extraction_rejects_invalid_models() -> None:
    with pytest.raises(ValueError, match="no model"):
        _phase0_state_dict({})
    with pytest.raises(ValueError, match="missing"):
        _phase0_state_dict({"model": {}})
    state = {
        "encoder.glyph_embedding.weight": torch.zeros(6000, 32),
        "encoder.cnn.0.weight": torch.zeros(64, 32, 3, 3),
        "encoder.cnn.2.weight": torch.zeros(64, 64, 3, 3),
        "encoder.cnn.4.weight": torch.zeros(64, 64, 3, 3),
        "encoder.projection.weight": torch.zeros(256, 5211),
        "core.core.weight_ih_l0": torch.zeros(768, 256),
        "critic_linear.weight": torch.zeros(1, 256),
        "action_parameterization.distribution_linear.weight": torch.zeros(7, 256),
    }
    with pytest.raises(ValueError, match="eight-action"):
        _phase0_state_dict({"model": state})
