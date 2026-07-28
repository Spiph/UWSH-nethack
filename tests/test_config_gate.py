import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from pydantic import ValidationError
from safetensors.torch import load_file

import ups.gate as gate_module
from ups.config import load_config
from ups.gate import (
    REQUIRED_NUMERICAL_CHECKS,
    _has_geometry,
    _null_ensemble_counts,
    evaluate_gate,
)
from ups.verifier import _manifest_checks, _population_checks, verify_artifacts
from ups.workflow import (
    _phase0_state_dict,
    evaluate,
    extract_updates,
    launch_population,
    population_jobs,
    reproduce,
    train,
    validate_existing_stage,
)

ROOT = Path(__file__).parents[1]


def test_config_hash_is_stable_and_schema_is_strict(tmp_path: Path) -> None:
    config = load_config(ROOT / "configs/smoke.yaml")
    assert config.digest == load_config(ROOT / "configs/smoke.yaml").digest
    raw = (ROOT / "configs/smoke.yaml").read_text()
    bad = tmp_path / "bad.yaml"
    bad.write_text(raw + "\nunknown: true\n")
    with pytest.raises(ValidationError):
        load_config(bad)


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
    assert result["verified"] is True


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


def test_population_launcher_records_resumable_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(ROOT / "configs/smoke.yaml").model_copy(
        update={"artifact_root": tmp_path / "run"}
    )
    monkeypatch.setenv("SOL_COMMIT", "7c272b66e6ebe72ca008526d33f7e2e40e660af5")

    def fake_run(command: list[str], check: bool) -> None:
        assert check is True
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
    assert payload["status"] == "TRAINED_AWAITING_EVALUATION"
    assert len(payload["jobs"][0]["checkpoints"]) == 2


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
            json.dumps({"success_rate": 0.5, "median_return": 1.0})
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
        json.dumps({"env": config.environments[0], "seed": config.seeds[0]})
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
    stage = extract_updates(config)
    payload = json.loads(stage.read_text())
    assert payload["status"] == "EXTRACTED_UNQUALIFIED"
    exported = next((config.artifact_root / "weights").glob("*.safetensors"))
    assert load_file(exported)["encoder.projection.weight"].shape == (256, 5211)


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
