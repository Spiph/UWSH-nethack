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
from ups.workflow import (
    _phase0_state_dict,
    extract_updates,
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
    monkeypatch.setattr(gate_module, "ARTIFACT_VERIFIER_IMPLEMENTED", True)
    report = evaluate_gate(config, evidence)
    assert report["decision"] == "PASS"
    assert report["phase_one_authorized"] is True
    evidence["completeness"]["null_ensembles"].pop("independent_low_rank")
    assert evaluate_gate(config, evidence)["decision"] == "NO_GO"


def test_gate_contract_rejects_missing_geometry_and_control_manifests() -> None:
    assert _has_geometry({"geometry": None}, "encoder") is False
    assert _has_geometry({"geometry": {"encoder": None}}, "encoder") is False
    assert _null_ensemble_counts({"null_ensembles": None}, 1000) is False


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
