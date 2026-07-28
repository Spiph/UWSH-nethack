import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from ups.config import load_config
from ups.gate import REQUIRED_NUMERICAL_CHECKS, evaluate_gate
from ups.workflow import reproduce, train, validate_existing_stage

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
    stage = train(config, sol_smoke=True)
    payload = json.loads(stage.read_text())
    assert payload["status"] == "SMOKE_ONLY"
    assert payload["command"][-1] == "--smoke"
