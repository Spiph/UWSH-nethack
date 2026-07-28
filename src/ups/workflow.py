"""Resumable Phase Zero orchestration.

The reduced workflow validates mechanics only. It never promotes smoke evidence to
preregistered evidence and therefore necessarily produces NO_GO.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
import zarr  # type: ignore[import-untyped]
from safetensors.numpy import save_file

from ups.alignment import permute_mlp
from ups.artifacts import atomic_json, sha256_path, write_manifest
from ups.config import Phase0Config
from ups.gate import evaluate_gate, markdown_report
from ups.lora import compose_lora, merged_equivalent, rotate_factors
from ups.metrics import action_agreement, action_kl, linear_cka, normalized_value_rmse
from ups.model import RecurrentNLEPolicy
from ups.nulls import gaussian_norm_matched, spectrum_matched_orientation
from ups.subspace import fit_svd, hosvd, hosvd_reconstruct


def stage_path(config: Phase0Config, name: str) -> Path:
    return config.artifact_root / "stages" / f"{name}.json"


def record_stage(config: Phase0Config, name: str, payload: dict[str, Any]) -> Path:
    path = stage_path(config, name)
    artifact = payload.get("checkpoint") or payload.get("zarr")
    integrity = sha256_path(Path(artifact)) if artifact is not None else None
    atomic_json(
        path,
        {
            "config_hash": config.digest,
            "stage": name,
            "artifact_sha256": integrity,
            **payload,
        },
    )
    write_manifest(config, name, [path])
    return path


def validate_existing_stage(config: Phase0Config, name: str) -> bool:
    path = stage_path(config, name)
    if not path.exists():
        return False
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if value.get("config_hash") != config.digest:
        raise RuntimeError(f"stale configuration hash in {path}")
    if value.get("status") not in {"COMPLETE", "SMOKE_ONLY"}:
        return False
    artifact = value.get("checkpoint") or value.get("zarr")
    if artifact is not None and not Path(artifact).exists():
        raise RuntimeError(f"missing stage artifact {artifact}")
    if artifact is not None and value.get("artifact_sha256") != sha256_path(Path(artifact)):
        raise RuntimeError(f"corrupt stage artifact {artifact}")
    return True


def train(config: Phase0Config, reduced: bool = False, sol_smoke: bool = False) -> Path:
    """Create deterministic smoke checkpoints or declare the full APPO launch contract."""
    if sol_smoke:
        if config.training.runtime != "sol_patched":
            raise RuntimeError("SOL APPO smoke runs require training.runtime=sol_patched")
        if os.environ.get("SOL_COMMIT") != "7c272b66e6ebe72ca008526d33f7e2e40e660af5":
            raise RuntimeError("SOL APPO smoke runs must execute in the pinned research container")
        environment = config.environments[0]
        experiment = (
            f"{config.study}-{environment.removesuffix('-v0').lower()}-seed{config.seeds[0]}-smoke"
        )
        command = [
            sys.executable,
            "-m",
            "ups.sol_train",
            "--environment",
            environment,
            "--seed",
            str(config.seeds[0]),
            "--artifact-root",
            str(config.artifact_root),
            "--experiment",
            experiment,
            "--max-steps",
            str(config.training.max_environment_steps),
            "--smoke",
        ]
        subprocess.run(command, check=True)
        checkpoint_root = config.artifact_root / "sample_factory" / experiment
        return record_stage(
            config,
            "train",
            {
                "status": "SMOKE_ONLY",
                "checkpoint": str(checkpoint_root),
                "full_population": False,
                "command": command,
            },
        )
    if not reduced:
        return record_stage(
            config,
            "train",
            {
                "status": "NOT_EXECUTED",
                "reason": "full APPO population must run via the pinned rl container",
                "expected_jobs": len(config.seeds) * len(config.environments),
                "phase_one_authorized": False,
            },
        )
    torch.manual_seed(config.seeds[0])
    model = RecurrentNLEPolicy(glyph_vocab=128, crop_size=5, hidden_size=32)
    checkpoint = config.artifact_root / "checkpoints" / "smoke-policy.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "step": config.training.max_environment_steps,
            "config_hash": config.digest,
        },
        checkpoint,
    )
    return record_stage(
        config,
        "train",
        {"status": "SMOKE_ONLY", "checkpoint": str(checkpoint), "full_population": False},
    )


def collect_states(config: Phase0Config, reduced: bool = False) -> Path:
    sequences = config.evaluation_buffer.sequences if reduced else 0
    root = config.artifact_root / "states.zarr"
    group = zarr.open_group(str(root), mode="w")
    group.attrs.update(
        {
            "config_hash": config.digest,
            "behavior": "synthetic mechanics fixture" if reduced else "not collected",
            "smoke_only": reduced,
        }
    )
    rng = np.random.default_rng(1101)
    group.create_dataset(
        "glyphs_crop",
        data=rng.integers(
            0, 128, size=(sequences, config.evaluation_buffer.sequence_length, 5, 5), dtype=np.int16
        ),
        chunks=(min(8, max(sequences, 1)), config.evaluation_buffer.sequence_length, 5, 5),
    )
    group.create_dataset(
        "reset_masks",
        data=np.zeros((sequences, config.evaluation_buffer.sequence_length), dtype=np.bool_),
    )
    return record_stage(
        config,
        "collect-states",
        {
            "status": "SMOKE_ONLY" if reduced else "NOT_EXECUTED",
            "sequences": sequences,
            "zarr": str(root),
        },
    )


def numerical_evidence(config: Phase0Config) -> tuple[dict[str, bool], dict[str, float]]:
    rng = np.random.default_rng(7)
    a = rng.normal(size=(2, 7))
    b = rng.normal(size=(5, 2))
    rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    rotated_a, rotated_b = rotate_factors(a, b, rotation)
    update = compose_lora(a, b, alpha=4)
    rotation_ok = np.allclose(update, compose_lora(rotated_a, rotated_b, alpha=4))
    merge_ok = merged_equivalent(rng.normal(size=(5, 7)), a, b, rng.normal(size=(3, 7)), 4)

    # Re-Basin compensation check on a synthetic two-layer ReLU network.
    w1, bias, w2 = rng.normal(size=(4, 3)), rng.normal(size=4), rng.normal(size=(2, 4))
    permutation = np.array([2, 0, 3, 1])
    pw1, pbias, pw2 = permute_mlp(w1, bias, w2, permutation)
    x = rng.normal(size=(20, 3))
    before = np.maximum(0, x @ w1.T + bias) @ w2.T
    after = np.maximum(0, x @ pw1.T + pbias) @ pw2.T

    tensor = rng.normal(size=(4, 5, 3))
    factors = hosvd(tensor)
    reconstructed = hosvd_reconstruct(tensor, factors, list(tensor.shape))
    population = rng.normal(size=(8, 12))
    basis = fit_svd(population, 0.95)
    svd_ok = np.linalg.norm(basis.reconstruct(population) - population) <= np.linalg.norm(
        population - population.mean(0)
    )

    gaussian = gaussian_norm_matched(update, rng)
    spectrum = spectrum_matched_orientation(update, rng)
    null_ok = np.allclose(np.linalg.norm(gaussian), np.linalg.norm(update)) and np.allclose(
        np.linalg.svd(spectrum, compute_uv=False), np.linalg.svd(update, compute_uv=False)
    )

    reference = rng.normal(size=(64, 8))
    candidate = reference + rng.normal(scale=1e-3, size=reference.shape)
    features = rng.normal(size=(64, 12))
    metrics = {
        "action_kl": action_kl(reference, candidate),
        "action_agreement": action_agreement(reference, candidate),
        "feature_cka": linear_cka(features, features + rng.normal(scale=1e-3, size=features.shape)),
        "normalized_value_rmse": normalized_value_rmse(reference[:, 0], candidate[:, 0]),
    }
    checks = {
        "lora_composition": bool(np.allclose(update, 2 * (b @ a))),
        "lora_rotation_invariance": bool(rotation_ok),
        "merged_adapter_equivalence": merge_ok,
        "alignment_invariance": bool(np.allclose(before, after)),
        "null_preservation": bool(null_ok),
        # Convention: samples are zero-centered before SVD; HOSVD uses mode-n
        # unfoldings and left singular vectors, matching the official UniSub method.
        # This mechanics check validates conventions, but it is not the required
        # reproduction on the four public adapters and therefore cannot pass it.
        "adapter_svd_hosvd_reproduction": False,
        "svd_hosvd_mechanics": bool(np.allclose(tensor, reconstructed, atol=1e-10) and svd_ok),
    }
    return checks, metrics


def analyze(config: Phase0Config, reduced: bool = False) -> Path:
    checks, metrics = numerical_evidence(config)
    table = pd.DataFrame([{"scope": "mechanics-smoke", **metrics}])
    path = config.artifact_root / "metrics" / "functional.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    basis_path = config.artifact_root / "bases" / "mechanics.safetensors"
    basis_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"basis": np.eye(4, dtype=np.float32)}, str(basis_path))
    runtime_verified = (
        config.training.runtime == "sol_patched"
        and os.environ.get("SOL_COMMIT") == "7c272b66e6ebe72ca008526d33f7e2e40e660af5"
    )
    evidence = {
        "config_hash": config.digest,
        "evidence_class": "SMOKE_ONLY" if reduced else "INCOMPLETE",
        "numerical_checks": checks,
        "metrics": metrics,
        "completeness": {
            "runtime_profile": config.training.runtime,
            "locked_runtime_compatible": runtime_verified,
            "full_population": False,
            "all_policies_success_ge_075": False,
            "minimum_policy_success": None,
            "evaluation_sequences": config.evaluation_buffer.sequences if reduced else 0,
            "sequence_length": config.evaluation_buffer.sequence_length if reduced else 0,
            "null_replicates": config.analysis.null_replicates if reduced else 0,
        },
    }
    evidence_path = config.artifact_root / "evidence.json"
    atomic_json(evidence_path, evidence)
    write_manifest(config, "analyze", [path, basis_path, evidence_path])
    return evidence_path


def gate(config: Phase0Config) -> Path:
    evidence_path = config.artifact_root / "evidence.json"
    if evidence_path.exists():
        with evidence_path.open(encoding="utf-8") as stream:
            evidence = json.load(stream)
        if evidence.get("config_hash") != config.digest:
            raise RuntimeError("stale evidence config hash")
    else:
        evidence = {}
    report = evaluate_gate(config, evidence)
    json_path = config.artifact_root / "phase0_gate.json"
    markdown_path = config.artifact_root / "phase0_gate.md"
    atomic_json(json_path, report)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    write_manifest(config, "gate-phase0", [json_path, markdown_path])
    return json_path


def reproduce(config: Phase0Config, reduced: bool) -> Path:
    steps: tuple[tuple[str, Callable[[], Path]], ...] = (
        ("train", lambda: train(config, reduced)),
        ("collect-states", lambda: collect_states(config, reduced)),
        ("analyze", lambda: analyze(config, reduced)),
    )
    for name, operation in steps:
        if not validate_existing_stage(config, name):
            operation()
    rerun = config.artifact_root / "RERUN.md"
    rerun.parent.mkdir(parents=True, exist_ok=True)
    rerun.write_text(
        "# Exact rerun\n\n"
        "\n```bash\nuv run ups reproduce phase0 "
        f"--config configs/{'smoke' if reduced else 'phase0'}.yaml"
        f"{' --reduced' if reduced else ''}\n```\n",
        encoding="utf-8",
    )
    return gate(config)
