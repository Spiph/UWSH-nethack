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
from safetensors.torch import save_file as save_torch_file

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


def train(
    config: Phase0Config,
    reduced: bool = False,
    sol_smoke: bool = False,
    resume: bool = False,
    plan_only: bool = False,
) -> Path:
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
        if resume:
            command.append("--resume")
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
                "resume": resume,
            },
        )
    if not reduced:
        return launch_population(config, plan_only=plan_only)
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


def population_jobs(config: Phase0Config) -> list[dict[str, Any]]:
    """Create the immutable, duplicate-free 4-task x 3-seed job registry."""
    jobs: list[dict[str, Any]] = []
    for environment in config.environments:
        for seed in config.seeds:
            slug = environment.removesuffix("-v0").lower()
            jobs.append(
                {
                    "environment": environment,
                    "seed": seed,
                    "experiment": f"{config.study}-{slug}-seed{seed}",
                    "max_environment_steps": config.training.max_environment_steps,
                    "evaluation_interval": config.training.evaluation_interval,
                    "evaluation_episodes": config.training.evaluation_episodes,
                    "minimum_success": config.training.minimum_success,
                    "cohort": "independent_initialization",
                    "base_checkpoint": None,
                }
            )
    return jobs


def launch_population(config: Phase0Config, plan_only: bool = False) -> Path:
    """Launch exactly the preregistered jobs in resumable 100k-step chunks.

    The launcher intentionally stops at ``TRAINED_AWAITING_EVALUATION``. No
    checkpoint becomes a retained policy until the fixed-episode evaluator records
    the preregistered success threshold.
    """
    jobs = population_jobs(config)
    population_root = config.artifact_root / "population"
    population_root.mkdir(parents=True, exist_ok=True)
    plan_path = population_root / "population_plan.json"
    atomic_json(plan_path, {"config_hash": config.digest, "jobs": jobs})
    if plan_only:
        return record_stage(
            config,
            "train",
            {
                "status": "PLAN_ONLY",
                "checkpoint": str(population_root),
                "jobs": jobs,
                "command": None,
            },
        )
    if config.training.runtime != "sol_patched":
        raise RuntimeError("the Phase Zero population requires the pinned SOL runtime")
    if os.environ.get("SOL_COMMIT") != "7c272b66e6ebe72ca008526d33f7e2e40e660af5":
        raise RuntimeError("run the Phase Zero population inside the pinned research container")
    job_records: list[dict[str, Any]] = []
    interval = config.training.evaluation_interval
    for job in jobs:
        experiment_root = config.artifact_root / "sample_factory" / job["experiment"]
        checkpoints: list[dict[str, Any]] = []
        for target in range(interval, config.training.max_environment_steps + interval, interval):
            command = [
                sys.executable,
                "-m",
                "ups.sol_train",
                "--environment",
                job["environment"],
                "--seed",
                str(job["seed"]),
                "--artifact-root",
                str(config.artifact_root),
                "--experiment",
                job["experiment"],
                "--max-steps",
                str(target),
            ]
            if target > interval or experiment_root.is_dir():
                command.append("--resume")
            subprocess.run(command, check=True)
            checkpoint = _latest_checkpoint(experiment_root)
            if checkpoint is None:
                raise RuntimeError(
                    f"no checkpoint produced for {job['experiment']} at {target} steps"
                )
            checkpoints.append(
                {
                    "target_environment_steps": target,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_path(checkpoint),
                    "command": command,
                    "evaluation": "AWAITING_FIXED_200_EPISODE_EVALUATION",
                }
            )
        job_records.append({**job, "checkpoints": checkpoints})
    report = population_root / "training_report.json"
    atomic_json(
        report,
        {
            "config_hash": config.digest,
            "jobs": job_records,
            "status": "TRAINED_AWAITING_EVALUATION",
            "qualified_population": False,
        },
    )
    return record_stage(
        config,
        "train",
        {
            "status": "TRAINED_AWAITING_EVALUATION",
            "checkpoint": str(population_root),
            "jobs": job_records,
            "training_report": str(report),
            "full_population": len(job_records) == len(jobs),
            "phase_one_authorized": False,
        },
    )


def evaluate(config: Phase0Config) -> Path:
    """Evaluate every recorded checkpoint on its fixed 200-episode task seeds."""
    report_path = config.artifact_root / "population" / "training_report.json"
    if not report_path.is_file():
        return record_stage(config, "evaluate", {"status": "AWAITING_CHECKPOINTS"})
    if os.environ.get("SOL_COMMIT") != "7c272b66e6ebe72ca008526d33f7e2e40e660af5":
        raise RuntimeError("Phase Zero evaluation must run in the pinned SOL container")
    training = json.loads(report_path.read_text(encoding="utf-8"))
    fixed_seeds = dict(zip(config.environments, config.evaluation_buffer.fixed_seeds, strict=True))
    summaries: list[dict[str, Any]] = []
    for job in training.get("jobs", []):
        for checkpoint_record in job.get("checkpoints", []):
            command = [
                sys.executable,
                "-m",
                "ups.sol_evaluate",
                "--checkpoint",
                checkpoint_record["checkpoint"],
                "--episodes",
                str(config.training.evaluation_episodes),
                "--eval-seed",
                str(fixed_seeds[job["environment"]]),
                "--artifact-root",
                str(config.artifact_root),
                "--environment",
                job["environment"],
                "--experiment",
                f"{job['experiment']}-step{checkpoint_record['target_environment_steps']}",
            ]
            subprocess.run(command, check=True)
            summary = json.loads(
                (
                    config.artifact_root
                    / "evaluations"
                    / (
                        f"{job['experiment']}-step"
                        f"{checkpoint_record['target_environment_steps']}.json"
                    )
                ).read_text(encoding="utf-8")
            )
            summaries.append(
                {
                    **job,
                    **checkpoint_record,
                    "success_rate": summary["success_rate"],
                    "median_return": summary["median_return"],
                    "qualified": summary["success_rate"] >= config.training.minimum_success,
                }
            )
    table_path = config.artifact_root / "evaluations" / "policy_registry.parquet"
    pd.DataFrame(summaries).to_parquet(table_path, index=False)
    evaluation_report = config.artifact_root / "evaluations" / "evaluation_report.json"
    final_summaries = [
        row
        for row in summaries
        if row["target_environment_steps"] == config.training.max_environment_steps
    ]
    final_pairs = {(row["environment"], row["seed"]) for row in final_summaries}
    expected_pairs = {(job["environment"], job["seed"]) for job in population_jobs(config)}
    qualified_final = (
        bool(final_summaries)
        and len(final_pairs) == len(expected_pairs)
        and final_pairs == expected_pairs
        and all(row["qualified"] for row in final_summaries)
    )
    atomic_json(
        evaluation_report,
        {
            "config_hash": config.digest,
            "episodes_per_checkpoint": config.training.evaluation_episodes,
            "fixed_eval_seeds": fixed_seeds,
            "records": len(summaries),
            "qualified_population": qualified_final,
            "policy_registry": str(table_path),
        },
    )
    return record_stage(
        config,
        "evaluate",
        {
            "status": "QUALIFIED" if qualified_final else "EVALUATED_UNQUALIFIED",
            "checkpoint": str(config.artifact_root / "evaluations"),
            "evaluation_report": str(evaluation_report),
            "qualified_population": qualified_final,
        },
    )


def _latest_checkpoint(experiment: Path) -> Path | None:
    checkpoints = sorted(experiment.glob("checkpoint_p0/checkpoint_*.pth"))
    return checkpoints[-1] if checkpoints else None


def _phase0_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    model = payload.get("model")
    if not isinstance(model, dict):
        raise ValueError("Sample Factory checkpoint has no model state dictionary")
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in model.items()
        if isinstance(value, torch.Tensor)
    }
    required = {
        "encoder.glyph_embedding.weight",
        "encoder.cnn.0.weight",
        "encoder.cnn.2.weight",
        "encoder.cnn.4.weight",
        "encoder.projection.weight",
        "core.core.weight_ih_l0",
        "critic_linear.weight",
        "action_parameterization.distribution_linear.weight",
    }
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError(f"incompatible Phase Zero Sample Factory checkpoint; missing {missing}")
    if tuple(state["action_parameterization.distribution_linear.weight"].shape) != (8, 256):
        raise ValueError(
            "checkpoint does not expose the required eight-action, 256-unit policy head"
        )
    return state


def extract_updates(config: Phase0Config) -> Path:
    """Convert SOL checkpoints to SafeTensors and propagate fixed-eval qualification."""
    source = config.artifact_root / "sample_factory"
    export_root = config.artifact_root / "weights"
    expected = {(environment, seed) for environment in config.environments for seed in config.seeds}
    seen: set[tuple[str, int]] = set()
    records: list[dict[str, Any]] = []
    if source.is_dir():
        for experiment in sorted(path for path in source.iterdir() if path.is_dir()):
            config_path = experiment / "config.json"
            checkpoint = _latest_checkpoint(experiment)
            if not config_path.is_file() or checkpoint is None:
                continue
            with config_path.open(encoding="utf-8") as stream:
                sf_config = json.load(stream)
            environment, seed = sf_config.get("env"), sf_config.get("seed")
            if not isinstance(environment, str) or not isinstance(seed, int):
                raise ValueError(f"invalid Sample Factory config at {config_path}")
            if (environment, seed) not in expected:
                continue
            payload = torch.load(checkpoint, map_location="cpu")
            state = _phase0_state_dict(payload)
            destination = export_root / f"{environment.removesuffix('-v0')}-seed{seed}.safetensors"
            destination.parent.mkdir(parents=True, exist_ok=True)
            save_torch_file(
                state, str(destination), metadata={"format": "sol-sample-factory-phase0-v1"}
            )
            metadata_path = destination.with_suffix(".json")
            atomic_json(
                metadata_path,
                {
                    "environment": environment,
                    "seed": seed,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256_path(checkpoint),
                    "weights": str(destination),
                    "state_keys": sorted(state),
                    "train_step": payload.get("train_step"),
                    "environment_steps": payload.get("env_steps"),
                },
            )
            records.append(json.loads(metadata_path.read_text(encoding="utf-8")))
            seen.add((environment, seed))
    missing = sorted(f"{environment}:seed{seed}" for environment, seed in expected - seen)
    evaluation_report = config.artifact_root / "evaluations" / "evaluation_report.json"
    qualified_population = False
    if evaluation_report.is_file():
        try:
            evaluation = json.loads(evaluation_report.read_text(encoding="utf-8"))
            qualified_population = (
                evaluation.get("config_hash") == config.digest
                and evaluation.get("qualified_population") is True
                and not missing
            )
        except (OSError, json.JSONDecodeError):
            qualified_population = False
    report = export_root / "extraction.json"
    atomic_json(
        report,
        {
            "config_hash": config.digest,
            "records": records,
            "expected_population": len(expected),
            "extracted_population": len(seen),
            "missing": missing,
            "qualified_population": qualified_population,
            "qualification_source": str(evaluation_report) if evaluation_report.is_file() else None,
        },
    )
    status = (
        "EXTRACTED_QUALIFIED"
        if records and qualified_population
        else "EXTRACTED_UNQUALIFIED"
        if records
        else "AWAITING_UPSTREAM_ARTIFACTS"
    )
    return record_stage(
        config,
        "extract-updates",
        {"status": status, "checkpoint": str(export_root), "extraction_report": str(report)},
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
