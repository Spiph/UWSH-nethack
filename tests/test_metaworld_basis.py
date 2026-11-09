from __future__ import annotations

import base64
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from scripts.metaworld_alignment import apply_sac_permutations, distribution_outputs
from scripts.metaworld_basis import (
    BasisArtifact,
    BasisFitSpec,
    SourcePolicy,
    TaskRelations,
    TaskSplitManifest,
    actor_hash_from_vector,
    fit_source_basis,
)
from scripts.metaworld_geometry import actor_vector, hash_actor, load_actor_vector
from scripts.metaworld_protocol import Goal, GoalBank, GoalManifest, SeedStreams

Array = npt.NDArray[np.float64]


def goal_bank(split: str, offset: int, task: str = "synthetic-v3") -> GoalBank:
    return GoalBank(
        split,
        offset,
        tuple(
            Goal(
                task,
                base64.b64encode(f"synthetic-payload-{offset + index}".encode()).decode(),
                (float(offset + index),),
            )
            for index in range(3)
        ),
    )


def goals() -> GoalManifest:
    return GoalManifest((goal_bank("source", 0), goal_bank("development", 10)))


def policies(
    rows: Array,
    *,
    layout: list[dict[str, Any]] | None = None,
    initial: Array | None = None,
    independent: bool = False,
    dtype: str = "float64",
) -> list[SourcePolicy]:
    layout = layout or [
        {"name": "weight", "shape": [rows.shape[1]], "start": 0, "end": rows.shape[1]}
    ]
    result = []
    for index, vector in enumerate(rows):
        start = np.zeros(rows.shape[1]) if initial is None else initial.copy()
        if independent:
            start += index + 1
        streams = SeedStreams.from_seed(
            1000 + index, actor_seed=777 + index if independent else 777
        )
        result.append(
            SourcePolicy(
                policy_id=f"synthetic-policy-{index}",
                training_run_id=f"synthetic-training-{index}",
                checkpoint_step=100,
                task_class="synthetic-v3",
                library_id="synthetic-independent-library-0",
                initialization_condition="independent_init" if independent else "shared_init",
                training_goal_bank_sha256=goals().for_fitting()[0].sha256,
                actor_sha256=actor_hash_from_vector(vector, layout, dtype),
                initial_actor_sha256=actor_hash_from_vector(start, layout, dtype),
                rng_streams=streams.__dict__,
                vector=vector,
                initial_vector=start,
                layout=layout,
                parameter_dtype=dtype,
            )
        )
    return result


def task_split() -> TaskSplitManifest:
    return TaskSplitManifest(("synthetic-v3",), "development_only", "synthetic-unit-test-only")


def flat_spec(rank: int = 1, **kwargs: Any) -> BasisFitSpec:
    return BasisFitSpec(
        "flat_population_pca", "final_weights", "raw", "fixed_predeclared", rank=rank, **kwargs
    )


def fit(rows: Array, spec: BasisFitSpec | None = None) -> BasisArtifact:
    return fit_source_basis(
        policies(rows),
        goals(),
        spec or flat_spec(),
        task_split=task_split(),
        relations=TaskRelations(),
    )


def test_known_low_rank_unseen_vector_and_actual_storage_counts() -> None:
    center = np.arange(6, dtype=np.float64)
    direction = np.array([1.0, 2.0, -1.0, 0.0, 1.0, 3.0])
    rows = np.stack([center - direction, center, center + direction])
    artifact = fit(rows)
    oracle = artifact.oracle_project(center + 2.5 * direction)
    np.testing.assert_allclose(oracle.reconstructed, center + 2.5 * direction, atol=1e-12)
    np.testing.assert_array_equal(artifact.decode(np.zeros(1)), artifact.reference)
    assert oracle.family_sha256 == artifact.family_sha256
    assert "privileged_target_weights" in oracle.information_access
    assert artifact.metadata["centering"]["stacked_axes"] == [0]
    assert artifact.metadata["storage"]["deployment_shared_scalars"] == 12
    assert artifact.metadata["storage"]["per_target_coefficient_scalars"] == 1
    assert artifact.metadata["storage"]["artifact_array_scalars"] == 33
    assert artifact.metadata["independent_library_replicates"] == 1
    assert artifact.metadata["fitting_goal_manifest_sha256"] == goals().sha256
    assert artifact.metadata["task_split_manifest_sha256"] == task_split().sha256


def test_exact_source_reference_is_not_silently_projected_into_the_pca_mean_family() -> None:
    rows = np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    sources = policies(rows)
    spec = flat_spec(reference_rule="source_policy", reference_policy_id=sources[1].policy_id)
    artifact = fit_source_basis(
        sources, goals(), spec, task_split=task_split(), relations=TaskRelations()
    )
    np.testing.assert_array_equal(artifact.decode([0]), rows[1])
    oracle = artifact.oracle_project(np.array([0.5, 0.0]))
    np.testing.assert_allclose(oracle.reconstructed, [0.5, 1.0], atol=1e-12)
    np.testing.assert_array_equal(oracle.reconstructed, artifact.decode(oracle.coefficients))
    assert oracle.family_sha256 == artifact.family_sha256


def test_scaled_basis_oracle_uses_raw_weight_metric_and_the_same_decoder() -> None:
    direction = np.array([1.0, 10.0])
    artifact = fit(
        np.stack([-direction, np.zeros(2), direction]),
        flat_spec(normalization="source_coordinate_std"),
    )
    oracle = artifact.oracle_project(np.array([1.0, 0.0]))
    np.testing.assert_allclose(oracle.reconstructed, np.array([1.0, 10.0]) / 101, atol=1e-12)
    np.testing.assert_array_equal(oracle.reconstructed, artifact.decode(oracle.coefficients))


def test_exclusion_precedes_center_normalization_and_rank_selection() -> None:
    rows = np.array([[-2.0, 0.0, 1.0], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [1e10, -1e10, 3e10]])
    sources = policies(rows)
    spec = flat_spec(normalization="source_coordinate_std")
    baseline = fit_source_basis(
        sources[:3], goals(), spec, task_split=task_split(), relations=TaskRelations()
    )
    excluded = fit_source_basis(
        sources,
        goals(),
        spec,
        excluded_policy_ids=(sources[3].policy_id,),
        task_split=task_split(),
        relations=TaskRelations(),
    )
    assert baseline.family_sha256 == excluded.family_sha256
    assert sources[3].policy_id not in excluded.metadata["source_policy_ids"]
    changed = policies(rows.copy())
    poisoned = np.array([7e12, 5e13, -2e14])
    changed[3] = replace(
        changed[3],
        vector=poisoned,
        actor_sha256=actor_hash_from_vector(poisoned, sources[0].layout, "float64"),
    )
    repeated = fit_source_basis(
        changed,
        goals(),
        spec,
        excluded_policy_ids=(sources[3].policy_id,),
        task_split=task_split(),
        relations=TaskRelations(),
    )
    assert repeated.sha256 == excluded.sha256
    with pytest.raises(ValueError, match="remaining source"):
        fit_source_basis(
            sources,
            goals(),
            replace(spec, reference_rule="source_policy", reference_policy_id=sources[3].policy_id),
            excluded_policy_ids=(sources[3].policy_id,),
            task_split=task_split(),
            relations=TaskRelations(),
        )


def test_final_goal_and_teacher_target_data_are_denied_before_fitting() -> None:
    sources = policies(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    final_goals = GoalManifest(
        (*goals().banks, goal_bank("final_test", 20)), transfer_regime="within_family"
    )
    with pytest.raises(ValueError, match="final-test goals"):
        fit_source_basis(
            sources, final_goals, flat_spec(), task_split=task_split(), relations=TaskRelations()
        )
    with pytest.raises(ValueError, match="never targets or teachers"):
        replace(sources[0], training_split="final_test")
    with pytest.raises(ValueError, match="never targets or teachers"):
        replace(sources[0], data_kind="teacher_actions")
    for split, kind in (
        ("final_test", "metrics"),
        ("development", "teacher_actions"),
        ("development", "target_weights"),
    ):
        evidence = {
            "stage": "rank_selection",
            "split": split,
            "kind": kind,
            "identifier": "synthetic-evidence",
            "goal_bank_sha256": goals().for_tuning()[0].sha256,
        }
        with pytest.raises(ValueError):
            fit_source_basis(
                sources,
                goals(),
                flat_spec(selection_evidence=(evidence,)),
                task_split=task_split(),
                relations=TaskRelations(),
            )


def test_development_metrics_are_declared_but_never_actor_fitting_rows() -> None:
    evidence = {
        "stage": "rank_selection",
        "split": "development",
        "kind": "metrics",
        "identifier": "a" * 64,
        "goal_bank_sha256": goals().for_tuning()[0].sha256,
    }
    artifact = fit(np.array([[-1.0, 0.0], [1.0, 0.0]]), flat_spec(selection_evidence=(evidence,)))
    assert artifact.metadata["source_rows"] == 2
    assert artifact.metadata["data_access"]["rank_reference_selection"] == [evidence]
    assert artifact.metadata["data_access"]["target_weights_teacher_actions"] == "not accessed"


def test_mixed_libraries_conditions_and_bad_initialization_hashes_fail_closed() -> None:
    sources = policies(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    for changed in (
        replace(sources[1], library_id="different-independent-library"),
        replace(sources[1], initialization_condition="independent_init"),
    ):
        with pytest.raises(ValueError, match="one independent library"):
            fit_source_basis(
                [sources[0], changed],
                goals(),
                flat_spec(),
                task_split=task_split(),
                relations=TaskRelations(),
            )
    with pytest.raises(ValueError, match="effective initialization hash"):
        replace(sources[0], initial_actor_sha256="0" * 64)
    distinct = np.array([1.0, 2.0])
    changed = replace(
        sources[1],
        initial_vector=distinct,
        initial_actor_sha256=actor_hash_from_vector(distinct, sources[1].layout, "float64"),
    )
    with pytest.raises(ValueError, match="shared initialization"):
        fit_source_basis(
            [sources[0], changed],
            goals(),
            flat_spec(),
            task_split=task_split(),
            relations=TaskRelations(),
        )
    independent = policies(np.array([[-1.0, 0.0], [1.0, 0.0]]), independent=True)
    assert (
        fit_source_basis(
            independent, goals(), flat_spec(), task_split=task_split(), relations=TaskRelations()
        ).metadata["initialization_condition"]
        == "independent_init"
    )


def test_checkpoints_are_basis_rows_not_independent_library_replicates() -> None:
    sources = policies(np.array([[-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    sources[2] = replace(
        sources[2],
        training_run_id=sources[0].training_run_id,
        rng_streams=sources[0].rng_streams,
        checkpoint_step=200,
    )
    with pytest.raises(ValueError, match="repeated checkpoints"):
        fit_source_basis(
            sources, goals(), flat_spec(), task_split=task_split(), relations=TaskRelations()
        )
    artifact = fit_source_basis(
        sources,
        goals(),
        flat_spec(allow_repeated_checkpoints=True),
        task_split=task_split(),
        relations=TaskRelations(),
    )
    assert artifact.metadata["source_rows"] == 3
    assert artifact.metadata["source_training_run_count"] == 2
    assert artifact.metadata["independent_library_replicates"] == 1
    sources[2] = replace(sources[2], checkpoint_step=100)
    with pytest.raises(ValueError, match="duplicate checkpoint"):
        fit_source_basis(
            sources,
            goals(),
            flat_spec(allow_repeated_checkpoints=True),
            task_split=task_split(),
            relations=TaskRelations(),
        )


def test_flat_centered_rank_bound_zero_rank_and_spectrum_selection() -> None:
    rows = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="N-1"):
        fit(rows, flat_spec(2))
    with pytest.raises(ValueError, match="numerical source rank"):
        fit(np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]]), flat_spec(2))
    zero = fit(np.zeros((2, 3)), flat_spec(0))
    assert zero.coefficient_count == 0
    np.testing.assert_array_equal(zero.decode([]), np.zeros(3))
    spec = BasisFitSpec(
        "flat_population_pca",
        "final_weights",
        "raw",
        "source_explained_variance",
        variance_threshold=0.95,
    )
    assert fit(rows, spec).coefficient_count == 1
    for estimator in ("unspecified", "hosvd_over_model_axis"):
        with pytest.raises(ValueError, match="explicit supported estimator"):
            fit(rows, replace(spec, estimator=estimator))


def test_common_base_deltas_store_one_real_base_and_reject_per_initialization_offsets() -> None:
    rows = np.array([[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    initial = np.array([0.125, 0.5])
    sources = policies(rows, initial=initial)
    spec = replace(
        flat_spec(), representation="common_base_deltas", common_base_policy_id=sources[0].policy_id
    )
    artifact = fit_source_basis(
        sources, goals(), spec, task_split=task_split(), relations=TaskRelations()
    )
    final_weight_artifact = fit_source_basis(
        sources, goals(), flat_spec(), task_split=task_split(), relations=TaskRelations()
    )
    np.testing.assert_allclose(artifact.reference, rows.mean(axis=0), atol=1e-12)
    # A common base cancels under elementwise population centering. This is
    # bookkeeping/provenance, not a new flat-PCA initialization intervention.
    for coordinates in ([0.0], [1.0], [-2.0]):
        np.testing.assert_allclose(
            artifact.decode(coordinates), final_weight_artifact.decode(coordinates), atol=1e-12
        )
    assert "cancels" in artifact.metadata["representation_note"]
    np.testing.assert_allclose(
        artifact.oracle_project(np.array([2.0, 0.0])).reconstructed, [2.0, 0.0], atol=1e-12
    )
    with pytest.raises(ValueError, match="single deployable affine reference"):
        fit_source_basis(
            sources,
            goals(),
            replace(spec, representation="per_initialization_deltas"),
            task_split=task_split(),
            relations=TaskRelations(),
        )


def tensor_fixture() -> tuple[list[SourcePolicy], BasisFitSpec]:
    direction = np.array([1.0, -1.0, 2.0])
    mean = np.array([1.0, 2.0, 3.0])[:, None]
    cores = np.array([[1.0, -1.0, 2.0, -2.0], [-2.0, 0.0, 1.0, 1.0], [1.0, 1.0, -3.0, 1.0]])
    rows = np.stack(
        [
            np.concatenate([(mean + direction[:, None] * core).reshape(-1), [index, -index]])
            for index, core in enumerate(cores)
        ]
    )
    layout = [
        {"name": "matrix", "shape": [3, 4], "start": 0, "end": 12},
        {"name": "bias", "shape": [2], "start": 12, "end": 14},
    ]
    spec = BasisFitSpec(
        "uwsh_v3_parameter_hosvd",
        "final_weights",
        "raw",
        "fixed_predeclared",
        parameter_modes={"matrix": (0,), "bias": ()},
        mode_ranks={"matrix": (1,), "bias": ()},
    )
    return policies(rows, layout=layout), spec


def test_uwsh_v3_centers_nonparameter_modes_and_counts_all_uncompressed_coefficients() -> None:
    sources, spec = tensor_fixture()
    artifact = fit_source_basis(
        sources, goals(), spec, task_split=task_split(), relations=TaskRelations()
    )
    assert artifact.coefficient_count == 6  # 1x4 core plus both uncompressed bias scalars.
    assert artifact.metadata["model_axis_compressed"] is False
    matrix, bias = artifact.metadata["blocks"]
    assert matrix["centering_stacked_axes"] == [0, 2]
    assert matrix["compressed_stacked_axes"] == [1]
    assert matrix["core_shape"] == [1, 4]
    assert bias["core_shape"] == [2]
    assert artifact.metadata["storage"]["deployment_shared_scalars"] == 17
    target = np.concatenate(
        [
            (
                np.array([1.0, 2.0, 3.0])[:, None]
                + np.array([1.0, -1.0, 2.0])[:, None] * np.array([3.0, 4.0, 1.0, -2.0])
            ).reshape(-1),
            [7.0, 8.0],
        ]
    )
    oracle = artifact.oracle_project(target)
    np.testing.assert_allclose(oracle.reconstructed, target, atol=1e-12)
    np.testing.assert_allclose(artifact.decode(oracle.coefficients), target, atol=1e-12)
    flat = fit_source_basis(
        sources, goals(), flat_spec(), task_split=task_split(), relations=TaskRelations()
    )
    assert flat.coefficient_count == 1
    assert flat.metadata["estimator_revision"] != artifact.metadata["estimator_revision"]
    assert (
        fit_source_basis(
            sources,
            goals(),
            replace(
                spec, rank_rule="source_explained_variance", mode_ranks={}, variance_threshold=1.0
            ),
            task_split=task_split(),
            relations=TaskRelations(),
        ).coefficient_count
        == 6
    )


def test_parameter_mode_rank_can_exceed_number_of_models_minus_one() -> None:
    sources, spec = tensor_fixture()
    matrix = np.eye(3, 4)
    rows = np.stack([np.concatenate([sign * matrix.reshape(-1), np.zeros(2)]) for sign in (-1, 1)])
    two_models = policies(rows, layout=sources[0].layout)
    artifact = fit_source_basis(
        two_models,
        goals(),
        replace(spec, mode_ranks={"matrix": (2,), "bias": ()}),
        task_split=task_split(),
        relations=TaskRelations(),
    )
    assert artifact.metadata["source_rows"] == 2
    assert artifact.metadata["blocks"][0]["core_shape"] == [2, 4]
    assert artifact.coefficient_count == 10


def test_hosvd_all_parameter_axes_use_elementwise_model_mean() -> None:
    sources, spec = tensor_fixture()
    both = replace(
        spec,
        parameter_modes={"matrix": (0, 1), "bias": (0,)},
        mode_ranks={"matrix": (1, 2), "bias": (1,)},
    )
    artifact = fit_source_basis(
        sources, goals(), both, task_split=task_split(), relations=TaskRelations()
    )
    assert artifact.metadata["blocks"][0]["centering_stacked_axes"] == [0]
    np.testing.assert_allclose(
        artifact.reference, np.stack([source.vector for source in sources]).mean(axis=0), atol=1e-12
    )
    for source in sources:
        np.testing.assert_allclose(
            artifact.oracle_project(source.vector).reconstructed, source.vector, atol=1e-12
        )
    with pytest.raises(ValueError, match="never the model axis"):
        fit_source_basis(
            sources,
            goals(),
            replace(spec, parameter_modes={"matrix": (-1,), "bias": ()}),
            task_split=task_split(),
            relations=TaskRelations(),
        )


def test_basis_roundtrip_rejects_tampering_and_preserves_actual_affine_family(
    tmp_path: Path,
) -> None:
    sources, spec = tensor_fixture()
    for index, selected in enumerate((spec, flat_spec())):
        artifact = fit_source_basis(
            sources, goals(), selected, task_split=task_split(), relations=TaskRelations()
        )
        path = tmp_path / f"synthetic-{index}.npz"
        saved = artifact.save(path)
        assert saved["archive_bytes"] == path.stat().st_size
        assert saved["artifact_sha256"] == artifact.sha256
        restored = BasisArtifact.load(path)
        assert restored.sha256 == artifact.sha256
        assert restored.family_sha256 == artifact.family_sha256
        coordinates = np.arange(artifact.coefficient_count, dtype=np.float64)
        np.testing.assert_array_equal(restored.decode(coordinates), artifact.decode(coordinates))
        with pytest.raises(FileExistsError):
            artifact.save(path)
        with np.load(path, allow_pickle=False) as saved:
            entries = {name: saved[name] for name in saved.files}
        entries["reference"] = entries["reference"].copy() + 1
        corrupted = tmp_path / f"corrupt-{index}.npz"
        np.savez(corrupted, **entries)
        with pytest.raises(ValueError, match=r"hash mismatch|affine reference"):
            BasisArtifact.load(corrupted)


def test_actor_npz_adapter_matches_existing_geometry_schema_and_effective_hashes(
    tmp_path: Path,
) -> None:
    actor = TinyActor()
    vector, layout = actor_vector(actor)
    sources = policies(np.stack([vector, vector]), layout=layout, initial=vector, dtype="float32")
    assert sources[0].actor_sha256 == hash_actor(actor)
    for name in ("actor", "initial"):
        np.savez_compressed(
            tmp_path / f"{name}.npz", vector=vector, layout_json=np.asarray(json.dumps(layout))
        )
    metadata = sources[0].metadata()
    for name in ("layout", "raw_vector_sha256", "initial_vector_sha256"):
        metadata.pop(name)
    restored = SourcePolicy.from_actor_npz(
        tmp_path / "actor.npz", tmp_path / "initial.npz", metadata=metadata
    )
    np.testing.assert_array_equal(restored.vector, vector)
    assert restored.initial_actor_sha256 == hash_actor(actor)
    with pytest.raises(ValueError, match="source-only provenance first"):
        SourcePolicy.from_actor_npz(
            Path("must-not-open"),
            Path("must-not-open"),
            metadata=metadata | {"training_split": "final_test"},
        )


class TinyActor(torch.nn.Module):
    def __init__(self, activation: str = "tanh") -> None:
        super().__init__()
        self.latent_pi = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Tanh() if activation == "tanh" else torch.nn.ReLU(),
            torch.nn.Linear(4, 3),
            torch.nn.Tanh() if activation == "tanh" else torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(3, 2)
        self.log_std = torch.nn.Linear(3, 2)

    def outputs(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.latent_pi(observations)
        return self.mu(hidden), self.log_std(hidden)


@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_exact_permutations_and_source_alignment_preserve_mu_and_log_std(activation: str) -> None:
    torch.manual_seed(91)
    actor = TinyActor(activation)
    vector, layout = actor_vector(actor)
    permutation = [[2, 0, 3, 1], [2, 0, 1]]
    permuted = apply_sac_permutations(vector, layout, permutation)
    observations = torch.linspace(-1.0, 1.0, 33).reshape(11, 3)
    before = actor.outputs(observations)
    load_actor_vector(actor, permuted, layout)
    after = actor.outputs(observations)
    for left, right in zip(before, after, strict=True):
        torch.testing.assert_close(left, right, atol=1e-6, rtol=0)
    numpy_outputs = distribution_outputs(
        permuted, layout, observations.numpy().astype(np.float64), activation
    )
    for expected, actual in zip(after, numpy_outputs, strict=True):
        np.testing.assert_allclose(expected.detach().numpy(), actual, atol=1e-6)
    modified = vector.copy()
    mu_bias = next(item for item in layout if item["name"] == "mu.bias")
    modified[mu_bias["start"] : mu_bias["end"]] += np.float32(0.125)
    sources = policies(
        np.stack([vector, modified, apply_sac_permutations(modified, layout, permutation)]),
        layout=layout,
        initial=vector,
        dtype="float32",
    )
    raw_copy = [source.vector.copy() for source in sources]
    raw = fit_source_basis(
        sources, goals(), flat_spec(), task_split=task_split(), relations=TaskRelations()
    )
    aligned = fit_source_basis(
        sources,
        goals(),
        replace(
            flat_spec(),
            coordinate_condition="source_aligned",
            alignment_anchor_policy_id=sources[0].policy_id,
            alignment_activation=activation,
            alignment_probe_seed=37,
        ),
        task_split=task_split(),
        relations=TaskRelations(),
    )
    assert raw.family_sha256 != aligned.family_sha256
    assert not raw.metadata["alignment"]
    for source, original, report in zip(
        sources, raw_copy, aligned.metadata["alignment"], strict=True
    ):
        np.testing.assert_array_equal(source.vector, original)
        transformed = apply_sac_permutations(source.vector, layout, report["permutations"])
        original_outputs = distribution_outputs(
            source.vector, layout, observations.numpy().astype(np.float64), activation
        )
        aligned_outputs = distribution_outputs(
            transformed, layout, observations.numpy().astype(np.float64), activation
        )
        for original_array, aligned_array in zip(original_outputs, aligned_outputs, strict=True):
            np.testing.assert_allclose(original_array, aligned_array, atol=1e-12)
        assert report["max_abs_mu_error"] < 1e-8
        assert report["max_abs_log_std_error"] < 1e-8
        assert report["aligned_vector_sha256"]
    with pytest.raises(ValueError, match="remaining source"):
        fit_source_basis(
            sources,
            goals(),
            replace(
                flat_spec(),
                coordinate_condition="source_aligned",
                alignment_anchor_policy_id=sources[2].policy_id,
                alignment_activation=activation,
            ),
            excluded_policy_ids=(sources[2].policy_id,),
            task_split=task_split(),
            relations=TaskRelations(),
        )


def test_task_classes_skill_groups_and_near_far_are_separate_explicit_declarations() -> None:
    taxonomy = TaskRelations(
        {"synthetic_articulated_skill": ("open-v3", "close-v3")},
        (("open-v3", "close-v3"),),
        (("open-v3", "other-v3"),),
        "synthetic-unit-test-only; not a project task-panel decision",
    )
    taxonomy.validate({"open-v3", "close-v3", "other-v3"})
    assert taxonomy.relationship("open-v3", "close-v3") == "declared_near"
    assert taxonomy.relationship("open-v3", "other-v3") == "declared_far"
    assert taxonomy.relationship("open-v3", "open-v3") == "same_benchmark_class"
    with pytest.raises(ValueError, match="explicit prospective decision"):
        replace(taxonomy, decision_reference=None).validate({"open-v3", "close-v3", "other-v3"})
    with pytest.raises(ValueError, match="no near/far"):
        TaskRelations().relationship("open-v3", "close-v3")
    with pytest.raises(ValueError, match="conflicting"):
        replace(taxonomy, far_pairs=taxonomy.near_pairs).validate({"open-v3", "close-v3"})


def test_input_and_returned_metadata_mutations_cannot_change_frozen_artifact() -> None:
    rows = np.array([[-1.0, 0.0], [1.0, 0.0]])
    artifact = fit(rows)
    before = artifact.sha256
    returned = artifact.metadata
    returned["source_policy_ids"].clear()
    reference = artifact.reference
    reference[:] = 123
    rows[:] = 999
    assert artifact.sha256 == before
    with pytest.raises(ValueError, match="finite coefficient"):
        artifact.decode([np.nan])
    with pytest.raises(ValueError, match="dimensions differ"):
        artifact.oracle_project(np.zeros(3))
    with pytest.raises(ValueError, match="real"):
        artifact.decode([1.0j])


@pytest.mark.parametrize("damage", ["rng", "layout", "vector"])
def test_mutated_source_records_cannot_bypass_fitting_validation(damage: str) -> None:
    sources = policies(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    if damage == "rng":
        sources[0].rng_streams["training"] += 1
    elif damage == "layout":
        sources[0].layout[0]["name"] = "undeclared_parameter"
    else:
        sources[0].vector.flags.writeable = True
        sources[0].vector[0] += 1
    with pytest.raises(ValueError, match="changed after declaration"):
        fit_source_basis(
            sources, goals(), flat_spec(), task_split=task_split(), relations=TaskRelations()
        )


def test_mutated_specification_is_rejected_and_no_alignment_anchor_leaks_from_holdout() -> None:
    sources, spec = tensor_fixture()
    spec.parameter_modes["matrix"] = (1,)
    with pytest.raises(ValueError, match="specification changed"):
        fit_source_basis(sources, goals(), spec, task_split=task_split(), relations=TaskRelations())


def test_declared_development_selection_does_not_accept_excluded_source_weights() -> None:
    sources = policies(np.array([[-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    evidence = {
        "stage": "rank_selection",
        "split": "source",
        "kind": "source_weights",
        "identifier": sources[2].policy_id,
        "goal_bank_sha256": sources[2].training_goal_bank_sha256,
    }
    with pytest.raises(ValueError, match="excluded or target weights"):
        fit_source_basis(
            sources,
            goals(),
            flat_spec(selection_evidence=(evidence,)),
            excluded_policy_ids=(sources[2].policy_id,),
            task_split=task_split(),
            relations=TaskRelations(),
        )


def test_task_class_roles_are_separate_from_source_own_development_goals() -> None:
    sources = policies(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    classes = TaskSplitManifest(
        ("synthetic-v3",),
        "heldout_task_classes",
        "synthetic-role-declaration-only",
        development_target_classes=("development-target-v3",),
        final_target_classes=("final-target-v3",),
    )
    role_goals = GoalManifest(
        (*goals().banks, goal_bank("development", 30, "development-target-v3"))
    )
    artifact = fit_source_basis(
        sources, role_goals, flat_spec(), task_split=classes, relations=TaskRelations()
    )
    assert artifact.metadata["task_split_manifest_sha256"] == classes.sha256
    assert artifact.metadata["task_role_access"]["source_own_development_goals_allowed"] is True
    target = replace(sources[0], task_class="final-target-v3")
    with pytest.raises(ValueError, match="task-class roles"):
        fit_source_basis(
            [target, sources[1]],
            role_goals,
            flat_spec(),
            task_split=classes,
            excluded_policy_ids=(target.policy_id,),
            relations=TaskRelations(),
        )
    mislabeled_final = GoalManifest(
        (*goals().banks, goal_bank("development", 40, "final-target-v3"))
    )
    with pytest.raises(ValueError, match="task-class roles"):
        fit_source_basis(
            sources, mislabeled_final, flat_spec(), task_split=classes, relations=TaskRelations()
        )
    with pytest.raises(ValueError, match="held-out task classes"):
        replace(classes, development_target_classes=("synthetic-v3",))


def test_within_class_final_role_and_sealed_final_commitment_need_no_final_payloads() -> None:
    sources = policies(np.array([[-1.0, 0.0], [1.0, 0.0]]))
    within = TaskSplitManifest(
        ("synthetic-v3",),
        "within_class_goals",
        "synthetic-within-class-only",
        final_target_classes=("synthetic-v3",),
    )
    within_artifact = fit_source_basis(
        sources, goals(), flat_spec(), task_split=within, relations=TaskRelations()
    )
    assert (
        within_artifact.metadata["task_role_access"]["named_final_class_overlap_locally_checked"]
        is True
    )
    sealed = TaskSplitManifest(
        ("synthetic-v3",),
        "heldout_task_classes",
        "synthetic-sealed-role-only",
        sealed_final_commitment_sha256="c" * 64,
    )
    artifact = fit_source_basis(
        sources, goals(), flat_spec(), task_split=sealed, relations=TaskRelations()
    )
    assert artifact.metadata["task_split_manifest"]["sealed_final_commitment_sha256"] == "c" * 64
    assert (
        artifact.metadata["task_role_access"]["named_final_class_overlap_locally_checked"] is False
    )
    assert (
        "not locally inspectable"
        in artifact.metadata["task_role_access"]["sealed_or_unassigned_final_overlap"]
    )
    with pytest.raises(ValueError, match="silent typos"):
        fit_source_basis(
            sources,
            goals(),
            flat_spec(),
            task_split=within,
            excluded_policy_ids=("misspelled-policy",),
            relations=TaskRelations(),
        )
