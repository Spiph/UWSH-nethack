from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from scripts.metaworld_adaptation_contract import (
    InteractionLimitExceeded,
    RandomControlFamily,
    TargetInteractionLedger,
)
from scripts.metaworld_basis import (
    BasisArtifact,
    BasisFitSpec,
    SourcePolicy,
    TaskRelations,
    TaskSplitManifest,
    actor_hash_from_vector,
    fit_source_basis,
)
from scripts.metaworld_protocol import Goal, GoalBank, GoalManifest, SeedStreams

Array = npt.NDArray[np.float64]


def goal_manifest() -> GoalManifest:
    return GoalManifest(
        (
            GoalBank("source", 0, (Goal("synthetic-v3", "eA==", (0.0,)),)),
            GoalBank("development", 1, (Goal("synthetic-v3", "eA==", (1.0,)),)),
        )
    )


def fit(
    rows: Array, spec: BasisFitSpec, layout: list[dict[str, Any]] | None = None
) -> BasisArtifact:
    layout = layout or [
        {"name": "weight", "shape": [rows.shape[1]], "start": 0, "end": rows.shape[1]}
    ]
    initial = np.zeros(rows.shape[1], dtype=np.float64)
    goals = goal_manifest()
    sources = [
        SourcePolicy(
            policy_id=f"synthetic-policy-{index}",
            training_run_id=f"synthetic-run-{index}",
            checkpoint_step=1,
            task_class="synthetic-v3",
            library_id="synthetic-library",
            initialization_condition="shared_init",
            training_goal_bank_sha256=goals.for_fitting()[0].sha256,
            actor_sha256=actor_hash_from_vector(row, layout, "float64"),
            initial_actor_sha256=actor_hash_from_vector(initial, layout, "float64"),
            rng_streams=SeedStreams.from_seed(100 + index, actor_seed=777).__dict__,
            vector=row,
            initial_vector=initial,
            layout=layout,
            parameter_dtype="float64",
        )
        for index, row in enumerate(rows)
    ]
    return fit_source_basis(
        sources,
        goals,
        spec,
        relations=TaskRelations(),
        task_split=TaskSplitManifest(("synthetic-v3",), "development_only", "synthetic-test"),
    )


def flat(rank: int = 2, *, normalized: bool = True) -> BasisArtifact:
    rows = np.random.default_rng(10).normal(size=(5, 7)) * np.array([1, 10, 3, 40, 5, 60, 7])
    return fit(
        rows,
        BasisFitSpec(
            "flat_population_pca",
            "final_weights",
            "raw",
            "fixed_predeclared",
            rank=rank,
            normalization="source_coordinate_std" if normalized else "none",
        ),
    )


def tensor(modes: tuple[int, ...] = (0, 1), ranks: tuple[int, ...] = (2, 1)) -> BasisArtifact:
    rows = np.random.default_rng(11).normal(size=(5, 8))
    layout = [
        {"name": "matrix", "shape": [2, 3], "start": 0, "end": 6},
        {"name": "bias", "shape": [2], "start": 6, "end": 8},
    ]
    return fit(
        rows,
        BasisFitSpec(
            "uwsh_v3_parameter_hosvd",
            "final_weights",
            "raw",
            "fixed_predeclared",
            parameter_modes={"matrix": modes, "bias": ()},
            mode_ranks={"matrix": ranks, "bias": ()},
        ),
        layout,
    )


def assert_rng_unchanged(before: Any) -> None:
    after: Any = np.random.get_state()
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_scaled_flat_matches_whole_row_gram_and_preserves_local_rng_and_reference() -> None:
    parent = flat()
    directions = parent.deployment_arrays()["flat_directions"]
    gram = directions @ directions.T
    assert abs(gram[0, 1]) > 1  # Norm-only row matching would miss this covariance.
    before = np.random.get_state()
    control = RandomControlFamily.from_basis(parent, seed=51)
    assert_rng_unchanged(before)
    random = control.deployment_arrays()["flat_directions"]
    np.testing.assert_allclose(random @ random.T, gram, rtol=1e-12, atol=1e-10)
    assert not np.allclose(random, directions)
    np.testing.assert_array_equal(
        control.decode(np.zeros(control.coefficient_count)), parent.reference
    )
    assert control.metadata["estimator"] == "matched_random_flat"
    assert control.metadata["parent_artifact_sha256"] == parent.sha256
    assert control.metadata["parent_family_sha256"] == parent.family_sha256
    assert "spec" not in control.metadata
    assert (
        control.metadata["source_coordinate_condition"]
        == parent.metadata["spec"]["coordinate_condition"]
    )
    assert control.metadata["estimator_revision"] != parent.metadata["estimator_revision"]
    assert control.metadata["control"]["orientation_null_informative"]
    assert RandomControlFamily.from_basis(parent, seed=51).sha256 == control.sha256
    assert RandomControlFamily.from_basis(parent, seed=52).family_sha256 != control.family_sha256


def test_constant_flat_control_uses_no_random_generator(monkeypatch: pytest.MonkeyPatch) -> None:
    parent = flat(rank=0)
    monkeypatch.setattr(
        np.random, "default_rng", lambda *args: pytest.fail("constant family sampled RNG")
    )
    control = RandomControlFamily.from_basis(parent, seed=0)
    assert control.coefficient_count == 0
    assert control.family_sha256 == parent.family_sha256
    np.testing.assert_array_equal(control.decode([]), parent.reference)
    assert control.metadata["control"]["constant_family"]
    assert not control.metadata["control"]["orientation_null_informative"]


def test_parameter_factor_control_keeps_core_bias_counts_and_uses_structured_flags() -> None:
    parent = tensor()
    before = np.random.get_state()
    control = RandomControlFamily.from_basis(parent, seed=17)
    assert_rng_unchanged(before)
    assert control.metadata["estimator"] == "matched_random_parameter_factors"
    assert control.metadata["blocks"] == parent.metadata["blocks"]
    assert control.coefficient_count == parent.coefficient_count == 4
    arrays = control.deployment_arrays()
    for key in control.metadata["deployment_basis_keys"]:
        factor = arrays[key]
        np.testing.assert_allclose(factor.T @ factor, np.eye(factor.shape[1]), atol=1e-14)
    flags = control.metadata["control"]
    assert flags["structure"] == "structured_parameter_factor_control"
    assert flags["unstructured_global_orientation_null"] is False
    assert flags["factors"][0]["full_rank"] is True
    assert flags["blocks"][0]["effectively_same_block_space"] is False
    assert flags["blocks"][1]["effectively_same_block_space"] is True  # Uncompressed biases.
    assert flags["all_blocks_coordinate_only"] is False
    coefficients = np.array([0.0, 0.0, 0.5, -0.2])
    decoded = control.decode(coefficients)
    np.testing.assert_array_equal(decoded[:6], parent.reference[:6])
    np.testing.assert_allclose(decoded[6:], parent.reference[6:] + np.array([0.5, -0.2]))
    expected_scalars = sum(value.size for value in arrays.values())
    assert control.metadata["storage"]["deployment_shared_bytes"] == expected_scalars * 8
    assert "flat_directions" not in arrays  # No expanded K x P parameter map.


def test_full_rank_factors_are_flagged_as_coordinate_only_control() -> None:
    control = RandomControlFamily.from_basis(tensor(ranks=(2, 3)), seed=1)
    flags = control.metadata["control"]
    assert flags["all_blocks_coordinate_only"] is True
    assert flags["orientation_null_informative"] is False
    assert all(row["full_rank"] for row in flags["factors"])


def test_zero_core_and_uncompressed_bias_control_never_draws_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parent = tensor(ranks=(0, 1))
    monkeypatch.setattr(np.random, "default_rng", lambda *args: pytest.fail("zero core drew RNG"))
    control = RandomControlFamily.from_basis(parent, seed=8)
    assert control.coefficient_count == 2
    assert control.family_sha256 == parent.family_sha256
    assert control.metadata["control"]["blocks"][0]["zero_core"]
    assert control.metadata["control"]["all_blocks_coordinate_only"]
    np.testing.assert_array_equal(control.decode([0.0, 0.0]), parent.reference)


def test_control_roundtrip_requires_exact_parent_and_honest_seed_provenance(tmp_path: Path) -> None:
    parent = flat()
    control = RandomControlFamily.from_basis(parent, seed=3)
    path = tmp_path / "control.npz"
    control.save(path)
    loaded = RandomControlFamily.load(path, parent=parent)
    assert loaded.sha256 == control.sha256
    np.testing.assert_array_equal(loaded.decode([0.1, -0.2]), control.decode([0.1, -0.2]))
    with pytest.raises(FileExistsError):
        control.save(path)
    with pytest.raises(ValueError, match="provenance"):
        RandomControlFamily.load(path, parent=flat(normalized=False))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    envelope = json.loads(str(arrays["metadata_json"].item()))
    envelope["metadata"]["estimator"] = "flat_population_pca"
    arrays["metadata_json"] = np.asarray(json.dumps(envelope))
    np.savez_compressed(tmp_path / "forged.npz", **arrays)
    with pytest.raises(ValueError, match="provenance"):
        RandomControlFamily.load(tmp_path / "forged.npz", parent=parent)


def test_returned_arrays_and_metadata_cannot_mutate_control() -> None:
    control = RandomControlFamily.from_basis(flat(), seed=9)
    committed = control.sha256
    control.deployment_arrays()["reference"][:] = 99
    control.metadata["storage"]["deployment_shared_bytes"] = -1
    assert control.sha256 == committed


def test_only_verified_parent_and_unsigned_control_seed_are_accepted() -> None:
    with pytest.raises(TypeError, match="BasisArtifact"):
        invalid: Any = np.zeros((3, 4))
        RandomControlFamily.from_basis(invalid, seed=1)
    parent = flat()
    for seed in (True, -1, 2**32):
        with pytest.raises(ValueError, match="seed"):
            RandomControlFamily.from_basis(parent, seed=seed)


def ledger(**overrides: int) -> TargetInteractionLedger:
    caps = dict(selection_cap=10, learning_cap=20, evaluation_cap=10, total_cap=30, max_restarts=1)
    return TargetInteractionLedger(**(caps | overrides))


def test_selection_warmup_learning_evaluation_axes_do_not_double_charge() -> None:
    costs = ledger()
    selection = costs.begin_attempt("selection", 3, label="source selection probe")
    warmup = costs.begin_attempt("learning", 5, warmup=True)
    learning = costs.begin_attempt("learning", 7)
    evaluation = costs.begin_attempt("evaluation", 4)
    for identifier in (selection, warmup, learning, evaluation):
        costs.complete_attempt(identifier)
    snapshot = costs.snapshot()
    assert snapshot["charged"] == {"selection": 3, "learning": 12, "evaluation": 4}
    assert snapshot["axes"] == {"learning_only": 12, "all_interactions": 19}
    assert snapshot["observed_axes"] == snapshot["axes"]
    assert snapshot["warmup_learning_subset"] == {"charged": 5, "observed": 5}
    assert all(not attempt["selection_replay_reuse"] for attempt in snapshot["attempts"])


def test_failed_and_pending_attempts_stay_charged_across_restart() -> None:
    costs = ledger()
    failed = costs.begin_attempt("selection", 5)
    costs.fail_attempt(failed, reason="backend failed", actual_steps=2)
    pending = costs.begin_attempt("learning", 4, warmup=True)
    costs.restart(reason="learner initialization failure")
    snapshot = costs.snapshot()
    assert snapshot["axes"] == {"learning_only": 4, "all_interactions": 9}
    assert snapshot["observed_axes"] == {"learning_only": 0, "all_interactions": 2}
    assert snapshot["attempts"][pending]["status"] == "interrupted"
    assert snapshot["unobserved_conservative_charge"] == 7
    assert snapshot["actual_steps_unknown_attempt_ids"] == [pending]
    assert snapshot["observed_axes_are_lower_bounds"] is True
    assert snapshot["warmup_learning_subset"]["charged"] == 4
    with pytest.raises(ValueError, match="resolved"):
        costs.complete_attempt(pending)
    with pytest.raises(InteractionLimitExceeded, match="restart"):
        costs.restart(reason="another failure")
    costs.begin_attempt("learning", 16)
    with pytest.raises(InteractionLimitExceeded):
        costs.begin_attempt("learning", 1)
    assert costs.learning_steps == 20


def test_total_cap_denies_work_before_attempt_and_snapshots_are_detached() -> None:
    costs = ledger(total_cap=5)
    costs.begin_attempt("selection", 3)
    saved = costs.snapshot()
    with pytest.raises(InteractionLimitExceeded):
        costs.begin_attempt("evaluation", 3)
    assert len(costs.snapshot()["attempts"]) == 1
    costs.begin_attempt("learning", 2)
    assert costs.total_steps == 5
    assert saved["axes"]["all_interactions"] == 3
    saved["charged"]["selection"] = 0
    assert costs.snapshot()["charged"]["selection"] == 3


def test_partial_failure_is_explicit_and_warmup_cannot_be_selection() -> None:
    costs = ledger()
    identifier = costs.begin_attempt("learning", 4)
    with pytest.raises(ValueError, match="complete"):
        costs.complete_attempt(identifier, actual_steps=3)
    costs.fail_attempt(identifier, reason="truncated attempt", actual_steps=3)
    assert costs.snapshot()["charged"]["learning"] == 4
    assert costs.snapshot()["observed"]["learning"] == 3
    with pytest.raises(ValueError, match="warmup"):
        costs.begin_attempt("selection", 1, warmup=True)
    with pytest.raises(ValueError, match="bounded"):
        unknown = costs.begin_attempt("evaluation", 1)
        costs.fail_attempt(unknown, reason="failure", actual_steps=2)


@pytest.mark.parametrize("cap", [-1, True, 1.5])
def test_caps_are_explicit_nonnegative_integers(cap: Any) -> None:
    with pytest.raises(ValueError, match="caps"):
        ledger(learning_cap=cap)


def test_full_rank_flat_control_reports_unchanged_parameter_span() -> None:
    parent = fit(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        BasisFitSpec("flat_population_pca", "final_weights", "raw", "fixed_predeclared", rank=2),
    )
    control = RandomControlFamily.from_basis(parent, seed=4)
    assert control.metadata["control"]["full_parameter_span"] is True
    assert control.metadata["control"]["orientation_null_informative"] is False
    assert control.coefficient_count == parent.coefficient_count == 2
