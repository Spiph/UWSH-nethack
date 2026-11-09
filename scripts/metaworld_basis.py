"""Offline source-only affine policy families, with explicit estimator provenance.

Parameter-mode conventions follow UWSH arXiv:2512.05117v3, Algorithm 1:
https://arxiv.org/html/2512.05117v3#alg1. Parameter axes are zero-based here;
the stacked model axis is estimation-only and is never compressed.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from scripts.metaworld_alignment import (
    Layout,
    align_sac_vectors,
    apply_sac_permutations,
    validate_layout,
)
from scripts.metaworld_protocol import GoalManifest, SeedStreams, stable_hash

Array = npt.NDArray[np.float64]
SCHEMA_VERSION = 1
ESTIMATOR_REVISIONS = {
    "flat_population_pca": "centered_population_pca_v1",
    "uwsh_v3_parameter_hosvd": "arxiv_2512.05117v3_algorithm1_parameter_axes_v1",
}


def _vector(value: Any, name: str) -> Array:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.array(value, dtype=np.float64, copy=True)
    if result.ndim != 1 or not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    result.flags.writeable = False
    return result


def _array_hash(array: Array) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def actor_hash_from_vector(vector: Array, layout: Layout, dtype: str) -> str:
    """Reproduce geometry.hash_actor from the saved vector and original dtype."""
    validate_layout(layout, vector.size)
    if dtype not in ("float32", "float64"):
        raise ValueError("the original actor parameter dtype must be float32 or float64")
    digest = hashlib.sha256()
    for item in layout:
        values = vector[item["start"] : item["end"]].astype(dtype)
        if not np.isfinite(values).all() or not np.array_equal(
            values.astype(np.float64), vector[item["start"] : item["end"]]
        ):
            raise ValueError(
                "saved actor vector does not exactly represent its declared parameter dtype"
            )
        digest.update(item["name"].encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SourcePolicy:
    policy_id: str
    training_run_id: str
    checkpoint_step: int
    task_class: str
    library_id: str
    initialization_condition: str
    training_goal_bank_sha256: str
    actor_sha256: str
    initial_actor_sha256: str
    rng_streams: dict[str, int]
    vector: Array
    initial_vector: Array
    layout: Layout
    parameter_dtype: str = "float32"
    training_split: str = "source"
    data_kind: str = "actor_weights"
    _commitment: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.training_split != "source" or self.data_kind != "actor_weights":
            raise ValueError(
                "fitting accepts source-trained actor weights, never targets or teachers"
            )
        for name in ("policy_id", "training_run_id", "task_class", "library_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"missing {name}")
        if type(self.checkpoint_step) is not int or self.checkpoint_step < 0:
            raise ValueError("checkpoint steps must be nonnegative integers")
        if self.initialization_condition not in ("shared_init", "independent_init"):
            raise ValueError("initialization condition must be explicitly shared or independent")
        SeedStreams(**self.rng_streams)
        vector = _vector(self.vector, "source actor")
        initial = _vector(self.initial_vector, "saved effective initial actor")
        if initial.shape != vector.shape:
            raise ValueError("initial and final actor shapes differ")
        layout = validate_layout(self.layout, vector.size)
        if actor_hash_from_vector(vector, layout, self.parameter_dtype) != self.actor_sha256:
            raise ValueError("source actor hash disagrees with its saved array")
        if (
            actor_hash_from_vector(initial, layout, self.parameter_dtype)
            != self.initial_actor_sha256
        ):
            raise ValueError("effective initialization hash disagrees with its saved array")
        object.__setattr__(self, "vector", vector)
        object.__setattr__(self, "initial_vector", initial)
        object.__setattr__(self, "layout", layout)
        object.__setattr__(self, "rng_streams", dict(self.rng_streams))
        object.__setattr__(self, "_commitment", stable_hash(self.metadata()))

    def validate_integrity(self) -> None:
        if stable_hash(self.metadata()) != self._commitment:
            raise ValueError("source policy arrays or provenance changed after declaration")
        SeedStreams(**self.rng_streams)

    def metadata(self) -> dict[str, Any]:
        return {
            name: copy.deepcopy(getattr(self, name))
            for name in self.__dataclass_fields__
            if name not in ("vector", "initial_vector", "_commitment")
        } | {
            "raw_vector_sha256": _array_hash(self.vector),
            "initial_vector_sha256": _array_hash(self.initial_vector),
        }

    @classmethod
    def from_actor_npz(
        cls, actor_path: Path, initial_path: Path, *, metadata: dict[str, Any]
    ) -> SourcePolicy:
        if (
            metadata.get("training_split") != "source"
            or metadata.get("data_kind") != "actor_weights"
        ):
            raise ValueError("actor NPZ access requires declared source-only provenance first")
        with (
            np.load(actor_path, allow_pickle=False) as actor,
            np.load(initial_path, allow_pickle=False) as initial,
        ):
            layout = json.loads(str(actor["layout_json"].item()))
            if json.loads(str(initial["layout_json"].item())) != layout:
                raise ValueError("initial/final NPZ parameter layouts differ")
            return cls(
                **metadata, vector=actor["vector"], initial_vector=initial["vector"], layout=layout
            )


@dataclass(frozen=True)
class TaskSplitManifest:
    """Task-class roles, distinct from source actors' within-class goal splits."""

    source_classes: tuple[str, ...]
    transfer_regime: str
    decision_reference: str
    development_target_classes: tuple[str, ...] = ()
    final_target_classes: tuple[str, ...] = ()
    sealed_final_commitment_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("source_classes", "development_target_classes", "final_target_classes"):
            values = getattr(self, name)
            if not isinstance(values, (tuple, list)):
                raise ValueError("task-class roles must be explicit finite sequences")
            object.__setattr__(self, name, tuple(values))
        self.validate()

    def validate(self) -> None:
        for values in (
            self.source_classes,
            self.development_target_classes,
            self.final_target_classes,
        ):
            if len(set(values)) != len(values) or any(
                not isinstance(value, str) or not value for value in values
            ):
                raise ValueError("task-class roles contain invalid or duplicate class IDs")
        if not self.source_classes or not self.decision_reference:
            raise ValueError("source classes and a prospective task-role decision are required")
        if self.sealed_final_commitment_sha256 is not None and (
            len(self.sealed_final_commitment_sha256) != 64
            or any(
                letter not in "0123456789abcdef" for letter in self.sealed_final_commitment_sha256
            )
        ):
            raise ValueError("sealed final task roles require a SHA256 commitment")
        source, development, final = map(
            set, (self.source_classes, self.development_target_classes, self.final_target_classes)
        )
        if self.transfer_regime == "development_only":
            if development or final or self.sealed_final_commitment_sha256:
                raise ValueError(
                    "development-only source calibration has no declared transfer targets"
                )
        elif self.transfer_regime == "within_class_goals":
            if not development.union(final).issubset(source):
                raise ValueError("within-class goal transfer requires known source task classes")
        elif self.transfer_regime == "heldout_task_classes":
            if source.intersection(development) or final.intersection(source.union(development)):
                raise ValueError(
                    "held-out task classes must exclude source/development-target classes"
                )
        else:
            raise ValueError("unknown task-class transfer regime")

    @property
    def sha256(self) -> str:
        return stable_hash(asdict(self))


@dataclass(frozen=True)
class TaskRelations:
    """User-supplied taxonomy; benchmark task classes never imply a skill split."""

    skill_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    near_pairs: tuple[tuple[str, str], ...] = ()
    far_pairs: tuple[tuple[str, str], ...] = ()
    decision_reference: str | None = None

    def validate(self, declared_classes: set[str]) -> None:
        near = {tuple(sorted(pair)) for pair in self.near_pairs}
        far = {tuple(sorted(pair)) for pair in self.far_pairs}
        classes = set().union(*map(set, self.skill_groups.values())) if self.skill_groups else set()
        for pair in (*self.near_pairs, *self.far_pairs):
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("near/far relations require two distinct benchmark task classes")
            classes.update(pair)
        if near.intersection(far) or not classes.issubset(declared_classes):
            raise ValueError("conflicting or undeclared task relationships")
        if (classes or near or far) and not self.decision_reference:
            raise ValueError(
                "task grouping/near-far rules require an explicit prospective decision"
            )

    def relationship(self, left: str, right: str) -> str:
        if left == right:
            return "same_benchmark_class"
        pair = tuple(sorted((left, right)))
        if pair in {tuple(sorted(value)) for value in self.near_pairs}:
            return "declared_near"
        if pair in {tuple(sorted(value)) for value in self.far_pairs}:
            return "declared_far"
        raise ValueError("no near/far relationship has been declared")


@dataclass(frozen=True)
class BasisFitSpec:
    estimator: str
    representation: str
    coordinate_condition: str
    rank_rule: str
    rank: int | None = None
    variance_threshold: float | None = None
    parameter_modes: dict[str, tuple[int, ...]] = field(default_factory=dict)
    mode_ranks: dict[str, tuple[int, ...]] = field(default_factory=dict)
    reference_rule: str = "estimator_center"
    reference_policy_id: str | None = None
    common_base_policy_id: str | None = None
    normalization: str = "none"
    alignment_anchor_policy_id: str | None = None
    alignment_activation: str | None = None
    alignment_iterations: int = 10
    alignment_probe_seed: int = 0
    allow_repeated_checkpoints: bool = False
    selection_evidence: tuple[dict[str, str], ...] = ()
    _commitment: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("parameter_modes", "mode_ranks", "selection_evidence"):
            object.__setattr__(self, name, copy.deepcopy(getattr(self, name)))
        object.__setattr__(self, "_commitment", stable_hash(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            key: copy.deepcopy(getattr(self, key))
            for key in self.__dataclass_fields__
            if key != "_commitment"
        }

    def validate(self) -> None:
        if stable_hash(self.to_dict()) != self._commitment:
            raise ValueError("basis specification changed after declaration")
        if self.estimator not in ESTIMATOR_REVISIONS:
            raise ValueError(
                "choose an explicit supported estimator; flat PCA is not parameter HOSVD"
            )
        if self.representation not in ("final_weights", "common_base_deltas"):
            raise ValueError("per-initialization deltas have no single deployable affine reference")
        if (self.representation == "common_base_deltas") != (
            self.common_base_policy_id is not None
        ):
            raise ValueError("common-base deltas require exactly one retained source initial actor")
        if self.coordinate_condition not in ("raw", "source_aligned"):
            raise ValueError("raw and source-aligned coordinates must be separate artifacts")
        if self.coordinate_condition == "source_aligned":
            if not self.alignment_anchor_policy_id or self.alignment_activation not in (
                "relu",
                "tanh",
            ):
                raise ValueError(
                    "alignment requires a source anchor and declared pointwise activation"
                )
        elif self.alignment_anchor_policy_id or self.alignment_activation:
            raise ValueError("raw artifacts cannot silently apply alignment")
        if self.reference_rule not in ("estimator_center", "source_policy"):
            raise ValueError("unknown affine reference rule")
        if (self.reference_rule == "source_policy") != (self.reference_policy_id is not None):
            raise ValueError("a source-anchored reference requires exactly one source policy ID")
        if self.normalization not in ("none", "source_coordinate_std"):
            raise ValueError("normalization must be an implemented source-only parameter transform")
        if self.estimator == "uwsh_v3_parameter_hosvd" and self.normalization != "none":
            raise ValueError(
                "the pinned Algorithm 1 comparator does not whiten parameter coordinates"
            )
        if self.rank_rule == "fixed_predeclared":
            if self.variance_threshold is not None:
                raise ValueError("fixed ranks cannot also select a source variance threshold")
            if self.estimator == "flat_population_pca" and (
                type(self.rank) is not int or self.rank < 0
            ):
                raise ValueError("flat PCA requires an explicit nonnegative integer rank")
        elif self.rank_rule == "source_explained_variance":
            if (
                self.rank is not None
                or self.mode_ranks
                or self.variance_threshold is None
                or not 0 < self.variance_threshold <= 1
            ):
                raise ValueError(
                    "source variance selection requires only a finite threshold in (0,1]"
                )
        else:
            raise ValueError("rank selection must be fixed or source-spectrum based")


def _canonical_columns(factor: Array) -> Array:
    result = factor.copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1
    return result


def _rank(
    singular: Array, shape: tuple[int, ...], requested: int | None, threshold: float | None
) -> int:
    tolerance = max(shape) * np.finfo(np.float64).eps * (singular[0] if singular.size else 0)
    available = int(np.count_nonzero(singular > tolerance))
    if threshold is not None:
        energy = singular**2
        return (
            min(int(np.searchsorted(np.cumsum(energy) / energy.sum(), threshold) + 1), available)
            if available
            else 0
        )
    if type(requested) is not int or not 0 <= requested <= available:
        raise ValueError(f"requested rank exceeds numerical source rank {available}")
    return requested


def _mode_product(tensor: Array, matrix: Array, mode: int) -> Array:
    return np.asarray(
        np.moveaxis(np.tensordot(matrix, tensor, axes=(1, mode)), 0, mode), dtype=np.float64
    )


@dataclass(frozen=True)
class OracleProjection:
    family_sha256: str
    coefficients: Array
    reconstructed: Array
    information_access: str = "privileged_target_weights; diagnostic_only"


class BasisArtifact:
    def __init__(self, metadata: dict[str, Any], arrays: dict[str, Array]):
        if any(np.iscomplexobj(value) for value in arrays.values()):
            raise ValueError("basis artifact arrays must be real")
        self._metadata = copy.deepcopy(metadata)
        self._arrays = {
            key: np.array(value, dtype=np.float64, copy=True) for key, value in arrays.items()
        }
        self._validate()
        for array in self._arrays.values():
            array.flags.writeable = False

    @property
    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    @property
    def reference(self) -> Array:
        return self._arrays["reference"].copy()

    @property
    def coefficient_count(self) -> int:
        return int(self._metadata["coefficient_count"])

    def _array_records(self) -> dict[str, Any]:
        return {
            key: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": _array_hash(value),
            }
            for key, value in self._arrays.items()
        }

    @property
    def sha256(self) -> str:
        return stable_hash({"metadata": self._metadata, "arrays": self._array_records()})

    @property
    def family_sha256(self) -> str:
        """Identify an exact coordinate map, not equivalence of affine spans."""
        keys = ["reference"] + self._metadata["deployment_basis_keys"]
        return stable_hash(
            {
                "layout": self._metadata["layout"],
                "blocks": self._metadata["blocks"],
                "arrays": {key: _array_hash(self._arrays[key]) for key in keys},
            }
        )

    def _validate(self) -> None:
        meta = self._metadata
        if (
            meta.get("schema_version") != SCHEMA_VERSION
            or meta.get("estimator") not in ESTIMATOR_REVISIONS
        ):
            raise ValueError("unknown basis artifact schema or estimator")
        if meta.get("estimator_revision") != ESTIMATOR_REVISIONS[meta["estimator"]]:
            raise ValueError("basis estimator revision mismatch")
        declared_spec = BasisFitSpec(**meta["spec"])
        declared_spec.validate()
        declared_tasks = TaskSplitManifest(**meta["task_split_manifest"])
        if declared_tasks.sha256 != meta["task_split_manifest_sha256"]:
            raise ValueError("task-class role manifest content/hash mismatch")
        if declared_spec.estimator != meta["estimator"]:
            raise ValueError("artifact and fitting specification name different estimators")
        reference = _vector(self._arrays["reference"], "affine reference")
        validate_layout(meta["layout"], reference.size)
        if type(meta["source_rows"]) is not int or meta["source_rows"] < 2:
            raise ValueError("basis fitting requires at least two source rows")
        if type(meta["coefficient_count"]) is not int or meta["coefficient_count"] < 0:
            raise ValueError("coefficient count must be a nonnegative integer")
        for key in ("estimator_center", "normalization_scale", "common_base"):
            if self._arrays[key].shape != reference.shape:
                raise ValueError("stored affine transforms differ from the full actor layout")
        if np.any(self._arrays["normalization_scale"] <= 0):
            raise ValueError("normalization scales must be positive")
        records = meta["source_policy_records"]
        if (
            len(records) != meta["source_rows"]
            or [row["policy_id"] for row in records] != meta["source_policy_ids"]
            or len(set(meta["source_policy_ids"])) != len(records)
            or meta["independent_library_replicates"] != 1
            or any(
                row["library_id"] != meta["library_id"]
                or row["initialization_condition"] != meta["initialization_condition"]
                or row["training_split"] != "source"
                or row["data_kind"] != "actor_weights"
                or row["layout"] != meta["layout"]
                or row["task_class"] not in declared_tasks.source_classes
                for row in records
            )
        ):
            raise ValueError("artifact source-library/condition/access provenance is inconsistent")
        if declared_spec.reference_policy_id is None:
            if not np.array_equal(reference, self._arrays["estimator_center"]):
                raise ValueError(
                    "deployed affine reference disagrees with its estimator-center rule"
                )
        else:
            index = meta["source_policy_ids"].index(declared_spec.reference_policy_id)
            reference_hash = (
                meta["alignment"][index]["aligned_vector_sha256"]
                if declared_spec.coordinate_condition == "source_aligned"
                else records[index]["raw_vector_sha256"]
            )
            if _array_hash(reference) != reference_hash:
                raise ValueError("deployed affine reference is not the declared source policy")
        for key, array in self._arrays.items():
            if not np.isfinite(array).all():
                raise ValueError(f"nonfinite artifact array: {key}")
        if meta["estimator"] == "flat_population_pca":
            directions = self._arrays["flat_directions"]
            if (
                directions.shape != (self.coefficient_count, reference.size)
                or self.coefficient_count > meta["source_rows"] - 1
            ):
                raise ValueError("invalid centered flat basis rank or dimensions")
        else:
            count = 0
            if len(meta["blocks"]) != len(meta["layout"]):
                raise ValueError("parameter-mode cores must cover every actor tensor")
            for block, item in zip(meta["blocks"], meta["layout"], strict=True):
                if any(block[key] != item[key] for key in ("name", "shape", "start", "end")):
                    raise ValueError("parameter-mode core blocks disagree with the actor layout")
                modes = block["parameter_modes"]
                if (
                    modes != sorted(set(modes))
                    or any(
                        type(axis) is not int or not 0 <= axis < len(item["shape"])
                        for axis in modes
                    )
                    or len(block["factor_keys"]) != len(modes)
                    or len(block["core_shape"]) != len(item["shape"])
                    or any(
                        type(rank) is not int
                        or not 0 <= rank <= size
                        or (axis not in modes and rank != size)
                        for axis, (rank, size) in enumerate(
                            zip(block["core_shape"], item["shape"], strict=True)
                        )
                    )
                    or block["compressed_stacked_axes"] != [axis + 1 for axis in modes]
                    or block["centering_stacked_axes"]
                    != [0, *(axis + 1 for axis in range(len(item["shape"])) if axis not in modes)]
                ):
                    raise ValueError("parameter-mode axes/core sizes violate Algorithm 1")
                count += math.prod(block["core_shape"])
                for axis, key in zip(block["parameter_modes"], block["factor_keys"], strict=True):
                    factor = self._arrays[key]
                    if factor.shape != (
                        block["shape"][axis],
                        block["core_shape"][axis],
                    ) or not np.allclose(factor.T @ factor, np.eye(factor.shape[1]), atol=1e-10):
                        raise ValueError("invalid orthonormal parameter-mode factors")
            if count != self.coefficient_count or meta.get("model_axis_compressed") is not False:
                raise ValueError(
                    "parameter cores must count all modes without compressing the model axis"
                )
        actual_scalars = sum(array.size for array in self._arrays.values())
        deployment_scalars = reference.size + sum(
            self._arrays[key].size for key in meta["deployment_basis_keys"]
        )
        if (
            meta["storage"]["artifact_array_scalars"] != actual_scalars
            or meta["storage"]["deployment_shared_scalars"] != deployment_scalars
            or meta["storage"]["artifact_array_bytes"] != 8 * actual_scalars
            or meta["storage"]["deployment_shared_bytes"] != 8 * deployment_scalars
            or meta["storage"]["per_target_coefficient_scalars"] != self.coefficient_count
            or meta["storage"]["per_target_coefficient_bytes_float64"] != 8 * self.coefficient_count
        ):
            raise ValueError("artifact storage accounting mismatch")

    def decode(self, coefficients: Any) -> Array:
        if np.iscomplexobj(coefficients):
            raise ValueError("policy coefficients must be real")
        coordinates = np.asarray(coefficients, dtype=np.float64)
        if coordinates.shape != (self.coefficient_count,) or not np.isfinite(coordinates).all():
            raise ValueError(
                "one finite coefficient is required for every declared degree of freedom"
            )
        result = self.reference
        if self._metadata["estimator"] == "flat_population_pca":
            result += coordinates @ self._arrays["flat_directions"]
        else:
            offset = 0
            for block in self._metadata["blocks"]:
                size = math.prod(block["core_shape"])
                core = coordinates[offset : offset + size].reshape(block["core_shape"])
                for axis, key in zip(block["parameter_modes"], block["factor_keys"], strict=True):
                    core = _mode_product(core, self._arrays[key], axis)
                result[block["start"] : block["end"]] += core.reshape(-1)
                offset += size
        return result

    def _coordinates(self, vector: Array) -> Array:
        target = _vector(vector, "projection weights")
        if target.shape != self._arrays["reference"].shape:
            raise ValueError("projection actor dimensions differ from the frozen family")
        delta = target - self._arrays["reference"]
        if self._metadata["estimator"] == "flat_population_pca":
            return np.asarray(
                np.linalg.lstsq(self._arrays["flat_directions"].T, delta, rcond=None)[0],
                dtype=np.float64,
            )
        cores = []
        for block in self._metadata["blocks"]:
            core = delta[block["start"] : block["end"]].reshape(block["shape"])
            for axis, key in zip(block["parameter_modes"], block["factor_keys"], strict=True):
                core = _mode_product(core, self._arrays[key].T, axis)
            cores.append(core.reshape(-1))
        return np.concatenate(cores)

    def oracle_project(self, target_weights: Array) -> OracleProjection:
        coordinates = self._coordinates(target_weights)
        return OracleProjection(self.family_sha256, coordinates, self.decode(coordinates))

    def save(self, path: Path) -> dict[str, Any]:
        if path.exists():
            raise FileExistsError("existing basis artifacts are preserved")
        envelope = {
            "metadata": self._metadata,
            "array_records": self._array_records(),
            "sha256": self.sha256,
            "family_sha256": self.family_sha256,
        }
        with path.open("xb") as handle:
            np.savez_compressed(
                handle,
                metadata_json=np.asarray(json.dumps(envelope, sort_keys=True, allow_nan=False)),
                **self._arrays,
            )
        return {
            "artifact_sha256": self.sha256,
            "family_sha256": self.family_sha256,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "archive_bytes": path.stat().st_size,
            "array_storage": copy.deepcopy(self._metadata["storage"]),
        }

    @classmethod
    def load(cls, path: Path) -> BasisArtifact:
        with np.load(path, allow_pickle=False) as archive:
            envelope = json.loads(str(archive["metadata_json"].item()))
            if set(archive.files) != {"metadata_json", *envelope["array_records"]}:
                raise ValueError("basis archive contains missing or undeclared arrays")
            result = cls(
                envelope["metadata"], {key: archive[key] for key in envelope["array_records"]}
            )
        if (
            result._array_records() != envelope["array_records"]
            or result.sha256 != envelope["sha256"]
            or result.family_sha256 != envelope["family_sha256"]
        ):
            raise ValueError("basis artifact content/hash mismatch")
        return result


def _validate_selection(
    spec: BasisFitSpec,
    goals: GoalManifest,
    policies: list[SourcePolicy],
    excluded_classes: tuple[str, ...],
) -> None:
    source_ids = {policy.policy_id for policy in policies}
    source_banks = {policy.training_goal_bank_sha256 for policy in policies}
    banks = {
        bank.sha256: bank.split
        for bank in goals.banks
        if bank.goals[0].task not in excluded_classes
        and (bank.split == "development" or bank.sha256 in source_banks)
    }
    for evidence in spec.selection_evidence:
        if set(evidence) != {"stage", "split", "kind", "identifier", "goal_bank_sha256"}:
            raise ValueError("selection evidence must state stage, split, kind, ID and goal bank")
        if evidence["stage"] not in ("rank_selection", "reference_selection"):
            raise ValueError(
                "selection evidence cannot replace source-only basis/transform fitting"
            )
        if (
            evidence["split"] not in ("source", "development")
            or banks.get(evidence["goal_bank_sha256"]) != evidence["split"]
        ):
            raise ValueError("selection cannot access final-test or undeclared goal data")
        if evidence["kind"] == "source_weights":
            if evidence["split"] != "source" or evidence["identifier"] not in source_ids:
                raise ValueError(
                    "rank/reference selection cannot inspect excluded or target weights"
                )
        elif evidence["kind"] != "metrics":
            raise ValueError("teacher actions and target weights are not selection inputs")
        elif len(evidence["identifier"]) != 64 or any(
            letter not in "0123456789abcdef" for letter in evidence["identifier"]
        ):
            raise ValueError("external selection metrics require a SHA256 content commitment")


def fit_source_basis(
    policies: list[SourcePolicy],
    goals: GoalManifest,
    spec: BasisFitSpec,
    *,
    excluded_policy_ids: tuple[str, ...] = (),
    excluded_task_classes: tuple[str, ...] = (),
    relations: TaskRelations,
    task_split: TaskSplitManifest,
) -> BasisArtifact:
    spec.validate()
    task_split.validate()
    if goals.for_evaluation("final_test"):
        raise ValueError(
            "actual final-test goals must not enter the fitting API; use outer commitments"
        )
    if len({policy.policy_id for policy in policies}) != len(policies):
        raise ValueError("source policy IDs must be unique")
    source_classes = set(task_split.source_classes)
    development_classes = source_classes | set(task_split.development_target_classes)
    if (
        any(policy.task_class not in source_classes for policy in policies)
        or any(bank.goals[0].task not in source_classes for bank in goals.for_fitting())
        or any(bank.goals[0].task not in development_classes for bank in goals.for_tuning())
    ):
        raise ValueError(
            "weights or goal banks violate declared source/development task-class roles"
        )
    declared_task_classes = (
        source_classes
        | set(task_split.development_target_classes)
        | set(task_split.final_target_classes)
    )
    if not set(excluded_policy_ids).issubset({policy.policy_id for policy in policies}) or not set(
        excluded_task_classes
    ).issubset(declared_task_classes):
        raise ValueError(
            "fold exclusions must name declared policies/task classes, not silent typos"
        )
    excluded_task_classes = tuple(
        sorted(
            set(excluded_task_classes)
            | (
                set(task_split.final_target_classes)
                if task_split.transfer_regime == "heldout_task_classes"
                else set()
            )
        )
    )
    # Exclusion precedes alignment, normalization, centering and rank selection.
    retained = [
        policy
        for policy in policies
        if policy.policy_id not in excluded_policy_ids
        and policy.task_class not in excluded_task_classes
    ]
    if len(retained) < 2:
        raise ValueError("at least two remaining source actors are required")
    for policy in retained:
        policy.validate_integrity()
    if (
        len({policy.library_id for policy in retained}) != 1
        or len({policy.initialization_condition for policy in retained}) != 1
    ):
        raise ValueError("fit one independent library and initialization condition per artifact")
    goals.validate_source_policy_records(
        [policy.metadata() | {"task": policy.task_class} for policy in retained]
    )
    layouts = [policy.layout for policy in retained]
    if any(layout != layouts[0] for layout in layouts):
        raise ValueError("all source actors must share the complete ordered parameter interface")
    if len({policy.parameter_dtype for policy in retained}) != 1:
        raise ValueError("all source actors must share their original parameter dtype")
    run_ids = {policy.training_run_id for policy in retained}
    if len({(policy.training_run_id, policy.checkpoint_step) for policy in retained}) != len(
        retained
    ):
        raise ValueError("duplicate checkpoint identities cannot be additional basis rows")
    if len(run_ids) != len(retained) and not spec.allow_repeated_checkpoints:
        raise ValueError(
            "repeated checkpoints require explicit declaration and do not add library replicates"
        )
    init_by_run: dict[str, tuple[str, int]] = {}
    for policy in retained:
        evidence = (policy.initial_actor_sha256, policy.rng_streams["actor"])
        if (
            policy.training_run_id in init_by_run
            and init_by_run[policy.training_run_id] != evidence
        ):
            raise ValueError("a training run cannot change its saved effective initialization")
        init_by_run[policy.training_run_id] = evidence
    distinct_initials = len({value[0] for value in init_by_run.values()})
    condition = retained[0].initialization_condition
    if condition == "shared_init" and (
        distinct_initials != 1 or len({value[1] for value in init_by_run.values()}) != 1
    ):
        raise ValueError(
            "shared initialization requires equal saved hashes/arrays and effective actor seeds"
        )
    if condition == "shared_init" and any(
        not np.array_equal(policy.initial_vector, retained[0].initial_vector) for policy in retained
    ):
        raise ValueError("shared initialization saved arrays differ")
    if condition == "independent_init" and (
        distinct_initials != len(run_ids)
        or len({value[1] for value in init_by_run.values()}) != len(run_ids)
    ):
        raise ValueError(
            "independent initialization requires distinct effective arrays/hashes and actor seeds"
        )
    _validate_selection(spec, goals, retained, excluded_task_classes)
    declared_classes = declared_task_classes
    relations.validate(declared_classes)
    ids = [policy.policy_id for policy in retained]
    for identifier in (
        spec.reference_policy_id,
        spec.common_base_policy_id,
        spec.alignment_anchor_policy_id,
    ):
        if identifier is not None and identifier not in ids:
            raise ValueError("references and alignment anchors must be remaining source policies")
    layout = layouts[0]
    raw = np.stack([policy.vector for policy in retained])
    transformed = raw.copy()
    alignment: list[dict[str, Any]] = []
    if spec.coordinate_condition == "source_aligned":
        transformed, alignment = align_sac_vectors(
            raw,
            layout,
            anchor=ids.index(str(spec.alignment_anchor_policy_id)),
            activation=str(spec.alignment_activation),
            iterations=spec.alignment_iterations,
            probe_seed=spec.alignment_probe_seed,
        )
        for index, report in enumerate(alignment):
            report["aligned_vector_sha256"] = _array_hash(transformed[index])
            report["aligned_initial_vector_sha256"] = _array_hash(
                apply_sac_permutations(
                    retained[index].initial_vector, layout, report["permutations"]
                )
            )
    base = np.zeros(raw.shape[1], dtype=np.float64)
    if spec.common_base_policy_id:
        index = ids.index(spec.common_base_policy_id)
        base = retained[index].initial_vector.copy()
        if alignment:
            base = apply_sac_permutations(base, layout, alignment[index]["permutations"])
    represented = transformed - base
    arrays: dict[str, Array] = {"common_base": base}
    blocks: list[dict[str, Any]] = []
    deployment_keys = []
    coefficient_count = 0
    if spec.estimator == "flat_population_pca":
        if spec.parameter_modes or spec.mode_ranks:
            raise ValueError("flat population PCA does not accept parameter-mode settings")
        if spec.rank is not None and spec.rank > len(retained) - 1:
            raise ValueError("centered flat population rank cannot exceed N-1")
        center = represented.mean(axis=0)
        centered = represented - center
        scale = np.ones_like(center)
        if spec.normalization == "source_coordinate_std":
            deviation = centered.std(axis=0)
            scale = np.where(deviation > 1e-12, deviation, 1.0)
        normalized = centered / scale
        _, singular, vt = np.linalg.svd(normalized, full_matrices=False)
        coefficient_count = _rank(singular, normalized.shape, spec.rank, spec.variance_threshold)
        if coefficient_count > len(retained) - 1:
            raise ValueError("centered flat population rank cannot exceed N-1")
        arrays.update(
            estimator_center=base + center,
            normalization_scale=scale,
            flat_singular_values=singular,
            flat_directions=_canonical_columns(vt.T)[:, :coefficient_count].T * scale,
        )
        deployment_keys.append("flat_directions")
        centering = {"stacked_axes": [0], "meaning": "elementwise mean over source models"}
    else:
        if spec.rank is not None or set(spec.parameter_modes) != {item["name"] for item in layout}:
            raise ValueError(
                "parameter HOSVD needs explicit parameter modes for every actor tensor"
            )
        if spec.rank_rule == "fixed_predeclared" and set(spec.mode_ranks) != set(
            spec.parameter_modes
        ):
            raise ValueError("fixed HOSVD ranks must be declared for every parameter tensor")
        estimator_center = np.empty(raw.shape[1], dtype=np.float64)
        for index, item in enumerate(layout):
            name, shape = item["name"], item["shape"]
            modes = spec.parameter_modes[name]
            if tuple(sorted(set(modes))) != modes or any(
                type(axis) is not int or not 0 <= axis < len(shape) for axis in modes
            ):
                raise ValueError(
                    "parameter modes are unique zero-based parameter axes, never the model axis"
                )
            ranks = spec.mode_ranks.get(name, ())
            if spec.rank_rule == "fixed_predeclared" and len(ranks) != len(modes):
                raise ValueError("one rank is required for each selected parameter mode")
            population = represented[:, item["start"] : item["end"]].reshape(
                (len(retained), *shape)
            )
            mean_axes = (0, *(axis + 1 for axis in range(len(shape)) if axis not in modes))
            mean = population.mean(axis=mean_axes, keepdims=True)[0]
            centered_tensor = population - mean
            estimator_center[item["start"] : item["end"]] = base[
                item["start"] : item["end"]
            ] + np.broadcast_to(mean, shape).reshape(-1)
            core_shape, factor_keys = list(shape), []
            arrays[f"mean_{index}"] = np.asarray(mean, dtype=np.float64)
            for position, axis in enumerate(modes):
                unfolded = np.moveaxis(centered_tensor, axis + 1, 0).reshape(shape[axis], -1)
                u, singular, _ = np.linalg.svd(unfolded, full_matrices=False)
                rank = _rank(
                    singular,
                    unfolded.shape,
                    ranks[position] if ranks else None,
                    spec.variance_threshold,
                )
                core_shape[axis] = rank
                key = f"factor_{index}_{axis}"
                arrays[key] = _canonical_columns(u)[:, :rank]
                arrays[f"singular_{index}_{axis}"] = singular
                factor_keys.append(key)
                deployment_keys.append(key)
            coefficient_count += math.prod(core_shape)
            blocks.append(
                item
                | {
                    "parameter_modes": list(modes),
                    "compressed_stacked_axes": [axis + 1 for axis in modes],
                    "centering_stacked_axes": list(mean_axes),
                    "core_shape": core_shape,
                    "factor_keys": factor_keys,
                }
            )
        arrays["estimator_center"] = estimator_center
        arrays["normalization_scale"] = np.ones(raw.shape[1], dtype=np.float64)
        centering = {"meaning": "UWSH v3 Algorithm 1: model axis plus nondecomposed parameter axes"}
    arrays["reference"] = (
        transformed[ids.index(spec.reference_policy_id)].copy()
        if spec.reference_policy_id
        else arrays["estimator_center"].copy()
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "estimator": spec.estimator,
        "estimator_revision": ESTIMATOR_REVISIONS[spec.estimator],
        "spec": spec.to_dict(),
        "layout": layout,
        "source_rows": len(retained),
        "source_policy_ids": ids,
        "source_training_run_count": len(run_ids),
        "independent_library_replicates": 1,
        "library_id": retained[0].library_id,
        "initialization_condition": condition,
        "source_policy_records": [policy.metadata() for policy in retained],
        "source_goal_bank_hashes": sorted(
            {policy.training_goal_bank_sha256 for policy in retained}
        ),
        "fitting_goal_manifest_sha256": goals.sha256,
        "task_split_manifest": asdict(task_split),
        "task_split_manifest_sha256": task_split.sha256,
        "task_role_access": {
            "source_weight_fit_allowlist": list(task_split.source_classes),
            "source_own_development_goals_allowed": True,
            "development_target_metrics_allowlist": list(task_split.development_target_classes),
            "named_final_class_overlap_locally_checked": bool(task_split.final_target_classes),
            "sealed_or_unassigned_final_overlap": (
                "not locally inspectable; confirmation freeze must verify"
            ),
            "final_target_goals_or_weights": "not fitting inputs",
        },
        "excluded_policy_ids": list(excluded_policy_ids),
        "excluded_task_classes": list(excluded_task_classes),
        "task_relations": asdict(relations),
        "alignment": alignment,
        "centering": centering,
        "model_axis_compressed": False,
        "coefficient_count": coefficient_count,
        "blocks": blocks,
        "deployment_basis_keys": deployment_keys,
        "normalization_rule": "source population standard deviation, ddof=0; <=1e-12 replaced by1"
        if spec.normalization != "none"
        else "none",
        "representation_note": (
            "common-base subtraction cancels under elementwise population centering, "
            "up to numerical roundoff; it is not a separate flat-PCA geometry intervention"
            if spec.estimator == "flat_population_pca"
            else "feature-wise parameter-mode centering need not cancel every common-base component"
        ),
        "reference_note": (
            "an exact source-policy anchor can translate the truncated centered affine family; "
            "decode and oracle both use the stored deployed reference"
        ),
        "family_identifier_note": (
            "hash of the exact reference and coefficient map; unequal hashes do not prove "
            "different affine spans"
        ),
        "storage": {
            "artifact_array_scalars": sum(value.size for value in arrays.values()),
            "artifact_array_bytes": sum(value.nbytes for value in arrays.values()),
            "deployment_shared_scalars": raw.shape[1]
            + sum(arrays[key].size for key in deployment_keys),
            "deployment_shared_bytes": 8
            * (raw.shape[1] + sum(arrays[key].size for key in deployment_keys)),
            "per_target_coefficient_scalars": coefficient_count,
            "per_target_coefficient_bytes_float64": 8 * coefficient_count,
            "critic_optimizer_storage": "reported separately by the adaptation runner",
        },
        "data_access": {
            "center_basis_alignment_normalization": "remaining source actors only",
            "rank_reference_selection": list(spec.selection_evidence),
            "final_test_goals": "not accessed",
            "target_weights_teacher_actions": "not accessed",
        },
    }
    return BasisArtifact(metadata, arrays)
