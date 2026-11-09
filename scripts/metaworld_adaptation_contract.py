"""Source-bound matched random affine controls and conservative target-cost accounting.

No environment, actor training, or project resource allocation is performed here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import numpy.typing as npt

from scripts.metaworld_basis import BasisArtifact
from scripts.metaworld_protocol import stable_hash

Array = npt.NDArray[np.float64]
CONTROL_REVISIONS = {
    "matched_random_flat": "local_gaussian_qr_whole_row_gram_v1",
    "matched_random_parameter_factors": "structured_local_gaussian_qr_parameter_factors_v1",
}


class AffineFamilyProtocol(Protocol):
    @property
    def metadata(self) -> dict[str, Any]: ...

    @property
    def parameterization(self) -> str: ...

    @property
    def coefficient_count(self) -> int: ...

    @property
    def sha256(self) -> str: ...

    @property
    def family_sha256(self) -> str: ...

    def deployment_arrays(self) -> dict[str, npt.NDArray[Any]]: ...


def _array_hash(array: Array) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _family_hash(metadata: dict[str, Any], arrays: dict[str, Array]) -> str:
    return stable_hash(
        {
            "layout": metadata["layout"],
            "blocks": metadata["blocks"],
            "arrays": {key: _array_hash(array) for key, array in arrays.items()},
        }
    )


class RandomControlFamily:
    """Preserve source reference, coordinate count and declared map scales.

    Flat controls preserve B @ B.T, including off-diagonal terms introduced by
    source normalization. Parameter-mode controls rotate factors individually;
    their block structure is retained and they are explicitly structured nulls.
    """

    def __init__(self, parent: BasisArtifact, *, seed: int) -> None:
        if not isinstance(parent, BasisArtifact):
            raise TypeError("random controls require a verified BasisArtifact")
        parent.validate_integrity()
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("control seed must be an unsigned 32-bit integer")
        source = parent.metadata
        arrays = parent.deployment_arrays()
        if _family_hash(source, arrays) != parent.family_sha256:
            raise ValueError("parent deployment map and family hash disagree")
        self._parameterization = parent.parameterization
        self._coefficient_count = parent.coefficient_count
        rng: np.random.Generator | None = None

        def orientation(size: int, rank: int) -> Array:
            nonlocal rng
            if rank == 0:
                return np.empty((size, 0), dtype=np.float64)
            if not 0 < rank <= size:
                raise ValueError("random orientation rank exceeds its ambient dimension")
            if rng is None:
                rng = np.random.default_rng(seed)
            q, r = np.linalg.qr(rng.standard_normal((size, rank)), mode="reduced")
            q *= np.where(np.diag(r) < 0, -1.0, 1.0)
            return np.asarray(q, dtype=np.float64)

        if self.parameterization == "flat":
            estimator = "matched_random_flat"
            basis = arrays["flat_directions"]
            rows, parameters = basis.shape
            if rows == 0:
                random_basis = np.empty_like(basis)
                effective_rank = 0
            else:
                u, singular, _ = np.linalg.svd(basis, full_matrices=False)
                if rows > parameters:
                    raise ValueError(
                        "flat control requires no more coefficient rows than parameters"
                    )
                random_basis = (u * singular) @ orientation(parameters, rows).T
                effective_rank = int(np.linalg.matrix_rank(basis))
                expected_gram, actual_gram = basis @ basis.T, random_basis @ random_basis.T
                if not np.isfinite(expected_gram).all() or not np.allclose(
                    actual_gram, expected_gram, rtol=1e-10, atol=1e-10
                ):
                    raise ValueError("random control failed to preserve the whole raw row Gram")
            arrays["flat_directions"] = random_basis
            limitations: dict[str, Any] = {
                "structure": "global_flat_orientation",
                "matching_metric": "whole_raw_row_gram_including_off_diagonal_terms",
                "effective_rank": effective_rank,
                "constant_family": effective_rank == 0,
                "full_parameter_span": effective_rank == parameters,
                "orientation_null_informative": 0 < effective_rank < parameters,
            }
        elif self.parameterization == "parameter_modes":
            estimator = "matched_random_parameter_factors"
            factors, blocks = [], []
            for block in source["blocks"]:
                zero_core = math.prod(block["core_shape"]) == 0
                block_factor_flags = []
                for axis, key in zip(block["parameter_modes"], block["factor_keys"], strict=True):
                    size, rank = arrays[key].shape
                    if not np.allclose(
                        arrays[key].T @ arrays[key], np.eye(rank), rtol=1e-10, atol=1e-10
                    ):
                        raise ValueError(
                            "parent parameter factors must be orthonormal for matched controls"
                        )
                    if not zero_core:
                        arrays[key] = orientation(size, rank)
                    flag = {
                        "key": key,
                        "parameter_axis": axis,
                        "ambient_size": size,
                        "rank": rank,
                        "full_rank": rank == size,
                        "zero_rank": rank == 0,
                        "skipped_zero_core": zero_core,
                        "orientation_can_change_mode_span": 0 < rank < size,
                    }
                    factors.append(flag)
                    block_factor_flags.append(flag)
                changes_span = not zero_core and any(
                    flag["orientation_can_change_mode_span"] for flag in block_factor_flags
                )
                blocks.append(
                    {
                        "name": block["name"],
                        "zero_core": zero_core,
                        "coefficient_count": math.prod(block["core_shape"]),
                        "effectively_same_block_space": not changes_span,
                        "orientation_can_change_block_span": changes_span,
                    }
                )
            limitations = {
                "structure": "structured_parameter_factor_control",
                "unstructured_global_orientation_null": False,
                "factors": factors,
                "blocks": blocks,
                "all_blocks_coordinate_only": all(
                    block["effectively_same_block_space"] for block in blocks
                ),
                "orientation_null_informative": any(
                    block["orientation_can_change_block_span"] for block in blocks
                ),
            }
        else:
            raise ValueError("unknown verified parent parameterization")
        shared_scalars = sum(array.size for array in arrays.values())
        self._metadata = {
            "schema_version": 1,
            "estimator": estimator,
            "estimator_revision": CONTROL_REVISIONS[estimator],
            "seed": seed,
            "parameterization": self.parameterization,
            "coefficient_count": self.coefficient_count,
            "parent_artifact_sha256": parent.sha256,
            "parent_family_sha256": parent.family_sha256,
            "parent_estimator": source["estimator"],
            "parent_estimator_revision": source["estimator_revision"],
            "source_alignment_activation": source["spec"].get("alignment_activation"),
            "source_coordinate_condition": source["spec"]["coordinate_condition"],
            "layout": source["layout"],
            "blocks": source["blocks"],
            "deployment_basis_keys": source["deployment_basis_keys"],
            "control": limitations,
            "reference_provenance": (
                "exact unchanged parent reference; zero coefficients reproduce it"
            ),
            "information_access": "verified source-fitted deployment map only; no target data",
            "family_identifier_note": (
                "exact coordinate-map hash; unequal hashes do not prove different affine spans"
            ),
            "storage": {
                "artifact_array_scalars": shared_scalars,
                "artifact_array_bytes": shared_scalars * 8,
                "deployment_shared_scalars": shared_scalars,
                "deployment_shared_bytes": shared_scalars * 8,
                "per_target_coefficient_scalars": self.coefficient_count,
                "per_target_coefficient_bytes_float64": self.coefficient_count * 8,
                "critic_optimizer_storage": "reported separately by the adaptation consumer",
            },
        }
        self._arrays = {
            key: np.array(array, dtype=np.float64, copy=True) for key, array in arrays.items()
        }
        for array in self._arrays.values():
            if not np.isfinite(array).all():
                raise ValueError("nonfinite random control deployment array")
            array.flags.writeable = False
        self._commitment = self.sha256

    @classmethod
    def from_basis(cls, parent: BasisArtifact, *, seed: int) -> RandomControlFamily:
        return cls(parent, seed=seed)

    @property
    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    @property
    def parameterization(self) -> str:
        return self._parameterization

    @property
    def coefficient_count(self) -> int:
        return self._coefficient_count

    @property
    def sha256(self) -> str:
        return stable_hash(
            {
                "metadata": self._metadata,
                "arrays": {key: _array_hash(value) for key, value in self._arrays.items()},
            }
        )

    @property
    def family_sha256(self) -> str:
        return _family_hash(self._metadata, self._arrays)

    def deployment_arrays(self) -> dict[str, Array]:
        if self.sha256 != self._commitment:
            raise ValueError("random control changed after construction")
        return {key: value.copy() for key, value in self._arrays.items()}

    def decode(self, coefficients: Any) -> Array:
        if np.iscomplexobj(coefficients):
            raise ValueError("coefficients must be real")
        coordinates = np.asarray(coefficients, dtype=np.float64)
        if coordinates.shape != (self.coefficient_count,) or not np.isfinite(coordinates).all():
            raise ValueError("one finite coefficient is required per declared degree of freedom")
        arrays = self.deployment_arrays()
        result = arrays["reference"].copy()
        if self.parameterization == "flat":
            result += coordinates @ arrays["flat_directions"]
        else:
            offset = 0
            for block in self._metadata["blocks"]:
                size = math.prod(block["core_shape"])
                if size:
                    core = coordinates[offset : offset + size].reshape(block["core_shape"])
                    for axis, key in zip(
                        block["parameter_modes"], block["factor_keys"], strict=True
                    ):
                        core = np.moveaxis(np.tensordot(arrays[key], core, axes=(1, axis)), 0, axis)
                    result[block["start"] : block["end"]] += core.reshape(-1)
                offset += size
        return result

    def save(self, path: Path) -> dict[str, str]:
        arrays = self.deployment_arrays()
        envelope = {
            "metadata": self.metadata,
            "sha256": self.sha256,
            "family_sha256": self.family_sha256,
        }
        with path.open("xb") as handle:
            np.savez_compressed(
                handle, metadata_json=np.asarray(json.dumps(envelope, sort_keys=True)), **arrays
            )
        return {
            "artifact_sha256": self.sha256,
            "family_sha256": self.family_sha256,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    @classmethod
    def load(cls, path: Path, *, parent: BasisArtifact) -> RandomControlFamily:
        """Verify the saved recipe against its parent; numerical-stack drift fails closed."""
        with np.load(path, allow_pickle=False) as archive:
            envelope = json.loads(str(archive["metadata_json"].item()))
            result = cls.from_basis(parent, seed=envelope["metadata"]["seed"])
            arrays = result.deployment_arrays()
            if (
                envelope["metadata"] != result.metadata
                or envelope["sha256"] != result.sha256
                or envelope["family_sha256"] != result.family_sha256
                or set(archive.files) != {"metadata_json", *arrays}
                or any(
                    archive[key].dtype != np.float64 or not np.array_equal(archive[key], value)
                    for key, value in arrays.items()
                )
            ):
                raise ValueError("random control content, recipe or parent provenance mismatch")
        return result


class InteractionLimitExceeded(RuntimeError):
    pass


class TargetInteractionLedger:
    """Reserve target interactions before work; failed/restarted attempts never refund costs."""

    KINDS = ("selection", "learning", "evaluation")

    def __init__(
        self,
        *,
        selection_cap: int,
        learning_cap: int,
        evaluation_cap: int,
        total_cap: int,
        max_restarts: int,
    ) -> None:
        self._caps = {
            "selection": selection_cap,
            "learning": learning_cap,
            "evaluation": evaluation_cap,
            "total": total_cap,
            "restarts": max_restarts,
        }
        if any(type(value) is not int or value < 0 for value in self._caps.values()):
            raise ValueError("all caller-defined resource caps must be nonnegative integers")
        self._charged = dict.fromkeys(self.KINDS, 0)
        self._attempts: list[dict[str, Any]] = []
        self._restarts: list[dict[str, Any]] = []

    def begin_attempt(
        self,
        kind: Literal["selection", "learning", "evaluation"],
        steps: int,
        *,
        warmup: bool = False,
        label: str = "",
    ) -> int:
        if (
            kind not in self.KINDS
            or type(steps) is not int
            or steps <= 0
            or type(warmup) is not bool
        ):
            raise ValueError("declare a target bucket, positive integer steps and boolean warmup")
        if warmup and kind != "learning":
            raise ValueError("warmup is a subset of learning, never an additional bucket")
        if (
            self._charged[kind] + steps > self._caps[kind]
            or self.total_steps + steps > self._caps["total"]
        ):
            raise InteractionLimitExceeded("target interaction cap would be exceeded before work")
        self._charged[kind] += steps
        identifier = len(self._attempts)
        self._attempts.append(
            {
                "id": identifier,
                "kind": kind,
                "reserved_steps": steps,
                "warmup": warmup,
                "label": str(label),
                "generation": len(self._restarts),
                "status": "pending",
                "actual_steps": None,
                "selection_replay_reuse": False,
            }
        )
        return identifier

    def _pending(self, identifier: int) -> dict[str, Any]:
        if type(identifier) is not int or not 0 <= identifier < len(self._attempts):
            raise ValueError("unknown attempt identifier")
        attempt = self._attempts[identifier]
        if attempt["status"] != "pending":
            raise ValueError("attempt is already resolved or interrupted")
        return attempt

    def complete_attempt(self, identifier: int, *, actual_steps: int | None = None) -> None:
        attempt = self._pending(identifier)
        actual_steps = attempt["reserved_steps"] if actual_steps is None else actual_steps
        if type(actual_steps) is not int or actual_steps != attempt["reserved_steps"]:
            raise ValueError(
                "complete attempts must execute all reserved steps; report partial failure"
            )
        attempt.update(status="complete", actual_steps=actual_steps)

    def fail_attempt(
        self, identifier: int, *, reason: str, actual_steps: int | None = None
    ) -> None:
        attempt = self._pending(identifier)
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or (
                actual_steps is not None
                and (
                    type(actual_steps) is not int
                    or not 0 <= actual_steps <= attempt["reserved_steps"]
                )
            )
        ):
            raise ValueError("failed attempts need a reason and bounded observed steps or unknown")
        attempt.update(status="failed", actual_steps=actual_steps, reason=reason)

    def restart(self, *, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("restarts require a reason")
        if len(self._restarts) >= self._caps["restarts"]:
            raise InteractionLimitExceeded("restart cap exhausted; interaction costs are not reset")
        for attempt in self._attempts:
            if attempt["status"] == "pending":
                attempt.update(status="interrupted", reason=reason)
        self._restarts.append({"reason": reason, "charged_before_restart": dict(self._charged)})

    @property
    def learning_steps(self) -> int:
        return self._charged["learning"]

    @property
    def total_steps(self) -> int:
        return sum(self._charged.values())

    def snapshot(self) -> dict[str, Any]:
        observed = {
            kind: sum(
                attempt["actual_steps"] or 0
                for attempt in self._attempts
                if attempt["kind"] == kind
            )
            for kind in self.KINDS
        }
        return copy.deepcopy(
            {
                "schema_version": 1,
                "caps": self._caps,
                "charged": self._charged,
                "observed": observed,
                "axes": {
                    "learning_only": self.learning_steps,
                    "all_interactions": self.total_steps,
                },
                "observed_axes": {
                    "learning_only": observed["learning"],
                    "all_interactions": sum(observed.values()),
                },
                "warmup_learning_subset": {
                    "charged": sum(
                        attempt["reserved_steps"] for attempt in self._attempts if attempt["warmup"]
                    ),
                    "observed": sum(
                        attempt["actual_steps"] or 0
                        for attempt in self._attempts
                        if attempt["warmup"]
                    ),
                },
                "unobserved_conservative_charge": self.total_steps - sum(observed.values()),
                "actual_steps_unknown_attempt_ids": [
                    attempt["id"] for attempt in self._attempts if attempt["actual_steps"] is None
                ],
                "observed_axes_are_lower_bounds": any(
                    attempt["actual_steps"] is None for attempt in self._attempts
                ),
                "attempts": self._attempts,
                "restarts": self._restarts,
                "selection_probe_replay_reuse": "not authorized or implied by accounting",
                "cost_rule": (
                    "charge reservations before work; never refund failed or interrupted attempts"
                ),
            }
        )
