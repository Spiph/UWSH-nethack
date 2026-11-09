"""Privileged diagnostics against the actual materialized affine actor map.

This module never fits a map or initializes a learner. A target weight vector is
accepted only by the explicitly privileged projection function. Least squares
uses float64 arithmetic on the exported buffers' actual values; the returned
execution coefficients are rounded to the declared materialization precision.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from scripts.metaworld_alignment import SAC_PARAMETERS, validate_layout
from scripts.metaworld_protocol import stable_hash

Array = npt.NDArray[Any]
DESCRIPTOR_KEYS = {
    "schema_version",
    "basis_artifact_sha256",
    "family_sha256",
    "executed_coordinate_sha256",
    "materialization_dtype",
    "parameterization",
    "coefficient_count",
    "layout",
    "blocks",
    "arrays",
    "descriptor_sha256",
}


def _array_hash(array: Array) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_executed_map(descriptor: dict[str, Any]) -> None:
    """Check descriptor shape/content binding before using exported buffers."""
    if set(descriptor) != DESCRIPTOR_KEYS or descriptor["schema_version"] != 1:
        raise ValueError("unsupported executed-map descriptor")
    if descriptor["materialization_dtype"] not in ("float32", "float64"):
        raise ValueError("unsupported executed-map precision")
    for key in (
        "basis_artifact_sha256",
        "family_sha256",
        "executed_coordinate_sha256",
        "descriptor_sha256",
    ):
        value = descriptor[key]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("invalid executed-map provenance hash")
    arrays, layout, blocks = descriptor["arrays"], descriptor["layout"], descriptor["blocks"]
    if not isinstance(arrays, dict) or "reference" not in arrays:
        raise ValueError("missing executed-map arrays")
    dtype = np.dtype(descriptor["materialization_dtype"])
    if any(
        not isinstance(array, np.ndarray) or array.dtype != dtype or not np.isfinite(array).all()
        for array in arrays.values()
    ):
        raise ValueError("executed-map arrays must have the declared finite precision")
    reference = arrays["reference"]
    if reference.ndim != 1:
        raise ValueError("executed reference must be a parameter vector")
    validate_layout(layout, reference.size)
    if tuple(item["name"] for item in layout) != SAC_PARAMETERS:
        raise ValueError("executed map must cover the complete ordered SAC actor")
    count = descriptor["coefficient_count"]
    if type(count) is not int or count < 0:
        raise ValueError("invalid executed coefficient count")
    if descriptor["parameterization"] == "flat":
        keys = ["reference", "flat_directions"]
        if (
            set(arrays) != set(keys)
            or arrays["flat_directions"].shape != (count, reference.size)
            or blocks
        ):
            raise ValueError("flat executed-map dimensions disagree")
    elif descriptor["parameterization"] == "parameter_modes":
        keys = ["reference"]
        total = 0
        if len(blocks) != len(layout):
            raise ValueError("parameter cores must cover every actor tensor")
        for item, block in zip(layout, blocks, strict=True):
            modes, shape = block["parameter_modes"], block["core_shape"]
            if (
                any(block[key] != item[key] for key in ("name", "shape", "start", "end"))
                or len(shape) != len(item["shape"])
                or modes != sorted(set(modes))
                or len(modes) != len(block["factor_keys"])
                or any(type(axis) is not int or not 0 <= axis < len(shape) for axis in modes)
            ):
                raise ValueError("executed parameter-core metadata disagrees")
            for axis, (rank, width) in enumerate(zip(shape, item["shape"], strict=True)):
                if (
                    type(rank) is not int
                    or not 0 <= rank <= width
                    or (axis not in modes and rank != width)
                ):
                    raise ValueError("executed core has an uncounted or invalid dimension")
            for axis, key in zip(modes, block["factor_keys"], strict=True):
                if key not in arrays or arrays[key].shape != (item["shape"][axis], shape[axis]):
                    raise ValueError("executed factor dimensions disagree")
                keys.append(key)
            total += math.prod(shape)
        if total != count or len(set(keys)) != len(keys) or set(arrays) != set(keys):
            raise ValueError("executed-map coefficient/buffer accounting disagrees")
    else:
        raise ValueError("unknown executed-map parameterization")
    actual = stable_hash(
        {
            "layout": layout,
            "blocks": blocks,
            "arrays": {key: _array_hash(arrays[key]) for key in keys},
        }
    )
    if actual != descriptor["executed_coordinate_sha256"]:
        raise ValueError("executed-map content hash mismatch")
    envelope = {
        key: value
        for key, value in descriptor.items()
        if key not in ("arrays", "descriptor_sha256")
    }
    envelope["arrays"] = {key: _array_hash(array) for key, array in arrays.items()}
    if stable_hash(envelope) != descriptor["descriptor_sha256"]:
        raise ValueError("executed-map descriptor provenance/hash mismatch")


def _mode_product(array: Array, matrix: Array, axis: int) -> Array:
    return np.moveaxis(np.tensordot(matrix, array, axes=([1], [axis])), 0, axis)


def _decode(descriptor: dict[str, Any], coefficients: Array, dtype: npt.DTypeLike) -> Array:
    arrays = descriptor["arrays"]
    reference = np.asarray(arrays["reference"], dtype=dtype)
    values = np.asarray(coefficients, dtype=dtype)
    if descriptor["parameterization"] == "flat":
        return np.asarray(reference + values @ np.asarray(arrays["flat_directions"], dtype=dtype))
    pieces = []
    offset = 0
    for block in descriptor["blocks"]:
        size = math.prod(block["core_shape"])
        core = values[offset : offset + size].reshape(block["core_shape"])
        for axis, key in zip(block["parameter_modes"], block["factor_keys"], strict=True):
            core = _mode_product(core, np.asarray(arrays[key], dtype=dtype), axis)
        pieces.append(core.reshape(-1))
        offset += size
    return np.asarray(reference + np.concatenate(pieces))


def decode_executed_map(descriptor: dict[str, Any], coefficients: Array) -> Array:
    """Decode at execution precision; kernel reduction rounding can differ from Torch."""
    validate_executed_map(descriptor)
    values = np.asarray(coefficients)
    if values.shape != (descriptor["coefficient_count"],) or not np.isfinite(values).all():
        raise ValueError("invalid executed-map coefficients")
    result = _decode(descriptor, values, descriptor["materialization_dtype"])
    if not np.isfinite(result).all():
        raise ValueError("executed-map decoding overflowed")
    return result


@dataclass(frozen=True)
class ExecutedOracleProjection:
    family_sha256: str
    executed_coordinate_sha256: str
    coefficients: Array
    reconstruction: Array
    least_squares_coefficients_float64: Array
    least_squares_reconstruction_float64: Array
    least_squares_residual_l2: float
    materialized_residual_l2: float
    information_access: str = "privileged_target_weights; diagnostic_only"
    numerical_note: str = (
        "float64 least squares against actual executed buffers, followed by execution-dtype "
        "coefficient rounding; materialized reconstruction is not promised to be optimal "
        "after rounding and NumPy/Torch reduction order may differ"
    )


def privileged_oracle_project_executed_map(
    descriptor: dict[str, Any], target_weights: Array
) -> ExecutedOracleProjection:
    """Fit diagnostic coordinates without modifying a map or any learning state."""
    validate_executed_map(descriptor)
    target = np.asarray(target_weights, dtype=np.float64)
    arrays = descriptor["arrays"]
    if target.shape != arrays["reference"].shape or not np.isfinite(target).all():
        raise ValueError("privileged target must be a finite complete actor vector")
    delta = target - np.asarray(arrays["reference"], dtype=np.float64)
    if descriptor["parameterization"] == "flat":
        coordinates = np.linalg.lstsq(
            np.asarray(arrays["flat_directions"], dtype=np.float64).T, delta, rcond=None
        )[0]
    else:
        cores = []
        for block in descriptor["blocks"]:
            core = delta[block["start"] : block["end"]].reshape(block["shape"])
            for axis, key in zip(block["parameter_modes"], block["factor_keys"], strict=True):
                # Rounded float32 factors need not remain orthonormal. A transpose
                # would project into a different numerical map.
                core = _mode_product(core, np.linalg.pinv(arrays[key].astype(np.float64)), axis)
            cores.append(core.reshape(-1))
        coordinates = np.concatenate(cores)
    reconstruction64 = _decode(descriptor, coordinates, np.float64)
    execution_coordinates = coordinates.astype(descriptor["materialization_dtype"])
    reconstruction = decode_executed_map(descriptor, execution_coordinates)
    if not np.isfinite(execution_coordinates).all():
        raise ValueError("oracle coordinates overflow the execution precision")
    for array in (coordinates, execution_coordinates, reconstruction64, reconstruction):
        array.flags.writeable = False
    return ExecutedOracleProjection(
        family_sha256=descriptor["family_sha256"],
        executed_coordinate_sha256=descriptor["executed_coordinate_sha256"],
        coefficients=execution_coordinates,
        reconstruction=reconstruction,
        least_squares_coefficients_float64=coordinates,
        least_squares_reconstruction_float64=reconstruction64,
        least_squares_residual_l2=float(np.linalg.norm(reconstruction64 - target)),
        materialized_residual_l2=float(np.linalg.norm(reconstruction.astype(np.float64) - target)),
    )
