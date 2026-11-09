"""Differentiable frozen affine actor maps; SAC owns its action distribution.

Only coefficients are trainable. Raw mu/log_std heads are evaluated here without
clamping, sampling, squashing or log-probability calculations. Construction pins
an explicit floating dtype; device-only moves preserve it. Lossy dtype changes
require reconstruction from the verified source artifact.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn
from torch.nn import functional as functional

from scripts.metaworld_alignment import SAC_PARAMETERS, validate_layout
from scripts.metaworld_protocol import stable_hash

SCHEMA_VERSION = 1
DTYPES = {"float32": torch.float32, "float64": torch.float64}


class AffineFamily(Protocol):
    """A verified map; controls may have different honest estimator provenance."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    @property
    def parameterization(self) -> str: ...

    @property
    def sha256(self) -> str: ...

    @property
    def family_sha256(self) -> str: ...

    @property
    def coefficient_count(self) -> int: ...

    def deployment_arrays(self) -> dict[str, npt.NDArray[Any]]: ...


def _tensor_hash(tensor: Tensor) -> str:
    array = tensor.detach().cpu().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_hash(array: npt.NDArray[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class AffineCoefficientActor(nn.Module):
    """A complete two-hidden-layer SAC actor with one trainable coordinate vector."""

    def __init__(
        self,
        artifact: AffineFamily,
        *,
        activation: str,
        dtype: torch.dtype = torch.float64,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        if activation not in ("relu", "tanh"):
            raise ValueError("declare relu or tanh for the two pointwise hidden layers")
        if dtype not in DTYPES.values():
            raise ValueError("construct the actor with explicit float32 or float64 precision")
        self._basis_metadata = artifact.metadata
        aligned_activation = self._basis_metadata.get("spec", {}).get(
            "alignment_activation", self._basis_metadata.get("source_alignment_activation")
        )
        if aligned_activation is not None and aligned_activation != activation:
            raise ValueError("activation conflicts with the source-alignment declaration")
        self._basis_artifact_sha256 = artifact.sha256
        self._family_sha256 = artifact.family_sha256
        self._parameterization = artifact.parameterization
        if self._parameterization not in ("flat", "parameter_modes"):
            raise ValueError("unsupported affine coordinate parameterization")
        self._activation = activation
        self._layout = copy.deepcopy(self._basis_metadata["layout"])
        arrays = artifact.deployment_arrays()
        validate_layout(self._layout, arrays["reference"].size)
        if tuple(item["name"] for item in self._layout) != SAC_PARAMETERS:
            raise ValueError(
                "require the exact ordered SAC layout, including both heads and biases"
            )
        self._validate_head_shapes()
        self._buffer_keys = ("reference", *self._basis_metadata["deployment_basis_keys"])
        if len(set(self._buffer_keys)) != len(self._buffer_keys) or set(arrays) != set(
            self._buffer_keys
        ):
            raise ValueError("deployment buffers have duplicate or inconsistent keys")
        self._validate_map(arrays, artifact.coefficient_count)
        if (
            stable_hash(
                {
                    "layout": self._layout,
                    "blocks": self._basis_metadata["blocks"],
                    "arrays": {key: _array_hash(arrays[key]) for key in self._buffer_keys},
                }
            )
            != self.family_sha256
        ):
            raise ValueError(
                "exported deployment map disagrees with its declared logical family hash"
            )
        for key, array in arrays.items():
            if "." in key or key in ("coefficients", "_extra_state"):
                raise ValueError("invalid deployment buffer name")
            tensor = torch.tensor(array, dtype=dtype, device=device)
            if not torch.isfinite(tensor).all():
                raise ValueError("deployment array overflows the selected floating precision")
            self.register_buffer(key, tensor)
        self.coefficients = nn.Parameter(
            torch.zeros(artifact.coefficient_count, dtype=dtype, device=device)
        )
        self._expected_executed_hash = self._coordinate_hash()

    def _validate_map(self, arrays: dict[str, npt.NDArray[Any]], count: int) -> None:
        size = arrays["reference"].size
        if (
            type(count) is not int
            or count < 0
            or count != self._basis_metadata["coefficient_count"]
        ):
            raise ValueError("coefficient count disagrees with family metadata")
        if arrays["reference"].shape != (size,) or any(
            value.dtype != np.float64 or not np.isfinite(value).all() for value in arrays.values()
        ):
            raise ValueError("logical deployment arrays must be finite float64 tensors")
        if self._parameterization == "flat":
            if (
                self._buffer_keys != ("reference", "flat_directions")
                or arrays["flat_directions"].shape != (count, size)
                or self._basis_metadata["blocks"]
            ):
                raise ValueError("flat coordinate map dimensions disagree with the complete actor")
        else:
            blocks = self._basis_metadata["blocks"]
            if len(blocks) != len(self._layout):
                raise ValueError("parameter-mode cores must cover every actor tensor")
            total, keys = 0, []
            for item, block in zip(self._layout, blocks, strict=True):
                modes, core_shape = block["parameter_modes"], block["core_shape"]
                if (
                    any(block[key] != item[key] for key in ("name", "shape", "start", "end"))
                    or len(core_shape) != len(item["shape"])
                    or modes != sorted(set(modes))
                    or len(modes) != len(block["factor_keys"])
                ):
                    raise ValueError("parameter-mode core metadata conflicts with the actor layout")
                for axis, (rank, width) in enumerate(zip(core_shape, item["shape"], strict=True)):
                    if (
                        type(rank) is not int
                        or not 0 <= rank <= width
                        or (axis not in modes and rank != width)
                    ):
                        raise ValueError(
                            "uncompressed actor dimensions must be counted as coefficients"
                        )
                for axis, key in zip(modes, block["factor_keys"], strict=True):
                    if (
                        type(axis) is not int
                        or not 0 <= axis < len(core_shape)
                        or arrays[key].shape != (item["shape"][axis], core_shape[axis])
                    ):
                        raise ValueError(
                            "parameter-mode factor dimensions disagree with core metadata"
                        )
                    keys.append(key)
                total += math.prod(core_shape)
            if total != count or tuple(keys) != self._buffer_keys[1:]:
                raise ValueError("parameter-mode coefficient/buffer accounting is inconsistent")
        if self._basis_metadata["storage"]["deployment_shared_bytes"] != 8 * sum(
            value.size for value in arrays.values()
        ):
            raise ValueError("logical shared-map storage accounting is inconsistent")

    def _validate_head_shapes(self) -> None:
        shapes = {item["name"]: item["shape"] for item in self._layout}
        first, second = shapes["latent_pi.0.weight"], shapes["latent_pi.2.weight"]
        if (
            len(first) != 2
            or len(second) != 2
            or second[1] != first[0]
            or shapes["latent_pi.0.bias"] != [first[0]]
            or shapes["latent_pi.2.bias"] != [second[0]]
            or len(shapes["mu.weight"]) != 2
            or shapes["mu.weight"][1] != second[0]
            or shapes["mu.bias"] != [shapes["mu.weight"][0]]
            or shapes["log_std.weight"] != shapes["mu.weight"]
            or shapes["log_std.bias"] != shapes["mu.bias"]
        ):
            raise ValueError("SAC hidden/head shapes disagree with the complete actor interface")

    @property
    def family_sha256(self) -> str:
        """Original float64 artifact identity, distinct from its executed precision."""
        return self._family_sha256

    @property
    def executed_coordinate_sha256(self) -> str:
        return self._expected_executed_hash

    @property
    def coefficient_count(self) -> int:
        return self.coefficients.numel()

    @property
    def features_dim(self) -> int:
        return int(self._layout[0]["shape"][1])

    @property
    def action_dim(self) -> int:
        return int(next(item for item in self._layout if item["name"] == "mu.bias")["shape"][0])

    def _buffer(self, key: str) -> Tensor:
        value = self.get_buffer(key)
        if value is None:
            raise ValueError("missing frozen deployment buffer")
        return value

    def _coordinate_hash(self) -> str:
        return stable_hash(
            {
                "layout": self._layout,
                "blocks": self._basis_metadata["blocks"],
                "arrays": {key: _tensor_hash(self._buffer(key)) for key in self._buffer_keys},
            }
        )

    def _assert_frozen(self) -> None:
        if self._coordinate_hash() != self._expected_executed_hash:
            raise ValueError("frozen deployment buffers changed; reload from the verified artifact")

    def _binding(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "basis_artifact_sha256": self._basis_artifact_sha256,
            "family_sha256": self.family_sha256,
            "executed_coordinate_sha256": self.executed_coordinate_sha256,
            "activation": self._activation,
            "materialization_dtype": str(self.coefficients.dtype).removeprefix("torch."),
            "coefficient_count": self.coefficient_count,
            "parameterization": self._parameterization,
        }

    def logical_costs(self) -> dict[str, Any]:
        buffer_scalars = sum(self._buffer(key).numel() for key in self._buffer_keys)
        buffer_bytes = sum(
            self._buffer(key).numel() * self._buffer(key).element_size()
            for key in self._buffer_keys
        )
        coefficient_bytes = self.coefficients.numel() * self.coefficients.element_size()
        return {
            "decoded_actor_parameter_scalars": self._buffer("reference").numel(),
            "logical_shared_map_scalars": buffer_scalars,
            "materialized_shared_map_bytes": buffer_bytes,
            "source_artifact_shared_map_bytes_float64": self._basis_metadata["storage"][
                "deployment_shared_bytes"
            ],
            "logical_per_target_coefficient_scalars": self.coefficient_count,
            "per_target_coefficient_bytes": coefficient_bytes,
            "state_dict_tensor_bytes": buffer_bytes + coefficient_bytes,
            "export_payload_tensor_bytes": buffer_bytes + coefficient_bytes,
            "checkpoint_note": "full actor checkpoints include the frozen map and coefficients",
            "critic_optimizer_and_metadata_storage": "reported separately by the consumer",
        }

    @property
    def metadata(self) -> dict[str, Any]:
        self._assert_frozen()
        return {
            **self._binding(),
            "basis_metadata": copy.deepcopy(self._basis_metadata),
            "device": str(self.coefficients.device),
            "costs": self.logical_costs(),
            "head_semantics": "raw_mu_and_unclamped_log_std; distribution owned by SAC",
            "precision_note": "float64 origin identity and explicit executed-coordinate hash",
        }

    def decode(self, coefficients: Tensor | None = None) -> Tensor:
        values = self.coefficients if coefficients is None else coefficients
        if (
            values.shape != self.coefficients.shape
            or values.dtype != self.coefficients.dtype
            or values.device != self.coefficients.device
            or not torch.isfinite(values).all()
        ):
            raise ValueError("coefficients must match the declared finite shape, dtype and device")
        reference = self._buffer("reference")
        if self._parameterization == "flat":
            return reference + values @ self._buffer("flat_directions")
        pieces = []
        offset = 0
        for block in self._basis_metadata["blocks"]:
            size = math.prod(block["core_shape"])
            core = values[offset : offset + size].reshape(block["core_shape"])
            for axis, key in zip(block["parameter_modes"], block["factor_keys"], strict=True):
                core = torch.movedim(
                    torch.tensordot(self._buffer(key), core, dims=([1], [axis])), 0, axis
                )
            pieces.append(core.reshape(-1))
            offset += size
        return reference + torch.cat(pieces)

    def parameter_dict(self) -> dict[str, Tensor]:
        vector = self.decode()
        return {
            item["name"]: vector[item["start"] : item["end"]].reshape(item["shape"])
            for item in self._layout
        }

    def head_parameters(self, features: Tensor) -> tuple[Tensor, Tensor]:
        if (
            features.ndim < 1
            or features.shape[-1] != self.features_dim
            or features.dtype != self.coefficients.dtype
            or features.device != self.coefficients.device
            or not torch.isfinite(features).all()
        ):
            raise ValueError(
                "features must match the declared finite actor interface and precision"
            )
        parameters = self.parameter_dict()
        hidden = features
        for layer in (0, 2):
            hidden = functional.linear(
                hidden,
                parameters[f"latent_pi.{layer}.weight"],
                parameters[f"latent_pi.{layer}.bias"],
            )
            hidden = torch.relu(hidden) if self._activation == "relu" else torch.tanh(hidden)
        return (
            functional.linear(hidden, parameters["mu.weight"], parameters["mu.bias"]),
            functional.linear(hidden, parameters["log_std.weight"], parameters["log_std.bias"]),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return self.head_parameters(features)

    def _apply(
        self, fn: Callable[[Tensor], Tensor], recurse: bool = True
    ) -> AffineCoefficientActor:
        probe = fn(torch.empty(0, dtype=self.coefficients.dtype, device=self.coefficients.device))
        if probe.dtype != self.coefficients.dtype:
            raise ValueError("choose dtype at construction from the verified source artifact")
        # The pinned Torch versions expose these private hook signatures without
        # annotations; declare their checked calling contract locally.
        cast(Callable[..., nn.Module], super()._apply)(fn, recurse=recurse)
        return self

    def get_extra_state(self) -> dict[str, Any]:
        self._assert_frozen()
        return copy.deepcopy(self._binding())

    def set_extra_state(self, state: Any) -> None:
        if state != self._binding():
            raise ValueError(
                "checkpoint family, origin, activation or precision conflicts with this actor"
            )

    def _load_from_state_dict(
        self,
        state_dict: Mapping[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        expected = {"coefficients", "_extra_state", *self._buffer_keys}
        actual = {key[len(prefix) :] for key in state_dict if key.startswith(prefix)}
        errors = []
        if actual != expected or state_dict.get(prefix + "_extra_state") != self._binding():
            errors.append("checkpoint family/origin/activation/precision or state keys conflict")
        for key in self._buffer_keys:
            incoming = state_dict.get(prefix + key)
            current = self._buffer(key)
            if (
                not isinstance(incoming, Tensor)
                or incoming.dtype != current.dtype
                or incoming.shape != current.shape
                or not torch.equal(incoming.detach().cpu(), current.detach().cpu())
            ):
                errors.append(f"frozen buffer differs: {key}")
        coordinates = state_dict.get(prefix + "coefficients")
        if (
            not isinstance(coordinates, Tensor)
            or coordinates.dtype != self.coefficients.dtype
            or coordinates.shape != self.coefficients.shape
            or not torch.isfinite(coordinates).all()
        ):
            errors.append("invalid trainable coefficient state")
        if errors:
            error_msgs.append(prefix + "; ".join(errors))
            return
        cast(Callable[..., None], super()._load_from_state_dict)(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def export_payload(self) -> dict[str, Any]:
        metadata = self.metadata
        state = {
            key: value.detach().cpu().clone() if isinstance(value, Tensor) else copy.deepcopy(value)
            for key, value in self.state_dict().items()
        }
        return {"schema_version": SCHEMA_VERSION, "metadata": metadata, "state_dict": state}

    def export_executed_map(self) -> dict[str, Any]:
        """Export copied actual buffers for a separately privileged oracle diagnostic.

        The descriptor contains no coefficients or target information. Float32
        execution uses this map, rather than the original float64 artifact map.
        """
        self._assert_frozen()
        descriptor: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "basis_artifact_sha256": self._basis_artifact_sha256,
            "family_sha256": self.family_sha256,
            "executed_coordinate_sha256": self.executed_coordinate_sha256,
            "materialization_dtype": str(self.coefficients.dtype).removeprefix("torch."),
            "parameterization": self._parameterization,
            "coefficient_count": self.coefficient_count,
            "layout": copy.deepcopy(self._layout),
            "blocks": copy.deepcopy(self._basis_metadata["blocks"]),
            "arrays": {
                key: self._buffer(key).detach().cpu().numpy().copy() for key in self._buffer_keys
            },
        }
        descriptor["descriptor_sha256"] = stable_hash(
            {
                **{key: value for key, value in descriptor.items() if key != "arrays"},
                "arrays": {key: _array_hash(value) for key, value in descriptor["arrays"].items()},
            }
        )
        return descriptor

    @classmethod
    def from_payload(
        cls,
        payload: dict[str, Any],
        *,
        artifact: AffineFamily,
        device: str | torch.device = "cpu",
    ) -> AffineCoefficientActor:
        if (
            set(payload) != {"schema_version", "metadata", "state_dict"}
            or payload["schema_version"] != SCHEMA_VERSION
        ):
            raise ValueError("unsupported coefficient actor payload schema")
        metadata = copy.deepcopy(payload["metadata"])
        if (
            metadata.get("basis_artifact_sha256") != artifact.sha256
            or metadata.get("family_sha256") != artifact.family_sha256
        ):
            raise ValueError("payload origin disagrees with the verified shared basis artifact")
        dtype = DTYPES.get(metadata.get("materialization_dtype"))
        if dtype is None:
            raise ValueError("unsupported coefficient payload precision")
        result = cls(artifact, activation=metadata["activation"], dtype=dtype, device=device)
        expected = result.metadata
        metadata.pop("device", None)
        expected.pop("device", None)
        # Basis NPZ/JSON round-trips normalize tuples to lists while preserving
        # the canonical artifact hash and declared map.
        if stable_hash(metadata) != stable_hash(expected):
            raise ValueError(
                "coefficient payload metadata/executed map differs from the verified artifact"
            )
        result.load_state_dict(payload["state_dict"], strict=True)
        return result
