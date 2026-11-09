from __future__ import annotations

import base64
import copy
import io
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch.autograd.gradcheck import gradcheck

from scripts.metaworld_affine_actor import AffineCoefficientActor
from scripts.metaworld_affine_oracle import (
    decode_executed_map,
    privileged_oracle_project_executed_map,
    validate_executed_map,
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
from scripts.metaworld_geometry import actor_vector, load_actor_vector
from scripts.metaworld_protocol import Goal, GoalBank, GoalManifest, SeedStreams, stable_hash

Array = npt.NDArray[np.float64]


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread() -> Iterator[None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


class TinyActor(torch.nn.Module):
    def __init__(self, activation: str) -> None:
        super().__init__()
        self.latent_pi = torch.nn.Sequential(
            torch.nn.Linear(3, 4, dtype=torch.float64),
            torch.nn.ReLU() if activation == "relu" else torch.nn.Tanh(),
            torch.nn.Linear(4, 3, dtype=torch.float64),
            torch.nn.ReLU() if activation == "relu" else torch.nn.Tanh(),
        )
        self.mu = torch.nn.Linear(3, 2, dtype=torch.float64)
        self.log_std = torch.nn.Linear(3, 2, dtype=torch.float64)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.latent_pi(features)
        return self.mu(hidden), self.log_std(hidden)


def synthetic_artifact(estimator: str = "flat", *, shift: float = 0.0) -> BasisArtifact:
    with torch.random.fork_rng(devices=[]):
        cpu_generator = torch.Generator(device="cpu").manual_seed(91)
        torch.random.set_rng_state(cpu_generator.get_state())
        actor = TinyActor("tanh")
    base, layout = actor_vector(actor)
    direction = np.linspace(-0.2, 0.3, base.size)
    source_goal = Goal("synthetic-v3", base64.b64encode(b"synthetic-source").decode(), (0.0,))
    dev_goal = Goal("synthetic-v3", base64.b64encode(b"synthetic-development").decode(), (1.0,))
    source = GoalBank("source", 1, (source_goal,))
    goals = GoalManifest((source, GoalBank("development", 2, (dev_goal,))))
    policies = []
    for index, value in enumerate((-1, 0, 1)):
        vector = base + value * direction + shift
        policies.append(
            SourcePolicy(
                policy_id=f"synthetic-{index}",
                training_run_id=f"synthetic-run-{index}",
                checkpoint_step=100,
                task_class="synthetic-v3",
                library_id="synthetic-library",
                initialization_condition="shared_init",
                training_goal_bank_sha256=source.sha256,
                actor_sha256=actor_hash_from_vector(vector, layout, "float64"),
                initial_actor_sha256=actor_hash_from_vector(base, layout, "float64"),
                rng_streams=SeedStreams.from_seed(100 + index, actor_seed=123).__dict__,
                vector=vector,
                initial_vector=base,
                layout=layout,
                parameter_dtype="float64",
            )
        )
    if estimator == "flat":
        spec = BasisFitSpec(
            "flat_population_pca", "final_weights", "raw", "fixed_predeclared", rank=1
        )
    else:
        modes: dict[str, tuple[int, ...]] = {
            item["name"]: (0,) if len(item["shape"]) == 2 else () for item in layout
        }
        spec = BasisFitSpec(
            "uwsh_v3_parameter_hosvd",
            "final_weights",
            "raw",
            "fixed_predeclared",
            parameter_modes=modes,
            mode_ranks={key: (1,) if value else () for key, value in modes.items()},
        )
    return fit_source_basis(
        policies,
        goals,
        spec,
        task_split=TaskSplitManifest(
            ("synthetic-v3",), "development_only", "synthetic-unit-test-only"
        ),
        relations=TaskRelations(),
    )


@pytest.mark.parametrize("estimator", ["flat", "modes"])
@pytest.mark.parametrize("activation", ["relu", "tanh"])
def test_reference_and_numpy_decoder_agree_for_every_actor_tensor(
    estimator: str, activation: str
) -> None:
    artifact = synthetic_artifact(estimator)
    actor = AffineCoefficientActor(artifact, activation=activation)
    reference = TinyActor(activation)
    load_actor_vector(reference, artifact.reference, artifact.metadata["layout"])
    features = torch.linspace(-1.0, 1.0, 36, dtype=torch.float64).reshape(2, 2, 3, 3)
    for expected, actual in zip(reference(features), actor(features), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    np.testing.assert_array_equal(actor.decode().detach().numpy(), artifact.reference)
    assert actor.family_sha256 == artifact.family_sha256
    assert actor.executed_coordinate_sha256 == artifact.family_sha256
    assert tuple(actor.parameter_dict()) == tuple(
        item["name"] for item in artifact.metadata["layout"]
    )
    values = torch.linspace(-0.1, 0.2, actor.coefficient_count, dtype=torch.float64)
    np.testing.assert_allclose(
        actor.decode(values).detach().numpy(), artifact.decode(values.numpy()), atol=1e-12
    )
    assert actor.head_parameters(features)[0].shape == (2, 2, 3, 2)


@pytest.mark.parametrize("estimator", ["flat", "modes"])
def test_decode_gradcheck_and_both_head_gradients_match_finite_differences(estimator: str) -> None:
    actor = AffineCoefficientActor(synthetic_artifact(estimator), activation="tanh")
    values = torch.linspace(
        -0.01, 0.02, actor.coefficient_count, dtype=torch.float64, requires_grad=True
    )
    assert gradcheck(actor.decode, (values,), eps=1e-6, atol=1e-7)
    with torch.no_grad():
        actor.coefficients.copy_(values)
    features = torch.linspace(-0.7, 0.9, 15, dtype=torch.float64).reshape(5, 3)
    mean, log_std = actor.head_parameters(features)
    assert torch.autograd.grad(mean.sum(), actor.coefficients, retain_graph=True)[0].abs().sum() > 0
    assert (
        torch.autograd.grad(log_std.sum(), actor.coefficients, retain_graph=True)[0].abs().sum() > 0
    )
    loss = mean.square().mean() + log_std.square().mean()
    expected = torch.autograd.grad(loss, actor.coefficients)[0]
    finite = torch.empty_like(expected)
    original = actor.coefficients.detach().clone()
    with torch.no_grad():
        for index in range(actor.coefficient_count):
            actor.coefficients.copy_(original)
            actor.coefficients[index] += 1e-6
            left = sum(head.square().mean() for head in actor(features))
            actor.coefficients[index] -= 2e-6
            right = sum(head.square().mean() for head in actor(features))
            finite[index] = (left - right) / 2e-6
        actor.coefficients.copy_(original)
    torch.testing.assert_close(expected, finite, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("estimator", ["flat", "modes"])
def test_only_coefficients_train_and_frozen_map_is_unchanged(estimator: str) -> None:
    actor = AffineCoefficientActor(synthetic_artifact(estimator), activation="tanh")
    assert list(dict(actor.named_parameters())) == ["coefficients"]
    assert all(not value.requires_grad for value in actor.buffers())
    frozen = {name: tensor.clone() for name, tensor in actor.named_buffers()}
    original = actor.coefficients.detach().clone()
    executed = actor.executed_coordinate_sha256
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)
    optimizer.zero_grad()
    mean, log_std = actor(torch.ones(4, 3, dtype=torch.float64))
    (mean.sum() + log_std.sum()).backward()
    optimizer.step()
    assert not torch.equal(original, actor.coefficients)
    assert actor.metadata["executed_coordinate_sha256"] == executed
    for name, value in actor.named_buffers():
        torch.testing.assert_close(value, frozen[name], rtol=0, atol=0)


@pytest.mark.parametrize("estimator", ["flat", "modes"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_payload_roundtrip_with_verified_reloaded_artifact_and_honest_costs(
    estimator: str, dtype: torch.dtype, tmp_path: Path
) -> None:
    artifact = synthetic_artifact(estimator)
    actor = AffineCoefficientActor(artifact, activation="tanh", dtype=dtype)
    np.testing.assert_array_equal(
        actor.decode().detach().numpy(),
        artifact.reference.astype(str(dtype).removeprefix("torch.")),
    )
    with torch.no_grad():
        actor.coefficients.copy_(torch.linspace(-0.03, 0.04, actor.coefficient_count, dtype=dtype))
    if dtype == torch.float32:
        assert actor.executed_coordinate_sha256 != actor.family_sha256
    exported = actor.export_payload()
    stream = io.BytesIO()
    torch.save(exported, stream)
    stream.seek(0)
    loaded = torch.load(stream, map_location="cpu", weights_only=True)
    path = tmp_path / "synthetic-source-basis.npz"
    artifact.save(path)
    restored = AffineCoefficientActor.from_payload(loaded, artifact=BasisArtifact.load(path))
    assert restored.family_sha256 == actor.family_sha256
    assert restored.executed_coordinate_sha256 == actor.executed_coordinate_sha256
    torch.testing.assert_close(restored.decode(), actor.decode(), rtol=0, atol=0)
    features = torch.ones(2, 3, dtype=dtype)
    for left, right in zip(actor(features), restored(features), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    costs = actor.logical_costs()
    actual_bytes = sum(
        value.numel() * value.element_size()
        for value in exported["state_dict"].values()
        if isinstance(value, torch.Tensor)
    )
    assert costs["state_dict_tensor_bytes"] == costs["export_payload_tensor_bytes"] == actual_bytes
    assert costs["logical_per_target_coefficient_scalars"] == artifact.coefficient_count
    assert actual_bytes > costs["per_target_coefficient_bytes"]
    assert (
        costs["logical_shared_map_scalars"]
        == artifact.metadata["storage"]["deployment_shared_scalars"]
    )


def test_dtype_changes_fail_before_mutation_and_device_only_moves_work() -> None:
    actor = AffineCoefficientActor(synthetic_artifact(), activation="tanh")
    before = {name: value.clone() for name, value in actor.named_buffers()}
    before["coefficients"] = actor.coefficients.detach().clone()
    for cast in (actor.float, actor.half, lambda: actor.to(dtype=torch.float32)):
        with pytest.raises(ValueError, match="dtype at construction"):
            cast()
        assert all(value.dtype == torch.float64 for value in actor.buffers())
        for name, tensor in actor.named_buffers():
            torch.testing.assert_close(tensor, before[name], rtol=0, atol=0)
        torch.testing.assert_close(actor.coefficients, before["coefficients"], rtol=0, atol=0)
    assert actor.to(device=torch.device("cpu")) is actor
    with pytest.raises(ValueError, match="actor interface and precision"):
        actor.head_parameters(torch.ones(2, 3, dtype=torch.float32))
    with pytest.raises(ValueError, match="actor interface and precision"):
        actor.head_parameters(torch.empty(2, 3, dtype=torch.float64, device="meta"))


@pytest.mark.parametrize(
    "damage", ["origin", "executed_hash", "activation", "buffer", "coefficient_shape"]
)
def test_payload_conflicts_and_frozen_buffer_mutations_fail_closed(damage: str) -> None:
    artifact = synthetic_artifact()
    actor = AffineCoefficientActor(artifact, activation="tanh")
    payload = actor.export_payload()
    if damage == "origin":
        payload["metadata"]["basis_artifact_sha256"] = "0" * 64
    elif damage == "executed_hash":
        payload["metadata"]["executed_coordinate_sha256"] = "0" * 64
    elif damage == "activation":
        payload["metadata"]["activation"] = "sigmoid"
    elif damage == "buffer":
        payload["state_dict"]["reference"] += 1
    else:
        payload["state_dict"]["coefficients"] = torch.zeros(actor.coefficient_count + 1)
    with pytest.raises((ValueError, RuntimeError)):
        AffineCoefficientActor.from_payload(payload, artifact=artifact)
    torch.testing.assert_close(actor.decode(), torch.tensor(artifact.reference), rtol=0, atol=0)


def test_parent_state_dict_load_rejects_map_conflict_before_coefficients_change() -> None:
    actor = AffineCoefficientActor(synthetic_artifact(), activation="tanh")
    parent = torch.nn.Module()
    parent.add_module("affine", actor)
    state = copy.deepcopy(parent.state_dict())
    state["affine.coefficients"] += 9
    state["affine.reference"] += 1
    original = actor.coefficients.detach().clone()
    with pytest.raises(RuntimeError, match="frozen buffer differs"):
        parent.load_state_dict(state)
    torch.testing.assert_close(actor.coefficients, original, rtol=0, atol=0)


def test_public_export_is_copied_and_estimator_labels_do_not_select_the_decoder() -> None:
    artifact = synthetic_artifact()
    arrays = artifact.deployment_arrays()
    arrays["reference"] += 99
    artifact.validate_integrity()
    np.testing.assert_array_equal(artifact.reference, artifact.deployment_arrays()["reference"])

    class SyntheticControl:
        @property
        def metadata(self) -> dict[str, Any]:
            return artifact.metadata | {
                "estimator": "synthetic_same_map_control",
                "estimator_revision": "test_only_v1",
            }

        parameterization = artifact.parameterization
        coefficient_count = artifact.coefficient_count
        family_sha256 = artifact.family_sha256
        sha256 = stable_hash({"parent": artifact.sha256, "control": "synthetic_same_map"})

        def deployment_arrays(self) -> dict[str, Array]:
            return artifact.deployment_arrays()

    control = AffineCoefficientActor(SyntheticControl(), activation="tanh")
    expected = artifact.decode(np.array([0.4]))
    np.testing.assert_allclose(
        control.decode(torch.tensor([0.4], dtype=torch.float64)).detach().numpy(), expected
    )
    assert control.metadata["basis_metadata"]["estimator"] == "synthetic_same_map_control"

    class SyntheticAlignedControl(SyntheticControl):
        @property
        def metadata(self) -> dict[str, Any]:
            result = super().metadata
            result.pop("spec")
            result["source_alignment_activation"] = "tanh"
            return result

    AffineCoefficientActor(SyntheticAlignedControl(), activation="tanh")
    with pytest.raises(ValueError, match="source-alignment"):
        AffineCoefficientActor(SyntheticAlignedControl(), activation="relu")


def test_parameter_mode_cores_count_all_head_bias_coefficients() -> None:
    artifact = synthetic_artifact("modes")
    actor = AffineCoefficientActor(artifact, activation="tanh")
    blocks = artifact.metadata["blocks"]
    offsets, offset = {}, 0
    for block in blocks:
        offsets[block["name"]] = offset
        offset += math.prod(block["core_shape"])
    with torch.no_grad():
        actor.coefficients[offsets["mu.bias"]] = 2
        actor.coefficients[offsets["log_std.bias"]] = 3
    expected = artifact.reference
    for item in artifact.metadata["layout"]:
        if item["name"] == "mu.bias":
            expected[item["start"]] += 2
        if item["name"] == "log_std.bias":
            expected[item["start"]] += 3
    np.testing.assert_allclose(actor.decode().detach().numpy(), expected, atol=1e-12)
    assert offset == actor.coefficient_count


@pytest.mark.parametrize("estimator", ["flat", "modes"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_executed_map_oracle_uses_actual_precision_and_same_decoder(
    estimator: str, dtype: torch.dtype
) -> None:
    actor = AffineCoefficientActor(synthetic_artifact(estimator), activation="tanh", dtype=dtype)
    descriptor = actor.export_executed_map()
    validate_executed_map(descriptor)
    coefficients = np.linspace(-0.3, 0.4, actor.coefficient_count)
    target = decode_executed_map(descriptor, coefficients).astype(np.float64)
    # The target is a synthetic vector only; privileged coefficients are not
    # supplied to a learning constructor or persisted as an adaptation result.
    result = privileged_oracle_project_executed_map(descriptor, target)
    assert result.information_access == "privileged_target_weights; diagnostic_only"
    assert result.family_sha256 == actor.family_sha256
    assert result.executed_coordinate_sha256 == actor.executed_coordinate_sha256
    assert result.coefficients.dtype == descriptor["arrays"]["reference"].dtype
    assert "not promised to be optimal" in result.numerical_note
    with torch.no_grad():
        actor.coefficients.copy_(torch.tensor(result.coefficients.copy(), dtype=dtype))
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    np.testing.assert_allclose(
        actor.decode().detach().numpy(), result.reconstruction, rtol=tolerance, atol=tolerance
    )
    np.testing.assert_allclose(result.reconstruction, target, rtol=tolerance, atol=tolerance)
    # Consumers receive copies of the frozen map.
    descriptor["arrays"]["reference"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        privileged_oracle_project_executed_map(descriptor, target)
    actor.export_executed_map()


def test_parameter_oracle_uses_pseudoinverse_for_actual_nonorthonormal_factors() -> None:
    actor = AffineCoefficientActor(
        synthetic_artifact("modes"), activation="tanh", dtype=torch.float32
    )
    descriptor = actor.export_executed_map()
    # Synthesize an honestly rebound numerical map with scaled, nonorthonormal
    # factors. This amplifies the rounding issue and makes transpose incorrect.
    for key, array in descriptor["arrays"].items():
        if key != "reference":
            array *= np.float32(1.7)
    _rebind_synthetic_descriptor(descriptor)
    target = decode_executed_map(
        descriptor, np.linspace(-0.2, 0.5, descriptor["coefficient_count"])
    ).astype(np.float64)
    result = privileged_oracle_project_executed_map(descriptor, target)
    # Verify the separable solve against a dense synthetic least-squares map.
    precise = copy.deepcopy(descriptor)
    precise["arrays"] = {key: array.astype(np.float64) for key, array in precise["arrays"].items()}
    precise["materialization_dtype"] = "float64"
    _rebind_synthetic_descriptor(precise)
    reference = precise["arrays"]["reference"]
    directions = np.column_stack(
        [
            decode_executed_map(precise, coordinate) - reference
            for coordinate in np.eye(actor.coefficient_count)
        ]
    )
    expected = np.linalg.lstsq(directions, target - reference, rcond=None)[0]
    np.testing.assert_allclose(result.least_squares_coefficients_float64, expected, atol=1e-12)
    assert result.least_squares_residual_l2 < 1e-6


def _rebind_synthetic_descriptor(descriptor: dict[str, Any]) -> None:
    """Make a new test map, never claim it is the parent's fitted family."""
    import hashlib

    hashes = {}
    for key, array in descriptor["arrays"].items():
        digest = hashlib.sha256()
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
        hashes[key] = digest.hexdigest()
    descriptor["executed_coordinate_sha256"] = stable_hash(
        {"layout": descriptor["layout"], "blocks": descriptor["blocks"], "arrays": hashes}
    )
    descriptor["family_sha256"] = stable_hash({"synthetic_map": hashes})
    descriptor["basis_artifact_sha256"] = stable_hash({"synthetic_origin": hashes})
    descriptor["descriptor_sha256"] = stable_hash(
        {
            **{
                key: value
                for key, value in descriptor.items()
                if key not in ("arrays", "descriptor_sha256")
            },
            "arrays": hashes,
        }
    )


@pytest.mark.parametrize("estimator", ["flat", "modes"])
def test_rank_zero_executed_oracle_and_reference_are_well_defined(estimator: str) -> None:
    actor = AffineCoefficientActor(synthetic_artifact(estimator), activation="tanh")
    descriptor = actor.export_executed_map()
    descriptor["coefficient_count"] = 0
    if estimator == "flat":
        descriptor["arrays"]["flat_directions"] = np.empty((0, actor.decode().numel()))
    else:
        descriptor["arrays"] = {"reference": descriptor["arrays"]["reference"]}
        for index, block in enumerate(descriptor["blocks"]):
            key = f"synthetic_zero_factor_{index}"
            block["parameter_modes"] = [0]
            block["factor_keys"] = [key]
            block["core_shape"][0] = 0
            descriptor["arrays"][key] = np.empty((block["shape"][0], 0))
    _rebind_synthetic_descriptor(descriptor)
    target = descriptor["arrays"]["reference"] + 0.5
    result = privileged_oracle_project_executed_map(descriptor, target)
    assert result.coefficients.shape == (0,)
    np.testing.assert_array_equal(result.reconstruction, descriptor["arrays"]["reference"])
    assert result.least_squares_residual_l2 > 0


def test_executed_descriptor_rejects_origin_and_shape_tampering() -> None:
    actor = AffineCoefficientActor(synthetic_artifact(), activation="tanh", dtype=torch.float32)
    descriptor = actor.export_executed_map()
    descriptor["family_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="provenance/hash"):
        validate_executed_map(descriptor)
    descriptor = actor.export_executed_map()
    descriptor["coefficient_count"] += 1
    with pytest.raises(ValueError, match="dimensions"):
        validate_executed_map(descriptor)
