"""Synthetic cross-module checks; no SB3, environments or experiment data."""

from __future__ import annotations

import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import torch
from test_metaworld_affine_actor import synthetic_artifact

from scripts.metaworld_adaptation_contract import RandomControlFamily
from scripts.metaworld_affine_actor import AffineCoefficientActor
from scripts.metaworld_affine_oracle import privileged_oracle_project_executed_map
from scripts.metaworld_basis import BasisArtifact


@pytest.fixture(scope="module", autouse=True)
def one_cpu_thread() -> Iterator[None]:
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(original)


@pytest.mark.parametrize("estimator", ["flat", "modes"])
def test_real_random_control_actor_oracle_and_verified_reload(
    estimator: str, tmp_path: Path
) -> None:
    parent = synthetic_artifact(estimator)
    control = RandomControlFamily.from_basis(parent, seed=173)
    actor = AffineCoefficientActor(control, activation="tanh", dtype=torch.float32)
    coefficients = torch.linspace(-0.2, 0.3, actor.coefficient_count, dtype=torch.float32)
    with torch.no_grad():
        actor.coefficients.copy_(coefficients)
    executed = actor.decode().detach().numpy().copy()
    # The control's NumPy API uses its float64 origin. The actor executes cast
    # reference/factors, so agreement here explicitly permits rounding.
    np.testing.assert_allclose(
        executed,
        control.decode(coefficients.numpy()).astype(np.float32),
        rtol=1e-6,
        atol=1e-7,
    )
    assert actor.metadata["basis_metadata"]["estimator"] == control.metadata["estimator"]
    assert actor.family_sha256 == control.family_sha256
    assert actor.executed_coordinate_sha256 != actor.family_sha256

    descriptor = actor.export_executed_map()
    diagnostic = privileged_oracle_project_executed_map(descriptor, executed)
    assert diagnostic.information_access == "privileged_target_weights; diagnostic_only"
    assert diagnostic.executed_coordinate_sha256 == actor.executed_coordinate_sha256
    np.testing.assert_allclose(diagnostic.reconstruction, executed, rtol=1e-6, atol=1e-7)
    # Privileged coordinates remain diagnostic outputs. They never initialize
    # the actor or change the practical coefficient state saved below.
    torch.testing.assert_close(actor.coefficients, coefficients, rtol=0, atol=0)

    parent_path, control_path = tmp_path / "parent.npz", tmp_path / "control.npz"
    parent.save(parent_path)
    control.save(control_path)
    verified_parent = BasisArtifact.load(parent_path)
    verified_control = RandomControlFamily.load(control_path, parent=verified_parent)
    assert verified_control.sha256 == control.sha256
    stream = io.BytesIO()
    torch.save(actor.export_payload(), stream)
    stream.seek(0)
    payload = torch.load(stream, map_location="cpu", weights_only=True)
    restored = AffineCoefficientActor.from_payload(payload, artifact=verified_control)
    assert restored.family_sha256 == actor.family_sha256
    assert restored.executed_coordinate_sha256 == actor.executed_coordinate_sha256
    assert restored.export_executed_map()["descriptor_sha256"] == descriptor["descriptor_sha256"]
    torch.testing.assert_close(restored.coefficients, coefficients, rtol=0, atol=0)
    torch.testing.assert_close(restored.decode(), actor.decode(), rtol=0, atol=0)
    np.testing.assert_allclose(
        verified_control.decode(coefficients.numpy()),
        control.decode(coefficients.numpy()),
        rtol=0,
        atol=0,
    )
