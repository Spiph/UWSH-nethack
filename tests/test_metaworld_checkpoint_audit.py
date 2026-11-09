from __future__ import annotations

import copy
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import torch

from scripts.metaworld_checkpoint_audit import audit_saved_actors, main
from scripts.metaworld_geometry import actor_vector, hash_actor


class TinyActor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Tanh())
        self.head = torch.nn.Linear(3, 1)


def _fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, npt.NDArray[np.float64], list[dict[str, Any]]]:
    torch.manual_seed(7)
    actor = TinyActor()
    vector, layout = actor_vector(actor)
    root = tmp_path / "artifacts"
    policies = root / "policies"
    policies.mkdir(parents=True)

    payload = io.BytesIO()
    torch.save({f"actor.{name}": value for name, value in actor.state_dict().items()}, payload)
    with zipfile.ZipFile(policies / "tiny.zip", "w") as archive:
        archive.writestr("tiny/policy.pth", payload.getvalue())
    np.savez_compressed(
        policies / "tiny.actor.npz",
        vector=vector,
        layout_json=np.asarray(json.dumps(layout)),
    )

    record: dict[str, Any] = {
        "policy_id": "tiny",
        "model_file": "/old/location/policies/tiny.zip",
        "actor_file": "/old/location/policies/tiny.actor.npz",
        "actor_layout": copy.deepcopy(layout),
        "final_actor_sha256": hash_actor(actor),
    }
    return {"source_runs": [record]}, root, vector, layout


def test_matching_sb3_checkpoint_and_actor_archive_pass(tmp_path: Path) -> None:
    manifest, root, _, _ = _fixture(tmp_path)

    report = audit_saved_actors(manifest, root)

    assert report["valid"] is True
    assert report["scope"] == "checkpoint_parameter_integrity"
    assert "environment behavior" in report["scope_note"]
    assert report["source_runs"][0]["valid"] is True
    assert report["source_runs"][0]["parameter_vector_size"] > 0


def test_altered_archived_vector_is_reported_as_run_failure(tmp_path: Path) -> None:
    manifest, root, vector, layout = _fixture(tmp_path)
    changed = vector.copy()
    changed[0] += 1.0
    np.savez_compressed(
        root / "policies" / "tiny.actor.npz",
        vector=changed,
        layout_json=np.asarray(json.dumps(layout)),
    )

    report = audit_saved_actors(manifest, root)

    assert report["valid"] is False
    assert report["source_runs"][0]["valid"] is False
    assert "differs from checkpoint" in report["source_runs"][0]["errors"][0]


def test_archive_layout_must_match_manifest_and_checkpoint(tmp_path: Path) -> None:
    manifest, root, vector, layout = _fixture(tmp_path)
    bad_layout = copy.deepcopy(layout)
    bad_layout[0]["name"] = "wrong.weight"
    np.savez_compressed(
        root / "policies" / "tiny.actor.npz",
        vector=vector,
        layout_json=np.asarray(json.dumps(bad_layout)),
    )

    report = audit_saved_actors(manifest, root)

    assert report["valid"] is False
    assert "archive layout does not match manifest" in report["source_runs"][0]["errors"][0]


def test_final_actor_hash_must_match(tmp_path: Path) -> None:
    manifest, root, _, _ = _fixture(tmp_path)
    record = manifest["source_runs"][0]
    record["final_actor_sha256"] = "0" * 64

    report = audit_saved_actors(manifest, root)

    assert report["valid"] is False
    assert "hash does not match" in report["source_runs"][0]["errors"][0]


def test_missing_checkpoint_and_archive_are_reported(tmp_path: Path) -> None:
    manifest, root, _, _ = _fixture(tmp_path)
    (root / "policies" / "tiny.zip").unlink()
    (root / "policies" / "tiny.actor.npz").unlink()

    report = audit_saved_actors(manifest, root)

    assert report["valid"] is False
    error = report["source_runs"][0]["errors"][0]
    assert "tiny.zip" in error
    assert "tiny.actor.npz" in error


def test_each_source_run_is_reported_even_when_another_fails(tmp_path: Path) -> None:
    manifest, root, _, _ = _fixture(tmp_path)
    manifest["source_runs"].append({"policy_id": "missing"})

    report = audit_saved_actors(manifest, root)

    assert report["valid"] is False
    assert len(report["source_runs"]) == 2
    assert report["source_runs"][0]["valid"] is True
    assert report["source_runs"][1]["valid"] is False


def test_cli_creates_output_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, root, _, _ = _fixture(tmp_path)
    manifest_path = root / "pilot_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_path = tmp_path / "new" / "validation" / "audit.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "metaworld_checkpoint_audit",
            "--manifest",
            str(manifest_path),
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()

    assert exit_code == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["valid"] is True
