"""Audit saved Meta-World actor checkpoint parameter integrity."""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import torch

if TYPE_CHECKING:
    from scripts.metaworld_geometry import actor_vector, hash_actor, load_actor_vector
else:
    from .metaworld_geometry import actor_vector, hash_actor, load_actor_vector

SCOPE_NOTE = (
    "This audit tests checkpoint parameter integrity only; environment behavior and "
    "policy competence are separate measurements."
)


def _generic_actor(actor_state: OrderedDict[str, torch.Tensor]) -> torch.nn.Module:
    """Register checkpoint parameters in an ordinary Module hierarchy."""
    root = torch.nn.Module()
    for name, tensor in actor_state.items():
        components = name.split(".")
        parent = root
        for component in components[:-1]:
            try:
                parent.get_submodule(component)
            except AttributeError:
                parent.add_module(component, torch.nn.Module())
            parent = parent.get_submodule(component)
        parent.register_parameter(
            components[-1], torch.nn.Parameter(tensor.detach().clone(), requires_grad=False)
        )
    return root


def _read_actor_checkpoint(path: Path) -> OrderedDict[str, torch.Tensor]:
    with zipfile.ZipFile(path) as archive:
        policy_names = [name for name in archive.namelist() if name.endswith("/policy.pth")]
        if "policy.pth" in archive.namelist():
            policy_names.append("policy.pth")
        if len(policy_names) != 1:
            raise ValueError(f"expected exactly one policy.pth entry, found {len(policy_names)}")
        raw_state = torch.load(
            io.BytesIO(archive.read(policy_names[0])), map_location="cpu", weights_only=True
        )

    if not isinstance(raw_state, dict):
        raise ValueError("policy.pth does not contain a state dictionary")
    actor_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in raw_state.items():
        if isinstance(key, str) and key.startswith("actor."):
            name = key.removeprefix("actor.")
            if not name or not isinstance(value, torch.Tensor):
                raise ValueError(f"invalid actor state entry {key!r}")
            actor_state[name] = value
    if not actor_state:
        raise ValueError("policy.pth contains no actor.* parameters")
    if len(actor_state) != len(set(actor_state)):
        raise ValueError("policy.pth contains duplicate actor parameter names")
    return actor_state


def _load_archive(path: Path) -> tuple[npt.NDArray[Any], list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as archive:
        if not {"vector", "layout_json"}.issubset(archive.files):
            raise ValueError("actor archive must contain vector and layout_json")
        vector = np.asarray(archive["vector"])
        layout_value = archive["layout_json"]
        if layout_value.ndim != 0:
            raise ValueError("layout_json must be a scalar string")
        layout = json.loads(str(layout_value.item()))
    if not isinstance(layout, list):
        raise ValueError("layout_json must encode a list")
    if vector.ndim != 1 or not np.issubdtype(vector.dtype, np.number):
        raise ValueError("actor vector must be a one-dimensional numeric array")
    if not np.isfinite(vector).all():
        raise ValueError("actor vector must contain only finite values")
    return vector, cast(list[dict[str, Any]], layout)


def _audit_run(record: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    policy_id = record.get("policy_id")
    result: dict[str, Any] = {"policy_id": policy_id, "valid": False, "errors": []}
    try:
        model_file = record.get("model_file")
        actor_file = record.get("actor_file")
        if not isinstance(model_file, str) or not isinstance(actor_file, str):
            raise ValueError("model_file and actor_file must be path strings")
        checkpoint_path = artifact_root / "policies" / Path(model_file).name
        actor_path = artifact_root / "policies" / Path(actor_file).name
        missing = [str(path) for path in (checkpoint_path, actor_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("missing required artifact(s): " + ", ".join(missing))

        checkpoint_actor = _read_actor_checkpoint(checkpoint_path)
        vector, archived_layout = _load_archive(actor_path)
        manifest_layout = record.get("actor_layout")
        if not isinstance(manifest_layout, list):
            raise ValueError("manifest actor_layout must be a list")
        if archived_layout != manifest_layout:
            raise ValueError("archive layout does not match manifest actor_layout")

        actor = _generic_actor(checkpoint_actor)
        load_actor_vector(actor, vector, archived_layout)
        restored_parameters = dict(actor.named_parameters())
        for name, checkpoint_tensor in checkpoint_actor.items():
            restored = restored_parameters[name]
            if (
                restored.shape != checkpoint_tensor.shape
                or restored.dtype != checkpoint_tensor.dtype
                or not torch.equal(restored.detach().cpu(), checkpoint_tensor.detach().cpu())
            ):
                raise ValueError(f"restored parameter {name!r} differs from checkpoint")

        restored_vector, _ = actor_vector(actor)
        if not np.array_equal(restored_vector, vector):
            raise ValueError("actor vector round-trip is not exact")

        expected_hash = record.get("final_actor_sha256")
        actual_hash = hash_actor(actor)
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise ValueError("restored actor hash does not match final_actor_sha256")
        result.update(
            {
                "valid": True,
                "checkpoint": str(checkpoint_path),
                "actor_archive": str(actor_path),
                "parameter_count": len(checkpoint_actor),
                "parameter_vector_size": int(vector.size),
                "actor_sha256": actual_hash,
            }
        )
    except Exception as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
    return result


def audit_saved_actors(manifest: dict[str, Any], artifact_root: Path) -> dict[str, Any]:
    """Check each source run's actor NPZ against its SB3 checkpoint on CPU."""
    runs = manifest.get("source_runs") if isinstance(manifest, dict) else None
    run_results: list[dict[str, Any]] = []
    if not isinstance(runs, list):
        errors = ["manifest source_runs must be a list"]
    else:
        errors = []
        for index, record in enumerate(runs):
            if not isinstance(record, dict):
                run_results.append(
                    {
                        "policy_id": None,
                        "valid": False,
                        "errors": [f"source_runs[{index}] must be an object"],
                    }
                )
            else:
                run_results.append(_audit_run(record, Path(artifact_root)))
    return {
        "valid": bool(run_results) and not errors and all(run["valid"] for run in run_results),
        "scope": "checkpoint_parameter_integrity",
        "scope_note": SCOPE_NOTE,
        "source_runs": run_results,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    report = audit_saved_actors(manifest, args.manifest.parent)
    serialized = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
