"""Pure JSON/temporary-byte validation; no actual policies or runtime imports."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import metaworld_source_extension_plan as extension
from scripts.metaworld_development_plan import DevelopmentPlan, approved_configuration, derive_seed


def synthetic_parent() -> dict[str, Any]:
    base = DevelopmentPlan(approved_configuration())
    tasks: dict[str, Any] = {}
    manifests: dict[str, Any] = {}
    for task in base.tasks:
        banks: list[dict[str, Any]] = []
        for split, count, offset in (("source", 50, 0), ("development", 20, 100)):
            goals: list[dict[str, Any]] = []
            for index in range(count):
                vector = [float(index + offset), float(index + offset + 1)]
                payload = json.dumps(vector).encode()
                goals.append(
                    {
                        "task": task,
                        "random_vector": vector,
                        "payload_base64": base64.b64encode(payload).decode(),
                        "payload_sha256": hashlib.sha256(payload).hexdigest(),
                        "goal_id": extension.stable_hash({"task": task, "random_vector": vector}),
                    }
                )
            streams = (
                base.source_streams(task) if split == "source" else base.development_streams(task)
            )
            banks.append(
                {"split": split, "seed": derive_seed(streams["task_bank"], split), "goals": goals}
            )
        manifests[task] = {
            "schema_version": 1,
            "transfer_regime": "development_only",
            "heldout_task_classes": [],
            "banks": banks,
        }
        development = banks[1]
        seeds = list(range(100, 120))
        checkpoints = []
        for milestone in (100000,) if task == "reach-v3" else (100000, 250000):
            evaluation = [
                {
                    "episode": index,
                    "goal_index": index,
                    "goal_id": goal["goal_id"],
                    "goal_bank_sha256": extension.stable_hash(development),
                    "episode_seed": seeds[index],
                    "success": task == "reach-v3" and index < 16,
                    "length": 500,
                    "complete_episode": True,
                    "terminated": False,
                    "truncated": True,
                }
                for index, goal in enumerate(development["goals"])
            ]
            checkpoints.append(
                {
                    "target_steps": milestone,
                    "actual_steps": milestone,
                    "partial": False,
                    "evaluation": evaluation,
                    "evaluation_complete": True,
                    "passed": task == "reach-v3",
                    "artifacts": {},
                }
            )
        source = base.source_streams(task)
        tasks[task] = {
            "status": "PASS" if task == "reach-v3" else "FULL_BUDGET_FAIL",
            "actual_training_steps": checkpoints[-1]["actual_steps"],
            "checkpoints": checkpoints,
            "metadata": {
                "source_streams": source,
                "development_streams": base.development_streams(task),
                "evaluation_goal_order": list(range(20)),
                "evaluation_episode_seeds": seeds,
                "initial_actor_sha256": "f" * 64,
                "actor_layout": [],
                "initialization": {
                    "rng_streams": source,
                    "sac_constructor_seed": source["critic"],
                    "training_seed_applied_after_initialization": source["training"],
                    "replay_buffer_handle_timeout_termination": True,
                    "actor_state_sha256": "a" * 64,
                    "critic_state_sha256": "b" * 64,
                    "critic_target_state_sha256": "b" * 64,
                },
            },
        }
    return {
        "phase": "FINISHED",
        "status": "COMPLETE_FAIL",
        "errors": [],
        "original_scientific_plan": base.to_dict(),
        "tasks": tasks,
        "queue": [],
        "goal_manifests": manifests,
        "ledger": {
            "source_training_steps": 600000,
            "development_evaluation_steps": 50000,
            "preflight_steps": 4,
            "source_policy_starts": 6,
            "target_steps": 0,
            "final_test_steps": 0,
            "retries": 1,
            "technical_retries": 1,
            "total_agent_steps": 650004,
        },
    }


def test_accepted_plan_caps_and_source_identity_are_exact() -> None:
    config = extension.extension_configuration()
    assert config["authorization_url"].endswith("51#issuecomment-6749201546")
    assert config["training_deadline_utc"] == "2026-11-13T02:30:00+00:00"
    assert config["deadline_utc"] == "2026-11-13T03:00:00+00:00"
    assert config["limits"]["source_training_steps"] == 2100000
    assert config["limits"]["total_agent_steps"] == 2191000
    assert config["limits"]["artifact_bytes"] == 2**31
    assert extension.validate_parent(
        synthetic_parent(), DevelopmentPlan(approved_configuration())
    ) == ("push-v3", "pick-place-v3")


@pytest.mark.parametrize(
    "field,value",
    [
        ("goal_id", "wrong"),
        ("goal_bank_sha256", "wrong"),
        ("episode_seed", 999),
        ("goal_index", 1),
        ("length", 499),
        ("truncated", False),
        ("terminated", True),
        ("success", 1),
    ],
)
def test_retained_pass_requires_actual_complete_frozen_evaluation(field: str, value: Any) -> None:
    data = synthetic_parent()
    data["tasks"]["reach-v3"]["checkpoints"][0]["evaluation"][0][field] = value
    with pytest.raises(ValueError):
        extension.validate_parent(data, DevelopmentPlan(approved_configuration()))


def test_pass_flag_cannot_replace_threshold_evidence() -> None:
    data = synthetic_parent()
    data["tasks"]["reach-v3"]["checkpoints"][0]["evaluation"][0]["success"] = False
    with pytest.raises(ValueError, match="sixteen-of-twenty"):
        extension.validate_parent(data, DevelopmentPlan(approved_configuration()))


def test_budget_stopped_parent_preserves_pending_zero_step_policy() -> None:
    data = synthetic_parent()
    data.update(status="INCOMPLETE", stop_reason="training_reserve")
    task = data["tasks"]["pick-place-v3"]
    task.update(status="PENDING", actual_training_steps=0, checkpoints=[])
    data["ledger"].update(
        source_training_steps=350000,
        development_evaluation_steps=30000,
        total_agent_steps=380004,
    )
    assert extension.validate_parent(data, DevelopmentPlan(approved_configuration())) == (
        "push-v3",
        "pick-place-v3",
    )
    data.update(status="COMPLETE_FAIL")
    with pytest.raises(ValueError, match="COMPLETE_FAIL"):
        extension.validate_parent(data, DevelopmentPlan(approved_configuration()))


def test_full_budget_label_requires_its_completed_failed_evaluation() -> None:
    data = synthetic_parent()
    data["tasks"]["push-v3"]["checkpoints"][-1]["evaluation_complete"] = False
    with pytest.raises(ValueError, match="full-budget failure"):
        extension.validate_parent(data, DevelopmentPlan(approved_configuration()))


@pytest.mark.parametrize(
    "damage", ["streams", "bank", "payload", "order", "retry", "cost", "algorithm"]
)
def test_parent_protocol_and_cost_tampering_fails(damage: str) -> None:
    data = synthetic_parent()
    if damage == "streams":
        data["tasks"]["push-v3"]["metadata"]["source_streams"]["actor"] += 1
    elif damage == "bank":
        data["goal_manifests"]["push-v3"]["banks"][0]["seed"] += 1
    elif damage == "payload":
        data["goal_manifests"]["push-v3"]["banks"][0]["goals"][0]["payload_sha256"] = "bad"
    elif damage == "order":
        data["tasks"]["push-v3"]["metadata"]["evaluation_goal_order"].reverse()
    elif damage == "retry":
        data["ledger"]["retries"] = 0
    elif damage == "cost":
        data["ledger"]["total_agent_steps"] -= 1
    else:
        data["original_scientific_plan"]["algorithm"]["gamma"] = 0.5
    with pytest.raises(ValueError):
        extension.validate_parent(data, DevelopmentPlan(approved_configuration()))


@pytest.mark.parametrize(
    "status,reason,errors",
    [
        ("ERROR", "", []),
        ("INCOMPLETE", "", ["KeyboardInterrupt"]),
        ("INCOMPLETE", "supervisor_stop_file", []),
        ("INCOMPLETE", "incomplete_training_segment", []),
    ],
)
def test_no_error_or_user_interrupt_retry(status: str, reason: str, errors: Any) -> None:
    data = synthetic_parent()
    data.update(status=status, stop_reason=reason, errors=errors)
    with pytest.raises(ValueError):
        extension.validate_parent(data, DevelopmentPlan(approved_configuration()))


def temporary_binding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any]:
    config = extension.extension_configuration()
    for key in ("failed_root", "parent_root"):
        root = tmp_path / key
        root.mkdir()
        (root / "evidence.txt").write_bytes(key.encode())
        config[key] = str(root)
    parent = synthetic_parent()
    manifest = Path(config["parent_root"]) / "development_manifest.json"
    manifest.write_text(json.dumps(parent))
    monkeypatch.setattr(extension, "process_alive", lambda pid: False)
    monkeypatch.setattr(extension, "_validate_frozen_launch", lambda config, parent: None)
    binding = extension.bind_parent(config, old_pid=999999)
    return config, binding


def test_binding_freezes_both_roots_and_actual_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, binding = temporary_binding(tmp_path, monkeypatch)
    extension.verify_binding(config, binding)
    assert binding["prior_artifact_bytes"] == sum(row["bytes"] for row in binding["inventories"])
    for damage in ("reduced_bytes", "omit_root", "third_root", "task_list"):
        changed = copy.deepcopy(binding)
        if damage == "reduced_bytes":
            changed["prior_artifact_bytes"] -= 1
        elif damage == "omit_root":
            changed["inventories"].pop()
        elif damage == "third_root":
            changed["inventories"][0]["root"] = str(tmp_path)
        else:
            changed["unfinished_tasks"] = []
        with pytest.raises(ValueError):
            extension.verify_binding(config, changed)
    (Path(config["failed_root"]) / "evidence.txt").write_text("changed")
    with pytest.raises(ValueError, match="inventory changed"):
        extension.verify_binding(config, binding)


def test_missing_and_symlink_roots_cannot_hide_storage(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing nonsymlink"):
        extension.inventory(tmp_path / "missing")
    original = tmp_path / "original"
    original.mkdir()
    linked = tmp_path / "link"
    linked.symlink_to(original, target_is_directory=True)
    with pytest.raises(ValueError, match="existing nonsymlink"):
        extension.inventory(linked)


def test_frozen_launch_hash_and_code_authority_are_enforced(tmp_path: Path) -> None:
    config = extension.extension_configuration()
    config["parent_root"] = str(tmp_path)
    (tmp_path / "frozen_launch.json").write_text("{}")
    with pytest.raises(ValueError, match="prospective frozen launch"):
        extension._validate_frozen_launch(config, synthetic_parent())


def test_frozen_launch_binds_initialization_payloads_approval_and_failed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = extension.extension_configuration()
    failed, current = tmp_path / "failed", tmp_path / "current"
    failed.mkdir()
    current.mkdir()
    (failed / "evidence.bin").write_bytes(b"synthetic failed evidence")
    config.update(failed_root=str(failed), parent_root=str(current))
    observed = extension.inventory(failed)
    prior = {"files": observed["files"], "total_bytes": observed["bytes"]}
    prior_path = current / "prior_inventory.json"
    prior_path.write_text(json.dumps(prior))
    frozen = synthetic_parent()
    frozen.update(
        approval={
            "approved": True,
            "code_commit": extension.PARENT_CODE_COMMIT,
            "authorization_url": extension.PARENT_RETRY_AUTHORIZATION,
            "code_sha256": {"synthetic.py": "a" * 64},
        },
        code_sha256={"synthetic.py": "a" * 64},
        technical_retry_contract={
            "prior_inventory_sha256": extension.digest(prior_path),
            "prior_artifact_bytes": observed["bytes"],
        },
    )
    frozen_path = current / "frozen_launch.json"
    frozen_path.write_text(json.dumps(frozen))
    monkeypatch.setattr(extension, "PARENT_FROZEN_LAUNCH_SHA256", extension.digest(frozen_path))
    extension._validate_frozen_launch(config, copy.deepcopy(frozen))
    for damage in ("initialization", "payload", "approval"):
        parent = copy.deepcopy(frozen)
        if damage == "initialization":
            parent["tasks"]["push-v3"]["metadata"]["initial_actor_sha256"] = "c" * 64
        elif damage == "payload":
            parent["goal_manifests"]["push-v3"]["banks"][0]["goals"][0]["payload_base64"] = "bad"
        else:
            parent["approval"]["code_commit"] = "d" * 40
        with pytest.raises(ValueError):
            extension._validate_frozen_launch(config, parent)
    (failed / "evidence.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="failed-attempt evidence"):
        extension._validate_frozen_launch(config, frozen)


def test_residual_storage_cannot_be_negative_or_reset_to_full_cap() -> None:
    config = extension.extension_configuration()
    binding = {"parent_manifest": synthetic_parent(), "prior_artifact_bytes": 12345}
    plan = extension.ExtensionPlan(config, binding)
    assert plan.artifact_cap_bytes == 2**31 - 12345 - config["supervisor_reserve_bytes"]
    for prior in (-1, True, 2**31):
        with pytest.raises(ValueError):
            extension.ExtensionPlan(config, {**binding, "prior_artifact_bytes": prior})
