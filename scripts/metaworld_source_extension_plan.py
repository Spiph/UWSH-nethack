"""Dependency-free issue-51 amendment, late parent binding and resumed schedule."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.metaworld_development_plan import DevelopmentPlan, approved_configuration, derive_seed

AUTHORIZATION_URL = "https://github.com/Spiph/UWSH-nethack/issues/51#issuecomment-6749201546"
DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "experiments/metaworld/source_extension_2026-11-09.json"
)
OLD_ROOT = Path("/home/icannon/code/UWSH-nethack/artifacts/metaworld-development-2026-11-09-r2")
FAILED_ROOT = Path("/home/icannon/code/UWSH-nethack/artifacts/metaworld-development-2026-11-09")
PARENT_CODE_COMMIT = "c2cd64fd5fa697618f72d34ea3bf46a553ca8f4b"
# Immutable JSON captured before the existing source process learned anything.
# This binds exact goal payloads, episode seeds, initialization and runtime pins
# without loading a policy, pickle or benchmark environment during preparation.
PARENT_FROZEN_LAUNCH_SHA256 = "8398d1c7ebb0107923ec8dee501786387b30f3da0257011aed9997b97d837c74"
PARENT_RETRY_AUTHORIZATION = (
    "https://github.com/Spiph/UWSH-nethack/issues/43#issuecomment-6747418460"
)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def extension_configuration() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "decision_issue": 51,
        "authorization_url": AUTHORIZATION_URL,
        "original_plan_sha256": DevelopmentPlan(approved_configuration()).sha256,
        "parent_root": str(OLD_ROOT),
        "failed_root": str(FAILED_ROOT),
        "output": str(OLD_ROOT) + "-extension",
        "milestones": [100000, 250000, 500000, 1000000],
        "session_started_utc": "2026-11-09T15:35:00+00:00",
        "training_deadline_utc": "2026-11-13T02:30:00+00:00",
        "deadline_utc": "2026-11-13T03:00:00+00:00",
        "reserve_seconds": 1800,
        "preflight_seconds": 871,
        "prior_preflight_seconds_charged": 29,
        "supervisor_reserve_bytes": 16 * 1024**2,
        "limits": {
            "source_training_steps": 2100000,
            "development_evaluation_steps": 90000,
            "preflight_steps": 1000,
            "total_agent_steps": 2191000,
            "source_policy_starts": 8,
            "target_steps": 0,
            "final_test_steps": 0,
            "artifact_bytes": 2**31,
            "additional_sac_reloads": 2,
        },
        "source_policy_identity": "unchanged; necessary reloads are not independent policies",
        "parent_binding": "after PID exits and FINISHED; before extended learning",
        "partial_evaluation": "retain/charge partial; redo unfinished episode within remaining cap",
    }


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def inventory(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("parent artifact root must be an existing nonsymlink directory")
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("parent artifacts cannot contain symlinks")
        if path.is_file():
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
    return {"root": str(root), "files": records, "bytes": sum(row["bytes"] for row in records)}


def _residual_bytes(config: dict[str, Any], prior_bytes: Any) -> int:
    prior = _count(prior_bytes, "prior_artifact_bytes")
    reserve = _count(config["supervisor_reserve_bytes"], "supervisor_reserve_bytes")
    cap = _count(config["limits"]["artifact_bytes"], "artifact_bytes")
    remaining = cap - prior - reserve
    if remaining <= 0:
        raise ValueError("parent artifacts leave no approved storage for continuation")
    return remaining


def _count(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _task_protocol(parent: dict[str, Any], base: DevelopmentPlan, task: str) -> tuple[Any, Any]:
    metadata = parent["tasks"][task]["metadata"]
    if (
        metadata.get("source_streams") != base.source_streams(task)
        or metadata.get("development_streams") != base.development_streams(task)
        or metadata.get("evaluation_goal_order") != list(range(20))
    ):
        raise ValueError("parent changed the original source/development streams or goal order")
    seeds = metadata.get("evaluation_episode_seeds", [])
    if len(seeds) != 20 or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("parent lacks the twenty frozen development episode seeds")
    initialization = metadata.get("initialization", {})
    streams = base.source_streams(task)
    if (
        initialization.get("rng_streams") != streams
        or initialization.get("sac_constructor_seed") != streams["critic"]
        or initialization.get("training_seed_applied_after_initialization") != streams["training"]
        or initialization.get("replay_buffer_handle_timeout_termination") is not True
        or initialization.get("critic_state_sha256")
        != initialization.get("critic_target_state_sha256")
        or any(
            re.fullmatch(r"[0-9a-f]{64}", str(initialization.get(key, ""))) is None
            for key in ("actor_state_sha256", "critic_state_sha256", "critic_target_state_sha256")
        )
    ):
        raise ValueError("parent initialization differs from the original model/RNG protocol")
    manifest = parent["goal_manifests"][task]
    if (
        manifest.get("transfer_regime") != "development_only"
        or manifest.get("heldout_task_classes") != []
        or len(manifest.get("banks", [])) != 2
    ):
        raise ValueError("parent goal manifest must contain source/development data only")
    banks = {bank["split"]: bank for bank in manifest["banks"]}
    if set(banks) != {"source", "development"}:
        raise ValueError("parent must retain exactly source and development banks")
    seen: set[str] = set()
    for split, count in (("source", 50), ("development", 20)):
        bank = banks[split]
        streams = base.source_streams(task) if split == "source" else base.development_streams(task)
        if (
            bank.get("seed") != derive_seed(streams["task_bank"], split)
            or len(bank["goals"]) != count
        ):
            raise ValueError("parent bank seed/count differs from the original plan")
        for goal in bank["goals"]:
            vector = goal.get("random_vector", [])
            if (
                goal.get("task") != task
                or not vector
                or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in vector
                )
            ):
                raise ValueError("invalid frozen goal coordinates")
            goal_id = stable_hash({"task": task, "random_vector": vector})
            payload = base64.b64decode(goal["payload_base64"], validate=True)
            if (
                not payload
                or goal.get("goal_id") != goal_id
                or goal.get("payload_sha256") != hashlib.sha256(payload).hexdigest()
                or goal_id in seen
            ):
                raise ValueError("goal payload/hash mismatch or source/development leakage")
            seen.add(goal_id)
    return banks["development"], seeds


def _validate_completed_evaluation(row: dict[str, Any], bank: Any, seeds: Any) -> None:
    if (
        not evaluation_complete(row)
        or row.get("partial") is not False
        or row.get("target_steps") not in (100000, 250000)
        or row.get("actual_steps") != row.get("target_steps")
    ):
        raise ValueError("completed evaluation must retain twenty full truncated episodes")
    for index, episode in enumerate(row["evaluation"]):
        expected = {
            "episode": index,
            "goal_index": index,
            "goal_id": bank["goals"][index]["goal_id"],
            "goal_bank_sha256": stable_hash(bank),
            "episode_seed": seeds[index],
        }
        if (
            any(episode.get(key) != value for key, value in expected.items())
            or type(episode.get("success")) is not bool
        ):
            raise ValueError(
                "completed evaluation changed frozen goal IDs, order, seeds or success"
            )
    passed = sum(episode["success"] for episode in row["evaluation"]) >= 16
    if row.get("passed") is not passed:
        raise ValueError("checkpoint pass flag disagrees with the fixed sixteen-of-twenty gate")


def _validate_frozen_launch(config: dict[str, Any], parent: dict[str, Any]) -> None:
    root = Path(config["parent_root"])
    frozen_path = root / "frozen_launch.json"
    if digest(frozen_path) != PARENT_FROZEN_LAUNCH_SHA256:
        raise ValueError("parent prospective frozen launch changed from the original c2cd process")
    frozen = json.loads(frozen_path.read_text())
    approval = frozen["approval"]
    if (
        approval.get("approved") is not True
        or approval.get("code_commit") != PARENT_CODE_COMMIT
        or approval.get("authorization_url") != PARENT_RETRY_AUTHORIZATION
        or approval.get("code_sha256") != frozen.get("code_sha256")
    ):
        raise ValueError("parent did not use the authorized original frozen source code")
    fields = (
        "approval",
        "code_sha256",
        "plan",
        "plan_sha256",
        "original_scientific_plan",
        "technical_retry_contract",
        "technical_retry_contract_sha256",
        "goal_manifests",
        "prelearning_verification",
        "runtime",
        "session_started_utc",
        "training_deadline_utc",
        "absolute_deadline_utc",
    )
    if any(parent.get(key) != frozen.get(key) for key in fields):
        raise ValueError("final parent changed a prospective source/protocol/approval binding")
    metadata_fields = (
        "initialization",
        "initial_actor_sha256",
        "actor_layout",
        "source_streams",
        "development_streams",
        "evaluation_goal_order",
        "evaluation_episode_seeds",
    )
    for task in DevelopmentPlan(approved_configuration()).tasks:
        final = parent["tasks"][task]["metadata"]
        initial = frozen["tasks"][task]["metadata"]
        if any(final.get(key) != initial.get(key) for key in metadata_fields):
            raise ValueError("parent changed its original policy initialization or frozen protocol")
    prior_path = root / "prior_inventory.json"
    contract = frozen["technical_retry_contract"]
    if digest(prior_path) != contract["prior_inventory_sha256"]:
        raise ValueError("authorized failed-attempt inventory changed")
    prior = json.loads(prior_path.read_text())
    actual = inventory(Path(config["failed_root"]))
    if (
        {row["path"]: row for row in actual["files"]}
        != {row["path"]: row for row in prior["files"]}
        or actual["bytes"] != prior["total_bytes"]
        or actual["bytes"] != contract["prior_artifact_bytes"]
    ):
        raise ValueError("original failed-attempt evidence changed or its bytes were omitted")


def validate_parent(parent: dict[str, Any], base: DevelopmentPlan) -> tuple[str, ...]:
    if parent.get("phase") != "FINISHED" or parent.get("errors"):
        raise ValueError("parent must have finished without ERROR or user interruption")
    status = parent.get("status")
    reason = parent.get("stop_reason") or parent.get("cutoff_reason", "")
    if status not in ("COMPLETE_PASS", "COMPLETE_FAIL") and not (
        status == "INCOMPLETE"
        and isinstance(reason, str)
        and (
            reason == "training_reserve"
            or reason.startswith("absolute_utc_deadline")
            or reason.startswith("monotonic_run_budget")
        )
    ):
        raise ValueError("only a completed or budget-stopped parent can continue")
    if parent.get("original_scientific_plan", parent.get("plan")) != base.to_dict():
        raise ValueError("parent scientific plan differs from the approved original")
    tasks = parent.get("tasks", {})
    if set(tasks) != set(base.tasks) or tasks["reach-v3"].get("status") != "PASS":
        raise ValueError("all original tasks and the already-passing reach policy must be retained")
    if tasks["reach-v3"].get("actual_training_steps") != 100000:
        raise ValueError("reach must remain stopped at its original 100k pass")
    ledger = parent["ledger"]
    if ledger.get("source_policy_starts") != 6 or ledger.get("preflight_steps") != 4:
        raise ValueError(
            "parent must include the failed attempt exactly once (6 starts/4 preflight)"
        )
    if ledger.get("target_steps") != 0 or ledger.get("final_test_steps") != 0:
        raise ValueError("parent accessed target/final interactions")
    for key, cap in (
        ("source_training_steps", 750000),
        ("development_evaluation_steps", 60000),
        ("preflight_steps", 1000),
    ):
        if _count(ledger.get(key), key) > cap:
            raise ValueError("parent interaction counts exceed its original allocation")
    if ledger.get("total_agent_steps") != sum(
        ledger[key]
        for key in ("source_training_steps", "development_evaluation_steps", "preflight_steps")
    ):
        raise ValueError("parent aggregate cost does not include every interaction category")
    if ledger.get("retries") != 1 or ledger.get("technical_retries") != 1:
        raise ValueError("parent must retain the one explicitly authorized technical retry")
    for task in base.tasks:
        task_row = tasks[task]
        steps = _count(task_row.get("actual_training_steps"), "actual_training_steps")
        if steps > 250000 or task_row.get("status") not in ("PASS", "PENDING", "FULL_BUDGET_FAIL"):
            raise ValueError("parent task status/steps differ from its original finite schedule")
        bank, seeds = _task_protocol(parent, base, task)
        for checkpoint in task_row.get("checkpoints", []):
            if checkpoint.get("evaluation_complete") is True:
                _validate_completed_evaluation(checkpoint, bank, seeds)
        if task_row["status"] == "PASS":
            checkpoints = task_row.get("checkpoints", [])
            if (
                not checkpoints
                or checkpoints[-1].get("actual_steps") != steps
                or checkpoints[-1].get("target_steps") not in (100000, 250000)
                or checkpoints[-1].get("partial") is not False
                or not evaluation_complete(checkpoints[-1])
                or checkpoints[-1].get("passed") is not True
            ):
                raise ValueError("retained PASS lacks its complete matching passing checkpoint")
        if task_row["status"] == "FULL_BUDGET_FAIL":
            checkpoints = task_row.get("checkpoints", [])
            if (
                steps != 250000
                or not checkpoints
                or checkpoints[-1].get("actual_steps") != 250000
                or not evaluation_complete(checkpoints[-1])
                or checkpoints[-1].get("passed") is not False
            ):
                raise ValueError("parent full-budget failure lacks its completed 250k evidence")
    if status == "COMPLETE_PASS" and any(tasks[task]["status"] != "PASS" for task in base.tasks):
        raise ValueError("parent COMPLETE_PASS disagrees with its retained task statuses")
    if status == "COMPLETE_FAIL" and (
        any(tasks[task]["status"] == "PENDING" for task in base.tasks)
        or all(tasks[task]["status"] == "PASS" for task in base.tasks)
    ):
        raise ValueError("parent COMPLETE_FAIL disagrees with its retained task statuses")
    unfinished = tuple(task for task in base.tasks if tasks[task]["status"] != "PASS")
    if "reach-v3" in unfinished or len(unfinished) > 2:
        raise ValueError("only the same unfinished push/pick-place learners may reload")
    for task in unfinished:
        row = tasks[task]
        if (not row.get("checkpoints") and row["actual_training_steps"] != 0) or (
            row.get("checkpoints")
            and row["checkpoints"][-1]["actual_steps"] != row["actual_training_steps"]
        ):
            raise ValueError("parent lacks a current complete checkpoint inventory")
    return unfinished


def bind_parent(config: dict[str, Any], *, old_pid: int) -> dict[str, Any]:
    if process_alive(old_pid):
        raise ValueError("the original frozen process is still alive")
    base = DevelopmentPlan(approved_configuration())
    root = Path(config["parent_root"])
    manifest = root / "development_manifest.json"
    parent = json.loads(manifest.read_text())
    unfinished = validate_parent(parent, base)
    _validate_frozen_launch(config, parent)
    prior = [inventory(Path(config[key])) for key in ("failed_root", "parent_root")]
    total_bytes = sum(record["bytes"] for record in prior)
    if total_bytes + config["supervisor_reserve_bytes"] >= config["limits"]["artifact_bytes"]:
        raise ValueError("parent artifacts leave no approved storage for continuation")
    return {
        "schema_version": 1,
        "parent_manifest_sha256": digest(manifest),
        "parent_manifest": parent,
        "inventories": prior,
        "prior_artifact_bytes": total_bytes,
        "unfinished_tasks": list(unfinished),
        "old_pid": old_pid,
        "bound_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding_method": "PID gone + FINISHED + complete immutable file inventory",
    }


def verify_binding(config: dict[str, Any], binding: dict[str, Any]) -> None:
    if process_alive(binding["old_pid"]):
        raise ValueError("parent process unexpectedly exists during handoff")
    manifest = Path(config["parent_root"]) / "development_manifest.json"
    if digest(manifest) != binding["parent_manifest_sha256"]:
        raise ValueError("parent manifest changed after freezing")
    records = binding["inventories"]
    if (
        len(records) != 2
        or {row["root"] for row in records} != {config["failed_root"], config["parent_root"]}
        or sum(row["bytes"] for row in records) != binding["prior_artifact_bytes"]
    ):
        raise ValueError("binding must count both parent roots and actual cumulative bytes")
    for saved in records:
        if inventory(Path(saved["root"])) != saved:
            raise ValueError("parent artifact inventory changed after freezing")
    if json.loads(manifest.read_text()) != binding["parent_manifest"]:
        raise ValueError("bound parent content differs")
    unfinished = validate_parent(
        binding["parent_manifest"], DevelopmentPlan(approved_configuration())
    )
    if list(unfinished) != binding["unfinished_tasks"]:
        raise ValueError("bound unfinished tasks differ from the actual parent")
    _validate_frozen_launch(config, binding["parent_manifest"])
    _residual_bytes(config, binding["prior_artifact_bytes"])


def evaluation_complete(row: dict[str, Any]) -> bool:
    rows = row.get("evaluation", [])
    return (
        row.get("evaluation_complete") is True
        and len(rows) == 20
        and all(
            item.get("complete_episode") is True
            and item.get("length") == 500
            and item.get("terminated") is False
            and item.get("truncated") is True
            for item in rows
        )
    )


def continuation_schedule(parent: dict[str, Any], tasks: tuple[str, ...]) -> list[tuple[str, int]]:
    complete = {
        (task, row["target_steps"])
        for task in tasks
        for row in parent["tasks"][task]["checkpoints"]
        if evaluation_complete(row)
    }
    schedule: list[tuple[str, int]] = []
    unfinished_visits = [
        row
        for row in parent.get("queue", [])
        if row.get("task") in tasks and row.get("status") == "TRAINING"
    ]
    for visit in unfinished_visits:
        pair = visit["task"], visit["milestone"]
        if pair not in complete and pair not in schedule:
            schedule.append(pair)
    for milestone in (100000, 250000, 500000, 1000000):
        for task in tasks:
            pair = task, milestone
            if pair not in complete and pair not in schedule:
                if milestone < parent["tasks"][task]["actual_training_steps"]:
                    raise ValueError("a prior milestone was advanced without a complete evaluation")
                schedule.append(pair)
    return schedule


class ExtensionPlan:
    def __init__(self, config: dict[str, Any], binding: dict[str, Any]) -> None:
        if stable_hash(config) != stable_hash(extension_configuration()):
            raise ValueError("extension config differs from the exact approved amendment")
        self.config, self.binding = copy.deepcopy(config), copy.deepcopy(binding)
        self.base = DevelopmentPlan(approved_configuration())
        self.parent = self.binding["parent_manifest"]
        self.tasks = validate_parent(self.parent, self.base)
        self.output_path = Path(config["output"])
        self.artifact_cap_bytes = _residual_bytes(config, binding["prior_artifact_bytes"])
        self.milestones = (100000, 250000, 500000, 1000000)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    @property
    def sha256(self) -> str:
        return stable_hash(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.base.to_dict(),
            "extension": copy.deepcopy(self.config),
            "parent_binding_sha256": stable_hash(self.binding),
            "original_scientific_plan": self.base.to_dict(),
        }

    @property
    def deadline_utc(self) -> datetime:
        return datetime.fromisoformat(self.config["deadline_utc"])

    @property
    def training_deadline_utc(self) -> datetime:
        return datetime.fromisoformat(self.config["training_deadline_utc"])

    @property
    def preflight_seconds(self) -> int:
        return 871

    @property
    def max_training_steps(self) -> int:
        return 2100000

    @property
    def max_eval_steps(self) -> int:
        return 90000

    def ledger(self) -> dict[str, int]:
        return dict(self.config["limits"])

    def schedule(self) -> list[tuple[str, int]]:
        return continuation_schedule(self.parent, self.tasks)
