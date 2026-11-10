"""Prepare one explicitly approved replacement for the zero-learning serialization failure.

This is a separate, bounded attempt contract. It never loads a previous learner
or changes the scientific DevelopmentPlan, and does not authorize its own run.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import importlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from scripts.metaworld_development import DevelopmentRunner, code_hashes, digest
from scripts.metaworld_protocol import GoalManifest, stable_hash

PRIOR_BYTES = 3007135
REMAINING_BYTES = 2144476513
PRIOR_PHASE_SECONDS = 12
REMAINING_PHASE_SECONDS = 888
PRIOR_LEDGER = {
    "development_evaluation_steps": 0,
    "failed_step_calls": 0,
    "final_test_steps": 0,
    "optimizer_update_attempts": 0,
    "optimizer_updates": 0,
    "preflight_steps": 1,
    "source_policy_starts": 3,
    "source_training_steps": 0,
    "target_steps": 0,
    "total_agent_steps": 1,
}


class TechnicalRetryContract:
    def __init__(self, data: dict[str, Any], directory: Path) -> None:
        self.data = copy.deepcopy(data)
        self.directory = directory

    @classmethod
    def load(cls, path: Path) -> TechnicalRetryContract:
        return cls(json.loads(path.read_text()), path.resolve().parent)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.data)

    @property
    def sha256(self) -> str:
        return stable_hash(self.data)

    @property
    def prior_root(self) -> Path:
        return Path(self.data["prior_artifact_root"])

    @property
    def output_path(self) -> Path:
        return Path(self.data["replacement_artifact_root"])

    @property
    def inventory_path(self) -> Path:
        return self.directory / Path(str(self.data["prior_inventory_file"]))

    def validate(self, plan: Any) -> None:
        plan.validate()
        expected = {
            "schema_version": 1,
            "purpose": "one_zero_learning_technical_replacement",
            "decision_issue": 43,
            "approval_required": True,
            "base_plan_sha256": plan.sha256,
            "prior_artifact_root": str(plan.output_path),
            "replacement_artifact_root": str(Path(str(plan.output_path) + "-r2")),
            "prior_artifact_bytes": PRIOR_BYTES,
            "replacement_artifact_cap_bytes": REMAINING_BYTES,
            "cumulative_artifact_cap_bytes": 2**31,
            "prior_preflight_seconds_charged": PRIOR_PHASE_SECONDS,
            "replacement_preflight_seconds": REMAINING_PHASE_SECONDS,
            "additional_policy_starts": 3,
            "cumulative_policy_starts": 6,
            "technical_retries": 1,
            "prior_ledger": PRIOR_LEDGER,
            "session_started_utc": plan.session_started_utc.isoformat(),
            "training_deadline_utc": plan.training_deadline_utc.isoformat(),
            "deadline_utc": plan.deadline_utc.isoformat(),
            "cumulative_interaction_caps": {
                key: plan.ledger()[key]
                for key in (
                    "source_training_steps",
                    "development_evaluation_steps",
                    "preflight_steps",
                    "total_agent_steps",
                    "target_steps",
                    "final_test_steps",
                )
            },
        }
        if set(self.data) != {*expected, "prior_inventory_file", "prior_inventory_sha256"}:
            raise ValueError("unexpected or missing technical-retry contract fields")
        for key, value in expected.items():
            if stable_hash(self.data[key]) != stable_hash(value):
                raise ValueError(f"technical-retry contract differs at {key}")
        inventory_name = self.data["prior_inventory_file"]
        if not isinstance(inventory_name, str) or Path(inventory_name).name != inventory_name:
            raise ValueError("inventory must be a sibling file of the contract")
        if re.fullmatch(r"[0-9a-f]{64}", self.data["prior_inventory_sha256"]) is None:
            raise ValueError("invalid inventory hash")

    def verify_prior(self, plan: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        self.validate(plan)
        if digest(self.inventory_path) != self.data["prior_inventory_sha256"]:
            raise ValueError("prior inventory changed")
        inventory = json.loads(self.inventory_path.read_text())
        files = inventory["files"]
        expected: dict[str, Any] = {}
        for record in files:
            name = record["path"]
            if (
                not isinstance(name, str)
                or name in expected
                or Path(name).is_absolute()
                or ".." in Path(name).parts
            ):
                raise ValueError("invalid prior inventory path")
            expected[name] = record
        observed = set()
        for path in self.prior_root.rglob("*"):
            if path.is_symlink():
                raise ValueError("symlinks are forbidden in preserved prior evidence")
            if path.is_file():
                name = path.relative_to(self.prior_root).as_posix()
                observed.add(name)
                if name not in expected:
                    raise ValueError("unexpected file in prior evidence")
                record = expected[name]
                if path.stat().st_size != record["bytes"] or digest(path) != record["sha256"]:
                    raise ValueError(f"prior artifact changed: {name}")
        if observed != set(expected) or inventory["total_bytes"] != PRIOR_BYTES:
            raise ValueError("prior inventory coverage or size mismatch")
        if sum(row["bytes"] for row in files) != PRIOR_BYTES:
            raise ValueError("prior artifact total differs from the bounded contract")
        manifest = json.loads((self.prior_root / "development_manifest.json").read_text())
        if manifest["status"] != "ERROR" or stable_hash(manifest["ledger"]) != stable_hash(
            PRIOR_LEDGER
        ):
            raise ValueError("replacement is restricted to the original zero-learning failure")
        if manifest["plan_sha256"] != plan.sha256 or manifest["plan"] != plan.to_dict():
            raise ValueError("failed attempt used a different scientific plan")
        if not any(
            "pickle.PickleBuffer" in error and "has no len()" in error
            for error in manifest["errors"]
        ):
            raise ValueError("prior failure is not the authorized serialization incident")
        if set(manifest["tasks"]) != set(plan.tasks) or set(manifest["goal_manifests"]) != set(
            plan.tasks
        ):
            raise ValueError("prior attempt lacks the full frozen library")
        for task in plan.tasks:
            row = manifest["tasks"][task]
            if (
                row["status"] != "PENDING"
                or type(row["actual_training_steps"]) is not int
                or row["actual_training_steps"] != 0
                or row["checkpoints"]
            ):
                raise ValueError("prior task contains learning or evaluation outcomes")
            goals = GoalManifest.from_dict(manifest["goal_manifests"][task])
            if len(goals.for_fitting()[0].goals) != 50 or len(goals.for_tuning()[0].goals) != 20:
                raise ValueError("prior goal coverage differs from approved plan")
        return inventory, manifest

    def report(self, plan: Any) -> dict[str, Any]:
        self.verify_prior(plan)
        cumulative = dict(plan.ledger(), source_policy_starts=6, retries=1, technical_retries=1)
        remaining = dict(
            plan.ledger(),
            preflight_steps=999,
            total_agent_steps=810999,
            source_policy_starts=3,
            new_artifact_bytes=REMAINING_BYTES,
            technical_retries=0,
        )
        return {
            "technical_retry_contract_sha256": self.sha256,
            "approved": False,
            "base_plan_sha256": plan.sha256,
            "original_scientific_plan": plan.to_dict(),
            "prior_ledger": PRIOR_LEDGER,
            "remaining_attempt_caps": remaining,
            "cumulative_caps": cumulative,
            "remaining_preflight_seconds": REMAINING_PHASE_SECONDS,
            "prior_artifact_bytes": PRIOR_BYTES,
            "combined_artifact_cap_bytes": 2**31,
            "replacement_output": str(self.output_path),
            "deadline_utc": plan.deadline_utc.isoformat(),
            "training_deadline_utc": plan.training_deadline_utc.isoformat(),
        }


class ReplacementPlan:
    """Delegate the unchanged scientific plan; expose only explicit attempt bounds."""

    def __init__(self, base: Any, contract: TechnicalRetryContract) -> None:
        self.base, self.contract = base, contract
        self.output_path = contract.output_path
        self.artifact_cap_bytes = REMAINING_BYTES
        self.preflight_seconds = REMAINING_PHASE_SECONDS

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def ledger(self) -> dict[str, Any]:
        return dict(self.base.ledger(), source_policy_starts=6, retries=1, technical_retries=1)


def validate_retry_approval(
    plan: Any, contract: TechnicalRetryContract, approval: dict[str, Any]
) -> None:
    contract.validate(plan)
    if (
        approval.get("approved") is not True
        or approval.get("technical_retry_contract_sha256") != contract.sha256
    ):
        raise ValueError("a new explicit approval of this exact technical retry is required")
    if approval.get("plan_sha256") != plan.sha256:
        raise ValueError("retry approval changed the original scientific plan")
    if (
        re.fullmatch(
            r"https://github\.com/Spiph/UWSH-nethack/issues/43#issuecomment-[0-9]+",
            str(approval.get("authorization_url", "")),
        )
        is None
    ):
        raise ValueError("technical retry requires actual user approval recorded on issue 43")
    if (
        approval.get("code_sha256") != code_hashes()
        or re.fullmatch(r"[0-9a-f]{40}", str(approval.get("code_commit", ""))) is None
    ):
        raise ValueError("retry code hash or commit differs from the frozen approval")
    for key in ("session_started_utc", "deadline_utc", "training_deadline_utc"):
        if approval.get(key) != contract.data[key]:
            raise ValueError("retry approval must preserve original session deadlines")
    if approval.get("output") != str(contract.output_path):
        raise ValueError("retry approval output differs from the empty replacement location")


class TechnicalRetryRunner(DevelopmentRunner):
    def __init__(
        self, base: Any, contract: TechnicalRetryContract, approval: dict[str, Any], **kwargs: Any
    ) -> None:
        validate_retry_approval(base, contract, approval)
        self.base, self.contract = base, contract
        self.prior_inventory, self.prior_manifest = contract.verify_prior(base)
        if contract.output_path.exists() and any(contract.output_path.iterdir()):
            raise ValueError("replacement output is not empty")
        super().__init__(
            ReplacementPlan(base, contract),
            approval,
            approval_validator=lambda plan, record: validate_retry_approval(base, contract, record),
            prelearning_validator=self.verify_reconstruction,
            **kwargs,
        )
        self.ledger.update(PRIOR_LEDGER)
        self.ledger["technical_retries"] = 1
        self.ledger["retries"] = 1
        self.manifest["original_scientific_plan"] = base.to_dict()
        self.manifest["technical_retry_contract"] = contract.to_dict()
        self.manifest["technical_retry_contract_sha256"] = contract.sha256
        self.manifest["attempt_number"] = 2
        self.manifest["attempt_artifact_root"] = str(contract.output_path)
        self.manifest["continuation"] = (
            "one explicitly authorized zero-step reconstruction; within attempt 2 resident "
            "models/envs preserve optimizer/replay/RNG; no further restart"
        )
        self.store.json("technical_retry_contract.json", contract.to_dict())
        with self.store.open("prior_inventory.json", metadata=True) as handle:
            handle.write(contract.inventory_path.read_bytes())

    def verify_reconstruction(self, runner: DevelopmentRunner) -> dict[str, Any]:
        self.contract.verify_prior(self.base)
        checked = []
        for task in self.base.tasks:
            state = runner.states[task]
            if state.task != task:
                raise ValueError("replacement task identity differs")
            prior = self.prior_manifest["tasks"][task]
            if int(state.model.num_timesteps) != 0:
                raise ValueError("replacement learner is not a fresh zero-step instance")
            if state.goals.to_dict() != self.prior_manifest["goal_manifests"][task]:
                raise ValueError(f"reconstructed source/development payloads differ: {task}")
            fields = (
                "initial_actor_sha256",
                "actor_layout",
                "source_streams",
                "development_streams",
                "evaluation_goal_order",
                "evaluation_episode_seeds",
                "initialization",
            )
            for name in fields:
                if state.metadata.get(name) != prior["metadata"][name]:
                    raise ValueError(
                        f"reconstructed initialization/protocol differs: {task}/{name}"
                    )
            if hasattr(state.model, "_n_updates") and state.model._n_updates != 0:
                raise ValueError("replacement learner already performed updates")
            if hasattr(state.model, "replay_buffer") and (
                state.model.replay_buffer.pos != 0 or state.model.replay_buffer.full
            ):
                raise ValueError("replacement replay is not empty")
            checked.append(task)
        return {
            "all_three_reconstructed_before_learning": True,
            "tasks": checked,
            "prior_inventory_sha256": self.contract.data["prior_inventory_sha256"],
            "comparison": "exact payloads/streams/order/seeds and actor/critic/target hashes",
        }

    def persist(self) -> None:
        cumulative = dict(self.ledger)
        cumulative["total_agent_steps"] = sum(
            cumulative[key]
            for key in ("source_training_steps", "development_evaluation_steps", "preflight_steps")
        )
        self.manifest["technical_retry_accounting"] = {
            "prior_ledger": PRIOR_LEDGER,
            "attempt_ledger": {key: cumulative[key] - value for key, value in PRIOR_LEDGER.items()},
            "cumulative_ledger": cumulative,
            "prior_artifact_bytes": PRIOR_BYTES,
            "replacement_artifact_cap_bytes": REMAINING_BYTES,
            "cumulative_artifact_bytes_before_manifest": PRIOR_BYTES + self.store.size(),
            "cumulative_artifact_cap_bytes": 2**31,
            "preflight_seconds_charged_to_prior": PRIOR_PHASE_SECONDS,
            "additional_preflight_seconds_cap": REMAINING_PHASE_SECONDS,
        }
        super().persist()


@contextmanager
def prior_evidence_lock(contract: TechnicalRetryContract) -> Any:
    """Cooperative lock for replacement wrappers; external writers must also be absent."""
    with (contract.prior_root / "development_manifest.json").open("rb") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--plan", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args(argv)
    plan = importlib.import_module("scripts.metaworld_development_plan").DevelopmentPlan.load(
        args.config
    )
    contract = TechnicalRetryContract.load(args.contract)
    if not args.run:
        print(json.dumps(contract.report(plan), indent=2))
        return 0
    if args.approval is None:
        parser.error("--run requires a new approved issue-43 record")
    approval = json.loads(args.approval.read_text())
    validate_retry_approval(plan, contract, approval)
    from scripts.metaworld_reporting_sidecar import reporting_sidecar

    with prior_evidence_lock(contract):
        runner = TechnicalRetryRunner(plan, contract, approval)
        with reporting_sidecar(
            runner.store.root / "development_manifest.json", artifact_limit=runner.store.limit
        ):
            result = runner.run()
    print(json.dumps({"status": result["status"], "ledger": result["ledger"]}, indent=2))
    return 0 if result["status"].startswith("COMPLETE_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
