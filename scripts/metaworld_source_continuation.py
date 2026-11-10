"""Reload-only source-development amendment; never starts a replacement library."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import pickle
import re
import subprocess
import time
import traceback
import zipfile
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Any

from scripts.metaworld_deadlines import BudgetExpired, RunDeadline, parse_deadline, utc_now
from scripts.metaworld_development import (
    DevelopmentRunner,
    LearnerRNG,
    Runtime,
    Stopped,
    TaskState,
    code_hashes,
    digest,
)
from scripts.metaworld_protocol import (
    GoalManifest,
    SeedStreams,
    evaluation_order,
    validate_evaluation_coverage,
)
from scripts.metaworld_source_extension_plan import (
    AUTHORIZATION_URL,
    DEFAULT_CONFIG,
    ExtensionPlan,
    evaluation_complete,
    extension_configuration,
    stable_hash,
    verify_binding,
)


def validate_extension_approval(config: dict[str, Any], approval: dict[str, Any]) -> None:
    if (
        approval.get("approved") is not True
        or approval.get("authorization_url") != AUTHORIZATION_URL
        or approval.get("extension_sha256") != stable_hash(config)
        or approval.get("code_sha256") != code_hashes()
        or re.fullmatch(r"[0-9a-f]{40}", str(approval.get("code_commit", ""))) is None
    ):
        raise ValueError("actual issue-51 approval must bind the exact amendment and frozen code")
    for key in ("deadline_utc", "training_deadline_utc", "session_started_utc"):
        if approval.get(key) != config[key]:
            raise ValueError("extension approval changed the fixed inclusive session/cutoffs")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], text=True, timeout=3
    ).strip()
    if head != approval["code_commit"]:
        raise ValueError("execution checkout differs from the frozen approval commit")


def _artifact(root: Path, record: dict[str, Any]) -> Path:
    relative = Path(record["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("checkpoint paths must remain under the immutable parent root")
    path = root / relative
    if digest(path) != record["sha256"]:
        raise ValueError("selected checkpoint differs from its frozen file hash")
    return path


def selected_checkpoint(plan: ExtensionPlan, task: str) -> dict[str, Any]:
    parent = plan.parent["tasks"][task]
    if not parent["checkpoints"] or parent["actual_training_steps"] < 1000:
        raise ValueError("this actual-parent continuation requires a checkpoint beyond warmup")
    return copy.deepcopy(parent["checkpoints"][-1])


def gated_sac_type(sac: Any, check: Callable[[], None]) -> Any:
    """Check the policy/replay allocation boundary inside native classmethod load."""

    def setup(model: Any) -> None:
        check()
        sac._setup_model(model)

    return type("SourceContinuationSAC", (sac,), {"_setup_model": setup})


def inherited_monotonic_cutoff(
    binding: dict[str, Any], cutoff: float | None, deadline_utc: str
) -> float:
    window = binding.get("supervisor_window", {})
    if (
        cutoff is None
        or not math.isfinite(cutoff)
        or cutoff != window.get("monotonic_cutoff")
        or window.get("deadline_utc") != deadline_utc
        or not isinstance(window.get("started_at_utc"), str)
        or not isinstance(window.get("started_monotonic"), (int, float))
        or not math.isfinite(window["started_monotonic"])
        or window["started_monotonic"] >= cutoff
    ):
        raise ValueError("child cutoff must match the finite frozen supervisor window")
    started = parse_deadline(window["started_at_utc"])
    deadline = parse_deadline(deadline_utc)
    duration = (deadline - started).total_seconds()
    if duration <= 0 or cutoff != window["started_monotonic"] + duration:
        raise ValueError("frozen supervisor UTC and monotonic session bounds disagree")
    return cutoff


class ContinuationRuntime(Runtime):
    """Same native SAC learner/optimizer/replay, restored from the parent files."""

    def prepare(self, task: str) -> TaskState:
        runner = self.runner
        parent = runner.plan.parent["tasks"][task]
        checkpoint = selected_checkpoint(runner.plan, task)
        files = checkpoint["artifacts"]
        root = Path(runner.plan.config["parent_root"])
        goals = GoalManifest.from_dict(runner.plan.parent["goal_manifests"][task])
        metadata = copy.deepcopy(parent["metadata"])
        if (
            metadata["source_streams"] != runner.plan.source_streams(task)
            or metadata["development_streams"] != runner.plan.development_streams(task)
            or len(goals.for_fitting()[0].goals) != 50
            or len(goals.for_tuning()[0].goals) != 20
        ):
            raise ValueError("parent source/development protocol differs from the original design")
        metadata["initial_actor_artifact_root"] = str(root)
        runner.manifest.setdefault("recovery_attempts", {})[task] = {
            "verified": False,
            "status": "LOADING",
        }
        runner.persist()
        runner.check("initialization")
        zip_path = _artifact(root, files[".zip"])
        replay_path = _artifact(root, files[".replay.pkl"])
        rng_path = _artifact(root, files["rng"])
        runner.check("initialization")
        with self.preserve_rng():
            runner.check("initialization")
            env = self.pilot.make_env(
                task,
                SeedStreams(**metadata["source_streams"]).constructor,
                bank=goals.for_fitting()[0],
                streams=SeedStreams(**metadata["source_streams"]),
            )
            try:
                sac = importlib.import_module("stable_baselines3").SAC

                gated_sac = gated_sac_type(sac, lambda: runner.check("training"))
                runner.start_policy()
                model = gated_sac.load(zip_path, env=env, device="cuda", force_reset=False)
                runner.check("initialization")
                model.load_replay_buffer(replay_path)
                if int(model.num_timesteps) != parent["actual_training_steps"]:
                    raise ValueError("loaded timestep counter differs from the parent checkpoint")
                logger = importlib.import_module("stable_baselines3.common.logger")
                model.set_logger(
                    logger.configure(
                        folder=str(runner.store.root / "logs" / task), format_strings=[]
                    )
                )
                if self.geometry.hash_actor(model.actor) != files["actor_named_parameter_sha256"]:
                    raise ValueError("loaded actor differs from the frozen named-parameter hash")
                runner.check("initialization")
                with rng_path.open("rb") as handle:
                    rng_state = pickle.load(handle)
                history = files["environment"]["episode_history"]
                recovery = importlib.import_module("scripts.metaworld_source_recovery")
                plan = recovery.inspect_checkpoint_counters(
                    num_timesteps=int(model.num_timesteps),
                    replay_position=int(model.replay_buffer.pos),
                    replay_full=bool(model.replay_buffer.full),
                    replay_buffer_size=int(model.replay_buffer.buffer_size),
                    n_updates=int(model._n_updates),
                    episode_history=history,
                    learning_starts=runner.plan.algorithm["learning_starts"],
                    preflight_prefix_length=0,
                )
                before = self.learner_hashes(model, checkpoint=zip_path)

                def charge() -> None:
                    runner.check("preflight")
                    if runner.ledger["preflight_steps"] >= 1000:
                        raise Stopped("preflight_steps_limit")
                    runner.ledger["preflight_steps"] += 1

                controller = env.get_wrapper_attr("protocol_controller")()
                runner.manifest.setdefault("recovery_attempts", {})[task] = {
                    "verified": False,
                    "counters": plan.to_dict(),
                    "status": "RECONSTRUCTING",
                }
                runner.persist()
                report = recovery.recover_committed_environment(
                    model=model,
                    controller=controller,
                    history=history,
                    plan=plan,
                    check=lambda: runner.check("preflight"),
                    charge_preflight=charge,
                )
                after = self.learner_hashes(model)
                if before != after:
                    raise ValueError(
                        "environment reconstruction changed learner/replay/optimizer state"
                    )
                runner.manifest["recovery_attempts"][task].update(
                    verified=True, status="VERIFIED", report=report
                )
                runner.bind_env(env, "training")
                metadata["recovery"] = {
                    "counters": plan.to_dict(),
                    "verification": report,
                    "learner_hashes_before": before,
                    "learner_hashes_after": after,
                }
                metadata["physical_training_steps"] = plan.physical_steps
                metadata["physical_step_offset"] = plan.model_uncounted_physical_steps
                metadata["pending_optimizer_updates"] = plan.pending_optimizer_updates
                metadata["same_policy_identity"] = task
                metadata["continuation_parent_root"] = str(root)
                metadata["saved_learner_rng_file_sha256"] = files["rng"]["sha256"]
                metadata["diagnostic_history"] = (
                    "Monitor timing/history segmented at process continuation"
                )
                checkpoints = copy.deepcopy(parent["checkpoints"])
                for row in checkpoints:
                    row["artifact_root"] = str(root)
                state = TaskState(
                    task,
                    model,
                    env,
                    goals,
                    LearnerRNG(rng_state, self.torch, True),
                    metadata,
                    checkpoints=checkpoints,
                )
                return state
            except BaseException as error:
                runner.manifest["recovery_attempts"][task].update(
                    verified=False, status="ERROR", error=f"{type(error).__name__}: {error}"
                )
                runner.persist()
                env.close()
                raise

    def learner_hashes(self, model: Any, *, checkpoint: Path | None = None) -> dict[str, str]:
        import hashlib

        def hashed(value: Any) -> str:
            result = hashlib.sha256()

            def visit(item: Any) -> None:
                if isinstance(item, self.torch.Tensor):
                    visit(item.detach().cpu().numpy())
                elif isinstance(item, self.np.ndarray):
                    result.update(f"{item.dtype}:{item.shape}:".encode())
                    result.update(item.tobytes())
                elif isinstance(item, dict):
                    for key in sorted(item, key=str):
                        visit(key)
                        visit(item[key])
                elif isinstance(item, (tuple, list)):
                    for part in item:
                        visit(part)
                else:
                    result.update(repr(item).encode() + b";")

            visit(value)
            return result.hexdigest()

        replay = model.replay_buffer
        actual = {
            "policy": hashed(model.policy.state_dict()),
            "actor_optimizer": hashed(model.actor.optimizer.state_dict()),
            "critic_optimizer": hashed(model.critic.optimizer.state_dict()),
            "entropy_optimizer": hashed(model.ent_coef_optimizer.state_dict()),
            "log_entropy": hashed(model.log_ent_coef),
            "replay": hashed(
                {
                    key: getattr(replay, key)
                    for key in (
                        "observations",
                        "next_observations",
                        "actions",
                        "rewards",
                        "dones",
                        "timeouts",
                        "pos",
                        "full",
                    )
                }
            ),
        }
        if checkpoint is not None:
            names = {
                "policy": "policy.pth",
                "actor_optimizer": "actor.optimizer.pth",
                "critic_optimizer": "critic.optimizer.pth",
                "entropy_optimizer": "ent_coef_optimizer.pth",
                "log_entropy": "pytorch_variables.pth",
            }
            with zipfile.ZipFile(checkpoint) as archive:
                for key, name in names.items():
                    self.runner.check("initialization")
                    with archive.open(name) as handle:
                        expected = self.torch.load(handle, map_location="cpu", weights_only=True)
                    if key == "log_entropy":
                        expected = expected["log_ent_coef"]
                    if actual[key] != hashed(expected):
                        raise ValueError(
                            f"loaded {key} differs from the serialized native checkpoint"
                        )
        return actual


class ContinuationRunner(DevelopmentRunner):
    def __init__(self, plan: ExtensionPlan, approval: dict[str, Any], **kwargs: Any) -> None:
        validate_extension_approval(plan.config, approval)
        verify_binding(plan.config, plan.binding)
        plan.output_path.mkdir(parents=True, exist_ok=False)
        super().__init__(
            plan,
            approval,
            runtime_factory=kwargs.pop("runtime_factory", ContinuationRuntime),
            approval_validator=lambda selected, record: validate_extension_approval(
                selected.config, record
            ),
            **kwargs,
        )
        self.ledger.update(copy.deepcopy(plan.parent["ledger"]))
        self.manifest.update(
            parent_binding=copy.deepcopy(plan.binding),
            amendment_authorization=AUTHORIZATION_URL,
            original_session_preserved=True,
            resume_supported=True,
            continuation="same learner identities and verified checkpoint reconstruction",
            passed_tasks_not_reloaded=[task for task in plan.base.tasks if task not in plan.tasks],
            additional_sac_reloads=0,
            amendment_phase="post-outcome exploratory development",
        )
        self.stop_file: Path | None = None

    def check(self, kind: str) -> None:
        stop_file = getattr(self, "stop_file", None)
        if stop_file is not None and stop_file.exists():
            raise Stopped("supervisor_stop_file")
        super().check(kind)

    def start_policy(self) -> None:
        super().start_policy()
        self.manifest["additional_sac_reloads"] += 1
        if self.manifest["additional_sac_reloads"] > 2:
            raise Stopped("additional_sac_reloads_limit")

    def physical_steps(self, state: TaskState) -> int:
        return int(state.metadata["physical_training_steps"])

    def persist(self) -> None:
        self.ledger["total_agent_steps"] = sum(
            self.ledger[key]
            for key in ("source_training_steps", "development_evaluation_steps", "preflight_steps")
        )
        timing: dict[str, Any] = {}
        self.budget.record(timing)
        self.manifest.update(
            {
                key: value
                for key, value in timing.items()
                if key not in ("status", "validation_complete")
            }
        )
        if not timing["budget_compliant"] and self.manifest["status"] != "ERROR":
            self.manifest["status"] = "INCOMPLETE"
        tasks = copy.deepcopy(self.plan.parent["tasks"])
        for task, state in self.states.items():
            tasks[task] = {
                "status": state.status,
                "actual_training_steps": self.physical_steps(state),
                "model_timesteps": int(state.model.num_timesteps),
                "metadata": state.metadata,
                "checkpoints": state.checkpoints,
            }
        self.manifest.update(
            tasks=tasks,
            goal_manifests=self.plan.parent["goal_manifests"],
            phase=self.phase,
            training_reserve_stop_latched=self.training_stopped,
            cumulative_artifact_bytes_before_manifest=(
                self.plan.binding["prior_artifact_bytes"] + self.store.size()
            ),
            supervisor_storage_reserved_bytes=self.plan.config["supervisor_reserve_bytes"],
        )
        self.store.json("development_manifest.json", self.manifest)

    def checkpoint(self, state: TaskState, target: int, partial: bool) -> dict[str, Any]:
        physical, observed = self.physical_steps(state), int(state.model.num_timesteps)
        name = f"checkpoints/{state.task}/physical{physical}-model{observed}"
        if partial:
            name += "-partial"
        started = self.budget.monotonic()
        artifacts = self.runtime.save(state, name)
        row = {
            "target_steps": target,
            "actual_steps": physical,
            "physical_actual_steps": physical,
            "model_timesteps": observed,
            "artifact_root": str(self.store.root),
            "partial": partial,
            "artifacts": artifacts,
            "evaluation": [],
            "passed": False,
            "checkpoint_duration_seconds": self.budget.monotonic() - started,
            "checkpoint_started_at_utc": self.budget.now().isoformat(),
        }
        state.checkpoints.append(row)
        self.persist()
        return row

    def evaluate(self, state: TaskState, row: dict[str, Any]) -> None:
        self.check("evaluation")
        completed: list[dict[str, Any]] = []
        partial: list[dict[str, Any]] = []
        for record in row.get("evaluation", []):
            if record.get("complete_episode") and record.get("length") == 500:
                if record.get("episode") != len(completed):
                    raise ValueError("saved evaluation is not a complete ordered prefix")
                completed.append(record)
            else:
                partial.append(record)
        row["prior_partial_evaluation_rows"] = partial
        row["evaluation"] = completed
        row["evaluation_resume_note"] = (
            "old partial retained and charged; unfinished episode redone within cap"
        )
        env = self.runtime.evaluation_env(state)
        started = self.budget.monotonic()
        try:
            for _ in completed:
                self.check("evaluation")
                env.reset()
            for episode in range(len(completed), 20):
                self.check("evaluation")
                observation, _ = env.reset()
                total, length, success = 0.0, 0, False
                terminated = truncated = False
                try:
                    while not (terminated or truncated):
                        action, _ = state.model.predict(observation, deterministic=True)
                        observation, reward, terminated, truncated, info = env.step(action)
                        total += float(reward)
                        length += 1
                        success = success or bool(info.get("success", False))
                finally:
                    row["evaluation"].append(
                        {
                            "episode": episode,
                            "return": total,
                            "length": length,
                            "success": success,
                            "complete_episode": bool(terminated or truncated),
                            "terminated": bool(terminated),
                            "truncated": bool(truncated),
                            **self.runtime.episode_record(env),
                        }
                    )
                    self.persist()
            bank = state.goals.for_tuning()[0]
            validate_evaluation_coverage(row["evaluation"], bank, evaluation_order(bank, 20))
            self.check("evaluation")
            row["evaluation_complete"] = True
            row["passed"] = sum(item["success"] for item in row["evaluation"]) >= 16
        finally:
            row["continuation_evaluation_seconds"] = self.budget.monotonic() - started
            env.close()

    def run(self) -> dict[str, Any]:
        self.persist()
        try:
            self.check("initialization")
            if not self.plan.tasks:
                self.manifest["status"] = "NOOP_ALREADY_COMPLETE"
                return self.manifest
            self.phase = "PREFLIGHT"
            now = self.budget.now()
            self.preflight = RunDeadline(
                871,
                min(self.plan.deadline_utc, now + timedelta(seconds=871)),
                now=self.budget.now,
                monotonic=self.budget.monotonic,
            )
            self.runtime = self.runtime_factory(self)
            for task in self.plan.tasks:
                self.states[task] = self.runtime.prepare(task)
                self.persist()
            preserved = sum(
                int(row["actual_training_steps"])
                for task, row in self.plan.parent["tasks"].items()
                if task not in self.states
            )
            physical_sum = preserved + sum(
                self.physical_steps(state) for state in self.states.values()
            )
            if physical_sum != self.ledger["source_training_steps"]:
                raise ValueError("physical source histories differ from cumulative parent ledger")
            self.manifest["physical_source_reconciliation"] = {
                "sum_policy_physical_steps": physical_sum,
                "cumulative_parent_source_steps": self.ledger["source_training_steps"],
                "verified": True,
            }
            self.check("preflight")
            self.manifest["all_loaded_states_verified_before_learning"] = True
            self.manifest["prospective_frozen_at_utc"] = self.budget.now().isoformat()
            self.store.json("frozen_continuation.json", self.manifest)
            self.check("preflight")
            self.preflight = None
            self.phase = "SOURCE_DEVELOPMENT"
            for state in self.states.values():
                if state.metadata["pending_optimizer_updates"]:
                    before_updates = int(state.model._n_updates)
                    try:
                        with state.rng.activate(), self.optimizer_guard(state.model):
                            state.model.train(
                                gradient_steps=1, batch_size=self.plan.algorithm["batch_size"]
                            )
                        if int(state.model._n_updates) != before_updates + 1:
                            raise ValueError(
                                "pending optimizer work did not complete exactly one native update"
                            )
                    except BaseException:
                        self.checkpoint(state, self.physical_steps(state), True)
                        raise
                    state.metadata["pending_optimizer_updates_completed"] = 1
                    state.metadata["pending_optimizer_updates"] = 0
            for task, target in self.plan.schedule():
                state = self.states[task]
                if state.status == "PASS":
                    continue
                physical = self.physical_steps(state)
                self.check("training" if target > physical else "evaluation")
                if physical > target:
                    raise ValueError("cannot evaluate a historical milestone using later weights")
                visit = {
                    "task": task,
                    "milestone": target,
                    "increment": target - physical,
                    "status": "TRAINING",
                }
                self.manifest["queue"].append(visit)
                self.persist()
                error: BaseException | None = None
                before_ledger = self.ledger["source_training_steps"]
                try:
                    if target > physical:
                        with state.rng.activate(), self.optimizer_guard(state.model):
                            state.model.learn(
                                target - physical,
                                reset_num_timesteps=False,
                                callback=self.runtime.training_callback(state),
                            )
                except BaseException as caught:
                    error = caught
                finally:
                    state.metadata["physical_training_steps"] = (
                        physical + self.ledger["source_training_steps"] - before_ledger
                    )
                    state.metadata["physical_step_offset"] = self.physical_steps(state) - int(
                        state.model.num_timesteps
                    )
                partial = self.physical_steps(state) != target or error is not None
                if (
                    target == physical
                    and state.checkpoints[-1]["target_steps"] == target
                    and not state.metadata.get("pending_optimizer_updates_completed")
                ):
                    row = state.checkpoints[-1]
                    row.update(
                        parent_partial=row.get("partial", False),
                        partial=False,
                        actual_steps=physical,
                        physical_actual_steps=physical,
                        model_timesteps=int(state.model.num_timesteps),
                    )
                else:
                    row = self.checkpoint(state, target, partial)
                if error is not None:
                    raise error
                if partial:
                    raise Stopped("incomplete_training_segment")
                if not evaluation_complete(row):
                    with self.runtime.preserve_rng():
                        self.evaluate(state, row)
                if row.get("passed"):
                    self.check("evaluation")
                    state.status = "PASS"
                    state.metadata["earliest_passing_steps"] = target
                elif target == 1000000:
                    state.status = "FULL_BUDGET_FAIL"
                visit["status"] = state.status if state.status != "PENDING" else "CONTINUE"
                self.persist()
            self.check("completion")
            self.manifest["status"] = (
                "COMPLETE_PASS"
                if all(state.status == "PASS" for state in self.states.values())
                else "COMPLETE_FAIL"
            )
        except (BudgetExpired, Stopped) as error:
            self.manifest["status"] = "INCOMPLETE"
            self.manifest["stop_reason"] = str(error)
        except BaseException as error:
            self.manifest["status"] = "ERROR"
            self.manifest["errors"].append(f"{type(error).__name__}: {error}")
            self.manifest["traceback"] = traceback.format_exc()
        finally:
            if self.runtime is not None:
                try:
                    self.runtime.close()
                except BaseException as error:
                    self.manifest["status"] = "ERROR"
                    self.manifest["errors"].append(f"cleanup: {error}")
            self.phase = "FINISHED"
            self.persist()
        return self.manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--monotonic-cutoff", type=float)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if stable_hash(config) != stable_hash(extension_configuration()):
        raise ValueError("changed extension candidate")
    if args.plan:
        print(
            json.dumps(
                {
                    "extension": config,
                    "extension_sha256": stable_hash(config),
                    "code_sha256": code_hashes(),
                    "parent_bound": False,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    if args.approval is None or args.binding is None:
        raise ValueError("frozen approval and post-exit parent binding are required")
    approval = json.loads(args.approval.read_text())
    validate_extension_approval(config, approval)
    binding = json.loads(args.binding.read_text())
    plan = ExtensionPlan(config, binding)
    if utc_now() >= plan.deadline_utc:
        raise ValueError("extended cutoff has expired")
    cutoff = inherited_monotonic_cutoff(binding, args.monotonic_cutoff, config["deadline_utc"])
    entered_monotonic = time.monotonic()
    seconds = min((plan.deadline_utc - utc_now()).total_seconds(), cutoff - entered_monotonic)
    if seconds <= 0:
        raise ValueError("supervisor monotonic cutoff has expired")
    budget = RunDeadline(seconds, plan.deadline_utc)
    budget.started_monotonic = entered_monotonic
    runner = ContinuationRunner(plan, approval, budget=budget)
    runner.stop_file = args.stop_file
    from scripts.metaworld_reporting_sidecar import reporting_sidecar

    with reporting_sidecar(
        runner.store.root / "development_manifest.json", artifact_limit=runner.store.limit
    ):
        result = runner.run()
    print(json.dumps({"status": result["status"], "ledger": result["ledger"]}, sort_keys=True))
    return (
        0 if result["status"] in ("COMPLETE_PASS", "COMPLETE_FAIL", "NOOP_ALREADY_COMPLETE") else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
