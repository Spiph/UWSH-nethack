"""Unapproved scripted-only physical sentinel; preparation imports no native runtime."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

ENTRY_UTC, ENTRY_MONOTONIC = datetime.now(timezone.utc), time.monotonic()
ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/metaworld/source_sentinel_candidate.json"
ARTIFACTS = Path("/home/icannon/code/UWSH-nethack/artifacts")
FINAL = ARTIFACTS / "metaworld-development-2026-11-09-r2-extension"
SUPERVISOR = Path("/tmp/ups-development-20261109/source-extension-supervisor/status.json")
SOURCE_MODULES = (
    "metaworld_development",
    "metaworld_technical_retry",
    "metaworld_source_continuation",
    "metaworld_source_supervisor",
)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(data)
    return result.hexdigest()


def candidate_configuration() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "approved": False,
        "decision_issue": 56,
        "purpose": "scripted_source_only_physical_sentinel_not_learning",
        "window_seconds": 300,
        "outer_timeout_seconds": 300,
        "cleanup_seconds": 5,
        "attempts": 1,
        "automatic_retries": 0,
        "cpu_only": True,
        "artifact_bytes": 8 * 1024**2,
        "episode_horizon": 500,
        "source_diagnostic_steps": 1000,
        "development_steps": 0,
        "target_steps": 0,
        "final_test_steps": 0,
        "environment_constructors": 2,
        "scripted_controller_instances": 2,
        "source_actor_array_loads": 0,
        "sac_initializations": 0,
        "optimizer_updates": 0,
        "goal_index": 0,
        "split": "source",
        "action_boundary": "clip_all_four_to_minus1_plus1_no_extra_scale",
        "output": str(ARTIFACTS / "metaworld-scripted-source-sentinel-2026-11-10"),
        "source_pids": [205152, 362873, 400119],
        "evidence": {
            "original": {
                "path": str(
                    ARTIFACTS / "metaworld-development-2026-11-09-r2/development_manifest.json"
                ),
                "sha256": "1eb874f5d63f4d76d005eae101541c2a1305650f8373282891f0c044f62ac7e4",
            },
            "final": {
                "path": str(FINAL / "development_manifest.json"),
                "sha256": "2b751ea1d9c4d051c4edb89237987d360924d1b49d75f325ae4adcc472d94b74",
            },
            "supervisor": {
                "path": str(SUPERVISOR),
                "sha256": "8411451b003fec9c772009b0e6e4217ec88a70819b3cab69db8de0db1cfa85e9",
            },
        },
        "tasks": [
            {
                "task": "push-v3",
                "goal_id": "b5f92d7ca8c8a60f05ad4eefd48a87261c04dde6920d26a5693d9903b4d0beb5",
                "bank_sha256": "cf6c63bb66518c9c63eb0d47c0090c0814d913a3c47b0652feb09a28b7e025de",
                "episode_seed": 2562001696,
                "policy_module": "sawyer_push_v3_policy",
                "policy_class": "SawyerPushV3Policy",
                "policy_file_sha256": (
                    "921d7380bc04fce1834cf025fde0bdf4aa625fcf5ffbd93c7c5a85eb40e4cf53"
                ),
            },
            {
                "task": "pick-place-v3",
                "goal_id": "c5ad03653cdd7aa6f95449bd6bf13a774a1fd000f12bb45a022ca2c0390bde3f",
                "bank_sha256": "cb828e9a92dfee54f80f5758f069a5e4081edb9b28042fa13b77b70fa07fc712",
                "episode_seed": 1245230885,
                "policy_module": "sawyer_pick_place_v3_policy",
                "policy_class": "SawyerPickPlaceV3Policy",
                "policy_file_sha256": (
                    "3d64ddf57c5cf6ddcee28945005bbbe040964d7dd1ec8b68cfe55175ad5acfbc"
                ),
            },
        ],
        "versions": {
            "numpy": "1.26.4",
            "metaworld": "3.1.1",
            "mujoco": "3.3.0",
            "gymnasium": "1.1.1",
        },
    }


def code_hashes() -> dict[str, str]:
    return {path.name: digest(path) for path in sorted((ROOT / "scripts").glob("metaworld_*.py"))}


def _utc(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("fixed approval timestamps require a timezone")
    return result.astimezone(timezone.utc)


def source_processes() -> list[int]:
    found = []
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            words = (path / "cmdline").read_bytes().split(b"\0")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as error:
            raise ValueError("cannot verify that source development is idle") from error
        if any(
            word.decode(errors="replace") in {f"scripts.{name}" for name in SOURCE_MODULES}
            or Path(word.decode(errors="replace")).name in {f"{name}.py" for name in SOURCE_MODULES}
            for word in words
        ):
            found.append(int(path.name))
    return found


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def source_evidence(plan: dict[str, Any]) -> dict[str, Any]:
    records = {}
    for name, evidence in plan["evidence"].items():
        raw = Path(evidence["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != evidence["sha256"]:
            raise ValueError("source lifecycle evidence changed from the frozen final record")
        records[name] = json.loads(raw)
    if (
        any(records[name].get("phase") != "FINISHED" for name in ("original", "final"))
        or records["final"].get("status") != "COMPLETE_FAIL"
        or records["supervisor"].get("status") != "FINISHED"
        or records["supervisor"].get("returncode") != 0
    ):
        raise ValueError("all original, extension and supervisor work must be final")
    if any(alive(pid) for pid in plan["source_pids"]) or source_processes():
        raise ValueError("source development or its supervisor is still active")
    return cast(dict[str, Any], records["final"])


def validate_approval(
    plan: dict[str, Any],
    approval: dict[str, Any],
    *,
    now: datetime,
    hashes: dict[str, str],
    head: str,
) -> datetime:
    if approval.get("approved") is not True:
        raise ValueError("UNAPPROVED physical sentinel; no execution is authorized")
    if set(approval) != {
        "approved",
        "decision_issue",
        "authorization_url",
        "plan_sha256",
        "code_commit",
        "code_sha256",
        "session_started_utc",
        "deadline_utc",
        "output",
        "attempt",
        "automatic_retries",
    }:
        raise ValueError("approval contains missing or unexpected fields")
    issue = plan["decision_issue"]
    if (
        type(issue) is not int
        or issue <= 0
        or approval.get("decision_issue") != issue
        or approval.get("plan_sha256") != stable_hash(plan)
        or approval.get("code_sha256") != hashes
        or approval.get("code_commit") != head
        or re.fullmatch(r"[0-9a-f]{40}", head) is None
        or re.fullmatch(
            rf"https://github\.com/Spiph/UWSH-nethack/issues/{issue}#issuecomment-[0-9]+",
            str(approval.get("authorization_url")),
        )
        is None
        or approval.get("output") != plan["output"]
        or type(approval.get("attempt")) is not int
        or approval.get("attempt") != 1
        or type(approval.get("automatic_retries")) is not int
        or approval.get("automatic_retries") != 0
    ):
        raise ValueError("approval must bind the exact plan/code/commit and one attempt")
    start, cutoff = _utc(approval["session_started_utc"]), _utc(approval["deadline_utc"])
    if cutoff - start != timedelta(seconds=300) or not start <= now < cutoff:
        raise ValueError("approval must freeze a current inclusive five-minute window")
    return cutoff


class SentinelRunner:
    def __init__(
        self,
        plan: dict[str, Any],
        approval: dict[str, Any],
        source: dict[str, Any],
        budget: Any,
        store: Any,
        factory: Any,
    ) -> None:
        self.plan, self.budget, self.store, self.factory, self.source = (
            plan,
            budget,
            store,
            factory,
            source,
        )
        self.ledger = dict.fromkeys(
            (
                "environment_constructor_attempts",
                "environment_constructors",
                "scripted_constructor_attempts",
                "scripted_controllers",
                "reset_attempts",
                "resets",
                "source_diagnostic_step_attempts",
                "source_diagnostic_steps",
                "development_steps",
                "target_steps",
                "final_test_steps",
                "actor_array_loads",
                "sac_initializations",
                "optimizer_updates",
            ),
            0,
        )
        self.manifest: dict[str, Any] = {
            "status": "RUNNING",
            "validation_complete": False,
            "plan": copy.deepcopy(plan),
            "plan_sha256": stable_hash(plan),
            "approval": copy.deepcopy(approval),
            "ledger": self.ledger,
            "episodes": [],
            "errors": [],
            "interpretation": (
                "two scripted source-goal sentinels; not a competence estimate or transfer result"
            ),
        }

    def charge(self, key: str, maximum: int) -> None:
        self.budget.require(key)
        if self.ledger[key] >= maximum:
            raise RuntimeError(f"{key} cap exhausted before work")
        self.ledger[key] += 1

    def persist(self) -> None:
        self.budget.record(self.manifest)
        self.store.json("sentinel_manifest.json", self.manifest)

    def run(self) -> dict[str, Any]:
        runtime = None
        self.persist()
        try:
            self.budget.require("native_imports")
            runtime = self.factory(self)
            self.manifest["runtime"] = runtime.metadata()
            for request in self.plan["tasks"]:
                row = {
                    "task": request["task"],
                    "status": "RUNNING",
                    "length": 0,
                    "success": False,
                    "trace_rows": 0,
                    "complete_episode": False,
                }
                self.manifest["episodes"].append(row)
                handle = runtime.prepare(request)
                self.charge("reset_attempts", 2)
                observation, initial = runtime.reset(handle, request)
                self.ledger["resets"] += 1
                row["initial"] = initial
                trace = f"{request['task']}.jsonl"
                row["trace"] = trace
                self.persist()
                try:
                    with self.store.open(trace) as writer:
                        for step in range(500):
                            self.budget.require("scripted_inference")
                            raw_action, action = runtime.action(handle, observation)
                            self.charge("source_diagnostic_step_attempts", 1000)
                            following, reward, terminated, truncated, info = runtime.step(
                                handle, action
                            )
                            self.ledger["source_diagnostic_steps"] += 1
                            row["length"] += 1
                            row["success"] = (
                                row["success"] or info["success"] is True or info["success"] == 1
                            )
                            record = runtime.record(
                                handle,
                                step,
                                observation,
                                following,
                                raw_action,
                                action,
                                reward,
                                terminated,
                                truncated,
                                info,
                            )
                            writer.write(
                                (
                                    json.dumps(record, separators=(",", ":"), allow_nan=False)
                                    + "\n"
                                ).encode()
                            )
                            row["trace_rows"] += 1
                            observation = following
                            if terminated or truncated:
                                if terminated or not truncated or step != 499:
                                    raise ValueError(
                                        "sentinel violated full500-step truncation semantics"
                                    )
                                row["complete_episode"] = True
                            elif step == 499:
                                raise ValueError("sentinel failed to truncate at500")
                            if (step + 1) % 100 == 0:
                                self.persist()
                    row["trace_sha256"] = digest(self.store.root / trace)
                    self.budget.require("episode_completion_promotion")
                    row["status"] = "COMPLETE"
                finally:
                    if (self.store.root / trace).exists() and "trace_sha256" not in row:
                        row["trace_sha256"] = digest(self.store.root / trace)
            if (
                self.ledger["source_diagnostic_steps"] != 1000
                or self.ledger["environment_constructors"] != 2
                or self.ledger["scripted_controllers"] != 2
            ):
                raise RuntimeError("completed sentinel has unexpected resource counts")
            self.budget.require("completion")
            self.manifest.update(status="COMPLETE", validation_complete=True)
        except BaseException as error:
            self.manifest.update(
                status="INCOMPLETE"
                if isinstance(error, KeyboardInterrupt) or type(error).__name__ == "BudgetExpired"
                else "ERROR",
                validation_complete=False,
            )
            self.manifest["errors"].append(f"{type(error).__name__}: {error}")
            self.manifest["traceback"] = traceback.format_exc()
            if self.manifest["episodes"] and self.manifest["episodes"][-1]["status"] == "RUNNING":
                self.manifest["episodes"][-1]["status"] = self.manifest["status"]
        finally:
            if runtime is not None:
                try:
                    runtime.close()
                except BaseException as error:
                    self.manifest.update(status="ERROR", validation_complete=False)
                    self.manifest["errors"].append(f"cleanup: {error}")
            self.persist()
            self.budget.record(self.manifest)
            if not self.manifest.get("budget_compliant", False):
                self.manifest["validation_complete"] = False
                self.store.json("sentinel_manifest.json", self.manifest)
        return self.manifest


class NativeRuntime:
    def __init__(self, runner: SentinelRunner) -> None:
        self.runner = runner
        self.handles: list[dict[str, Any]] = []
        for package, expected in runner.plan["versions"].items():
            if version(package) != expected:
                raise ValueError(f"pinned {package} version mismatch")
        runner.budget.require("numpy_import")
        self.np = importlib.import_module("numpy")
        runner.budget.require("protocol_import")
        self.protocol = importlib.import_module("scripts.metaworld_protocol")
        runner.budget.require("metaworld_import")
        self.metaworld = importlib.import_module("metaworld")

    def metadata(self) -> dict[str, Any]:
        if "torch" in sys.modules or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise ValueError("sentinel must stay CPU-only without importing Torch")
        return {
            "versions": self.runner.plan["versions"],
            "device": "cpu",
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "scripted_commit": "7ea2b501c4a698c8533cdc55a396fe2734e2649d",
        }

    def prepare(self, request: dict[str, Any]) -> dict[str, Any]:
        runner, task = self.runner, request["task"]
        goals = self.protocol.GoalManifest.from_dict(runner.source["goal_manifests"][task])
        bank = goals.for_fitting()[0]
        if bank.sha256 != request["bank_sha256"] or bank.goals[0].goal_id != request["goal_id"]:
            raise ValueError("selected source goal differs from the frozen proposal")
        streams = self.protocol.SeedStreams(
            **runner.source["tasks"][task]["metadata"]["source_streams"]
        )
        policy_path = (
            Path(str(self.metaworld.__file__)).parent
            / "policies"
            / (request["policy_module"] + ".py")
        )
        if digest(policy_path) != request["policy_file_sha256"]:
            raise ValueError("official scripted controller differs from the pinned source")
        runner.budget.require("scripted_policy_import")
        policy_type = getattr(
            importlib.import_module("metaworld.policies." + request["policy_module"]),
            request["policy_class"],
        )
        runner.charge("environment_constructor_attempts", 2)
        env = self.protocol.make_protocol_env(task, streams, bank=bank, goal_order=(0,))
        runner.ledger["environment_constructors"] += 1
        handle = {"env": env, "bank": bank, "streams": streams}
        self.handles.append(handle)
        runner.charge("scripted_constructor_attempts", 2)
        handle["policy"] = policy_type()
        runner.ledger["scripted_controllers"] += 1
        return handle

    def reset(self, handle: dict[str, Any], request: dict[str, Any]) -> tuple[Any, Any]:
        observation, _ = handle["env"].reset()
        episode = self.protocol.evaluation_record(handle["env"])
        if (
            episode["goal_index"] != 0
            or episode["goal_id"] != request["goal_id"]
            or episode["goal_bank_sha256"] != request["bank_sha256"]
            or episode["episode_seed"] != request["episode_seed"]
        ):
            raise ValueError("realized sentinel goal/order/episode seed changed")
        return observation, {
            "observation": observation.tolist(),
            "episode": episode,
            "source_streams": handle["streams"].__dict__,
            "state": self.state(handle, observation),
        }

    def action(self, handle: dict[str, Any], observation: Any) -> tuple[Any, Any]:
        raw = self.np.asarray(handle["policy"].get_action(observation))
        if raw.shape != (4,) or not self.np.isfinite(raw).all():
            raise ValueError("scripted controller returned an invalid action")
        return raw, self.np.clip(raw, -1, 1).astype(self.np.float32)

    def step(self, handle: dict[str, Any], action: Any) -> Any:
        self.runner.budget.require("directly_before_physics_step")
        return handle["env"].step(action)

    def state(self, handle: dict[str, Any], observation: Any) -> dict[str, Any]:
        raw = handle["env"].unwrapped
        return {
            "tcp_center": self.np.asarray(raw.tcp_center).tolist(),
            "object_position": self.np.asarray(raw.get_body_com("obj")).tolist(),
            "object_initial_position": self.np.asarray(raw.obj_init_pos).tolist(),
            "goal_position": self.np.asarray(raw._target_pos).tolist(),
            "object_lift": float(observation[6] - raw.obj_init_pos[2]),
            "gripper_aperture_observation": float(observation[3]),
        }

    def record(
        self,
        handle: dict[str, Any],
        step: int,
        observation: Any,
        following: Any,
        raw_action: Any,
        action: Any,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "step": step,
            "observation": observation.tolist(),
            "next_observation": following.tolist(),
            "raw_scripted_action": raw_action.tolist(),
            "executed_clipped_action": action.tolist(),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "components": {
                key: float(info[key])
                for key in (
                    "success",
                    "near_object",
                    "grasp_success",
                    "grasp_reward",
                    "in_place_reward",
                    "obj_to_target",
                    "unscaled_reward",
                )
            },
            "state": self.state(handle, following),
        }

    def close(self) -> None:
        for handle in self.handles:
            handle["env"].close()


def approved_launch(plan: dict[str, Any], approval: dict[str, Any]) -> dict[str, Any]:
    if approval.get("approved") is not True:
        raise ValueError("UNAPPROVED physical sentinel")
    if plan != candidate_configuration():
        raise ValueError("only the exact proposed sentinel scope may be approved")
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=3
    ).strip()
    cutoff = validate_approval(
        plan, approval, now=datetime.now(timezone.utc), hashes=code_hashes(), head=head
    )
    source = source_evidence(plan)
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "scripts"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        timeout=3,
    )
    tracked = subprocess.check_output(
        ["git", "ls-files", "scripts"], cwd=ROOT, text=True, timeout=3
    ).splitlines()
    if any(f"scripts/{name}" not in tracked for name in approval["code_sha256"]):
        raise ValueError("approved runtime modules must all belong to the frozen commit")
    if any(
        name in sys.modules
        for name in ("numpy", "torch", "gymnasium", "metaworld", "stable_baselines3")
    ):
        raise ValueError("sentinel requires a fresh process before native imports")
    for key in (
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
    ):
        os.environ[key] = "" if key == "CUDA_VISIBLE_DEVICES" else "1"
    deadlines = importlib.import_module("scripts.metaworld_deadlines")
    seconds = min(300, (cutoff - ENTRY_UTC).total_seconds())
    if seconds <= 0:
        raise ValueError("inclusive sentinel setup window already expired")
    budget = deadlines.RunDeadline(seconds, cutoff)
    budget.started_monotonic, budget.started_utc = ENTRY_MONOTONIC, ENTRY_UTC
    budget.require("exclusive_attempt_claim")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    try:
        budget.require("artifact_store_import")
        store_type = importlib.import_module("scripts.metaworld_development").ArtifactStore
        store = store_type(output, plan["artifact_bytes"])
    except BaseException as error:
        payload = {
            "status": "ERROR",
            "validation_complete": False,
            "error": f"{type(error).__name__}: {error}",
            "approval": approval,
        }
        encoded = json.dumps(payload, sort_keys=True).encode()
        used = sum(path.stat().st_size for path in output.rglob("*") if path.is_file())
        if used + len(encoded) <= plan["artifact_bytes"]:
            with (output / "startup_failure.json").open("xb") as handle:
                handle.write(encoded)
        raise
    return SentinelRunner(plan, approval, source, budget, store, NativeRuntime).run()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args(argv)
    plan = json.loads(args.config.read_text())
    if plan != candidate_configuration():
        raise ValueError("sentinel plan differs from the exact proposed scope")
    if args.plan:
        print(
            json.dumps(
                {
                    "approved": False,
                    "plan": plan,
                    "plan_sha256": stable_hash(plan),
                    "code_sha256": code_hashes(),
                },
                indent=2,
            )
        )
        return 0
    if args.approval is None:
        raise ValueError("an actual resource approval is required")
    result = approved_launch(plan, json.loads(args.approval.read_text()))
    return 0 if result["validation_complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
