from __future__ import annotations

import copy
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from scripts.metaworld_deadlines import RunDeadline
from scripts.metaworld_development import Stopped, TaskState, code_hashes, digest
from scripts.metaworld_development_plan import DevelopmentPlan
from scripts.metaworld_protocol import Goal, GoalBank, GoalManifest, stable_hash
from scripts.metaworld_technical_retry import (
    PRIOR_BYTES,
    PRIOR_LEDGER,
    REMAINING_BYTES,
    ReplacementPlan,
    TechnicalRetryContract,
    TechnicalRetryRunner,
)


class SyntheticPlan:
    """No runtime execution: redirect a copied plan into a synthetic evidence fixture."""

    def __init__(self, output: Path) -> None:
        self.base = DevelopmentPlan.load("experiments/metaworld/development_2026-11-09.json")
        self.output_path = output

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def to_dict(self) -> dict[str, Any]:
        value = self.base.to_dict()
        value["output"] = str(self.output_path)
        return value

    @property
    def sha256(self) -> str:
        return stable_hash(self.to_dict())

    def validate(self) -> None:
        pass


def fake_goals(task: str) -> GoalManifest:
    return GoalManifest(
        tuple(
            GoalBank(
                split,
                1,
                tuple(Goal(task, "eA==", (float(index + offset),)) for index in range(count)),
            )
            for split, count, offset in (("source", 50, 0), ("development", 20, 100))
        )
    )


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Any, TechnicalRetryContract, dict[str, Any]]:
    plan = SyntheticPlan(tmp_path / "prior")
    plan.output_path.mkdir()
    template = json.loads(
        Path("experiments/metaworld/technical_retry_2026-11-09_candidate.json").read_text()
    )
    states: dict[str, Any] = {}
    manifests = {}
    for task in plan.tasks:
        metadata = {
            "initial_actor_sha256": "a" * 64,
            "actor_layout": [],
            "source_streams": plan.source_streams(task),
            "development_streams": plan.development_streams(task),
            "evaluation_goal_order": list(range(20)),
            "evaluation_episode_seeds": list(range(20)),
            "initialization": {
                "actor_state_sha256": "b" * 64,
                "critic_state_sha256": "c" * 64,
                "critic_target_state_sha256": "c" * 64,
            },
        }
        states[task] = {
            "status": "PENDING",
            "actual_training_steps": 0,
            "checkpoints": [],
            "metadata": metadata,
        }
        manifests[task] = fake_goals(task).to_dict()
    prior = {
        "status": "ERROR",
        "ledger": dict(PRIOR_LEDGER),
        "plan_sha256": plan.sha256,
        "plan": plan.to_dict(),
        "tasks": states,
        "goal_manifests": manifests,
        "errors": ["TypeError: object of type 'pickle.PickleBuffer' has no len()"],
    }
    (plan.output_path / "development_manifest.json").write_text(json.dumps(prior))
    length = (plan.output_path / "development_manifest.json").stat().st_size
    (plan.output_path / "padding").write_bytes(b"x" * (PRIOR_BYTES - length))
    inventory = {
        "total_bytes": PRIOR_BYTES,
        "files": [
            {"path": p.name, "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(plan.output_path.iterdir())
        ],
    }
    inventory_path = tmp_path / "inventory.json"
    inventory_path.write_text(json.dumps(inventory))
    template.update(
        base_plan_sha256=plan.sha256,
        prior_artifact_root=str(plan.output_path),
        replacement_artifact_root=str(plan.output_path) + "-r2",
        prior_inventory_file=inventory_path.name,
        prior_inventory_sha256=digest(inventory_path),
    )
    contract = TechnicalRetryContract(template, tmp_path)
    contract.validate(plan)
    return plan, contract, prior


def fake_approval(plan: Any, contract: TechnicalRetryContract) -> dict[str, Any]:
    return {
        "approved": True,
        "technical_retry_contract_sha256": contract.sha256,
        "plan_sha256": plan.sha256,
        "authorization_url": "https://github.com/Spiph/UWSH-nethack/issues/43#issuecomment-123",
        "code_commit": "a" * 40,
        "code_sha256": code_hashes(),
        **{
            key: contract.data[key]
            for key in ("session_started_utc", "training_deadline_utc", "deadline_utc")
        },
        "output": str(contract.output_path),
    }


def fake_budget(plan: Any) -> RunDeadline:
    return RunDeadline(
        18000,
        plan.deadline_utc,
        monotonic=lambda: 0,
        now=lambda: plan.session_started_utc + timedelta(hours=1),
    )


def populate_states(runner: TechnicalRetryRunner, prior: dict[str, Any]) -> None:
    for task, row in prior["tasks"].items():
        runner.states[task] = TaskState(
            task,
            SimpleNamespace(
                num_timesteps=0, _n_updates=0, replay_buffer=SimpleNamespace(pos=0, full=False)
            ),
            None,
            GoalManifest.from_dict(prior["goal_manifests"][task]),
            None,
            copy.deepcopy(row["metadata"]),
        )


def test_false_or_old_approval_cannot_create_attempt_two(fixture: Any) -> None:
    plan, contract, _ = fixture
    valid = fake_approval(plan, contract)
    for record in (
        {**valid, "approved": False},
        {
            **valid,
            "authorization_url": "https://github.com/Spiph/UWSH-nethack/issues/37#issuecomment-6744072078",
        },
        {**valid, "technical_retry_contract_sha256": "0" * 64},
    ):
        with pytest.raises(ValueError):
            TechnicalRetryRunner(plan, contract, record)
    assert not contract.output_path.exists()


def test_prior_tampering_or_extra_file_rejected_before_runtime(fixture: Any) -> None:
    plan, contract, _ = fixture
    (plan.output_path / "extra").write_text("mutation")
    with pytest.raises(ValueError, match="unexpected file"):
        TechnicalRetryRunner(
            plan, contract, fake_approval(plan, contract), budget=fake_budget(plan)
        )
    assert not contract.output_path.exists()


def test_inventory_snapshot_mutation_is_rejected(fixture: Any) -> None:
    plan, contract, _ = fixture
    contract.inventory_path.write_text("{}")
    with pytest.raises(ValueError, match="inventory changed"):
        contract.verify_prior(plan)


def test_unchanged_scientific_plan_and_cumulative_resource_accounting(fixture: Any) -> None:
    plan, contract, _ = fixture
    before = plan.to_dict(), plan.sha256
    proxy = ReplacementPlan(plan, contract)
    assert (proxy.to_dict(), proxy.sha256) == before
    assert proxy.output_path == contract.output_path
    assert proxy.artifact_cap_bytes == REMAINING_BYTES
    assert proxy.preflight_seconds == 888
    engine = TechnicalRetryRunner(
        plan, contract, fake_approval(plan, contract), budget=fake_budget(plan)
    )
    assert engine.ledger["source_policy_starts"] == 3
    assert engine.ledger["preflight_steps"] == 1
    for _ in range(3):
        engine.start_policy()
    with pytest.raises(Stopped, match="source_policy_starts_limit"):
        engine.start_policy()
    assert engine.ledger["source_policy_starts"] == 6
    assert engine.ledger["retries"] == 1
    assert engine.plan.ledger()["retries"] == 1
    called = []
    engine.ledger["preflight_steps"] = 1000
    env = engine.bind_env(
        SimpleNamespace(reset=lambda: None, step=lambda: called.append(True)), "preflight"
    )
    with pytest.raises(Stopped, match="preflight_steps_limit"):
        env.step()
    assert not called
    engine.persist()
    record = json.loads((contract.output_path / "development_manifest.json").read_text())
    assert record["technical_retry_accounting"]["attempt_ledger"]["source_policy_starts"] == 3
    assert record["technical_retry_accounting"]["cumulative_ledger"]["source_policy_starts"] == 6
    assert record["technical_retry_accounting"]["cumulative_artifact_cap_bytes"] == 2**31


def test_exact_reconstruction_and_mismatch_detection_before_learning(fixture: Any) -> None:
    plan, contract, prior = fixture
    engine = TechnicalRetryRunner(
        plan, contract, fake_approval(plan, contract), budget=fake_budget(plan)
    )
    populate_states(engine, prior)
    assert engine.verify_reconstruction(engine)["all_three_reconstructed_before_learning"]
    task = plan.tasks[2]
    engine.states[task].metadata["initialization"]["critic_target_state_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="initialization"):
        engine.verify_reconstruction(engine)


def test_prior_mutation_after_construction_rejected_before_learning(fixture: Any) -> None:
    plan, contract, prior = fixture
    engine = TechnicalRetryRunner(
        plan, contract, fake_approval(plan, contract), budget=fake_budget(plan)
    )
    populate_states(engine, prior)
    (plan.output_path / "padding").write_bytes(b"changed")
    with pytest.raises(ValueError, match="prior artifact changed"):
        engine.verify_reconstruction(engine)


@pytest.mark.parametrize(
    "field",
    [
        "additional_policy_starts",
        "replacement_preflight_seconds",
        "replacement_artifact_cap_bytes",
        "deadline_utc",
    ],
)
def test_retry_contract_cannot_expand_the_exception(fixture: Any, field: str) -> None:
    plan, contract, _ = fixture
    contract.data[field] = 999999 if field != "deadline_utc" else "2026-11-10T21:35:00+00:00"
    with pytest.raises(ValueError, match="differs"):
        contract.validate(plan)


def test_nonempty_replacement_directory_cannot_be_reused(fixture: Any) -> None:
    plan, contract, _ = fixture
    contract.output_path.mkdir()
    (contract.output_path / "sentinel").write_text("preserve")
    with pytest.raises(ValueError, match="not empty"):
        TechnicalRetryRunner(
            plan, contract, fake_approval(plan, contract), budget=fake_budget(plan)
        )
    assert (contract.output_path / "sentinel").read_text() == "preserve"


def test_last_task_mismatch_prevents_all_learning_and_preflight(fixture: Any) -> None:
    plan, contract, prior = fixture
    events: list[str] = []

    class FakeRuntime:
        def __init__(self, engine: TechnicalRetryRunner) -> None:
            self.engine = engine

        def prepare(self, task: str) -> TaskState:
            self.engine.start_policy()
            metadata = copy.deepcopy(prior["tasks"][task]["metadata"])
            if task == plan.tasks[-1]:
                metadata["initialization"]["critic_target_state_sha256"] = "wrong"
            return TaskState(
                task,
                SimpleNamespace(
                    num_timesteps=0,
                    _n_updates=0,
                    replay_buffer=SimpleNamespace(pos=0, full=False),
                    learn=lambda *args, **kwargs: events.append("learn"),
                ),
                None,
                GoalManifest.from_dict(prior["goal_manifests"][task]),
                None,
                metadata,
            )

        def save_initial(self, state: TaskState) -> None:
            with self.engine.store.open(f"initial/{state.task}.npz") as handle:
                handle.write(b"fake initial array")

        def preflight(self, states: list[TaskState]) -> None:
            events.append("preflight")

        def close(self) -> None:
            events.append("close")

    engine = TechnicalRetryRunner(
        plan,
        contract,
        fake_approval(plan, contract),
        runtime_factory=FakeRuntime,
        budget=fake_budget(plan),
    )
    result = engine.run()
    assert result["status"] == "ERROR"
    assert events == ["close"]
    assert result["ledger"]["source_policy_starts"] == 6
    assert result["ledger"]["source_training_steps"] == 0
    assert result["ledger"]["preflight_steps"] == 1
    assert not (contract.output_path / "frozen_launch.json").exists()


def test_unapproved_cli_rejects_without_output_or_rl_imports() -> None:
    import subprocess
    import sys

    code = """
import json
from datetime import timedelta
import sys
from pathlib import Path
from scripts.metaworld_technical_retry import main
contract = Path('experiments/metaworld/technical_retry_2026-11-09_candidate.json')
output = Path(json.loads(contract.read_text())['replacement_artifact_root'])
existed = output.exists()
try:
    main(['--config','experiments/metaworld/development_2026-11-09.json',
          '--contract',str(contract),'--run',
          '--approval','experiments/metaworld/technical_retry_2026-11-09_approval_UNAPPROVED.json'])
except ValueError as error:
    assert 'explicit approval' in str(error)
else:
    raise AssertionError('unapproved replacement was accepted')
assert output.exists() == existed
assert not {'torch','stable_baselines3','gymnasium','mujoco'}.intersection(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
