from __future__ import annotations

import base64
import copy
import importlib
import json
import os
import random
import subprocess
import sys
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from scripts import metaworld_protocol as protocol
from scripts.metaworld_protocol import (
    Goal,
    GoalBank,
    GoalManifest,
    LegacyRNG,
    SeedStreams,
    TaskController,
    bootstrap_mask,
    build_seeded_sac,
    evaluation_order,
    validate_evaluation_coverage,
)


class FakeSpace:
    def __init__(self) -> None:
        self.shape = (4,)
        self.low = -np.ones(4)
        self.high = np.ones(4)
        self.seed(0)

    def seed(self, seed: int) -> None:
        self._np_random = np.random.default_rng(seed)

    def sample(self) -> float:
        return float(self._np_random.random())


class FakeEnv:
    def __init__(self, *, reward_function_version: str = "v2") -> None:
        self.reward_function_version = reward_function_version
        self.action_space = FakeSpace()
        self.unwrapped = self
        self._partially_observable = False
        self.max_path_length = 500
        self.observation_space = SimpleNamespace(shape=(39,))
        self.np_random = np.random.default_rng()
        self.length = 0

    def seed(self, seed: int) -> None:
        self.np_random = np.random.default_rng(seed)
        self.action_space.seed(seed)

    def set_task(self, task: Any) -> None:
        self._last_rand_vec = np.array(json.loads(task.data))

    def reset(self, options: Any = None) -> tuple[Any, dict[str, Any]]:
        self.length = 0
        return self.draw(), {}

    def draw(self) -> tuple[float, ...]:
        return (float(self.np_random.random()), float(np.random.random()), random.random())

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        self.length += 1
        return self.draw(), 0.0, False, self.length == 500, {"success": self.length == 2}

    def close(self) -> None:
        pass


def goal(task: str, number: int) -> Goal:
    vector = (float(number), float(number + 1))
    payload = base64.b64encode(json.dumps(vector).encode()).decode()
    return Goal(task=task, payload_base64=payload, random_vector=vector)


def bank(split: str = "source", offset: int = 0, task: str = "reach-v3") -> GoalBank:
    return GoalBank(split, offset, tuple(goal(task, offset + index) for index in range(5)))


def controller(
    streams: SeedStreams | None = None, order: tuple[int, ...] | None = None
) -> TaskController:
    return TaskController(
        FakeEnv(), bank(), streams or SeedStreams.from_seed(42), SimpleNamespace, order
    )


def test_seed_streams_are_stable_named_and_validated() -> None:
    streams = SeedStreams.from_seed(123)
    assert streams == SeedStreams.from_seed(123)
    assert len(set(asdict(streams).values())) == len(asdict(streams))
    assert replace(streams, actor=99).task_bank == streams.task_bank
    with pytest.raises(ValueError, match="32-bit"):
        SeedStreams.from_seed(-1)
    with pytest.raises(ValueError, match="32-bit"):
        SeedStreams.from_seed(True)


def test_legacy_draws_restore_callers_even_on_failure() -> None:
    np.random.seed(123)
    random.seed(123)
    expected_numpy = np.random.RandomState(123).random()
    expected_python = random.Random(123).random()
    rng = LegacyRNG(77)
    with pytest.raises(RuntimeError), rng.activate():
        np.random.random(20)
        random.random()
        raise RuntimeError("upstream construction failed")
    assert np.random.random() == expected_numpy
    assert random.random() == expected_python


def test_goal_manifest_roundtrip_and_hash_check() -> None:
    manifest = GoalManifest((bank(), bank("development", 10)))
    record = json.loads(json.dumps(manifest.to_dict()))
    restored = GoalManifest.from_dict(record)
    assert restored == manifest
    assert restored.sha256 == manifest.sha256
    record["banks"][0]["goals"][0]["random_vector"][0] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        GoalManifest.from_dict(record)


def test_manifest_pins_reward_version_independently_of_task_suffix() -> None:
    record = GoalManifest((bank(),)).to_dict()
    record["environment"]["reward_function_version"] = "v1"
    with pytest.raises(ValueError, match="pinned environment"):
        GoalManifest.from_dict(record)


def test_split_goal_and_task_class_leakage_are_rejected() -> None:
    with pytest.raises(ValueError, match="leaks between splits"):
        GoalManifest((bank(), bank("development")))
    with pytest.raises(ValueError, match="held-out task class"):
        GoalManifest(
            (bank(), bank("final_test", 10)), ("reach-v3",), transfer_regime="heldout_family"
        )
    with pytest.raises(ValueError, match="explicitly held out"):
        GoalManifest((bank(), bank("final_test", 10, "push-v3")), transfer_regime="heldout_family")


def test_two_transfer_tiers_and_source_only_fitting_access() -> None:
    source, development = bank(), bank("development", 10)
    within = GoalManifest(
        (source, development, bank("final_test", 20)), transfer_regime="within_family"
    )
    assert within.for_fitting() == (source,)
    assert within.for_tuning() == (development,)
    heldout = GoalManifest(
        (source, development, bank("final_test", 20, "push-v3")),
        ("push-v3",),
        transfer_regime="heldout_family",
    )
    assert len(heldout.for_evaluation("final_test")) == 1
    heldout.validate_source_weights(["reach-v3"])
    heldout.validate_source_policy_records(
        [
            {
                "task": "reach-v3",
                "training_split": "source",
                "training_goal_bank_sha256": source.sha256,
            }
        ]
    )
    with pytest.raises(ValueError, match="source goal bank"):
        within.validate_source_policy_records(
            [
                {
                    "task": "reach-v3",
                    "training_split": "final_test",
                    "training_goal_bank_sha256": within.for_evaluation("final_test")[0].sha256,
                }
            ]
        )
    with pytest.raises(ValueError, match="source task classes"):
        heldout.validate_source_weights(["push-v3"])
    with pytest.raises(ValueError, match="development-only"):
        GoalManifest((bank("final_test"),))


def test_replica_goal_order_episode_draws_and_actions_ignore_global_rng() -> None:
    first, second = controller(), controller()
    for episode in range(8):
        first_obs, _ = first.reset(seed=episode)
        np.random.seed(episode)
        np.random.random(101)
        random.seed(episode)
        random.random()
        second_obs, _ = second.reset(seed=999)
        assert first_obs == second_obs
        assert first.env.action_space.sample() == second.env.action_space.sample()
        for _ in range(3):
            first_step = first.step(0)
            np.random.random(31)
            random.random()
            second_step = second.step(0)
            assert first_step[:4] == second_step[:4]
        left, right = first.episode_history[-1], second.episode_history[-1]
        assert left["goal_id"] == right["goal_id"]
        assert left["episode_seed"] == right["episode_seed"]


def test_episode_selection_action_and_model_streams_are_independent() -> None:
    seeds = SeedStreams.from_seed(42)
    original = controller(seeds)
    other_models = controller(replace(seeds, actor=9, critic=10, training=11))
    other_episodes = controller(replace(seeds, episode=12))
    other_selection = controller(replace(seeds, selection=13))
    other_action = controller(replace(seeds, action=14))
    for _ in range(20):
        left, _ = original.reset()
        same, _ = other_models.reset()
        different, _ = other_episodes.reset()
        other_selection.reset()
        same_action_env, _ = other_action.reset()
        assert left == same == same_action_env
        assert left != different
        assert original.env.action_space.sample() != other_action.env.action_space.sample()
    assert [r["goal_id"] for r in original.episode_history] == [
        r["goal_id"] for r in other_episodes.episode_history
    ]
    assert [r["episode_seed"] for r in original.episode_history] == [
        r["episode_seed"] for r in other_selection.episode_history
    ]
    assert [r["goal_id"] for r in original.episode_history] != [
        r["goal_id"] for r in other_selection.episode_history
    ]


def test_declared_order_any_time_success_and_complete_episode_coverage() -> None:
    goals = bank()
    order = evaluation_order(goals, 7)
    assert order == (0, 1, 2, 3, 4, 0, 1)
    env = controller(order=order)
    for _ in order:
        env.reset()
        for _ in range(500):
            _, _, terminated, truncated, info = env.step(0)
        assert terminated is False and truncated is True
        assert info["success"] is False
        assert env.episode_history[-1]["success"] is True
    validate_evaluation_coverage(env.episode_history, goals, order)
    with pytest.raises(ValueError, match="exhausted"):
        env.reset()
    with pytest.raises(ValueError, match="all declared"):
        validate_evaluation_coverage(env.episode_history[:-1], goals, order)
    rows = copy.deepcopy(env.episode_history)
    rows[0]["goal_id"] = rows[1]["goal_id"]
    with pytest.raises(ValueError, match="coverage"):
        validate_evaluation_coverage(rows, goals, order)


def test_interrupted_episodes_cannot_complete_evaluation_and_timeouts_bootstrap() -> None:
    env = controller(order=(0,))
    env.reset()
    for _ in range(3):
        env.step(0)
    assert env.episode_history[0]["success"] is True
    with pytest.raises(ValueError, match="full-episode"):
        validate_evaluation_coverage(env.episode_history, bank(), (0,))
    assert bootstrap_mask([False, False, True, True], [False, True, False, True]).tolist() == [
        1.0,
        1.0,
        0.0,
        0.0,
    ]


def test_bank_generation_uses_split_seed_and_restores_global_rng(monkeypatch: Any) -> None:
    def make_benchmark(task: str, seed: int) -> Any:
        np.random.seed(seed)
        tasks = [
            SimpleNamespace(data=json.dumps(np.random.random(2).tolist()).encode())
            for _ in range(5)
        ]
        return SimpleNamespace(train_tasks=tasks, train_classes={task: FakeEnv})

    fake_metaworld = SimpleNamespace(MT1=make_benchmark)
    monkeypatch.setattr(protocol, "_runtime", lambda: (fake_metaworld, None))
    seeds = SeedStreams.from_seed(99)
    np.random.seed(33)
    expected = np.random.RandomState(33).random()
    first = protocol.generate_goal_bank("reach-v3", seeds)
    assert np.random.random() == expected
    second = protocol.generate_goal_bank("reach-v3", replace(seeds, actor=100, training=101))
    assert first == second
    development = protocol.generate_goal_bank("reach-v3", seeds, split="development")
    final = protocol.generate_goal_bank("reach-v3", seeds, split="final_test")
    assert len({first.sha256, development.sha256, final.sha256}) == 3
    GoalManifest((first, development, final), transfer_regime="within_family")


class FakeWrapper:
    def __init__(self, env: Any, *, max_episode_steps: int = 500):
        self.env = env
        self.unwrapped = env.unwrapped
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.max_episode_steps = max_episode_steps

    def reset(self, *, options: Any = None) -> Any:
        return self.env.reset(options=options)

    def step(self, action: Any) -> Any:
        return self.env.step(action)

    def get_wrapper_attr(self, name: str) -> Any:
        return getattr(self, name)


def test_runtime_factory_pins_contract_and_does_not_consume_first_goal(monkeypatch: Any) -> None:
    fake_metaworld = SimpleNamespace(
        env_dict=SimpleNamespace(ALL_V3_ENVIRONMENTS={"reach-v3": FakeEnv}), Task=SimpleNamespace
    )
    fake_gym = SimpleNamespace(Wrapper=FakeWrapper, wrappers=SimpleNamespace(TimeLimit=FakeWrapper))
    monkeypatch.setattr(protocol, "_runtime", lambda: (fake_metaworld, fake_gym))
    seeds, goals = SeedStreams.from_seed(22), bank("development")
    env = protocol.make_protocol_env("reach-v3", seeds, bank=goals, goal_order=(3, 1))
    assert env.protocol_manifest()["episode_history"] == []
    assert env.unwrapped.reward_function_version == "v2"
    assert env.env.max_episode_steps == 500
    env.reset(seed=999)
    record = protocol.evaluation_record(env)
    assert record["episode"] == 0
    assert record["goal_id"] == goals.goals[3].goal_id
    assert record["goal"] == goals.goals[3].to_dict()


def test_task_payload_must_match_manifest_configuration() -> None:
    declared = goal("reach-v3", 0)
    inconsistent = replace(declared, random_vector=(100.0, 200.0))
    env = TaskController(
        FakeEnv(), GoalBank("source", 0, (inconsistent,)), SeedStreams.from_seed(0), SimpleNamespace
    )
    with pytest.raises(ValueError, match="payload disagrees"):
        env.reset()


def test_manifest_reconstruction_is_stable_in_separate_processes() -> None:
    record = GoalManifest((bank(), bank("development", 10))).to_dict()
    code = (
        "import json, runpy; from scripts.metaworld_protocol import GoalManifest, SeedStreams; "
        f"value=GoalManifest.from_dict(json.loads({json.dumps(json.dumps(record))})); "
        "env=runpy.run_path('tests/test_metaworld_protocol.py')['controller'](); "
        "observations=[env.reset()[0] for _ in range(4)]; "
        "print(value.sha256, SeedStreams.from_seed(42).selection, "
        "json.dumps(observations), json.dumps(env.episode_history))"
    )
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code], env={**os.environ, "PYTHONHASHSEED": str(seed)}, text=True
        )
        for seed in (1, 2)
    ]
    assert outputs[0] == outputs[1]


class FakePolicy:
    def __init__(self) -> None:
        self.actor = self.make_actor()
        self.critic = torch.nn.Linear(2, 1)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_parameter_ids = [id(parameter) for parameter in self.actor.parameters()]

    def make_actor(self) -> torch.nn.Linear:
        return torch.nn.Linear(2, 2)


class FakeSAC:
    def __init__(self, policy: str, env: Any, seed: int, **kwargs: Any) -> None:
        assert policy == "MlpPolicy"
        assert kwargs["replay_buffer_kwargs"]["handle_timeout_termination"] is True
        torch.manual_seed(seed)
        self.policy = FakePolicy()
        self.device = torch.device("cpu")
        self.action_space = FakeSpace()
        self.replay_buffer = SimpleNamespace(handle_timeout_termination=True)

    def set_random_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)


def test_sac_constructor_seed_cannot_override_actor_stream(monkeypatch: Any) -> None:
    original_import = importlib.import_module

    def fake_import(name: str) -> Any:
        return (
            SimpleNamespace(SAC=FakeSAC) if name == "stable_baselines3" else original_import(name)
        )

    monkeypatch.setattr(importlib, "import_module", fake_import)
    seeds = SeedStreams.from_seed(42)
    first, first_meta = build_seeded_sac(None, seeds)
    other_actor, actor_meta = build_seeded_sac(None, replace(seeds, actor=8))
    _, critic_meta = build_seeded_sac(None, replace(seeds, critic=9))
    _, training_meta = build_seeded_sac(None, replace(seeds, training=10))
    assert first_meta["actor_state_sha256"] != actor_meta["actor_state_sha256"]
    assert first_meta["critic_state_sha256"] == actor_meta["critic_state_sha256"]
    assert first_meta["actor_state_sha256"] == critic_meta["actor_state_sha256"]
    assert first_meta["critic_state_sha256"] != critic_meta["critic_state_sha256"]
    assert first_meta["critic_state_sha256"] == training_meta["critic_state_sha256"]
    assert first_meta["actor_state_sha256"] == training_meta["actor_state_sha256"]
    assert first.policy.actor_parameter_ids == [id(p) for p in first.policy.actor.parameters()]
    assert other_actor.policy.actor_parameter_ids == [
        id(p) for p in other_actor.policy.actor.parameters()
    ]
    assert first_meta["critic_state_sha256"] == first_meta["critic_target_state_sha256"]
    with pytest.raises(ValueError, match="time-limit bootstrapping"):
        build_seeded_sac(None, seeds, replay_buffer_kwargs={"handle_timeout_termination": False})
