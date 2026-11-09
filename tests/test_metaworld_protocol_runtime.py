"""Optional pinned-runtime constructor, replay and bounded physics checks."""

from __future__ import annotations

import importlib
import importlib.util
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from scripts.metaworld_deadlines import RunDeadline, guard_env
from scripts.metaworld_protocol import (
    GoalManifest,
    SeedStreams,
    build_seeded_sac,
    evaluation_record,
    generate_goal_bank,
    make_protocol_env,
)

RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("gymnasium", "stable_baselines3")
)
PHYSICS_AVAILABLE = RUNTIME_AVAILABLE and all(
    importlib.util.find_spec(name) is not None for name in ("metaworld", "mujoco")
)


@unittest.skipUnless(RUNTIME_AVAILABLE, "requires the isolated pinned Gymnasium/SB3 runtime")
class SB3ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.models: list[Any] = []
        self.counts: list[dict[str, int]] = []

    def tearDown(self) -> None:
        for model in self.models:
            model.get_env().close()
        for counts in self.counts:
            self.assertEqual(counts, {"reset": 0, "step": 0})

    def make_model(self, seeds: SeedStreams) -> tuple[Any, dict[str, Any]]:
        gym = importlib.import_module("gymnasium")
        counts = {"reset": 0, "step": 0}

        def reset(env: Any, **kwargs: Any) -> Any:
            counts["reset"] += 1
            raise AssertionError("model construction must not reset an environment")

        def step(env: Any, action: Any) -> Any:
            counts["step"] += 1
            raise AssertionError("unit validation must not collect environment transitions")

        synthetic = type("NoTransitionEnv", (gym.Env,), {"reset": reset, "step": step})()
        synthetic.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(39,), dtype=np.float64)
        synthetic.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        model, metadata = build_seeded_sac(
            synthetic,
            seeds,
            device="cpu",
            policy_kwargs={"net_arch": [8, 8]},
            buffer_size=32,
            batch_size=2,
            learning_starts=0,
            verbose=0,
        )
        self.models.append(model)
        self.counts.append(counts)
        self.assertEqual(model.num_timesteps, 0)
        self.assertEqual(model.seed, seeds.training)
        self.assertTrue(model.replay_buffer.handle_timeout_termination)
        optimizer_ids = {
            id(parameter)
            for group in model.policy.actor.optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(optimizer_ids, {id(parameter) for parameter in model.actor.parameters()})
        return model, metadata

    def test_actual_actor_critic_and_training_seed_separation(self) -> None:
        streams = SeedStreams.from_seed(123)
        _, baseline = self.make_model(streams)
        _, critic_changed = self.make_model(replace(streams, critic=7))
        _, actor_changed = self.make_model(replace(streams, actor=8))
        _, training_changed = self.make_model(replace(streams, training=9))
        self.assertEqual(baseline["actor_state_sha256"], critic_changed["actor_state_sha256"])
        self.assertNotEqual(baseline["critic_state_sha256"], critic_changed["critic_state_sha256"])
        self.assertNotEqual(baseline["actor_state_sha256"], actor_changed["actor_state_sha256"])
        self.assertEqual(baseline["critic_state_sha256"], actor_changed["critic_state_sha256"])
        self.assertEqual(baseline["actor_state_sha256"], training_changed["actor_state_sha256"])
        self.assertEqual(baseline["critic_state_sha256"], training_changed["critic_state_sha256"])
        self.assertEqual(baseline["critic_state_sha256"], baseline["critic_target_state_sha256"])

    def test_actual_replay_masks_timeouts_and_keeps_terminal_observations(self) -> None:
        model, _ = self.make_model(SeedStreams.from_seed(123))
        replay = model.replay_buffer
        for timeout in (True, False):
            model._last_obs = np.zeros((1, 39))
            model._store_transition(
                replay,
                np.zeros((1, 4)),
                -np.ones((1, 39)),  # synthetic autoreset observation
                np.zeros(1),
                np.ones(1, dtype=bool),
                [{"TimeLimit.truncated": timeout, "terminal_observation": np.ones(39)}],
            )
        samples = replay._get_samples(np.array([0, 1]))
        self.assertEqual(samples.dones.cpu().numpy().tolist(), [[0.0], [1.0]])
        np.testing.assert_array_equal(samples.next_observations.cpu().numpy(), np.ones((2, 39)))


@unittest.skipUnless(PHYSICS_AVAILABLE, "requires the isolated pinned Meta-World/MuJoCo runtime")
class PhysicsProtocolTests(unittest.TestCase):
    def test_paired_task_sequences_and_full_horizon_without_policy_training(self) -> None:
        budget = RunDeadline(60, datetime.now(timezone.utc) + timedelta(seconds=60))
        transitions = 0
        for task in ("reach-v3", "push-v3", "pick-place-v3"):
            with self.subTest(task=task):
                streams = SeedStreams.from_seed(2026110917)
                budget.require("source_bank_validation")
                source = generate_goal_bank(task, streams, split="source")
                budget.require("development_bank_validation")
                development = generate_goal_bank(task, streams, split="development")
                GoalManifest((source, development))
                envs = []
                try:
                    for _ in range(2):
                        budget.require("paired_environment_initialization")
                        envs.append(
                            guard_env(
                                make_protocol_env(
                                    task, streams, bank=development, goal_order=(0, 1)
                                ),
                                budget,
                            )
                        )
                    left, right = envs
                    for env in envs:
                        self.assertEqual(
                            env.get_wrapper_attr("protocol_manifest")()["episode_history"], []
                        )
                    for episode in range(2):
                        lo, li = left.reset(seed=11 + episode)
                        ro, ri = right.reset(seed=999 + episode)
                        np.testing.assert_array_equal(lo, ro)
                        self.assertEqual(
                            li["protocol"]["goal_id"], development.goals[episode].goal_id
                        )
                        self.assertEqual(li["protocol"]["goal_id"], ri["protocol"]["goal_id"])
                        self.assertEqual(
                            li["protocol"]["episode_seed"], ri["protocol"]["episode_seed"]
                        )
                        for _ in range(2):
                            ls = left.step(np.zeros(4, dtype=np.float32))
                            rs = right.step(np.zeros(4, dtype=np.float32))
                            transitions += 2
                            np.testing.assert_array_equal(ls[0], rs[0])
                            np.testing.assert_equal(ls[1:4], rs[1:4])
                    if task == "reach-v3":
                        for _ in range(498):
                            _, _, terminated, truncated, _ = left.step(
                                np.zeros(4, dtype=np.float32)
                            )
                            transitions += 1
                        self.assertIs(terminated, False)
                        self.assertIs(truncated, True)
                        record = evaluation_record(left)
                        self.assertEqual(record["length"], 500)
                        self.assertIs(record["complete_episode"], True)
                finally:
                    for env in envs:
                        env.close()
        self.assertEqual(transitions, 522)


if __name__ == "__main__":
    unittest.main()
