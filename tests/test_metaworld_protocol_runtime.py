"""Optional SB3 unit checks on synthetic spaces: no MuJoCo, reset, step or learn."""

from __future__ import annotations

import importlib
import importlib.util
import unittest
from dataclasses import replace
from typing import Any

import numpy as np

from scripts.metaworld_protocol import SeedStreams, build_seeded_sac

RUNTIME_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("gymnasium", "stable_baselines3")
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


if __name__ == "__main__":
    unittest.main()
