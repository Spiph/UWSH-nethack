"""Actual SB3 continuation on synthetic data, without Meta-World or research runs."""

from __future__ import annotations

import copy
import importlib
import importlib.util
import json
import pickle
import random
import tempfile
import unittest
import zipfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

try:
    PINNED_RUNTIME = (
        importlib.util.find_spec("gymnasium") is not None
        and importlib.util.find_spec("stable_baselines3") is not None
        and version("stable-baselines3") == "2.6.0"
    )
except PackageNotFoundError:
    PINNED_RUNTIME = False


@unittest.skipUnless(PINNED_RUNTIME, "requires the isolated pinned SB3 2.6.0 runtime")
class DevelopmentRuntimeTests(unittest.TestCase):
    torch: ClassVar[Any]
    np: ClassVar[Any]
    gym: ClassVar[Any]
    runtime: ClassVar[Any]
    protocol: ClassVar[Any]
    original_threads: ClassVar[int]

    @classmethod
    def setUpClass(cls) -> None:
        cls.torch = importlib.import_module("torch")
        cls.np = importlib.import_module("numpy")
        cls.gym = importlib.import_module("gymnasium")
        # Dynamic import keeps static type checking independent of optional RL
        # modules and the runtime-only legacy pilot helper.
        cls.runtime = importlib.import_module("scripts.metaworld_development")
        cls.protocol = importlib.import_module("scripts.metaworld_protocol")
        cls.original_threads = cls.torch.get_num_threads()
        cls.torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.torch.set_num_threads(cls.original_threads)

    def setUp(self) -> None:
        self.models: list[Any] = []

    def tearDown(self) -> None:
        for model in self.models:
            model.get_env().close()

    def make_model(self, seed: int) -> tuple[Any, Any, Any]:
        np = self.np
        counts = {"reset": 0, "step": 0}

        def reset(env: Any, *, seed: int | None = None, options: Any = None) -> Any:
            counts["reset"] += 1
            env.counter = 0
            return np.zeros(39, dtype=np.float32), {}

        def step(env: Any, action: Any) -> Any:
            counts["step"] += 1
            env.counter += 1
            observation = np.full(39, env.counter / 100, dtype=np.float32)
            reward = float(0.1 - np.square(action).sum())
            return observation, reward, False, False, {}

        env = type("SyntheticContinuationEnv", (self.gym.Env,), {"reset": reset, "step": step})()
        env.observation_space = self.gym.spaces.Box(-5, 5, shape=(39,), dtype=np.float32)
        env.action_space = self.gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        env.counter = 0
        env.counts = counts
        streams = self.protocol.SeedStreams.from_seed(seed)
        model, _ = self.protocol.build_seeded_sac(
            env,
            streams,
            device="cpu",
            policy_kwargs={"net_arch": [8, 8]},
            buffer_size=32,
            batch_size=2,
            learning_starts=0,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=0,
        )
        capsule = self.runtime.LearnerRNG(
            self.runtime.capture_rng(self.torch, False), self.torch, False
        )
        self.models.append(model)
        return model, env, capsule

    def optimizer_steps(self, optimizer: Any) -> set[int]:
        return {int(state["step"].item()) for state in optimizer.state.values()}

    def train_segment(self, model: Any, capsule: Any, steps: int) -> tuple[float, float, float]:
        with capsule.activate():
            model.learn(total_timesteps=steps, reset_num_timesteps=False, progress_bar=False)
            return random.random(), float(self.np.random.random()), float(self.torch.rand(1).item())

    def assert_rng_equal(self, left: Any, right: Any) -> None:
        self.assertEqual(left["python"], right["python"])
        self.assertEqual(left["numpy"][0], right["numpy"][0])
        self.np.testing.assert_array_equal(left["numpy"][1], right["numpy"][1])
        self.assertEqual(left["numpy"][2:], right["numpy"][2:])
        self.assertTrue(self.torch.equal(left["torch_cpu"], right["torch_cpu"]))
        self.assertEqual(left["torch_cuda"], right["torch_cuda"])

    def test_incremental_learning_preserves_replay_optimizer_and_environment(self) -> None:
        model, env, capsule = self.make_model(42)
        identities = {
            "actor": id(model.actor),
            "critic": id(model.critic),
            "replay": id(model.replay_buffer),
            "actor_optimizer": id(model.actor.optimizer),
            "critic_optimizer": id(model.critic.optimizer),
            "entropy_optimizer": id(model.ent_coef_optimizer),
            "entropy_coefficient": id(model.log_ent_coef),
            "vector_env": id(model.get_env()),
        }
        self.train_segment(model, capsule, 8)
        self.assertEqual(model.num_timesteps, 8)
        self.assertEqual(model.replay_buffer.pos, 8)
        self.assertEqual(env.counts, {"reset": 1, "step": 8})
        for optimizer in (model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer):
            self.assertEqual(self.optimizer_steps(optimizer), {8})
        previous_observation = model._last_obs.copy()
        self.train_segment(model, capsule, 12)
        self.assertEqual(model.num_timesteps, 20)
        self.assertEqual(model._n_updates, 20)
        self.assertEqual(model.replay_buffer.pos, 20)
        self.assertEqual(env.counts, {"reset": 1, "step": 20})
        self.np.testing.assert_array_equal(
            model.replay_buffer.observations[8, 0], previous_observation[0]
        )
        self.assertEqual(
            identities,
            {
                "actor": id(model.actor),
                "critic": id(model.critic),
                "replay": id(model.replay_buffer),
                "actor_optimizer": id(model.actor.optimizer),
                "critic_optimizer": id(model.critic.optimizer),
                "entropy_optimizer": id(model.ent_coef_optimizer),
                "entropy_coefficient": id(model.log_ent_coef),
                "vector_env": id(model.get_env()),
            },
        )
        for optimizer in (model.actor.optimizer, model.critic.optimizer, model.ent_coef_optimizer):
            self.assertEqual(self.optimizer_steps(optimizer), {20})
        self.assert_checkpoint_roundtrip(model, env, capsule)

    def assert_checkpoint_roundtrip(self, model: Any, env: Any, capsule: Any) -> None:
        geometry = importlib.import_module("scripts.metaworld_geometry")
        adapter = self.runtime.Runtime.__new__(self.runtime.Runtime)
        adapter.np = self.np
        adapter.geometry = geometry
        env.protocol_manifest = lambda: {"synthetic_unit_test": True}
        state = SimpleNamespace(model=model, env=env, rng=capsule)
        with tempfile.TemporaryDirectory(prefix="ups-synthetic-checkpoint-") as directory:
            store = self.runtime.ArtifactStore(Path(directory) / "artifacts", 1024**2)
            adapter.runner = SimpleNamespace(store=store)
            files = adapter.save(state, "checkpoints/synthetic_20")
            self.assertLess(store.size(), store.limit)
            for key in (".zip", ".replay.pkl", "actor", "rng"):
                artifact = store.root / files[key]["path"]
                self.assertEqual(self.runtime.digest(artifact), files[key]["sha256"])
            with zipfile.ZipFile(store.root / files[".zip"]["path"]) as archive:
                self.assertIsNone(archive.testzip())
            with (store.root / files["rng"]["path"]).open("rb") as handle:
                self.assert_rng_equal(pickle.load(handle), capsule.state)
            vector, layout = geometry.actor_vector(model.actor)
            with self.np.load(store.root / files["actor"]["path"], allow_pickle=False) as actor:
                self.np.testing.assert_array_equal(actor["vector"], vector)
                self.assertEqual(json.loads(actor["layout_json"].item()), layout)
            loaded = importlib.import_module("stable_baselines3").SAC.load(
                store.root / files[".zip"]["path"], device="cpu"
            )
            loaded.load_replay_buffer(store.root / files[".replay.pkl"]["path"])
            self.assertEqual(loaded.num_timesteps, 20)
            self.assertEqual(loaded.replay_buffer.pos, model.replay_buffer.pos)
            self.np.testing.assert_array_equal(
                loaded.replay_buffer.actions, model.replay_buffer.actions
            )
            for name, tensor in model.policy.state_dict().items():
                self.assertTrue(self.torch.equal(tensor, loaded.policy.state_dict()[name]), name)
            self.assertEqual(self.optimizer_steps(loaded.actor.optimizer), {20})
            self.assertEqual(self.optimizer_steps(loaded.critic.optimizer), {20})
            self.assertEqual(self.optimizer_steps(loaded.ent_coef_optimizer), {20})
            self.assertTrue(self.torch.equal(model.log_ent_coef, loaded.log_ent_coef))
            observation = self.np.zeros(39, dtype=self.np.float32)
            self.np.testing.assert_array_equal(
                model.predict(observation, deterministic=True)[0],
                loaded.predict(observation, deterministic=True)[0],
            )
            self.assertEqual(files["environment"], {"synthetic_unit_test": True})
        self.assertEqual(env.counts, {"reset": 1, "step": 20})

    def test_interleaved_learners_and_evaluation_preserve_source_rng_sequences(self) -> None:
        isolated = [self.make_model(seed) for seed in (43, 44)]
        interleaved = [self.make_model(seed) for seed in (43, 44)]
        expected_draws = []
        for model, _, capsule in isolated:
            expected_draws.append(
                (self.train_segment(model, capsule, 8), self.train_segment(model, capsule, 12))
            )
        first_draws = [self.train_segment(model, capsule, 8) for model, _, capsule in interleaved]
        before_capsules = [copy.deepcopy(capsule.state) for _, _, capsule in interleaved]
        caller = self.runtime.capture_rng(self.torch, False)
        try:
            # Stand-in evaluation consumes all three global streams; it never
            # activates a source capsule or steps a source training environment.
            for _ in range(11):
                random.random()
                self.np.random.random(17)
                self.torch.rand(19)
        finally:
            self.runtime.restore_rng(caller, self.torch, False)
        self.assert_rng_equal(caller, self.runtime.capture_rng(self.torch, False))
        for before, (_, _, capsule) in zip(before_capsules, interleaved, strict=True):
            self.assert_rng_equal(before, capsule.state)
        second_draws = [self.train_segment(model, capsule, 12) for model, _, capsule in interleaved]
        self.assertEqual(expected_draws, list(zip(first_draws, second_draws, strict=True)))
        for (reference, _, reference_rng), (actual, env, actual_rng) in zip(
            isolated, interleaved, strict=True
        ):
            self.assertEqual(actual.num_timesteps, 20)
            self.assertEqual(env.counts, {"reset": 1, "step": 20})
            for name, tensor in reference.policy.state_dict().items():
                self.assertTrue(self.torch.equal(tensor, actual.policy.state_dict()[name]), name)
            self.assertTrue(self.torch.equal(reference.log_ent_coef, actual.log_ent_coef))
            self.np.testing.assert_array_equal(
                reference.replay_buffer.actions[:20], actual.replay_buffer.actions[:20]
            )
            self.assert_rng_equal(reference_rng.state, actual_rng.state)


@unittest.skipUnless(PINNED_RUNTIME, "requires the isolated pinned SB3 2.6.0 runtime")
class ReplayBufferSerializationTests(unittest.TestCase):
    """Production-sized replay serialization; no environment or policy is created."""

    def replay(self) -> Any:
        np = importlib.import_module("numpy")
        gym = importlib.import_module("gymnasium")
        replay_type = importlib.import_module("stable_baselines3.common.buffers").ReplayBuffer
        replay = replay_type(
            buffer_size=50000,
            observation_space=gym.spaces.Box(-100, 100, shape=(39,), dtype=np.float64),
            action_space=gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
            device="cpu",
            handle_timeout_termination=True,
        )
        # Synthetic data insertion exercises replay state without environment interaction.
        replay.add(
            np.ones((1, 39)),
            np.full((1, 39), 2.0),
            np.ones((1, 4)),
            np.array([3.0]),
            np.array([False]),
            [{"TimeLimit.truncated": True}],
        )
        replay.observations[-1] = 17.0
        replay.next_observations[-1] = 19.0
        return replay

    def test_fifty_thousand_slot_replay_roundtrip_without_environment_or_policy(self) -> None:
        np = importlib.import_module("numpy")
        runtime = importlib.import_module("scripts.metaworld_development")
        replay = self.replay()
        with tempfile.TemporaryDirectory(prefix="ups-large-replay-") as directory:
            store = runtime.ArtifactStore(Path(directory) / "artifacts", 64 * 1024**2)
            with store.open("replay.pkl") as handle:
                pickle.dump(replay, handle, protocol=pickle.HIGHEST_PROTOCOL)
            with (store.root / "replay.pkl").open("rb") as handle:
                loaded = pickle.load(handle)
            self.assertEqual(loaded.buffer_size, 50000)
            self.assertEqual(loaded.pos, replay.pos)
            self.assertEqual(loaded.full, replay.full)
            self.assertEqual(loaded.handle_timeout_termination, replay.handle_timeout_termination)
            for field in (
                "observations",
                "next_observations",
                "actions",
                "rewards",
                "dones",
                "timeouts",
            ):
                np.testing.assert_array_equal(getattr(loaded, field), getattr(replay, field))
            self.assertGreater(store.size(), 65536)
            self.assertLessEqual(store.size(), store.limit)

    def test_fifty_thousand_slot_replay_respects_storage_quota(self) -> None:
        runtime = importlib.import_module("scripts.metaworld_development")
        replay = self.replay()
        with tempfile.TemporaryDirectory(prefix="ups-large-replay-cap-") as directory:
            store = runtime.ArtifactStore(Path(directory) / "artifacts", 1024**2)
            with (
                store.open("partial.pkl") as handle,
                self.assertRaisesRegex(runtime.Stopped, "artifact_storage_limit"),
            ):
                pickle.dump(replay, handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.assertLessEqual(store.size(), store.limit - store.reserve)
            store.json("failure.json", {"status": "INCOMPLETE"})
            self.assertLessEqual(store.size(), store.limit)


if __name__ == "__main__":
    unittest.main()
