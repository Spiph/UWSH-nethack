"""Pure Torch and fake-SB3 contract checks; never construct a real SAC or env.

The fake supplies the documented native constructor/distribution interfaces. It
does not establish integration correctness of the pinned SB3 implementation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import math
import random
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, ClassVar, cast
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from scripts.metaworld_protocol import SeedStreams, stable_hash


class SyntheticFamily:
    def __init__(self, *, random_control: bool = False) -> None:
        shapes = [
            ("latent_pi.0.weight", (3, 39)),
            ("latent_pi.0.bias", (3,)),
            ("latent_pi.2.weight", (3, 3)),
            ("latent_pi.2.bias", (3,)),
            ("mu.weight", (4, 3)),
            ("mu.bias", (4,)),
            ("log_std.weight", (4, 3)),
            ("log_std.bias", (4,)),
        ]
        layout, offset = [], 0
        for name, shape in shapes:
            count = int(np.prod(shape))
            layout.append(
                {"name": name, "shape": list(shape), "start": offset, "end": offset + count}
            )
            offset += count
        generator = np.random.default_rng(55)
        self.arrays = {
            "reference": generator.normal(0, 0.1, offset).astype(np.float64),
            "flat_directions": generator.normal(0, 0.1, (2, offset)).astype(np.float64),
        }
        self._metadata: dict[str, Any] = {
            "estimator": "matched_random_flat" if random_control else "flat_population_pca",
            "layout": layout,
            "blocks": [],
            "deployment_basis_keys": ["flat_directions"],
            "coefficient_count": 2,
            "storage": {"deployment_shared_bytes": sum(a.nbytes for a in self.arrays.values())},
        }
        hashes = {}
        for key, array in self.arrays.items():
            digest = hashlib.sha256()
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
            hashes[key] = digest.hexdigest()
        self.family_sha256 = stable_hash({"layout": layout, "blocks": [], "arrays": hashes})
        self.sha256 = stable_hash(
            {"family": self.family_sha256, "estimator": self._metadata["estimator"]}
        )
        if random_control:
            self._metadata["parent_family_sha256"] = self.family_sha256
            self._metadata["parent_artifact_sha256"] = stable_hash(
                {"family": self.family_sha256, "estimator": "flat_population_pca"}
            )
        self.parameterization = "flat"
        self.coefficient_count = 2

    @property
    def metadata(self) -> dict[str, Any]:
        return copy.deepcopy(self._metadata)

    def deployment_arrays(self) -> dict[str, Any]:
        return {key: array.copy() for key, array in self.arrays.items()}


class FakeFlatten(nn.Module):
    def __init__(self, observation_space: Any = None) -> None:
        super().__init__()

    def forward(self, observation: Tensor) -> Tensor:
        return observation.flatten(start_dim=1)


class FakeBox:
    def __init__(self, shape: tuple[int, ...], low: float = -1.0, high: float = 1.0) -> None:
        self.shape = shape
        self.low = np.full(shape, low)
        self.high = np.full(shape, high)

    def seed(self, seed: int) -> None:
        pass


class FakeDistribution:
    def actions_from_params(
        self, mean: Tensor, log_std: Tensor, deterministic: bool = False, **kwargs: Any
    ) -> Tensor:
        gaussian = mean + log_std.exp() * torch.randn_like(mean)
        return torch.tanh(mean if deterministic else gaussian)

    def log_prob_from_params(
        self, mean: Tensor, log_std: Tensor, **kwargs: Any
    ) -> tuple[Tensor, Tensor]:
        gaussian = mean + log_std.exp() * torch.randn_like(mean)
        actions = torch.tanh(gaussian)
        gaussian_log_prob = (
            -0.5 * ((gaussian - mean) * (-log_std).exp()).square()
            - log_std
            - 0.5 * math.log(2 * math.pi)
        )
        log_prob = (gaussian_log_prob - torch.log(1 - actions.square() + 1e-6)).sum(-1)
        return actions, log_prob


class FakeActor(nn.Module):
    def __init__(
        self,
        *,
        observation_space: Any,
        action_space: Any,
        net_arch: list[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: type[nn.Module] = nn.ReLU,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.net_arch, self.features_dim = net_arch, features_dim
        self.features_extractor, self.action_space = features_extractor, action_space
        self.latent_pi = nn.Sequential(
            nn.Linear(features_dim, net_arch[0]),
            activation_fn(),
            nn.Linear(net_arch[0], net_arch[1]),
            activation_fn(),
        )
        self.mu, self.log_std = nn.Linear(net_arch[-1], 4), nn.Linear(net_arch[-1], 4)
        self.action_dist = FakeDistribution()

    def extract_features(self, obs: Tensor, extractor: nn.Module) -> Tensor:
        return cast(Tensor, extractor(obs.float()))

    def get_action_dist_params(self, obs: Tensor) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        hidden = self.latent_pi(self.extract_features(obs, self.features_extractor))
        return self.mu(hidden), self.log_std(hidden).clamp(-20, 2), {}

    def forward(self, obs: Tensor, deterministic: bool = False) -> Tensor:
        mean, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.actions_from_params(mean, log_std, deterministic, **kwargs)

    def action_log_prob(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std, kwargs = self.get_action_dist_params(obs)
        return self.action_dist.log_prob_from_params(mean, log_std, **kwargs)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        return {}

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    def predict(
        self,
        observation: Any,
        state: Any = None,
        episode_start: Any = None,
        deterministic: bool = False,
    ) -> tuple[Any, Any]:
        with torch.no_grad():
            actions = self(torch.as_tensor(observation).reshape(-1, 39), deterministic)
        return actions.numpy(), state


class FakePolicy(nn.Module):
    def __init__(
        self,
        *,
        observation_space: Any,
        action_space: Any,
        lr_schedule: Any,
        net_arch: list[int],
        activation_fn: type[nn.Module],
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.actor_kwargs = {
            "observation_space": observation_space,
            "action_space": action_space,
            "net_arch": net_arch,
            "activation_fn": activation_fn,
        }
        self.actor = self.make_actor()
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_schedule(1))
        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_schedule(1))

    def _update_features_extractor(self, kwargs: dict[str, Any], extractor: Any) -> dict[str, Any]:
        return {**kwargs, "features_extractor": extractor or FakeFlatten(), "features_dim": 39}

    def make_actor(self, features_extractor: Any = None) -> Any:
        raise NotImplementedError

    def make_critic(self, features_extractor: Any = None) -> Any:
        return nn.Sequential(nn.Linear(43, 7), nn.ReLU(), nn.Linear(7, 2))

    def _get_constructor_parameters(self) -> dict[str, Any]:
        return {}


class FakeSAC:
    calls: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, policy: Any, env: Any, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        torch.manual_seed(kwargs["seed"])
        self.policy = policy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: kwargs["learning_rate"],
            **kwargs["policy_kwargs"],
        )
        self.n_envs, self.num_timesteps = 1, 0
        self.action_space = env.action_space
        self.log_ent_coef = nn.Parameter(torch.tensor(0.0))
        self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=kwargs["learning_rate"])
        self.replay_buffer = SimpleNamespace(handle_timeout_termination=True, size=lambda: 0)

    def set_random_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> Any:
    # Keep this active through the whole test, including torch.manual_seed,
    # which calls cuda.manual_seed_all even when is_available() is false.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
    modules: dict[str, ModuleType] = {}
    exports: dict[str, dict[str, Any]] = {
        "gymnasium": {"spaces": SimpleNamespace(Box=FakeBox)},
        "stable_baselines3": {"SAC": FakeSAC},
        "stable_baselines3.common.policies": {"ContinuousCritic": nn.Module},
        "stable_baselines3.common.torch_layers": {
            "BaseFeaturesExtractor": nn.Module,
            "FlattenExtractor": FakeFlatten,
        },
        "stable_baselines3.sac.policies": {
            "Actor": FakeActor,
            "SACPolicy": FakePolicy,
            "LOG_STD_MIN": -20,
            "LOG_STD_MAX": 2,
        },
    }
    for name, attributes in exports.items():
        module = ModuleType(name)
        module.__dict__.update(attributes)
        modules[name] = module
    name = "_mock_validated_metaworld_sac_adapter"
    path = Path(__file__).parents[1] / "scripts/metaworld_sac_adapter.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {**modules, name: result}):
        spec.loader.exec_module(result)
    FakeSAC.calls.clear()
    return result


def environment() -> Any:
    return SimpleNamespace(
        observation_space=FakeBox((39,)),
        action_space=FakeBox((4,)),
    )


def build(adapter: Any, arm: str, *, streams: SeedStreams | None = None) -> tuple[Any, Any]:
    model, metadata = adapter.build_matched_sac(
        environment(),
        streams or SeedStreams.from_seed(35),
        family=SyntheticFamily(random_control=arm == "random"),
        arm=arm,
        activation="relu",
        config=adapter.MatchedSACConfig(**configuration()),
    )
    return model, metadata


def configuration() -> dict[str, Any]:
    return {
        "learning_rate": 3e-4,
        "buffer_size": 32,
        "learning_starts": 10,
        "batch_size": 2,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_update_interval": 1,
    }


def test_zero_reference_full_and_coefficients_have_same_raw_heads_and_native_methods(
    adapter: Any,
) -> None:
    learned, _ = build(adapter, "learned")
    full, _ = build(adapter, "full")
    inputs = torch.randn(5, 39)
    for actual, expected in zip(
        learned.policy.actor.get_action_dist_params(inputs)[:2],
        full.policy.actor.get_action_dist_params(inputs)[:2],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert adapter.MatchedSACActor.forward is FakeActor.forward
    assert adapter.MatchedSACActor.action_log_prob is FakeActor.action_log_prob
    assert not hasattr(learned.policy.actor, "latent_pi")
    assert not hasattr(learned.policy.actor, "mu")
    assert not hasattr(learned.policy.actor, "log_std")


def test_gradients_optimizer_and_reload_leave_frozen_map_unchanged(adapter: Any) -> None:
    model, _ = build(adapter, "learned")
    actor = model.policy.actor
    decoder = actor.coefficient_actor
    assert list(dict(actor.named_parameters())) == ["coefficient_actor.coefficients"]
    before = {key: value.clone() for key, value in decoder.named_buffers()}
    initial_hash = adapter.state_sha256(actor)
    actions, log_prob = actor.action_log_prob(torch.randn(6, 39))
    (actions.square().mean() + log_prob.mean()).backward()
    assert decoder.coefficients.grad.abs().sum() > 0
    actor.optimizer.step()
    assert adapter.state_sha256(actor) != initial_hash
    for key, value in decoder.named_buffers():
        assert torch.equal(value, before[key])
    clone, _ = build(adapter, "learned")
    clone.policy.actor.load_state_dict(actor.state_dict())
    assert adapter.state_sha256(clone.policy.actor) == adapter.state_sha256(actor)
    corrupted = copy.deepcopy(actor.state_dict())
    corrupted["coefficient_actor._extra_state"]["executed_coordinate_sha256"] = "bad"
    with pytest.raises(RuntimeError, match="checkpoint"):
        clone.policy.actor.load_state_dict(corrupted)


def test_all_learning_arms_have_identical_fresh_critics_and_hyperparameters(adapter: Any) -> None:
    models = [build(adapter, arm) for arm in adapter.LEARNING_ARMS]
    assert len({metadata["critic_state_sha256"] for _, metadata in models}) == 1
    for model, metadata in models:
        assert metadata["critic_state_sha256"] == metadata["critic_target_state_sha256"]
        assert model.num_timesteps == 0
        assert model.replay_buffer.size() == 0
        assert metadata["training_seed_applied_after_initialization"] == (
            SeedStreams.from_seed(35).training
        )
    for kwargs in FakeSAC.calls:
        assert kwargs["ent_coef"] == "auto_1.0"
        assert kwargs["target_entropy"] == -4
        assert kwargs["use_sde"] is False
        assert kwargs["replay_buffer_kwargs"] == {"handle_timeout_termination": True}
    records = {metadata["arm"]: metadata for _, metadata in models}
    adapter.validate_matched_preflight(records)
    assert records["scratch"]["initial_reference_float32_bytes_sha256"] is None
    assert records["learned"]["actor_optimizer_state_tensor_bytes"] == 0
    assert records["learned"]["actor_frozen_buffer_bytes"] > 0
    assert records["learned"]["target_critic_stored_scalars"] > 0
    for key, value in (
        ("critic_state_sha256", "different"),
        ("construction_num_timesteps", 1),
        ("initial_reference_float32_bytes_sha256", "different"),
        ("activation", "tanh"),
    ):
        damaged = copy.deepcopy(records)
        damaged["random"][key] = value
        with pytest.raises(ValueError):
            adapter.validate_matched_preflight(damaged)


def test_initializer_isolation_and_training_stream_are_explicit(adapter: Any) -> None:
    random.seed(1)
    np.random.seed(2)
    torch.manual_seed(3)
    expected_numpy: Any = np.random.get_state()
    expected = random.getstate(), torch.get_rng_state()
    with adapter.isolated_seed(99):
        random.random()
        np.random.random()
        torch.rand(1)
    assert random.getstate() == expected[0]
    actual_numpy: Any = np.random.get_state()
    assert np.array_equal(actual_numpy[1], expected_numpy[1])
    assert torch.equal(torch.get_rng_state(), expected[1])
    _, metadata = build(adapter, "learned")
    actual = torch.rand(4)
    torch.manual_seed(metadata["rng_streams"]["training"])
    assert torch.equal(actual, torch.rand(4))


def test_reference_binding_hash_includes_nontensor_metadata(adapter: Any) -> None:
    model, _ = build(adapter, "full")
    actor = model.policy.actor
    before = adapter.state_sha256(actor)
    actor._actor_binding["arm"] = "scratch"
    assert adapter.state_sha256(actor) != before
    with pytest.raises(ValueError, match="verified family"):
        adapter.verified_policy_kwargs(
            {"family": SimpleNamespace(sha256="bad")}, family=SyntheticFamily()
        )
    saved = {"family": SyntheticFamily(), "arm": "learned"}
    verified = SyntheticFamily()
    assert adapter.verified_policy_kwargs(saved, family=verified)["family"] is verified


def test_frozen_reuse_has_no_optimizer_critic_or_learning_interface(adapter: Any) -> None:
    env = environment()
    family = SyntheticFamily()
    with pytest.raises(ValueError, match="source-policy reference"):
        adapter.FrozenSourcePolicy(
            env.observation_space, env.action_space, family=family, activation="relu"
        )
    family._metadata["spec"] = {
        "reference_rule": "source_policy",
        "reference_policy_id": "synthetic-source",
    }
    family._metadata["source_policy_ids"] = ["synthetic-source"]
    frozen = adapter.FrozenSourcePolicy(
        env.observation_space, env.action_space, family=family, activation="relu"
    )
    assert not any(parameter.requires_grad for parameter in frozen.actor.parameters())
    assert not hasattr(frozen.actor, "optimizer")
    assert not hasattr(frozen, "learn")
    assert not hasattr(frozen, "critic")
    actions, state = frozen.predict(np.zeros((2, 39), dtype=np.float32))
    assert actions.shape == (2, 4) and state is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"buffer_size": 0},
        {"batch_size": 0},
        {"learning_starts": -1},
        {"gradient_steps": True},
        {"gamma": float("nan")},
        {"tau": 0},
    ],
)
def test_invalid_matched_configuration_fails_before_construction(adapter: Any, kwargs: Any) -> None:
    with pytest.raises(ValueError):
        adapter.MatchedSACConfig(**{**configuration(), **kwargs})
    assert not FakeSAC.calls


def test_random_provenance_and_precision_changes_fail_closed(adapter: Any) -> None:
    with pytest.raises(ValueError, match="random-control provenance"):
        adapter.build_matched_sac(
            environment(),
            SeedStreams.from_seed(1),
            family=SyntheticFamily(),
            arm="random",
            activation="relu",
            config=adapter.MatchedSACConfig(**configuration()),
        )
    model, _ = build(adapter, "full")
    before = adapter.state_sha256(model.policy.actor)
    with pytest.raises(ValueError, match="precision"):
        model.policy.actor.double()
    assert before == adapter.state_sha256(model.policy.actor)


def test_actor_critic_training_initialization_streams_do_not_cross(adapter: Any) -> None:
    streams = SeedStreams.from_seed(45)
    _, baseline = build(adapter, "scratch", streams=streams)
    _, actor_changed = build(adapter, "scratch", streams=replace(streams, actor=123))
    _, critic_changed = build(adapter, "scratch", streams=replace(streams, critic=456))
    _, training_changed = build(adapter, "scratch", streams=replace(streams, training=789))
    assert baseline["actor_state_sha256"] != actor_changed["actor_state_sha256"]
    assert baseline["critic_state_sha256"] == actor_changed["critic_state_sha256"]
    assert baseline["actor_state_sha256"] == critic_changed["actor_state_sha256"]
    assert baseline["critic_state_sha256"] != critic_changed["critic_state_sha256"]
    assert baseline["actor_state_sha256"] == training_changed["actor_state_sha256"]
    assert baseline["critic_state_sha256"] == training_changed["critic_state_sha256"]


@pytest.mark.parametrize(
    "observation,actions",
    [
        (FakeBox((40,)), FakeBox((4,))),
        (SimpleNamespace(shape=(39,)), FakeBox((4,))),
        (FakeBox((39,)), FakeBox((4,), low=-2.0)),
        (FakeBox((39,)), FakeBox((4,), high=2.0)),
        (FakeBox((39,)), FakeBox((3,))),
    ],
)
def test_spaces_and_action_coordinates_fail_before_construction(
    adapter: Any, observation: Any, actions: Any
) -> None:
    env = SimpleNamespace(observation_space=observation, action_space=actions)
    with pytest.raises(ValueError, match="Box"):
        adapter.build_matched_sac(
            env,
            SeedStreams.from_seed(1),
            family=SyntheticFamily(),
            arm="full",
            activation="relu",
            config=adapter.MatchedSACConfig(**configuration()),
        )
    assert not FakeSAC.calls


@pytest.mark.parametrize("declaration", ["spec", "fallback"])
def test_frozen_source_activation_conflict_fails_before_native_actor(
    adapter: Any, declaration: str
) -> None:
    family = SyntheticFamily()
    family._metadata["spec"] = {
        "reference_rule": "source_policy",
        "reference_policy_id": "synthetic-source",
    }
    family._metadata["source_policy_ids"] = ["synthetic-source"]
    if declaration == "spec":
        family._metadata["spec"]["alignment_activation"] = "tanh"
    else:
        family._metadata["source_alignment_activation"] = "tanh"
    env = environment()
    with (
        patch.object(FakeActor, "__init__", side_effect=AssertionError("native actor constructed")),
        pytest.raises(ValueError, match="source-alignment declaration"),
    ):
        adapter.FrozenSourcePolicy(
            env.observation_space, env.action_space, family=family, activation="relu"
        )
    with pytest.raises(ValueError, match="source-alignment declaration"):
        adapter.build_matched_sac(
            env,
            SeedStreams.from_seed(1),
            family=family,
            arm="full",
            activation="relu",
            config=adapter.MatchedSACConfig(**configuration()),
        )
    assert not FakeSAC.calls


def test_fake_fixture_blocks_gpu_rng_discovery_and_intercepts_manual_seed(
    adapter: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_gpu_rng(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("pure CPU test attempted to inspect or mutate GPU RNG")

    for name in (
        "device_count",
        "get_rng_state",
        "get_rng_state_all",
        "set_rng_state",
        "set_rng_state_all",
    ):
        monkeypatch.setattr(torch.cuda, name, forbidden_gpu_rng)
    intercepted: list[int] = []
    monkeypatch.setattr(torch.cuda, "manual_seed_all", intercepted.append)
    assert torch.cuda.is_available() is False
    build(adapter, "learned")
    assert intercepted
