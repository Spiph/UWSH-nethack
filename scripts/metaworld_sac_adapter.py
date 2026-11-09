"""Optional pinned-SB3 actor adapters; construction never collects transitions.

Import this module only in the isolated SB3 2.6.0 runtime. The caller owns the
protocol environments, interaction ledger, evaluation streams and run allowance.
No target experiment or runtime smoke is authorized by this implementation.
"""

from __future__ import annotations

import copy
import hashlib
import random
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor
from stable_baselines3.sac.policies import LOG_STD_MAX, LOG_STD_MIN, Actor, SACPolicy
from torch import Tensor, nn

from scripts.metaworld_affine_actor import AffineCoefficientActor, AffineFamily
from scripts.metaworld_alignment import SAC_PARAMETERS, validate_layout
from scripts.metaworld_protocol import SeedStreams, stable_hash

LEARNING_ARMS = ("learned", "random", "full", "scratch")
LEARNED_ESTIMATORS = ("flat_population_pca", "uwsh_v3_parameter_hosvd")
RANDOM_ESTIMATORS = ("matched_random_flat", "matched_random_parameter_factors")


def validate_metaworld_spaces(observation_space: Any, action_space: Any) -> None:
    """Require the source actor's observation and physical-action coordinates."""
    if not isinstance(observation_space, spaces.Box) or observation_space.shape != (39,):
        raise ValueError("require a 39-dimensional Box observation space")
    if (
        not isinstance(action_space, spaces.Box)
        or action_space.shape != (4,)
        or not np.array_equal(action_space.low, np.full(4, -1.0))
        or not np.array_equal(action_space.high, np.full(4, 1.0))
    ):
        raise ValueError("require a four-dimensional Box action space with exact [-1, 1] bounds")


def _validate_arm_family(family: AffineFamily, arm: str) -> None:
    if arm not in LEARNING_ARMS:
        raise ValueError("unknown learning arm")
    estimator = family.metadata.get("estimator")
    if arm == "learned" and estimator not in LEARNED_ESTIMATORS:
        raise ValueError("the learned arm requires a learned source-only estimator")
    if arm == "random" and estimator not in RANDOM_ESTIMATORS:
        raise ValueError("the random arm requires explicitly labeled random-control provenance")


def _validate_activation(family: AffineFamily, activation: str) -> None:
    metadata = family.metadata
    declared = metadata.get("spec", {}).get(
        "alignment_activation", metadata.get("source_alignment_activation")
    )
    if declared is not None and declared != activation:
        raise ValueError("activation conflicts with the source-alignment declaration")


def state_sha256(module: nn.Module) -> str:
    """Hash tensor bytes AND canonical non-tensor extra state, including map IDs."""
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        if isinstance(value, Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(b"tensor")
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        else:
            digest.update(b"json")
            digest.update(stable_hash(value).encode())
    return digest.hexdigest()


@contextmanager
def isolated_seed(seed: int) -> Iterator[None]:
    """Synchronous construction isolation, including every available CUDA RNG."""
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("initialization seeds must be unsigned 32-bit integers")
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _architecture(family: AffineFamily) -> tuple[int, int]:
    metadata = family.metadata
    layout = validate_layout(metadata["layout"], family.deployment_arrays()["reference"].size)
    shapes = {item["name"]: item["shape"] for item in layout}
    if tuple(shapes) != SAC_PARAMETERS:
        raise ValueError("the reference must cover the complete ordered SAC actor")
    first, second = shapes["latent_pi.0.weight"], shapes["latent_pi.2.weight"]
    if (
        len(first) != 2
        or len(second) != 2
        or first[1] != 39
        or second[1] != first[0]
        or shapes["latent_pi.0.bias"] != [first[0]]
        or shapes["latent_pi.2.bias"] != [second[0]]
        or shapes["mu.weight"] != [4, second[0]]
        or shapes["mu.bias"] != [4]
        or shapes["log_std.weight"] != [4, second[0]]
        or shapes["log_std.bias"] != [4]
    ):
        raise ValueError("require the declared 39-observation, 4-action, two-hidden-layer actor")
    return int(first[0]), int(second[0])


class MatchedSACActor(Actor):
    """Native SAC distribution with either all actor weights or only coefficients."""

    def __init__(
        self,
        *args: Any,
        family: AffineFamily,
        arm: str,
        **kwargs: Any,
    ) -> None:
        if args:
            raise ValueError("use named native Actor constructor arguments")
        validate_metaworld_spaces(kwargs.get("observation_space"), kwargs.get("action_space"))
        _validate_arm_family(family, arm)
        if torch.get_default_dtype() != torch.float32:
            raise ValueError("the matched runtime requires float32 native actor initialization")
        activation_fn = kwargs.get("activation_fn", nn.ReLU)
        if activation_fn not in (nn.ReLU, nn.Tanh):
            raise ValueError("declare ReLU or Tanh without changing the source architecture")
        activation_name = "relu" if activation_fn is nn.ReLU else "tanh"
        _validate_activation(family, activation_name)
        if kwargs.get("use_sde", False):
            raise ValueError("the matched comparison uses native diagonal Gaussian SAC without SDE")
        self.family = family
        self.arm = arm
        self.activation_name = activation_name
        super().__init__(*args, **kwargs)
        if (
            tuple(self.net_arch) != _architecture(family)
            or self.features_dim != 39
            or self.action_space.shape != (4,)
            or type(self.features_extractor) is not FlattenExtractor
            or list(self.features_extractor.parameters())
        ):
            raise ValueError(
                "require the declared architecture and a parameter-free FlattenExtractor"
            )
        self.coefficient_actor: AffineCoefficientActor | None = None
        reference = family.deployment_arrays()["reference"].astype(np.float32)
        if not np.isfinite(reference).all():
            raise ValueError("the source reference overflows float32")
        self._actor_binding = {
            "family_sha256": family.family_sha256,
            "basis_artifact_sha256": family.sha256,
            "arm": arm,
            "activation": self.activation_name,
            "materialization_dtype": "float32",
            "initial_behavior": (
                "independent_native_initialization" if arm == "scratch" else "affine_reference"
            ),
            "initial_reference_float32_bytes_sha256": (
                None if arm == "scratch" else hashlib.sha256(reference.tobytes()).hexdigest()
            ),
        }
        if arm in ("learned", "random"):
            # Actor.__init__ consumes the native constructor draws. Remove these
            # modules before SACPolicy creates Adam; no unused trainable layers.
            del self.latent_pi, self.mu, self.log_std
            self.coefficient_actor = AffineCoefficientActor(
                family, activation=self.activation_name, dtype=torch.float32
            )
            if self.coefficient_actor.coefficient_count == 0:
                raise ValueError("zero-dimensional families use the nonlearning frozen interface")
        elif arm == "full":
            arrays = family.deployment_arrays()
            reference_tensor = torch.as_tensor(arrays["reference"], dtype=torch.float32)
            with torch.no_grad():
                parameters = dict(self.named_parameters())
                for item in family.metadata["layout"]:
                    parameters[item["name"]].copy_(
                        reference_tensor[item["start"] : item["end"]].reshape(item["shape"])
                    )

    def get_action_dist_params(self, obs: Any) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        if self.coefficient_actor is None:
            return super().get_action_dist_params(obs)
        features = self.extract_features(obs, self.features_extractor)
        mean, raw_log_std = self.coefficient_actor.head_parameters(features)
        return mean, torch.clamp(raw_log_std, LOG_STD_MIN, LOG_STD_MAX), {}

    def _get_constructor_parameters(self) -> dict[str, Any]:
        return {**super()._get_constructor_parameters(), "family": self.family, "arm": self.arm}

    def get_extra_state(self) -> dict[str, Any]:
        return copy.deepcopy(self._actor_binding)

    def set_extra_state(self, state: Any) -> None:
        if state != self.get_extra_state():
            raise ValueError("actor checkpoint family, arm, activation or precision mismatch")

    def _apply(self, fn: Callable[[Tensor], Tensor], recurse: bool = True) -> MatchedSACActor:
        if fn(torch.empty(0, dtype=torch.float32)).dtype != torch.float32:
            raise ValueError("reconstruct the matched actor to change its declared precision")
        cast(Callable[..., nn.Module], super()._apply)(fn, recurse=recurse)
        return self


class MatchedSACPolicy(SACPolicy):
    """All learning arms receive independently seeded, fresh twin critics."""

    def __init__(
        self,
        *args: Any,
        family: AffineFamily,
        arm: str,
        actor_seed: int,
        critic_seed: int,
        **kwargs: Any,
    ) -> None:
        if kwargs.get("share_features_extractor", False):
            raise ValueError("actor/critic feature sharing is excluded from the matched comparison")
        if kwargs.get("n_critics", 2) != 2:
            raise ValueError("the matched comparison requires fresh twin critics")
        if kwargs.get("features_extractor_class", FlattenExtractor) is not FlattenExtractor:
            raise ValueError("the matched comparison requires a parameter-free FlattenExtractor")
        self.family, self.arm = family, arm
        self.actor_seed, self.critic_seed = actor_seed, critic_seed
        super().__init__(*args, **kwargs)
        actor_parameters = list(self.actor.parameters())
        optimized = [p for group in self.actor.optimizer.param_groups for p in group["params"]]
        if len(optimized) != len(actor_parameters) or {id(p) for p in optimized} != {
            id(p) for p in actor_parameters
        }:
            raise ValueError("actor optimizer does not own exactly the declared actor parameters")
        if (
            isinstance(self.actor, MatchedSACActor)
            and self.actor.coefficient_actor is not None
            and (
                len(actor_parameters) != 1
                or actor_parameters[0] is not self.actor.coefficient_actor.coefficients
            )
        ):
            raise ValueError("coefficient adaptation must optimize only the coefficient vector")

    def make_actor(self, features_extractor: BaseFeaturesExtractor | None = None) -> Actor:
        kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        with isolated_seed(self.actor_seed):
            return MatchedSACActor(**kwargs, family=self.family, arm=self.arm).to(self.device)

    def make_critic(
        self, features_extractor: BaseFeaturesExtractor | None = None
    ) -> ContinuousCritic:
        # Each construction is isolated from actor size and basis representation.
        # SACPolicy still copies the online critic into its target as usual.
        with isolated_seed(self.critic_seed):
            return super().make_critic(features_extractor)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        return {
            **super()._get_constructor_parameters(),
            "family": self.family,
            "arm": self.arm,
            "actor_seed": self.actor_seed,
            "critic_seed": self.critic_seed,
        }


@dataclass(frozen=True)
class MatchedSACConfig:
    """Explicit common hyperparameters, to be frozen in the later run design."""

    learning_rate: float
    buffer_size: int
    learning_starts: int
    batch_size: int
    tau: float
    gamma: float
    train_freq: int
    gradient_steps: int
    target_update_interval: int

    def __post_init__(self) -> None:
        integers = (
            self.buffer_size,
            self.batch_size,
            self.train_freq,
            self.gradient_steps,
            self.target_update_interval,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError(
                "sizes, training frequency and gradient steps must be positive integers"
            )
        if type(self.learning_starts) is not int or self.learning_starts < 0:
            raise ValueError("learning_starts must count nonnegative training transitions")
        if (
            self.batch_size > self.buffer_size
            or not 0 < self.learning_rate < float("inf")
            or not 0 < self.tau <= 1
            or not 0 <= self.gamma <= 1
        ):
            raise ValueError("invalid matched SAC hyperparameters")


def build_matched_sac(
    env: Any,
    streams: SeedStreams,
    *,
    family: AffineFamily,
    arm: str,
    activation: str,
    config: MatchedSACConfig,
    device: str = "cpu",
) -> tuple[Any, dict[str, Any]]:
    """Construct one fresh learner; no learning, reset, step or selection is done.

    ``full`` uses the same cast reference as zero coefficients. ``scratch`` has
    its own seeded native initialization and explicitly different initial behavior.
    Learned/random references must be checked against each other by the caller.
    """
    if arm not in LEARNING_ARMS or activation not in ("relu", "tanh"):
        raise ValueError("declare a supported learning arm and activation")
    validate_metaworld_spaces(env.observation_space, env.action_space)
    _validate_activation(family, activation)
    _validate_arm_family(family, arm)
    architecture = _architecture(family)
    model = SAC(
        MatchedSACPolicy,
        env,
        seed=streams.critic,
        device=device,
        policy_kwargs={
            "family": family,
            "arm": arm,
            "actor_seed": streams.actor,
            "critic_seed": streams.critic,
            "net_arch": list(architecture),
            "activation_fn": nn.ReLU if activation == "relu" else nn.Tanh,
            "features_extractor_class": FlattenExtractor,
            "share_features_extractor": False,
            "n_critics": 2,
        },
        replay_buffer_kwargs={"handle_timeout_termination": True},
        optimize_memory_usage=False,
        use_sde=False,
        ent_coef="auto_1.0",
        target_entropy=-4.0,
        verbose=0,
        **asdict(config),
    )
    replay = model.replay_buffer
    if replay is None:
        raise ValueError("matched construction requires a replay buffer")
    if model.n_envs != 1 or model.num_timesteps != 0 or replay.size() != 0:
        raise ValueError("matched construction requires one environment and fresh empty replay")
    if replay.handle_timeout_termination is not True:
        raise ValueError("time-limit transitions must bootstrap")
    if model.log_ent_coef is None:
        raise ValueError("matched construction requires freshly learned entropy temperature")
    model.set_random_seed(streams.training)
    model.seed = streams.training
    model.action_space.seed(streams.action)
    actor = model.policy.actor
    decoder = actor.coefficient_actor
    actor_metadata = decoder.metadata if decoder is not None else actor.get_extra_state()
    metadata = {
        "arm": arm,
        "rng_streams": asdict(streams),
        "sac_constructor_seed": streams.critic,
        "training_seed_applied_after_initialization": streams.training,
        "config": asdict(config),
        "architecture": {"observation": 39, "action": 4, "hidden": list(architecture)},
        "activation": activation,
        "materialization_dtype": "float32",
        "action_coordinates": "physical_Box_minus_one_to_one; native_rescaling_is_identity",
        "actor": actor_metadata,
        "initial_behavior": actor.get_extra_state()["initial_behavior"],
        "initial_reference_float32_bytes_sha256": actor.get_extra_state()[
            "initial_reference_float32_bytes_sha256"
        ],
        "construction_num_timesteps": model.num_timesteps,
        "construction_replay_size": replay.size(),
        "actor_state_sha256": state_sha256(actor),
        "critic_state_sha256": state_sha256(model.policy.critic),
        "critic_target_state_sha256": state_sha256(model.policy.critic_target),
        "actor_trainable_scalars": sum(p.numel() for p in actor.parameters()),
        "critic_trainable_scalars": sum(p.numel() for p in model.policy.critic.parameters()),
        "target_critic_stored_scalars": sum(
            value.numel()
            for value in model.policy.critic_target.state_dict().values()
            if isinstance(value, Tensor)
        ),
        "actor_frozen_buffer_bytes": sum(
            value.numel() * value.element_size() for value in actor.buffers()
        ),
        "actor_state_tensor_bytes": _state_tensor_bytes(actor),
        "critic_state_tensor_bytes": _state_tensor_bytes(model.policy.critic),
        "target_critic_state_tensor_bytes": _state_tensor_bytes(model.policy.critic_target),
        "actor_optimizer_state_tensor_bytes": _optimizer_tensor_bytes(actor.optimizer),
        "critic_optimizer_state_tensor_bytes": _optimizer_tensor_bytes(
            model.policy.critic.optimizer
        ),
        "entropy_optimizer_state_tensor_bytes": (
            _optimizer_tensor_bytes(model.ent_coef_optimizer)
            if model.ent_coef_optimizer is not None
            else 0
        ),
        "actual_checkpoint_bytes": "caller measures checkpoint including metadata and artifacts",
        "entropy_trainable_scalars": int(model.log_ent_coef.numel()),
        "initial_entropy_coefficient": 1.0,
        "target_entropy": -4.0,
        "replay_buffer_handle_timeout_termination": True,
        "reference_selection": "predeclared_source_only; target selection is external and charged",
        "cost_note": "charge training, warmup, selection, evaluation and failed interactions",
    }
    return model, metadata


def _state_tensor_bytes(module: nn.Module) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in module.state_dict().values()
        if isinstance(value, Tensor)
    )


def _optimizer_tensor_bytes(optimizer: torch.optim.Optimizer) -> int:
    return sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, Tensor)
    )


def validate_matched_preflight(records: Mapping[str, Mapping[str, Any]]) -> None:
    """Reject mismatched construction records before the caller permits learning."""
    if set(records) != set(LEARNING_ARMS):
        raise ValueError("preflight requires all four learning arms")
    common = (
        "config",
        "rng_streams",
        "architecture",
        "activation",
        "materialization_dtype",
        "critic_state_sha256",
        "critic_target_state_sha256",
        "initial_entropy_coefficient",
        "target_entropy",
        "replay_buffer_handle_timeout_termination",
    )
    baseline = records["learned"]
    for arm, record in records.items():
        if (
            record.get("arm") != arm
            or record.get("construction_num_timesteps") != 0
            or record.get("construction_replay_size") != 0
            or any(key not in record or record[key] != baseline.get(key) for key in common)
            or record.get("critic_state_sha256") != record.get("critic_target_state_sha256")
        ):
            raise ValueError("learning arms differ in initialization, seed or training protocol")
        if arm != "scratch" and (
            not record.get("initial_reference_float32_bytes_sha256")
            or record.get("initial_reference_float32_bytes_sha256")
            != baseline.get("initial_reference_float32_bytes_sha256")
            or record.get("initial_behavior") != "affine_reference"
        ):
            raise ValueError("learned, random and full must have the identical float32 reference")
    if records["scratch"].get("initial_behavior") != "independent_native_initialization":
        raise ValueError("scratch must declare its different independent initial behavior")
    learned_actor = baseline.get("actor", {})
    random_actor = records["random"].get("actor", {})
    full_actor = records["full"].get("actor", {})
    parent = random_actor.get("basis_metadata", {})
    if (
        baseline.get("actor_trainable_scalars") != records["random"].get("actor_trainable_scalars")
        or parent.get("parent_artifact_sha256") != learned_actor.get("basis_artifact_sha256")
        or parent.get("parent_family_sha256") != learned_actor.get("family_sha256")
        or full_actor.get("basis_artifact_sha256") != learned_actor.get("basis_artifact_sha256")
        or full_actor.get("family_sha256") != learned_actor.get("family_sha256")
    ):
        raise ValueError("random/full arms are not bound to the same learned parent and dimension")


class FrozenReferencePolicy:
    """Nonlearning native-SB3 prediction interface; no critic, replay or optimizer."""

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        *,
        family: AffineFamily,
        activation: str,
        device: str = "cpu",
    ) -> None:
        if activation not in ("relu", "tanh"):
            raise ValueError("declare relu or tanh")
        _validate_activation(family, activation)
        validate_metaworld_spaces(observation_space, action_space)
        with isolated_seed(0):
            self.actor = MatchedSACActor(
                observation_space=observation_space,
                action_space=action_space,
                net_arch=list(_architecture(family)),
                features_extractor=FlattenExtractor(observation_space),
                features_dim=39,
                activation_fn=nn.ReLU if activation == "relu" else nn.Tanh,
                family=family,
                arm="full",
            ).to(device)
        self.actor.requires_grad_(False)
        self.actor.set_training_mode(False)

    def predict(
        self,
        observation: Any,
        state: tuple[np.ndarray[Any, Any], ...] | None = None,
        episode_start: np.ndarray[Any, Any] | None = None,
        deterministic: bool = True,
    ) -> tuple[np.ndarray[Any, Any], tuple[np.ndarray[Any, Any], ...] | None]:
        return self.actor.predict(observation, state, episode_start, deterministic)


class FrozenSourcePolicy(FrozenReferencePolicy):
    """Source reuse is allowed only for a verified actual source-policy anchor."""

    def __init__(self, *args: Any, family: AffineFamily, **kwargs: Any) -> None:
        spec = family.metadata.get("spec", {})
        if (
            spec.get("reference_rule") != "source_policy"
            or not spec.get("reference_policy_id")
            or spec["reference_policy_id"] not in family.metadata.get("source_policy_ids", [])
        ):
            raise ValueError("frozen source reuse requires an explicit source-policy reference")
        super().__init__(*args, family=family, **kwargs)


def verified_policy_kwargs(saved: Mapping[str, Any], *, family: AffineFamily) -> dict[str, Any]:
    """Supply to SAC.load(custom_objects={'policy_kwargs': ...}) before reconstruction.

    The external, verified artifact replaces the saved pickle object. The actor's
    state loader then checks its family/map metadata against that artifact.
    """
    result = copy.copy(dict(saved))
    previous = result.get("family")
    if previous is None or (
        previous.sha256 != family.sha256 or previous.family_sha256 != family.family_sha256
    ):
        raise ValueError("checkpoint does not bind the supplied verified family artifact")
    result["family"] = family
    return result
