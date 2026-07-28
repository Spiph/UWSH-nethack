# mypy: ignore-errors
"""SOL/Sample Factory integration for the Phase Zero MiniHack population.

This module is deliberately imported only inside the ``sol_patched`` container.
The development dependency set does not include SOL's patched NLE or its Sample
Factory fork, so importing this module on a normal workstation remains optional.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ups.model import NLE_ACTIONS

PHASE0_ENVIRONMENTS = (
    "MiniHack-Room-Random-5x5-v0",
    "MiniHack-Room-Dark-5x5-v0",
    "MiniHack-Room-Monster-5x5-v0",
    "MiniHack-Room-Trap-5x5-v0",
)


def _runtime_imports() -> tuple[Any, Any, Any, Any, Any]:
    """Import SOL dependencies lazily with an actionable error outside Docker."""
    try:
        import gymnasium as gym
        import torch
        from sample_factory.algo.utils.context import global_model_factory
        from sample_factory.envs.env_utils import register_env
        from sample_factory.model.encoder import Encoder
    except ImportError as error:  # pragma: no cover - exercised in the SOL image
        raise RuntimeError(
            "SOL integration requires the pinned research-cpu or research-gpu image; "
            "run this command through docker compose."
        ) from error
    return gym, torch, global_model_factory, register_env, Encoder


def compass_action_values() -> tuple[int, ...]:
    """Return NLE's canonical eight compass actions in study action order."""
    try:
        from nle.nethack import CompassDirection
    except ImportError as error:  # pragma: no cover - exercised in the SOL image
        raise RuntimeError("the patched NLE runtime is required") from error
    values = (
        CompassDirection.N,
        CompassDirection.E,
        CompassDirection.S,
        CompassDirection.W,
        CompassDirection.NE,
        CompassDirection.SE,
        CompassDirection.SW,
        CompassDirection.NW,
    )
    return tuple(int(value) for value in values)


def crop_glyphs(glyphs: np.ndarray, blstats: np.ndarray, crop_size: int) -> np.ndarray:
    """Extract a zero-padded square glyph crop around NLE's x/y BLStats fields."""
    if glyphs.ndim != 2:
        raise ValueError(f"glyphs must have shape [height, width], got {glyphs.shape}")
    if blstats.shape[0] < 2:
        raise ValueError("blstats must contain x and y coordinates")
    radius = crop_size // 2
    padded = np.pad(glyphs, radius, mode="constant")
    x = int(blstats[0]) + radius
    y = int(blstats[1]) + radius
    return padded[y - radius : y + radius + 1, x - radius : x + radius + 1].copy()


def make_phase0_env(
    full_env_name: str, cfg: Any, env_config: Any, render_mode: str | None = None
) -> Any:
    """Make a MiniHack room task with exactly eight compass actions and a glyph crop."""
    gym, _, _, _, _ = _runtime_imports()
    # MiniHack registers its Gymnasium task IDs as an import side effect.
    import minihack  # noqa: F401

    class Phase0Observation(gym.ObservationWrapper):
        def __init__(self, env: Any) -> None:
            super().__init__(env)
            spaces = env.observation_space.spaces.copy()
            crop_size = int(getattr(cfg, "phase0_crop_size", 9))
            spaces["glyphs_crop"] = gym.spaces.Box(
                low=0, high=5999, shape=(crop_size, crop_size), dtype=np.int16
            )
            self.crop_size = crop_size
            self.observation_space = gym.spaces.Dict(spaces)

        def observation(self, observation: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            result = dict(observation)
            result["glyphs_crop"] = crop_glyphs(
                np.asarray(observation["glyphs"]),
                np.asarray(observation["blstats"]),
                self.crop_size,
            )
            return result

    class EightCompassActions(gym.ActionWrapper):
        def __init__(self, env: Any) -> None:
            super().__init__(env)
            available_actions = tuple(env.unwrapped.actions)
            compass_actions = compass_action_values()
            try:
                self._actions = tuple(available_actions.index(action) for action in compass_actions)
            except ValueError as error:
                raise RuntimeError(
                    "MiniHack task does not expose every required compass direction"
                ) from error
            self.action_space = gym.spaces.Discrete(len(NLE_ACTIONS))

        def action(self, action: int) -> int:
            if not self.action_space.contains(action):
                raise ValueError(f"invalid Phase Zero action {action}")
            return self._actions[int(action)]

    environment = gym.make(full_env_name, render_mode=render_mode)
    return EightCompassActions(Phase0Observation(environment))


def make_phase0_encoder(cfg: Any, obs_space: Any) -> Any:
    """Build the preregistered glyph embedding + three-CNN-block encoder for SOL."""
    _, torch, _, _, Encoder = _runtime_imports()
    from torch import nn

    class Phase0Encoder(Encoder):
        def __init__(self, cfg: Any, obs_space: Any) -> None:
            super().__init__(cfg)
            crop_shape = obs_space["glyphs_crop"].shape
            if len(crop_shape) != 2 or crop_shape[0] != crop_shape[1]:
                raise ValueError("Phase Zero requires a square glyphs_crop observation")
            self.crop_size = int(crop_shape[0])
            self.glyph_embedding = nn.Embedding(6000, 32)
            self.cnn = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
            )
            blstats_size = int(obs_space["blstats"].shape[0])
            self.projection = nn.Linear(64 * self.crop_size * self.crop_size + blstats_size, 256)
            self.encoder_out_size = 256

        def forward(self, obs: dict[str, Any]) -> Any:
            glyphs = obs["glyphs_crop"].long().clamp_(0, 5999)
            if glyphs.ndim != 3:
                raise ValueError("SOL policy expects glyphs_crop with shape [batch, height, width]")
            embedded = self.glyph_embedding(glyphs).permute(0, 3, 1, 2)
            visual = self.cnn(embedded).flatten(start_dim=1)
            return torch.relu(self.projection(torch.cat((visual, obs["blstats"].float()), dim=1)))

        def get_out_size(self) -> int:
            return self.encoder_out_size

        def type_for_input_tensor(self, input_tensor_name: str) -> Any:
            if input_tensor_name in {"glyphs", "glyphs_crop"}:
                return torch.int64
            return torch.float32

    return Phase0Encoder(cfg, obs_space)


def register_phase0_components() -> None:
    """Register the fixed MiniHack tasks and the Phase Zero encoder with SOL."""
    _, _, global_model_factory, register_env, _ = _runtime_imports()
    for environment in PHASE0_ENVIRONMENTS:
        register_env(environment, make_phase0_env)
    global_model_factory().register_encoder_factory(make_phase0_encoder)
