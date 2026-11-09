"""Synthetic arrays/controllers only; no native learner, environment or GPU."""

from __future__ import annotations

import base64
import copy
import json
import random
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from scripts.metaworld_protocol import Goal, GoalBank, SeedStreams, TaskController
from scripts.metaworld_source_recovery import (
    inspect_checkpoint_counters,
    recover_committed_environment,
)


class Space:
    def __init__(self) -> None:
        self._np_random = np.random.default_rng(0)

    def sample(self) -> Any:
        return self._np_random.uniform(-1, 1, 4).astype(np.float32)


class Raw:
    def __init__(self) -> None:
        self.unwrapped = self
        self.action_space = Space()
        self._partially_observable = False
        self.length = 0
        self.value = 0.0

    def seed(self, seed: int) -> None:
        self.np_random = np.random.default_rng(seed)
        self.action_space._np_random = np.random.default_rng(seed)

    def set_task(self, task: Any) -> None:
        self._last_rand_vec = np.asarray(json.loads(task.data))

    def observation(self) -> Any:
        observation = np.zeros(39, dtype=np.float64)
        observation[:5] = (
            self.value,
            self.length,
            self.np_random.random(),
            np.random.random(),
            random.random(),
        )
        return observation

    def reset(self, options: Any = None) -> tuple[Any, Any]:
        self.length, self.value = 0, 0.0
        return self.observation(), {}

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Any]:
        self.length += 1
        self.value += float(np.sum(action))
        return self.observation(), self.value, False, self.length == 500, {"success": False}


class Monitor:
    def __init__(self, controller: TaskController) -> None:
        self.controller = controller
        self.needs_reset = True
        self.rewards: list[float] = []
        self.total_steps = 0

    def reset(self, *, seed: int | None = None) -> tuple[Any, Any]:
        observation, info = self.controller.reset(seed=seed)
        self.needs_reset = False
        self.rewards = []
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, Any]:
        assert not self.needs_reset
        result = self.controller.step(action)
        self.rewards.append(result[1])
        self.total_steps += 1
        self.needs_reset = result[2] or result[3]
        return result


class Policy:
    def __init__(self) -> None:
        self.unscaled = 0

    def unscale_action(self, action: Any) -> Any:
        self.unscaled += 1
        return action * np.float32(0.5) + np.float32(0.25)


def controller() -> TaskController:
    goals = []
    for index in range(3):
        vector = [float(index), float(index + 1)]
        payload = base64.b64encode(json.dumps(vector).encode()).decode()
        goals.append(Goal("push-v3", payload, tuple(vector)))
    return TaskController(
        Raw(), GoalBank("source", 4, tuple(goals)), SeedStreams.from_seed(44), SimpleNamespace
    )


def fixture(
    physical: int = 1002,
    *,
    lost_callback: bool = False,
    blocked_terminal_reset: bool = False,
    pending_update: bool = False,
) -> tuple[Any, Any, Any, Any]:
    old = controller()
    monitor = Monitor(old)
    policy = Policy()
    size = 500
    replay = SimpleNamespace(
        observations=np.zeros((size, 1, 39)),
        next_observations=np.zeros((size, 1, 39)),
        actions=np.zeros((size, 1, 4), dtype=np.float32),
        rewards=np.zeros((size, 1), dtype=np.float32),
        dones=np.zeros((size, 1), dtype=np.float32),
        timeouts=np.zeros((size, 1), dtype=np.float32),
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
        buffer_size=size,
    )
    committed = physical - int(lost_callback or blocked_terminal_reset)
    observed = physical - int(blocked_terminal_reset)
    observation = last_obs = None
    if physical:
        observation, _ = monitor.reset(seed=123)
    for step in range(physical):
        buffer_action = (
            old.env.action_space.sample()
            if step < 1000
            else np.full(4, (step % 13) / 20, dtype=np.float32)
        )
        new_observation, reward, terminated, truncated, _ = monitor.step(
            policy.unscale_action(buffer_action[None])[0]
        )
        terminal_observation = new_observation
        if truncated and not (blocked_terminal_reset and step == physical - 1):
            new_observation, _ = monitor.reset()
        if step < committed:
            ring = step % size
            replay.observations[ring, 0] = observation
            replay.next_observations[ring, 0] = terminal_observation
            replay.actions[ring, 0] = buffer_action
            replay.rewards[ring, 0] = reward
            replay.dones[ring, 0] = terminated or truncated
            replay.timeouts[ring, 0] = truncated
            last_obs = new_observation.copy()[None]
        observation = new_observation
    replay.pos, replay.full = committed % size, committed >= size
    fresh = controller()
    fresh_monitor = Monitor(fresh)
    vec = SimpleNamespace(
        num_envs=1,
        envs=[fresh_monitor],
        buf_obs={None: np.zeros((1, 39))},
        buf_rews=np.zeros(1, dtype=np.float32),
        buf_dones=np.zeros(1, dtype=bool),
        buf_infos=[{}],
        reset_infos=[{}],
        _seeds=[999],
        _options=[{"unused": True}],
    )
    model = SimpleNamespace(
        num_timesteps=observed,
        _n_updates=max(committed - 1000, 0) - int(pending_update),
        learning_starts=1000,
        replay_buffer=replay,
        env=vec,
        policy=Policy(),
        action_space=Space(),
        _last_obs=last_obs,
        ep_info_buffer=[{"r": 3.0}],
        expected_controller=old,
    )
    history = copy.deepcopy(old.episode_history)
    plan = inspect_checkpoint_counters(
        num_timesteps=observed,
        replay_position=replay.pos,
        replay_full=replay.full,
        replay_buffer_size=size,
        n_updates=model._n_updates,
        episode_history=history,
        preflight_prefix_length=0,
    )
    return model, fresh, history, plan


@pytest.mark.parametrize(
    "kwargs,counts",
    [
        ({}, (1002, 1002, 1002, 0, 0, 2)),
        ({"lost_callback": True}, (1002, 1002, 1001, 1, 0, 1)),
        ({"pending_update": True}, (1002, 1002, 1002, 0, 1, 2)),
        (
            {"physical": 1500, "blocked_terminal_reset": True},
            (1499, 1500, 1499, 1, 0, 499),
        ),
        ({"physical": 1000}, (1000, 1000, 1000, 0, 0, 0)),
    ],
)
def test_reconstruction_matches_every_transition_without_mutating_learner(
    kwargs: Any, counts: Any
) -> None:
    model, fresh, history, plan = fixture(**kwargs)
    assert (
        plan.observed_steps,
        plan.physical_steps,
        plan.committed_steps,
        plan.lost_observed_steps,
        plan.pending_optimizer_updates,
        plan.active_episode_length,
    ) == counts
    saved_last = model._last_obs.copy()
    replay_before = model.replay_buffer.actions.copy()
    ep_info = model.ep_info_buffer
    charged = []
    report = recover_committed_environment(
        model=model,
        controller=fresh,
        history=history,
        plan=plan,
        check=lambda: None,
        charge_preflight=lambda: charged.append(1),
    )
    assert report["verified"]
    assert len(charged) == plan.active_episode_length
    assert model.policy.unscaled == len(charged)
    assert model.num_timesteps == plan.observed_steps
    assert np.array_equal(model._last_obs, saved_last)
    assert np.array_equal(model.replay_buffer.actions, replay_before)
    assert model.ep_info_buffer is ep_info
    assert model.env.envs[0].total_steps == plan.physical_steps
    assert len(model.env.envs[0].rewards) == plan.active_episode_length
    assert fresh.length == plan.active_episode_length
    assert fresh.action_rng.bit_generator.state == _warmup_rng(fresh)


def _warmup_rng(fresh: Any) -> Any:
    rng = np.random.default_rng(fresh.streams.action)
    for _ in range(1000):
        rng.uniform(-1, 1, 4)
    return rng.bit_generator.state


def test_zero_step_checkpoint_stays_unreset() -> None:
    model, fresh, history, plan = fixture(physical=0)
    report = recover_committed_environment(
        model=model,
        controller=fresh,
        history=history,
        plan=plan,
        check=lambda: None,
        charge_preflight=lambda: pytest.fail("zero-step recovery stepped"),
    )
    assert report["unreset_zero_step_learner"]
    assert model._last_obs is None
    assert fresh.episode_history == []
    assert model.env.envs[0].needs_reset
    assert model.action_space._np_random is fresh.action_rng
    expected = np.random.default_rng(fresh.streams.action).uniform(-1, 1, 4).astype(np.float32)
    assert np.array_equal(model.action_space.sample(), expected)


def test_next_transition_retains_private_episode_rng_after_reconstruction() -> None:
    model, fresh, history, plan = fixture()
    recover_committed_environment(
        model=model,
        controller=fresh,
        history=history,
        plan=plan,
        check=lambda: None,
        charge_preflight=lambda: None,
    )
    action = np.full(4, 0.375, dtype=np.float32)
    actual = fresh.step(action)
    expected = model.expected_controller.step(action)
    assert np.array_equal(actual[0], expected[0])
    assert actual[1:4] == expected[1:4]
    assert fresh.legacy_rng.python_state == model.expected_controller.legacy_rng.python_state
    assert np.array_equal(
        fresh.legacy_rng.numpy_state[1], model.expected_controller.legacy_rng.numpy_state[1]
    )


@pytest.mark.parametrize("damage", ["observation", "next", "reward", "terminal", "last"])
def test_mismatch_stops_before_any_learning(damage: str) -> None:
    model, fresh, history, plan = fixture()
    index = plan.replay_tail_indices[0]
    if damage == "observation":
        model.replay_buffer.observations[index, 0, 0] += 1e-12
    elif damage == "next":
        model.replay_buffer.next_observations[index, 0, 0] += 1e-12
    elif damage == "reward":
        model.replay_buffer.rewards[index, 0] += 1
    elif damage == "terminal":
        model.replay_buffer.dones[index, 0] = 1
    else:
        model._last_obs[0, 0] += 1e-12
    with pytest.raises(ValueError, match=r"mismatch|terminal"):
        recover_committed_environment(
            model=model,
            controller=fresh,
            history=history,
            plan=plan,
            check=lambda: None,
            charge_preflight=lambda: None,
        )
    assert model._n_updates == 2


def test_all_history_seed_draws_are_verified_before_reset_or_step() -> None:
    model, fresh, history, plan = fixture()
    history[0]["episode_seed"] += 1
    with pytest.raises(ValueError, match="prefix disagrees"):
        recover_committed_environment(
            model=model,
            controller=fresh,
            history=history,
            plan=plan,
            check=lambda: None,
            charge_preflight=lambda: pytest.fail("invalid prefix stepped"),
        )
    assert fresh.episode_history == []


def test_counter_tampering_and_multiple_pending_updates_fail_closed() -> None:
    model, _, history, plan = fixture()
    arguments = {
        "num_timesteps": model.num_timesteps,
        "replay_position": model.replay_buffer.pos,
        "replay_full": model.replay_buffer.full,
        "replay_buffer_size": 500,
        "n_updates": model._n_updates,
        "episode_history": history,
        "preflight_prefix_length": 0,
    }
    for changes in (
        {"replay_position": 0},
        {"replay_full": False},
        {"n_updates": 0},
        {"n_updates": 3},
        {"num_timesteps": True},
    ):
        with pytest.raises(ValueError):
            inspect_checkpoint_counters(**{**arguments, **changes})
    assert plan.pending_optimizer_updates == 0


def test_ambiguous_warmup_requires_saved_box_rng_evidence() -> None:
    model, fresh, history, plan = fixture(physical=10)
    with pytest.raises(ValueError, match="warmup"):
        recover_committed_environment(
            model=model,
            controller=fresh,
            history=history,
            plan=plan,
            check=lambda: None,
            charge_preflight=lambda: pytest.fail("ambiguous warmup stepped"),
        )
    # The external checkpoint verifier may supply the Box state extracted BEFORE
    # SAC.load reseeds that saved space; the helper never reads a pickle itself.
    saved_rng = np.random.default_rng(fresh.streams.action)
    saved_rng.uniform(-1, 1, (11, 4))
    state = saved_rng.bit_generator.state
    report = recover_committed_environment(
        model=model,
        controller=fresh,
        history=history,
        plan=plan,
        check=lambda: None,
        charge_preflight=lambda: None,
        action_rng_state=state,
    )
    assert report["action_rng_from_saved_box"]
    assert fresh.action_rng.bit_generator.state == state


def test_guard_blocks_reset_and_each_later_step() -> None:
    model, fresh, history, plan = fixture()

    def stopped() -> None:
        raise RuntimeError("absolute UTC cutoff")

    with pytest.raises(RuntimeError, match="cutoff"):
        recover_committed_environment(
            model=model,
            controller=fresh,
            history=history,
            plan=plan,
            check=stopped,
            charge_preflight=lambda: pytest.fail("expired recovery stepped"),
        )
    assert fresh.episode_history == []
    charges: list[int] = []

    def after_first() -> None:
        if charges:
            raise RuntimeError("reserve")

    with pytest.raises(RuntimeError, match="reserve"):
        recover_committed_environment(
            model=model,
            controller=fresh,
            history=history,
            plan=plan,
            check=after_first,
            charge_preflight=lambda: charges.append(1),
        )
    assert charges == [1]
    assert fresh.length == 0


def test_changed_plan_is_rejected_before_reset() -> None:
    model, fresh, history, plan = fixture()
    with pytest.raises(ValueError, match="plan changed"):
        recover_committed_environment(
            model=model,
            controller=fresh,
            history=history,
            plan=replace(plan, active_episode_length=4),
            check=lambda: None,
            charge_preflight=lambda: pytest.fail("changed plan stepped"),
        )
