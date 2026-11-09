"""Bounded, fail-closed reconstruction of the frozen source pilot's environment.

This is not general checkpoint resume. The caller supplies already verified
checkpoints, reloads the native learner without resetting it, and permits no
learning until *all* recovery reports pass. No optional RL runtime is imported.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from scripts.metaworld_protocol import TaskController, stable_hash


@dataclass(frozen=True)
class RecoveryPlan:
    observed_steps: int
    physical_steps: int
    committed_steps: int
    lost_observed_steps: int
    model_uncounted_physical_steps: int
    pending_optimizer_updates: int
    source_history_offset: int
    episode_index: int
    active_episode_length: int
    replay_tail_indices: tuple[int, ...]
    buffer_size: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _physical_steps(history: Sequence[Mapping[str, Any]], observed: int, offset: int) -> int | None:
    if offset and (
        not history
        or history[0].get("length") != 1
        or history[0].get("complete_episode") is not False
    ):
        return None
    source = history[offset:]
    for index, row in enumerate(history):
        if row.get("episode") != index:
            return None
        length = row.get("length")
        if type(length) is not int or not 0 <= length <= 500:
            return None
    if not source:
        return 0 if observed == 0 else None
    for row in source[:-1]:
        if row["length"] != 500 or row.get("complete_episode") is not True:
            return None
    last = source[-1]
    physical = sum(int(row["length"]) for row in source)
    if physical not in (observed, observed + 1):
        return None
    if last["length"] == 500:
        # DummyVecEnv's automatic reset may be stopped after the physical step,
        # before collect_rollouts increments num_timesteps.
        if physical != observed + 1 or last.get("complete_episode") is not True:
            return None
    elif last.get("complete_episode") is not False:
        return None
    return physical


def inspect_checkpoint_counters(
    *,
    num_timesteps: int,
    replay_position: int,
    replay_full: bool,
    replay_buffer_size: int,
    n_updates: int,
    episode_history: Sequence[Mapping[str, Any]],
    learning_starts: int = 1000,
    preflight_prefix_length: int | None = None,
) -> RecoveryPlan:
    """Infer at most one lost transition or pending update, never invent either.

    The 500-step, single-env, train_freq=gradient_steps=1, no-retry pilot is
    required. The optional one-step preflight history prefix is validated and
    reported separately from source transitions. The runner must additionally
    reconcile these per-policy physical totals with the cumulative cost ledger.
    """
    observed = _integer(num_timesteps, "num_timesteps")
    position = _integer(replay_position, "replay_position")
    size = _integer(replay_buffer_size, "replay_buffer_size")
    updates = _integer(n_updates, "n_updates")
    warmup = _integer(learning_starts, "learning_starts")
    if size < 500 or position >= size or type(replay_full) is not bool:
        raise ValueError("require a valid replay ring with at least one complete episode")
    if preflight_prefix_length not in (None, 0, 1):
        raise ValueError("only zero or one explicit preflight history row is supported")
    candidates = [
        count
        for count in (observed, observed - 1)
        if count >= 0 and count % size == position and (count >= size) == replay_full
    ]
    if len(candidates) != 1:
        raise ValueError("replay counters do not identify N or N-1 committed transitions")
    committed = candidates[0]
    prefixes = (0, 1) if preflight_prefix_length is None else (preflight_prefix_length,)
    histories = [
        (offset, physical)
        for offset in prefixes
        if (physical := _physical_steps(episode_history, observed, offset)) is not None
        and 0 <= physical - committed <= 1
    ]
    if len(histories) != 1:
        raise ValueError("history does not identify one unambiguous physical-step prefix")
    offset, physical = histories[0]
    expected_updates = max(committed - warmup, 0)
    missing = expected_updates - updates
    if missing not in (0, 1) or (
        missing == 1 and (committed != observed or physical != committed or committed <= warmup)
    ):
        raise ValueError("optimizer counters do not identify zero or one pending update")
    length = committed % 500
    episode = committed // 500 + offset
    if observed > 0 and episode >= len(episode_history):
        raise ValueError("committed current episode is missing from saved history")
    return RecoveryPlan(
        observed,
        physical,
        committed,
        physical - committed,
        physical - observed,
        missing,
        offset,
        episode,
        length,
        tuple((committed - length + index) % size for index in range(length)),
        size,
    )


def _equal(actual: Any, expected: Any, name: str) -> None:
    left, right = np.asarray(actual), np.asarray(expected)
    if left.shape != right.shape or not np.isfinite(left).all() or not np.array_equal(left, right):
        raise ValueError(f"exact reconstruction mismatch: {name}")


def _validated_prefix(
    controller: TaskController, history: Sequence[Mapping[str, Any]], count: int
) -> tuple[np.random.Generator, np.random.Generator]:
    selection = np.random.default_rng(controller.streams.selection)
    episodes = np.random.default_rng(controller.streams.episode)
    for index, row in enumerate(history):
        goal_index = int(selection.integers(len(controller.bank.goals)))
        seed = int(episodes.integers(0, 2**32, dtype=np.uint64))
        goal = controller.bank.goals[goal_index]
        expected = {
            "episode": index,
            "goal_index": goal_index,
            "goal_id": goal.goal_id,
            "goal_bank_sha256": controller.bank.sha256,
            "episode_seed": seed,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("history goal/seed prefix disagrees with the frozen source stream")
    # Validation covers even a lost step's auto-reset. Restore the prefix before
    # the committed episode, allowing its reset to consume exactly its own draw.
    selection = np.random.default_rng(controller.streams.selection)
    episodes = np.random.default_rng(controller.streams.episode)
    for _ in range(count):
        selection.integers(len(controller.bank.goals))
        episodes.integers(0, 2**32, dtype=np.uint64)
    return selection, episodes


def recover_committed_environment(
    *,
    model: Any,
    controller: TaskController,
    history: Sequence[Mapping[str, Any]],
    plan: RecoveryPlan,
    check: Callable[[], None],
    charge_preflight: Callable[[], None],
    action_rng_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recover only the committed episode, charging attempts before physical steps.

    ``model`` is a loaded native SAC with one Monitor/DummyVecEnv and its saved
    replay attached. The callbacks must check the new allocation and reserve an
    actual preflight transition *before* each attempt. A failure may mutate the
    new disposable environment, but never the model/replay/old artifacts. The
    caller retains the failure report and forbids every learner from training.
    """
    check()
    if controller.goal_order is not None or controller.episode_history:
        raise ValueError("recovery requires a fresh source controller with random source order")
    replay, vec = model.replay_buffer, model.env
    if (
        vec.num_envs != 1
        or len(vec.envs) != 1
        or replay.n_envs != 1
        or replay.optimize_memory_usage
        or not replay.handle_timeout_termination
        or int(model.num_timesteps) != plan.observed_steps
    ):
        raise ValueError("recovery requires the verified one-env native replay configuration")
    fresh_plan = inspect_checkpoint_counters(
        num_timesteps=int(model.num_timesteps),
        replay_position=int(replay.pos),
        replay_full=bool(replay.full),
        replay_buffer_size=int(replay.buffer_size),
        n_updates=int(model._n_updates),
        episode_history=history,
        learning_starts=int(model.learning_starts),
        preflight_prefix_length=plan.source_history_offset,
    )
    if fresh_plan != plan:
        raise ValueError("recovery plan changed after counter inspection")
    if 0 < plan.observed_steps < model.learning_starts and action_rng_state is None:
        raise ValueError("interrupted warmup requires saved pre-load action-space RNG evidence")
    prefix = plan.episode_index if plan.observed_steps else len(history)
    selection, episodes = _validated_prefix(controller, history, prefix)
    controller.selection_rng, controller.episode_rng = selection, episodes
    controller.episode_history = [copy.deepcopy(dict(row)) for row in history[:prefix]]
    controller.active = False
    controller.action_rng = np.random.default_rng(controller.streams.action)
    controller.env.action_space._np_random = controller.action_rng
    # BaseAlgorithm.load restores a separately pickled action_space before its
    # setup seeds it. Reattach that sampling space to the controller's stream;
    # this is essential for an untouched zero-step learner's future warmup.
    model.action_space._np_random = controller.action_rng
    if action_rng_state is None:
        samples = min(plan.observed_steps, int(model.learning_starts))
        for _ in range(samples):
            check()
            controller.env.action_space.sample()
    else:
        controller.action_rng.bit_generator.state = copy.deepcopy(dict(action_rng_state))
        samples = None
    report: dict[str, Any] = {
        "method": "frozen_source_episode_replay_v1",
        "comparison": "exact_shape_and_values; reward_cast_to_replay_dtype",
        "plan": plan.to_dict(),
        "saved_history_sha256": stable_hash(list(history)),
        "action_samples_advanced": samples,
        "action_rng_from_saved_box": action_rng_state is not None,
        "reconstruction_steps": 0,
        "operational_monitor_restored": False,
        "historical_monitor_diagnostics": "segmented; old JSON and saved ep_info_buffer retained",
        "verified": False,
    }
    if plan.observed_steps == 0:
        if model._last_obs is not None or plan.committed_steps != 0:
            raise ValueError("zero-step checkpoint must retain an unreset learner")
        report["verified"] = True
        report["unreset_zero_step_learner"] = True
        return report
    if model._last_obs is None:
        raise ValueError("trained checkpoint is missing its committed last observation")
    monitor = vec.envs[0]
    if not hasattr(monitor, "needs_reset") or not hasattr(monitor, "rewards"):
        raise ValueError("expected the pinned native operational Monitor wrapper")
    check()
    saved_row = history[plan.episode_index]
    observation, reset_info = monitor.reset(seed=saved_row.get("forwarded_reset_seed"))
    last_reward: Any = np.float32(0)
    last_info: dict[str, Any] = {}
    for ring_index in plan.replay_tail_indices:
        check()
        _equal(observation, replay.observations[ring_index, 0], "replay observation")
        if replay.dones[ring_index, 0] or replay.timeouts[ring_index, 0]:
            raise ValueError("committed partial-episode replay unexpectedly contains a terminal")
        buffer_action = replay.actions[ring_index].copy()
        action = model.policy.unscale_action(buffer_action)[0]
        charge_preflight()
        check()
        observation, reward, terminated, truncated, last_info = monitor.step(action)
        report["reconstruction_steps"] += 1
        if terminated or truncated:
            raise ValueError("partial-episode reconstruction terminated unexpectedly")
        _equal(observation, replay.next_observations[ring_index, 0], "replay next observation")
        last_reward = np.asarray(reward, dtype=replay.rewards.dtype)
        _equal(last_reward, replay.rewards[ring_index, 0], "replay reward")
    _equal(np.asarray(observation)[None], model._last_obs, "saved model._last_obs")
    if controller.length != plan.active_episode_length or not controller.active:
        raise ValueError("reconstructed controller length/activity disagrees")
    if plan.physical_steps == plan.committed_steps:
        actual_row = controller.episode_history[-1]
        for field in ("length", "success", "complete_episode"):
            if actual_row.get(field) != saved_row.get(field):
                raise ValueError("committed protocol outcome disagrees with reconstructed episode")
    check()
    # Rebuild only operational vector/Monitor state. Saved native _last_obs and
    # ep_info_buffer remain untouched; no claim of historical Monitor wall times.
    if set(vec.buf_obs) != {None}:
        raise ValueError("recovery only supports the declared flat observation coordinates")
    vec.buf_obs[None][0] = observation
    vec.buf_rews[0] = last_reward
    vec.buf_dones[0] = plan.active_episode_length == 0 and plan.committed_steps > 0
    vec.buf_infos[0] = copy.deepcopy(last_info)
    vec.reset_infos[0] = copy.deepcopy(reset_info)
    vec._seeds = [None]
    vec._options = [{}]
    monitor.total_steps = plan.physical_steps
    monitor.needs_reset = False
    report["operational_monitor_restored"] = True
    report["recovered_history_sha256"] = stable_hash(controller.episode_history)
    report["goal_id"] = controller.episode_history[-1]["goal_id"]
    report["episode_seed"] = controller.episode_history[-1]["episode_seed"]
    report["verified"] = True
    return report
