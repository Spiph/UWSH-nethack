"""Synthetic arrays/plain pickle fixtures only; no native learners or environments."""

from __future__ import annotations

import io
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from scripts import metaworld_source_diagnosis as diagnosis


def replay() -> Any:
    size, position = 6, 2
    observations = np.zeros((size, 1, 39))
    next_observations = np.zeros_like(observations)
    for index in range(size):
        observations[index, 0, :18] = index
        observations[index, 0, 18:36] = index if index % 3 == 0 else index - 1
        next_observations[index, 0, :18] = index + 1
        next_observations[index, 0, 18:36] = index
        observations[index, 0, 36:39] = next_observations[index, 0, 36:39] = [1, 2, 3]
    values = dict(
        observations=observations,
        next_observations=next_observations,
        actions=np.arange(24, dtype=np.float32).reshape(size, 1, 4) / 24,
        rewards=np.array([0, 1, 5, 10, 0.5, 9.999], dtype=np.float32)[:, None],
        dones=np.array([0, 0, 1, 0, 0, 1], dtype=np.float32)[:, None],
        timeouts=np.array([0, 0, 1, 0, 0, 1], dtype=np.float32)[:, None],
    )
    indices = np.concatenate((np.arange(position, size), np.arange(position)))
    physical = {}
    for key, array in values.items():
        result = np.empty_like(array)
        result[indices] = array
        physical[key] = result
    return SimpleNamespace(
        **physical,
        buffer_size=size,
        pos=position,
        full=True,
        n_envs=1,
        optimize_memory_usage=False,
        handle_timeout_termination=True,
    )


def native_named_pickle(value: Any) -> bytes:
    # Only the fixture's class name resembles SB3. No SB3 import or native
    # constructor occurs; the restricted reader maps this name to plain Record.
    modules = {
        name: ModuleType(name)
        for name in (
            "stable_baselines3",
            "stable_baselines3.common",
            "stable_baselines3.common.buffers",
        )
    }
    cls = type("ReplayBuffer", (), {})
    cls.__module__ = "stable_baselines3.common.buffers"
    modules[cls.__module__].__dict__["ReplayBuffer"] = cls
    record = cls()
    record.__dict__.update(vars(value))
    with patch.dict(sys.modules, modules):
        return pickle.dumps(record, protocol=5)


def history(length: int = 0) -> list[dict[str, Any]]:
    return [
        {
            "episode": 0,
            "goal_id": "g0",
            "length": 500,
            "success": True,
            "complete_episode": True,
            "terminated": False,
            "truncated": True,
        },
        {
            "episode": 1,
            "goal_id": "g1",
            "length": length,
            "success": True,
            "complete_episode": False,
            "terminated": False,
            "truncated": False,
        },
    ]


def manifest_fixture(tmp_path: Path) -> Path:
    parent, current, failed = [tmp_path / name for name in ("parent", "extension", "failed")]
    for root in (parent, current, failed):
        root.mkdir()
    checkpoints = []
    for root, steps, partial in ((parent, 500, False), (current, 715, True)):
        replay_path, actor = root / "replay.pkl", root / "actor.npz"
        replay_path.write_bytes(native_named_pickle(replay()))
        actor.write_bytes(b"synthetic actor bytes: verified, never loaded")
        artifacts = {
            ".replay.pkl": {"path": replay_path.name, "sha256": diagnosis.digest(replay_path)},
            "actor": {"path": actor.name, "sha256": diagnosis.digest(actor)},
            "environment": {"episode_history": history(215 if partial else 0)},
        }
        row: dict[str, Any] = dict(
            actual_steps=steps,
            target_steps=1000 if partial else 500,
            partial=partial,
            artifacts=artifacts,
            evaluation_complete=not partial,
            evaluation=[
                dict(
                    length=500,
                    complete_episode=True,
                    terminated=False,
                    truncated=True,
                    success=index < 3,
                    return_=float(index),
                )
                for index in range(20)
            ]
            if not partial
            else [],
        )
        for episode in row["evaluation"]:
            episode["return"] = episode.pop("return_")
        if root == current:
            row["artifact_root"] = str(root)
        checkpoints.append(row)
    data = {
        "phase": "FINISHED",
        "status": "COMPLETE_FAIL",
        "ledger": {"source_training_steps": 715},
        "plan": {"extension": {"parent_root": str(parent)}},
        "parent_binding": {"inventories": [{"root": str(failed)}, {"root": str(parent)}]},
        "goal_manifests": {
            "push-v3": {
                "banks": [
                    {"split": "source", "goals": [{"goal_id": "g0"}, {"goal_id": "g1"}]},
                    {"split": "development", "goals": []},
                ]
            }
        },
        "tasks": {"push-v3": {"checkpoints": checkpoints}},
    }
    path = current / "development_manifest.json"
    path.write_text(json.dumps(data))
    return path


def test_restricted_unpickler_maps_native_names_to_plain_records() -> None:
    loaded = diagnosis.load_replay(io.BytesIO(native_named_pickle(replay())))
    assert isinstance(loaded, diagnosis.Record)
    assert np.array_equal(vars(loaded)["observations"], replay().observations)


def test_unknown_pickle_globals_and_persistent_native_references_are_rejected() -> None:
    class Unsafe:
        def __reduce__(self) -> Any:
            return os.system, ("this must never run",)

    with pytest.raises(ValueError, match="unsupported pickle global"):
        diagnosis.load_replay(io.BytesIO(pickle.dumps(Unsafe())))
    with pytest.raises(ValueError, match="persistent/native"):
        diagnosis.ArrayOnlyUnpickler(io.BytesIO()).persistent_load("torch storage")


def test_replay_statistics_respect_ring_bootstrap_and_frames() -> None:
    result = diagnosis.replay_summary(replay())
    assert result["samples"] == 6 and result["pos"] == 2
    assert result["reward_at_least_1"] == 4
    assert result["reward_at_least_5"] == 3
    assert result["reward_at_least_9_999"] == 2
    assert result["raw_dones"] == result["timeouts"] == 2
    assert result["true_dones"] == 0
    assert result["next_previous_frame_matches_current"]
    assert result["adjacent_nonterminal_next_observation_matches"]
    assert result["reset_previous_frame_matches_current"]
    assert result["goal_unchanged_within_transitions"]
    assert "not_tcp" in next(key for key in result if key.startswith("hand_object"))


def test_replay_detects_broken_frames_goals_and_nonfinite_values() -> None:
    value = replay()
    value.next_observations[2, 0, 18] += 0.25
    value.next_observations[2, 0, 36] += 0.25
    result = diagnosis.replay_summary(value)
    assert not result["next_previous_frame_matches_current"]
    assert not result["goal_unchanged_within_transitions"]
    value.actions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        diagnosis.replay_summary(value)


def test_partial_and_zero_length_source_rows_do_not_inflate_success_denominator() -> None:
    result = diagnosis.source_history_summary(history(215), {"g0", "g1"})
    assert result["physical_steps"] == 715
    assert result["complete_episodes"] == result["successful_complete_episodes"] == 1
    assert len(result["partial_episodes"]) == 1
    assert result["complete_episode_goal_coverage"] == 1
    assert diagnosis.source_history_summary(history(), {"g0", "g1"})["zero_length_rows"] == 1


def test_inherited_roots_actor_hashes_and_partial_checkpoints_are_preserved(tmp_path: Path) -> None:
    path = manifest_fixture(tmp_path)
    result = diagnosis.diagnose(path)
    assert len(result["checkpoints"]) == 2
    assert len(result["partial_checkpoints"]) == 1
    assert result["development_evaluations"][0]["successes"] == 3
    assert result["source_histories"]["push-v3"]["physical_steps"] == 715
    (tmp_path / "parent" / "actor.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash mismatch"):
        diagnosis.diagnose(path)


def test_unbound_or_escaping_artifact_paths_are_rejected(tmp_path: Path) -> None:
    path = manifest_fixture(tmp_path)
    data = json.loads(path.read_text())
    data["tasks"]["push-v3"]["checkpoints"][0]["artifacts"][".replay.pkl"]["path"] = (
        "../parent/replay.pkl"
    )
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="escapes"):
        diagnosis.diagnose(path)


def test_fresh_cli_imports_no_native_stack_and_preserves_existing_outputs(tmp_path: Path) -> None:
    path = manifest_fixture(tmp_path)
    output = tmp_path / "report"
    command = [
        sys.executable,
        "-m",
        "scripts.metaworld_source_diagnosis",
        "--manifest",
        str(path),
        "--output",
        str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)["native_modules_imported"] == []
    before = (output / "source_diagnosis.json").read_bytes()
    second = subprocess.run(command, capture_output=True, text=True)
    assert second.returncode != 0
    assert (output / "source_diagnosis.json").read_bytes() == before
    inside = subprocess.run(
        [*command[:-1], str(tmp_path / "parent" / "report")], capture_output=True, text=True
    )
    assert inside.returncode != 0 and not (tmp_path / "parent" / "report").exists()
