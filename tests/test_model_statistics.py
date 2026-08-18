import json
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import torch

from ups.config import load_config
from ups.model import NLE_ACTIONS, RecurrentNLEPolicy
from ups.sol import crop_glyphs
from ups.statistics import (
    aggregate_learning_curves,
    aggregate_seed_results,
    bootstrap_power_diagnostic,
    confidence_interval,
    hierarchical_bootstrap,
    write_statistics_report,
)
from ups.workflow import report

ROOT = Path(__file__).parents[1]


def test_policy_shapes_registry_and_lora() -> None:
    model = RecurrentNLEPolicy(glyph_vocab=128, crop_size=5, hidden_size=32, lora_rank=2)
    output = model(torch.zeros(3, 2, 5, 5), torch.zeros(3, 2, 27))
    assert output.logits.shape == (3, 2, len(NLE_ACTIONS))
    assert output.value.shape == (3, 2)
    assert output.state.shape == (1, 2, 32)
    assert "recurrent_core.gru" in model.module_registry


def test_hierarchical_bootstrap_is_deterministic() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    tasks = ["a", "a", "b", "b"]
    seeds = [0, 1, 0, 1]
    one = hierarchical_bootstrap(values, tasks, seeds, 50, np.random.default_rng(9))
    two = hierarchical_bootstrap(values, tasks, seeds, 50, np.random.default_rng(9))
    assert np.array_equal(one, two)
    low, high = confidence_interval(one)
    assert low <= high
    with pytest.raises(ValueError):
        confidence_interval(np.array([np.nan]))


def test_phase_zero_crop_uses_nethack_xy_coordinates_and_padding() -> None:
    glyphs = np.arange(9, dtype=np.int16).reshape(3, 3)
    center = crop_glyphs(glyphs, np.array([1, 1]), 3)
    assert np.array_equal(center, glyphs)
    corner = crop_glyphs(glyphs, np.array([0, 0]), 3)
    assert corner.shape == (3, 3)
    assert corner[1, 1] == glyphs[0, 0]
    with pytest.raises(ValueError, match="shape"):
        crop_glyphs(glyphs[None], np.array([0, 0]), 3)


def test_seed_aggregation_reports_per_environment_and_rejects_duplicates() -> None:
    frame = __import__("pandas").DataFrame(
        {
            "environment": ["easy", "easy", "hard", "hard"],
            "seed": [1, 2, 1, 2],
            "value": [0.4, 0.6, 0.2, 0.8],
        }
    )
    result = aggregate_seed_results(
        frame,
        expected_environments=["easy", "hard"],
        expected_seeds=[1, 2],
        bootstrap_replicates=100,
        bootstrap_seed=3,
    )
    assert result["n_seeds"].tolist() == [2, 2]
    assert result["mean"].tolist() == [0.5, 0.5]
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_seed_results(
            __import__("pandas").concat([frame, frame.iloc[[0]]], ignore_index=True)
        )


def test_learning_curve_common_grid_and_power_are_deterministic() -> None:
    pd = __import__("pandas")
    frame = pd.DataFrame(
        {
            "environment": ["easy"] * 4,
            "seed": [1, 2, 1, 2],
            "checkpoint": [10, 10, 20, 20],
            "value": [0.4, 0.6, 0.7, 0.9],
        }
    )
    curves = aggregate_learning_curves(
        frame, expected_seeds=[1, 2], bootstrap_replicates=100, bootstrap_seed=5
    )
    assert curves["checkpoint"].tolist() == [10, 20]
    one = bootstrap_power_diagnostic(
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        minimum_lift=0.2,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    two = bootstrap_power_diagnostic(
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        minimum_lift=0.2,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )
    assert one == two
    assert one["method"] == "centered independent seed bootstrap power diagnostic"
    assert 0.0 <= one["estimated_power_at_minimum_lift"] <= 1.0


def test_statistics_report_runs_end_to_end_on_complete_seed_grid(tmp_path: Path) -> None:
    source = load_config(ROOT / "configs/smoke.yaml")
    config = source.model_copy(
        update={
            "artifact_root": tmp_path / "confirmatory",
            "seeds": [0, 1],
            "environments": ["env-a", "env-b"],
            "training": source.training.model_copy(
                update={"max_environment_steps": 20, "evaluation_interval": 10}
            ),
            "evaluation_buffer": source.evaluation_buffer.model_copy(
                update={"fixed_seeds": [1101, 1401]}
            ),
            "analysis": source.analysis.model_copy(update={"bootstrap_replicates": 100}),
        }
    )
    evaluations = config.artifact_root / "evaluations"
    evaluations.mkdir(parents=True)
    registry_path = evaluations / "policy_registry.parquet"
    rows = []
    for environment_index, environment in enumerate(config.environments):
        for seed in config.seeds:
            for checkpoint in (10, 20):
                rows.append(
                    {
                        "environment": environment,
                        "seed": seed,
                        "target_environment_steps": checkpoint,
                        "success_rate": 0.2 + 0.1 * seed + 0.01 * checkpoint,
                        "mean_return": 0.1 * environment_index + checkpoint + seed,
                        "mean_length": 30.0 - checkpoint / 2 + seed,
                    }
                )
    pd.DataFrame(rows).to_parquet(registry_path, index=False)
    (evaluations / "evaluation_report.json").write_text(
        json.dumps(
            {
                "config_hash": config.digest,
                "episodes_per_checkpoint": config.training.evaluation_episodes,
                "policy_registry": str(registry_path),
            }
        ),
        encoding="utf-8",
    )

    stage = json.loads(report(config, minimum_lift=0.05, bootstrap_seed=7).read_text())
    assert stage["status"] == "COMPLETE"
    payload = json.loads(Path(stage["outputs"]["json"]).read_text())
    assert len(payload["seed_summaries"]) == 6
    assert {row["metric"] for row in payload["seed_summaries"]} == {
        "return",
        "success",
        "length",
    }
    assert all(row["n_seeds"] == 2 for row in payload["seed_summaries"])
    assert len(payload["learning_curves"]) == 12
    assert len(payload["individual_results"]) == 24
    assert set(payload["power_diagnostic"]["by_environment"]) == {"env-a", "env-b"}

    stale = json.loads((evaluations / "evaluation_report.json").read_text())
    stale["config_hash"] = "stale"
    (evaluations / "evaluation_report.json").write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="stale"):
        report(config)
    stale["config_hash"] = config.digest
    stale["policy_registry"] = "wrong.parquet"
    (evaluations / "evaluation_report.json").write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="link"):
        report(config)
    stale["policy_registry"] = str(registry_path)
    stale["episodes_per_checkpoint"] = 999
    (evaluations / "evaluation_report.json").write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="episode count"):
        report(config)
    stale["episodes_per_checkpoint"] = config.training.evaluation_episodes
    (evaluations / "evaluation_report.json").write_text(json.dumps(stale))
    incomplete = pd.DataFrame(rows).drop(columns="mean_length")
    incomplete.to_parquet(registry_path, index=False)
    with pytest.raises(ValueError, match="required report metrics"):
        report(config)
    incomplete = pd.DataFrame(rows)
    incomplete = incomplete[
        ~((incomplete["environment"] == "env-b") & (incomplete["target_environment_steps"] == 20))
    ]
    incomplete.to_parquet(registry_path, index=False)
    with pytest.raises(ValueError, match="checkpoint grid"):
        report(config)


def test_score_baseline_report_uses_return_not_minihack_success(tmp_path: Path) -> None:
    source = load_config(ROOT / "configs/smoke.yaml")
    config = source.model_copy(
        update={
            "study": "nethack-baseline",
            "artifact_root": tmp_path / "nethack-baseline",
            "seeds": [20, 21],
            "environments": ["NetHackScore-v0"],
            "training": source.training.model_copy(
                update={"max_environment_steps": 20, "evaluation_interval": 10}
            ),
            "evaluation_buffer": source.evaluation_buffer.model_copy(
                update={"fixed_seeds": [2101]}
            ),
            "analysis": source.analysis.model_copy(
                update={"bootstrap_replicates": 100, "practical_effect_size": 1.0}
            ),
        }
    )
    evaluations = config.artifact_root / "evaluations"
    evaluations.mkdir(parents=True)
    registry_path = evaluations / "policy_registry.parquet"
    pd.DataFrame(
        [
            {
                "environment": "NetHackScore-v0",
                "seed": seed,
                "target_environment_steps": checkpoint,
                # A score baseline must not be forced through this unrelated
                # MiniHack status field.
                "success_rate": 0.0,
                "mean_return": 2.0 * checkpoint + seed,
                "mean_length": 5000.0 - checkpoint,
                "mean_max_dungeon_depth": 1.0 + seed / 100,
            }
            for seed in config.seeds
            for checkpoint in (10, 20)
        ]
    ).to_parquet(registry_path, index=False)
    (evaluations / "evaluation_report.json").write_text(
        json.dumps(
            {
                "config_hash": config.digest,
                "episodes_per_checkpoint": config.training.evaluation_episodes,
                "policy_registry": str(registry_path),
            }
        ),
        encoding="utf-8",
    )

    stage = json.loads(report(config, bootstrap_seed=7).read_text())
    payload = json.loads(Path(stage["outputs"]["json"]).read_text())
    assert stage["primary_metric"] == "return"
    assert {row["metric"] for row in payload["seed_summaries"]} == {
        "return",
        "length",
        "max_depth",
    }
    assert payload["power_diagnostic"]["contrast"].startswith("return")
    assert payload["power_diagnostic"]["minimum_lift"] == 1.0


def test_statistics_rejects_invalid_or_pseudoreplicated_inputs(tmp_path: Path) -> None:
    valid = pd.DataFrame(
        {
            "environment": ["a", "a", "b", "b"],
            "seed": [0, 1, 0, 1],
            "value": [0.1, 0.2, 0.3, 0.4],
        }
    )
    with pytest.raises(ValueError, match="positive"):
        aggregate_seed_results(valid, bootstrap_replicates=0)
    for invalid, message in (
        (valid.drop(columns="value"), "missing required"),
        (valid.iloc[0:0], "empty"),
        (valid.assign(value="bad"), "numeric"),
        (valid.assign(value=[0.1, 0.2, np.nan, 0.4]), "non-finite"),
        (valid.assign(environment=["a", "a", "b", None]), "incomplete"),
    ):
        with pytest.raises(ValueError, match=message):
            aggregate_seed_results(invalid)
    with pytest.raises(ValueError, match="incomplete environments"):
        aggregate_seed_results(valid, expected_environments=["a", "b", "c"])
    with pytest.raises(ValueError, match="incomplete seeds"):
        aggregate_seed_results(valid, expected_seeds=[0, 1, 2])
    missing_unit = valid[~((valid["environment"] == "b") & (valid["seed"] == 1))]
    with pytest.raises(ValueError, match="replication units"):
        aggregate_seed_results(
            missing_unit,
            expected_environments=["a", "b"],
            expected_seeds=[0, 1],
        )

    curves = pd.concat(
        [valid.assign(checkpoint=10), valid.assign(checkpoint=20)], ignore_index=True
    )
    with pytest.raises(ValueError, match="requested checkpoint"):
        aggregate_learning_curves(curves, checkpoints=[30])
    incomplete_checkpoint = curves.drop(index=7)
    with pytest.raises(ValueError, match="incomplete seed curve"):
        aggregate_learning_curves(incomplete_checkpoint, expected_seeds=[0, 1])
    incomplete_environment = curves[
        ~((curves["environment"] == "b") & (curves["checkpoint"] == 20))
    ]
    with pytest.raises(ValueError, match="common checkpoint grid"):
        aggregate_learning_curves(incomplete_environment)

    with pytest.raises(ValueError, match=">=2"):
        bootstrap_power_diagnostic([0.1], [0.2], minimum_lift=0.1)
    with pytest.raises(ValueError, match="replicates"):
        bootstrap_power_diagnostic([0.1, 0.2], [0.2, 0.3], minimum_lift=0.1, bootstrap_replicates=0)
    with pytest.raises(ValueError, match="positive"):
        bootstrap_power_diagnostic([0.1, 0.2], [0.2, 0.3], minimum_lift=0.0)
    with pytest.raises(ValueError, match="finite"):
        confidence_interval(np.array([1.0]), level=1.0)
    with pytest.raises(ValueError, match="finite"):
        hierarchical_bootstrap(np.array([np.nan]), ["a"], [0], 1, np.random.default_rng(0))

    with pytest.raises(ValueError, match="seed_results"):
        write_statistics_report(pd.DataFrame(), tmp_path / "empty")
    with pytest.raises(ValueError, match="individual_results"):
        write_statistics_report(
            valid, tmp_path / "bad-individual", individual_results=pd.DataFrame()
        )
    outputs = write_statistics_report(valid, tmp_path / "minimal")
    assert set(outputs) == {"json", "csv", "parquet"}
