"""Future collection telemetry exposes incomplete episodes and nonfinite diagnostics."""

from __future__ import annotations

import json
import math

from scripts.metaworld_development import source_episode_summary, training_diagnostics


def test_partial_success_is_excluded_from_completed_source_statistics() -> None:
    rows = [
        {"complete_episode": True, "length": 500, "success": True},
        {"complete_episode": True, "length": 500, "success": False},
        {"complete_episode": False, "length": 215, "success": True},
        {"complete_episode": False, "length": 0, "success": False},
    ]
    summary = source_episode_summary(rows)
    assert summary["complete_episodes"] == 2
    assert summary["successful_complete_episodes"] == summary["recent_successes"] == 1
    assert rows[-2]["success"] is True  # Observation never changes the raw history.


def test_recent_source_window_retains_lifetime_denominator_separately() -> None:
    rows = [{"complete_episode": True, "length": 500, "success": True} for _ in range(10)]
    rows.extend({"complete_episode": True, "length": 500, "success": False} for _ in range(100))
    summary = source_episode_summary(rows)
    assert summary["complete_episodes"] == 110
    assert summary["successful_complete_episodes"] == 10
    assert summary["recent_complete_episodes"] == 100
    assert summary["recent_successes"] == 0


def test_nonfinite_metrics_remain_visible_without_invalid_json_numbers() -> None:
    values = {
        "train/actor_loss": float("nan"),
        "train/critic_loss": float("inf"),
        "train/ent_coef": 0.1,
        "train/non_numeric": object(),
        "time/fps": 42,
    }
    numeric, nonfinite = training_diagnostics(values)
    assert numeric == {"train/ent_coef": 0.1}
    assert nonfinite == ["train/actor_loss", "train/critic_loss"]
    json.dumps({"diagnostics": numeric, "nonfinite_diagnostic_keys": nonfinite}, allow_nan=False)
    actor_loss = values["train/actor_loss"]
    assert isinstance(actor_loss, float) and math.isnan(actor_loss)
