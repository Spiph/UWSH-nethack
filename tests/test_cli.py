from pathlib import Path

import yaml
from typer.testing import CliRunner

from ups.cli import app

ROOT = Path(__file__).parents[1]


def test_all_cli_commands(tmp_path: Path) -> None:
    raw = yaml.safe_load((ROOT / "configs/smoke.yaml").read_text())
    raw["artifact_root"] = str(tmp_path / "artifacts")
    config = tmp_path / "smoke.yaml"
    config.write_text(yaml.safe_dump(raw))
    runner = CliRunner()
    commands = [
        ["train", "--config", str(config), "--reduced"],
        ["evaluate", "--config", str(config)],
        ["report", "--config", str(config)],
        ["collect-states", "--config", str(config), "--reduced"],
        ["extract-updates", "--config", str(config)],
        ["align", "--config", str(config)],
        ["analyze", "--config", str(config), "--reduced"],
        ["nulls", "--config", str(config)],
        ["reconstruct", "--config", str(config)],
        ["gate", "phase0", "--config", str(config)],
        ["reproduce", "phase0", "--config", str(config), "--reduced"],
    ]
    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0, (command, result.output, result.exception)


def test_cli_help() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "collect-states" in result.output
    assert "report" in result.output
