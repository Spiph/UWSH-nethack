"""Stable command-line surface for Phase Zero."""

from __future__ import annotations

from pathlib import Path

import typer

from ups.artifacts import write_manifest
from ups.config import load_config
from ups.verifier import verify_artifacts
from ups.workflow import analyze as run_analyze
from ups.workflow import collect_states as run_collect_states
from ups.workflow import evaluate as run_evaluate
from ups.workflow import extract_updates as run_extract_updates
from ups.workflow import gate as run_gate
from ups.workflow import record_stage
from ups.workflow import reproduce as run_reproduce
from ups.workflow import train as run_train

app = typer.Typer(no_args_is_help=True)
gate_app = typer.Typer(no_args_is_help=True)
reproduce_app = typer.Typer(no_args_is_help=True)
app.add_typer(gate_app, name="gate")
app.add_typer(reproduce_app, name="reproduce")


def config_option() -> Path:
    return Path("configs/phase0.yaml")


@app.command()
def train(
    config: Path = typer.Option(config_option(), exists=True),
    reduced: bool = False,
    sol_smoke: bool = False,
    resume: bool = False,
    plan_only: bool = False,
    only_environment: str | None = typer.Option(
        None, help="Run one preregistered environment for a resumable pilot."
    ),
    only_seed: int | None = typer.Option(
        None, help="Run one preregistered seed for a resumable pilot."
    ),
) -> None:
    typer.echo(
        run_train(
            load_config(config),
            reduced,
            sol_smoke,
            resume,
            plan_only,
            only_environment,
            only_seed,
        )
    )


@app.command()
def evaluate(config: Path = typer.Option(config_option(), exists=True)) -> None:
    cfg = load_config(config)
    typer.echo(run_evaluate(cfg))


@app.command("check")
def check(
    config: Path = typer.Option(config_option(), exists=True),
    evidence: Path = typer.Option(..., exists=True, readable=True),
) -> None:
    """Independently verify Gate Zero artifacts and evidence claims."""
    cfg = load_config(config)
    import json

    result = verify_artifacts(cfg, json.loads(evidence.read_text(encoding="utf-8")))
    typer.echo(json.dumps(result, indent=2, sort_keys=True))
    if not result["verified"]:
        raise typer.Exit(code=1)


@app.command("collect-states")
def collect_states(
    config: Path = typer.Option(config_option(), exists=True), reduced: bool = False
) -> None:
    typer.echo(run_collect_states(load_config(config), reduced))


def placeholder(command: str, config: Path) -> None:
    cfg = load_config(config)
    path = record_stage(cfg, command, {"status": "AWAITING_UPSTREAM_ARTIFACTS"})
    write_manifest(cfg, command, [path])
    typer.echo(path)


@app.command("extract-updates")
def extract_updates(config: Path = typer.Option(config_option(), exists=True)) -> None:
    typer.echo(run_extract_updates(load_config(config)))


@app.command()
def align(config: Path = typer.Option(config_option(), exists=True)) -> None:
    placeholder("align", config)


@app.command()
def analyze(
    config: Path = typer.Option(config_option(), exists=True), reduced: bool = False
) -> None:
    typer.echo(run_analyze(load_config(config), reduced))


@app.command()
def nulls(config: Path = typer.Option(config_option(), exists=True)) -> None:
    placeholder("nulls", config)


@app.command()
def reconstruct(config: Path = typer.Option(config_option(), exists=True)) -> None:
    placeholder("reconstruct", config)


@gate_app.command("phase0")
def gate_phase0(config: Path = typer.Option(config_option(), exists=True)) -> None:
    typer.echo(run_gate(load_config(config)))


@reproduce_app.command("phase0")
def reproduce_phase0(
    config: Path = typer.Option(config_option(), exists=True), reduced: bool = False
) -> None:
    typer.echo(run_reproduce(load_config(config), reduced))


if __name__ == "__main__":
    app()
