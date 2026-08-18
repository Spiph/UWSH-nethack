# mypy: ignore-errors
"""Entrypoint used inside the pinned SOL container to launch one APPO policy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ups.sol import STUDY_ENVIRONMENTS, register_phase0_components


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--environment", choices=STUDY_ENVIRONMENTS, required=True)
    command.add_argument("--seed", type=int, required=True)
    command.add_argument("--artifact-root", type=Path, required=True)
    command.add_argument("--max-steps", type=int, required=True)
    command.add_argument("--experiment", required=True)
    command.add_argument("--checkpoint-retention", type=int, default=32)
    command.add_argument("--checkpoint-save-interval-seconds", type=int, default=3600)
    command.add_argument("--heartbeat-reporting-interval-seconds", type=int, default=3600)
    command.add_argument("--resume", action="store_true")
    command.add_argument("--smoke", action="store_true")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        from sample_factory.train import run_rl
        from sf_examples.nethack.train_nethack import parse_nethack_args
    except ImportError as error:  # pragma: no cover - requires the pinned image
        raise RuntimeError("run SOL APPO training via docker compose") from error

    register_phase0_components()
    train_dir = args.artifact_root / "sample_factory"
    train_dir.mkdir(parents=True, exist_ok=True)
    max_steps = 256 if args.smoke else args.max_steps
    sf_argv = [
        f"--env={args.environment}",
        f"--experiment={args.experiment}",
        f"--train_dir={train_dir}",
        f"--seed={args.seed}",
        f"--train_for_env_steps={max_steps}",
        "--use_rnn=True",
        "--rnn_type=gru",
        "--rnn_size=256",
        "--rnn_num_layers=1",
        "--actor_critic_share_weights=True",
        # Serial mode keeps CUDA tensors inside one process. This avoids CUDA IPC
        # handle failures in containerized single-GPU execution.
        "--serial_mode=True",
        "--async_rl=False",
        # The serial runtime must tolerate a recoverable host pause without
        # turning a checkpointed run into a scientific dropout.
        f"--heartbeat_reporting_interval={args.heartbeat_reporting_interval_seconds}",
        "--with_wandb=False",
        "--num_workers=1",
        "--num_envs_per_worker=2",
        # Two environments x 32 rollout steps yield 64 samples in synchronous
        # serial mode, so the learner batch must be an exact divisor/multiple.
        "--batch_size=64",
        "--rollout=32",
        # Each 100k segment saves its final checkpoint. Delaying periodic saves
        # beyond one segment prevents them from evicting scientific checkpoints.
        f"--save_every_sec={args.checkpoint_save_interval_seconds}",
        f"--keep_checkpoints={args.checkpoint_retention}",
        "--device=cpu" if args.smoke else "--device=gpu",
    ]
    if args.resume:
        sf_argv.append("--restart_behavior=resume")
    cfg = parse_nethack_args(sf_argv)
    return int(run_rl(cfg))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
