# Universal Policy Subspaces: Phase Zero

This repository implements the measurement harness for the first empirical stage
of the universal-policy-subspace study. The goal is to determine whether
independently trained policies share reusable weight-space coordinates that can
support reconstruction and coefficient-only adaptation. The reproducibility
rules, expanded five-seed cohort, uncertainty reporting, power analysis, and
qualitative validation are specified in [`docs/DEEP_RL_PROTOCOL.md`](docs/DEEP_RL_PROTOCOL.md),
following *Deep Reinforcement Learning that Matters*.

## Current empirical status

The earlier two-seed Random Room run under `artifacts/phase0/` is a pilot: it
shows that independently initialized policies can become competent, but it cannot establish a population
or cross-environment subspace result. The confirmatory milestone is five fixed,
shared seeds (10--14, disjoint from the pilot's 0--1) across six environments: Random, Dark, Monster, Trap, Ultimate Room,
and MazeWalk. Its target evidence is
competent comparable policies, aligned weights beyond matched nulls,
behavior-preserving reconstruction, and coefficient-only held-out adaptation.

The verifier remains fail-closed so incomplete evidence cannot be mistaken for
support of those claims. This is a validity guard, not the scientific result.

The confirmatory cohort uses the separate `artifacts/phase0-confirmatory-r2/`
namespace so its runs cannot resume or overwrite the observed pilot. Its report
is generated at `artifacts/phase0-confirmatory-r2/phase0_gate.{json,md}`. A valid
scientific `NO_GO` is an expected
outcome; it explicitly prohibits Phase One.

The earlier `artifacts/phase0-confirmatory/` execution is preserved but excluded
after a checkpoint-retention and task/seed-registry defect; see
[`docs/CONFIRMATORY_RUN_1_INCIDENT.md`](docs/CONFIRMATORY_RUN_1_INCIDENT.md).

`configs/nethack_baseline.yaml` adds a distinct five-seed `NetHackScore-v0`
behavioral baseline. It uses NLE's native 23 actions and score return, so it is
never pooled into the compatible eight-action MiniHack subspace population.

For command-by-command instructions, see [`docs/USAGE.md`](docs/USAGE.md).

## Reproduce

Python 3.10 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync --frozen --extra dev
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest --cov=ups --cov-fail-under=90
uv run ups reproduce phase0 --config configs/phase0.yaml
```

Run the reduced mechanics workflow separately:

```bash
uv run ups reproduce phase0 --config configs/smoke.yaml --reduced
```

It validates checkpoint/state/basis/table/manifest/report plumbing and intentionally
returns `NO_GO`. It is not evidence for the scientific hypotheses.

## CLI

The stable commands are:

```text
ups train
ups evaluate
ups report
ups collect-states
ups extract-updates
ups align
ups analyze
ups nulls
ups reconstruct
ups gate phase0
ups reproduce phase0
```

All commands validate the YAML schema and write manifests containing the complete
canonical config, SHA-256 config hash, output hashes, Git state, platform, Python,
and container provenance. Existing resumable stages are rejected if their config
hash is stale. Structured artifacts use SafeTensors, chunked Zarr, Parquet, and
JSON/Markdown.

## Scientific guardrails

- Gate thresholds live in `configs/phase0.yaml` and are immutable inputs.
- Missing, stale, incomplete, or non-finite evidence fails closed.
- The confirmatory study requires six tasks by five shared seeds (30 policies),
  200 evaluation episodes per policy,
  the 512 by 32 common replay buffer, and 1,000 null replicates.
- CNN/MLP permutations compensate adjacent layers and are function-tested.
- GRU symmetry is explicitly unresolved; recurrent comparisons are invariant
  principal-angle, projection, and representation comparisons only.
- LoRA analysis composes the identifiable update as scaled `B @ A`; tests cover
  factor gauge rotations and equivalence to merged weights.
- SVD is zero-centered and HOSVD uses mode-n unfolding/left singular vectors,
  matching the convention in the official `toshi2k2/unisub` implementation.

## Containers

```bash
docker compose build research-cpu
docker compose run --rm research-cpu --help
docker compose run --rm research-cpu reproduce phase0 --config configs/smoke.yaml --reduced
docker compose up tensorboard
```

`research-gpu` uses the NVIDIA runtime. The `sol_patched` profile follows SOL’s
working integration pattern: the image checks out SOL at commit
`7c272b66e6ebe72ca008526d33f7e2e40e660af5`, installs its Sample Factory fork,
builds its Cython extension, and installs its patched NLE package alongside
Gymnasium 1.0.0 and MiniHack 1.0.2. The image smoke test exercises both the
MiniHack environment and SOL’s Nethack APPO launcher. The upstream `appo` and
`nle` extras remain separately locked audit profiles because their release metadata
is incompatible; they are not selected together:

```bash
uv sync --extra appo
uv sync --extra nle
# uv rejects selecting both, by design.
```
