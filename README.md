# Universal Policy Subspaces: Phase Zero

This repository implements the fail-closed measurement harness for the Phase Zero
preregistration in `Universal_Policy_Subspaces_Research_Plan.pdf`. Phase One is out of
scope and is never launched.

## Current Gate Zero status

`NO_GO`. The full 12-policy, 2M-step-cap study has not been run, and the scientific
artifact verifier is intentionally fail-closed. Smoke evidence is permanently
labeled `SMOKE_ONLY` and cannot satisfy the gate's population checks.

The canonical report is generated at
`artifacts/phase0/phase0_gate.{json,md}`. A valid scientific `NO_GO` is an expected
outcome; it explicitly prohibits Phase One.

For command-by-command instructions, see [`docs/USAGE.md`](docs/USAGE.md). For the
current failures and the work required to complete the study, see
[`docs/PHASE0_COMPLETION_PLAN.md`](docs/PHASE0_COMPLETION_PLAN.md).

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
- The gate requires four tasks by three seeds, 200 evaluation episodes per policy,
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
