# Phase Zero harness: usage guide

This guide describes the commands that exist today, what each command consumes,
and where it writes artifacts. It is a guide to the current harness, not a claim
that the full Phase Zero study has already run. The canonical Gate Zero report is
currently `NO_GO`.

## 1. Prepare an environment

The local development path uses Python 3.10 and `uv`:

```bash
uv sync --frozen --extra dev
```

The SOL-compatible experiment runtime is intentionally isolated from the upstream
audit profiles:

```bash
uv sync --frozen --extra sol --extra adapter
```

Do not select `appo`, `nle`, and `sol` together. The upstream release metadata has
incompatible Gymnasium requirements; the `sol` profile installs the pinned SOL
Sample Factory fork and patched NLE in Docker.

For the reproducible container path:

```bash
docker compose build research-cpu
docker compose build research-gpu       # requires NVIDIA Container Toolkit
docker compose run --rm research-cpu --help
```

The image records the SOL commit and mounts `artifacts/`, `configs/`, and the
Hugging Face cache. The GPU service is the preferred path for actual APPO jobs.

## 2. Validate code and configuration

Run the checks before an experiment or after changing code/configuration:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest --cov=ups --cov-fail-under=90
docker compose config --quiet
uv lock --check
```

Configurations are strict, immutable-input YAML files. The full preregistration is
`configs/phase0.yaml`; the reduced mechanics fixture is `configs/smoke.yaml`.
Every manifest records the canonical configuration hash, source-tree hash, lockfile
hash, and runtime provenance. Reusing a stage with a stale hash is rejected.

## 3. Run the reduced mechanics workflow

This is the safe first run. It creates a deterministic checkpoint, a synthetic Zarr
state fixture, mechanics metrics, a SafeTensors basis, manifests, and a fail-closed
gate report:

```bash
uv run ups reproduce phase0 --config configs/smoke.yaml --reduced
```

Equivalent container command:

```bash
docker compose run --rm research-cpu \
  reproduce phase0 --config configs/smoke.yaml --reduced
```

Expected result: `artifacts/smoke/phase0_gate.json` with decision `NO_GO` and
`phase_one_authorized: false`. The smoke state data is synthetic mechanics data and
must not be used as scientific evidence.

## 4. Run the full Phase Zero workflow contract

The full command creates the exact 12-job registry and, inside the pinned SOL
container, launches each job in resumable 100,000-step chunks:

```bash
uv run ups reproduce phase0 --config configs/phase0.yaml
```

In the pinned container:

```bash
docker compose run --rm research-gpu \
  reproduce phase0 --config configs/phase0.yaml
```

Inspect the immutable job plan without launching workers:

```bash
docker compose run --rm research-gpu \
  train --config configs/phase0.yaml --plan-only
```

Training stops with `TRAINED_AWAITING_EVALUATION`; it does not qualify policies or
authorize the gate. The fixed 200-episode evaluator is still required before any
checkpoint can become retained-policy evidence. The expected canonical outputs are:

```text
artifacts/phase0/checkpoints/
artifacts/phase0/states.zarr/
artifacts/phase0/bases/
artifacts/phase0/metrics/
artifacts/phase0/manifests/
artifacts/phase0/evidence.json
artifacts/phase0/phase0_gate.json
artifacts/phase0/phase0_gate.md
```

## 5. SOL APPO integration smoke test

The Phase Zero SOL adapter registers the four room tasks, turns NLE's compass
actions into the fixed eight-action study interface, derives `glyphs_crop` from
the raw glyph map and BLStats coordinates, and supplies the 32-dimensional glyph
embedding plus three-CNN-block encoder to SOL's recurrent APPO policy.

Run one short APPO job before starting the population:

```bash
docker compose build research-cpu
docker compose run --rm --no-deps --entrypoint python research-cpu \
  -m ups.sol_train \
  --environment MiniHack-Room-Random-5x5-v0 \
  --seed 0 \
  --artifact-root artifacts/sol-smoke \
  --experiment phase0-random-seed0-smoke \
  --max-steps 2000000 \
  --smoke
```

The same smoke path is exposed through the stable CLI:

```bash
docker compose run --rm research-cpu \
  train --config configs/smoke.yaml --sol-smoke
```

Resume the same named Sample Factory experiment through the stable CLI with
`--resume`:

```bash
docker compose run --rm research-cpu \
  train --config configs/smoke.yaml --sol-smoke --resume
```

`--smoke` caps the job at 256 environment steps. It validates the SOL/NLE/APPO
integration only; it is not a retained policy or Gate Zero evidence. Omit
`--smoke` and use `research-gpu` only after the smoke run has produced a valid
Sample Factory checkpoint. `--resume` asks Sample Factory to continue the named
experiment directory.

## 6. Individual commands

All commands accept `--config PATH`; the default is `configs/phase0.yaml`.

| Command | Current behavior | Main output |
| --- | --- | --- |
| `ups train` | Plans or launches the exact 12 APPO jobs; `--plan-only` writes only the registry, while `--sol-smoke` runs one short job. | `population/population_plan.json`, `population/training_report.json` |
| `ups evaluate` | Replays each recorded checkpoint on fixed per-task evaluation seeds (200 episodes in the full config), writes the policy registry, and marks the population qualified or unqualified. If training has not produced a report, records `AWAITING_CHECKPOINTS`. | `evaluations/policy_registry.parquet`, `evaluations/evaluation_report.json`, `stages/evaluate.json` |
| `ups collect-states` | Creates synthetic mechanics states only with `--reduced`; full mode records `NOT_EXECUTED`. | `states.zarr`, `stages/collect-states.json` |
| `ups extract-updates` | Converts compatible SOL checkpoints to SafeTensors, or records missing upstream artifacts; exports remain explicitly unqualified. | `weights/*.safetensors`, `weights/extraction.json` |
| `ups align` | Records `AWAITING_UPSTREAM_ARTIFACTS`. | `stages/align.json` |
| `ups analyze` | Runs LoRA, permutation, null-invariant, SVD/HOSVD, and metric mechanics checks; it does not download or analyze the four public LoRA repositories yet. | `metrics/functional.parquet`, `bases/mechanics.safetensors`, `evidence.json` |
| `ups nulls` | Records `AWAITING_UPSTREAM_ARTIFACTS`. | `stages/nulls.json` |
| `ups reconstruct` | Records `AWAITING_UPSTREAM_ARTIFACTS`. | `stages/reconstruct.json` |
| `ups gate phase0` | Evaluates the evidence already present and writes the JSON/Markdown Gate Zero report. | `phase0_gate.json`, `phase0_gate.md` |

The nested commands are:

```bash
ups gate phase0 --config configs/phase0.yaml
ups reproduce phase0 --config configs/phase0.yaml
```

## 7. Inspecting results and provenance

Read the decision and authorization fields directly:

```bash
python - <<'PY'
import json
report = json.load(open("artifacts/phase0/phase0_gate.json"))
print(report["decision"], report["phase_one_authorized"])
PY
```

Use `artifacts/*/RERUN.md` for the exact command associated with a generated run.
Manifests are content-addressed and should be retained with any exported result.
The ignored `artifacts/` directory is local output; it is not a substitute for a
versioned study record or a verified scientific dataset.

## 8. TensorBoard

The compose service is available for local logs:

```bash
docker compose up tensorboard
```

Open <http://localhost:6006>. The current mechanics workflow does not produce a
full APPO training stream; TensorBoard becomes meaningful after the training
adapter writes learner/rollout summaries.
