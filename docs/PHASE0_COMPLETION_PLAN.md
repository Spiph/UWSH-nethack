# Phase Zero status, failures, and completion plan

## Current status

Gate Zero is `NO_GO`, and Phase One is explicitly unauthorized. The SOL-compatible
runtime is operational: the pinned Docker images build, MiniHack imports, the SOL
Nethack APPO launcher imports, the patched NLE package is visible, and the GPU
container executes CUDA work. The current report records that runtime check as
passing.

That runtime result is necessary but not sufficient. The repository currently
contains a fail-closed mechanics harness and orchestration contract, not the full
scientific Phase Zero dataset.

### Execution update (2026-07-28)

Step 2 is now partially enacted. `ups.sol` registers every required MiniHack room
task with SOL's Sample Factory fork, constrains the interface to the eight compass
actions, adds a 9×9 `glyphs_crop` observation, and registers the glyph embedding /
three-CNN-block / 256-dimensional recurrent-policy encoder. The patched CPU
container has been checked against all four environments. A 256-step APPO smoke
job completed with two vectorized environments (320 collected frames) and wrote a
Sample Factory checkpoint; reloading it produced identical logits and values on a
fixed observation.

This is integration evidence only. It does not qualify any policy, does not
replace the required 12-policy population, and does not change the Gate Zero
decision from `NO_GO`.

## What is still failing or incomplete

### 1. Real APPO population training is not implemented

The non-reduced `train` command writes `NOT_EXECUTED` and the expected job count
(four environments × three seeds). It does not launch Sample Factory workers, save
retained checkpoints at the 100k-step cadence, resume interrupted jobs, or stop on
the preregistered competence rule.

Required population:

- `MiniHack-Room-Random-5x5-v0`
- `MiniHack-Room-Dark-5x5-v0`
- `MiniHack-Room-Monster-5x5-v0`
- `MiniHack-Room-Trap-5x5-v0`
- seeds `0`, `1`, and `2`
- maximum 2,000,000 environment steps per policy
- 200 fixed evaluation episodes per retained policy
- at least 75% success for every retained policy

### 2. The state buffer is synthetic in reduced mode and absent in full mode

The reduced workflow writes random glyph crops and zero reset masks to Zarr. A real
run must collect 512 sequences × 32 steps, balanced across all four tasks, with
fixed seeds and behavior labels that prevent train/evaluation leakage. It must replay
identical observations and reset masks through original and reconstructed policies.

### 3. Public LoRA adapter reproduction is not implemented

The mechanics checks cover factor rotations, canonical `B @ A` composition, merged
weights, and SVD/HOSVD conventions. They do not yet download and verify the four
specified `TransferGraph/roberta-base-finetuned-lora-*` repositories, restrict the
analysis to query/value trunk layers, or reproduce the official UWSH conventions on
those artifacts.

### 4. Several CLI stages are explicit placeholders

`extract-updates`, `align`, `nulls`, and `reconstruct` currently record
`AWAITING_UPSTREAM_ARTIFACTS`. `evaluate` records `AWAITING_CHECKPOINTS`. The model,
null, alignment, and subspace modules provide building blocks, but the end-to-end
artifact handoff is not wired to trained policy checkpoints.

### 5. Basis fitting and held-out validation are not scientific yet

The reduced analyzer writes an identity mechanics basis. The real implementation
must fit centered bases on training policies only, use leave-one-task-out
cross-validation, choose the smallest rank reaching 95% training explained variance,
and validate reconstruction on the held-out task before any gate statistic is
computed.

### 6. Null ensembles and hierarchical statistics are not populated

The code contains norm/spectrum-preserving null primitives, but no 1,000-replicate
study artifact. The completed run must generate Gaussian norm-matched, spectrum-
matched orientation, independent low-rank, untrained, shuffled, aligned scratch,
and unaligned scratch controls, then compute task/seed hierarchical bootstrap
intervals and null-normalized effects.

### 7. Functional reconstruction evidence is absent

No real report currently contains the required encoder/actor held-out comparisons or
reconstructed-policy behavior. The final evidence must include action KL and
agreement, feature CKA, normalized value RMSE, success/return retention, principal
angles, projection distance, effective rank, and explained-variance curves.

### 8. Independent artifact verification is intentionally disabled

`ARTIFACT_VERIFIER_IMPLEMENTED` is `False`. This is a deliberate fail-closed guard:
caller-provided JSON cannot self-certify a scientific `PASS`. A verifier must check
schemas, hashes, source/config provenance, task/seed completeness, no seed leakage,
finite metrics, null invariants, and compatibility between every stage artifact.

## Completion plan

### Step 1 — Freeze the execution record

1. Keep `configs/phase0.yaml` immutable; record its canonical digest in the run
   manifest.
2. Record the Docker image digest, SOL commit, uv lock hash, Git commit, GPU/driver,
   and exact command line.
3. Create a clean artifact root for the scientific run; never mix it with smoke
   outputs.

### Step 2 — Implement and test the Sample Factory adapter

1. Register the recurrent policy with SOL’s Sample Factory fork.
2. Map `glyphs_crop` and `blstats` observations to the policy’s glyph embedding,
   CNN, GRU, actor, and critic modules.
3. Register the fixed eight-direction NLE action interface required by the study.
4. Add checkpoint save/resume, evaluation hooks every 100k steps, deterministic
   seeds, and per-policy manifests.
5. Run a short container smoke training job, resume it from a checkpoint, and verify
   logits/actions are reproducible before launching the population.

### Step 3 — Train and qualify the 12-policy population

1. Launch four tasks × three seeds in the GPU container (or an equivalent launcher).
2. Evaluate every 100k steps using fixed evaluation episodes.
3. Retain only policies meeting the 75% success threshold, recording failures rather
   than silently replacing them.
4. Refuse to proceed if any required task/seed is missing or below quality.

### Step 4 — Implement artifact extraction and alignment

1. Extract SafeTensors weights and layer-wise updates from every retained policy.
2. Add Git Re-Basin coordinate descent for convolution channels and MLP units,
   compensating adjacent weights and testing unchanged logits.
3. Exclude unsupported GRU permutations; use principal-angle, projection, and
   representation metrics for recurrent comparisons.
4. Add architecture/version checks and content hashes for every extracted layer.

### Step 5 — Reproduce the adapter reference analysis

1. Download the four public rank-1 RoBERTa LoRA repositories into a content-addressed
   cache.
2. Verify factor shapes, scale, query/value trunk-layer selection, and merged-model
   equivalence.
3. Run zero-centered layer SVD and mode-2/mode-3 HOSVD.
4. Compare numerical conventions and reconstruction against the official UWSH
   implementation; store the comparison as independently verifiable evidence.

### Step 6 — Collect leakage-safe states and fit bases

1. Collect the 512 × 32 common evaluation buffer balanced across environments, using
   fixed seeds and explicit trained/random behavior labels.
2. Split by task for leave-one-task-out fitting; never fit a held-out task’s basis.
3. Select the smallest rank reaching 95% training explained variance.
4. Materialize bases, projections, principal angles, effective ranks, and all split
   metadata in immutable artifacts.

### Step 7 — Generate controls and evaluate reconstructions

1. Generate all preregistered matched null/control populations with 1,000 replicates.
2. Replay identical sequences and reset masks through original, learned-basis, and
   control reconstructions.
3. Compute module-level geometry and functional metrics.
4. Run hierarchical bootstrap over tasks and seeds; retain random seeds and replicate
   manifests so intervals can be regenerated exactly.

### Step 8 — Build the verifier and run Gate Zero

1. Implement the independent verifier as a separate validation path from the metric
   producer.
2. Verify every artifact recursively, including hashes, schemas, finite values,
   population completeness, null invariants, and configuration freshness.
3. Enable `ARTIFACT_VERIFIER_IMPLEMENTED` only after verifier tests cover corruption,
   omission, stale configuration, seed leakage, and fabricated evidence.
4. Run `ups gate phase0 --config configs/phase0.yaml` and publish both JSON and
   Markdown reports.

## Gate conditions that must all pass

The learned-basis reconstruction must beat the spectrum-matched random-orientation
control for both encoder and actor aggregates with a strictly positive 95% hierarchical
bootstrap interval. The median null-normalized effect must be at least two null
standard deviations better than the matched null. Reconstructed policies must retain
at least 90% median success/return with an 80% lower confidence bound. Functional
thresholds are action KL ≤ 0.05, action agreement ≥ 95%, feature CKA ≥ 0.90, and
normalized value RMSE ≤ 0.10. Adapter composition/alignment/null checks and the
official SVD/HOSVD reproduction must also pass.

If any condition fails, publish `NO_GO`, preserve the failed evidence, and do not
launch Phase One. Thresholds must not be changed to rescue a failed result.
