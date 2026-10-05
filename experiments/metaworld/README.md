# One hour Meta World feasibility pilot

This isolated project supports the approved 18 source policy runs: reach, push and pick place; three independent library replicates; shared and independent actor initialization. Meta-World 3.1.1 requires Gymnasium 1.1 or newer and MuJoCo 3.3.0. Stable-Baselines3 2.6.0 is pinned because its changelog adds Gymnasium 1.1 support; 2.5.0 is incompatible with this benchmark release.

This pilot tests setup, competence and preliminary functional reuse within a one-hour local-use cap. It is exploratory. A partial or undertrained population cannot establish transfer or novelty. The first 25,000-step reach run is a throughput calibration, stored separately and excluded from the balanced cohort. The balanced cohort targets 4,000 environment steps per policy, with one evaluation episode per task; this shortened budget was chosen after measuring runtime and is not a confirmation test.

The laptop already has Python 3.10 and CUDA enabled PyTorch 2.2.2 in the repository environment. To avoid installing another multi gigabyte CUDA stack, the one time pilot reused that installed PyTorch from an isolated temporary environment. For a standalone reproduction, resolve and install this project's exact dependencies using uv. On this laptop, route the existing PyTorch site packages into the isolated interpreter with `PYTHONPATH` if CUDA is needed.

The standalone lock uses Torch 2.3.0 to meet Stable-Baselines3's declared requirement and NumPy 1.26.4 to match the locally validated NumPy version. The original pilot reused Torch 2.2.2 outside that declared dependency range; the complete locked environment still needs its own benchmark smoke check before another training run. The measurement-only commands below use the existing repository environment.

```bash
uv lock --project experiments/metaworld
uv sync --project experiments/metaworld --frozen
uv run --project experiments/metaworld --frozen python scripts/metaworld_hour_pilot.py \
  --output artifacts/metaworld-one-hour-pilot-r1
```

The script accepts a hard runtime budget and reserves time for evaluation. All derived artifacts go under the output directory. It records every planned policy and evaluation, including runs skipped or interrupted by the time cap. The realized command and manifests, rather than the example below, are authoritative for the pilot.

The public benchmark declares goal observations at indices 36 to 38 with zero width. The reset observation falls outside those declared bounds. The adapter broadens only those three goal bounds to infinity and leaves every observation value unchanged. It rejects observations outside the other declared bounds and checks shape and finiteness.

The measured plan is source policy training followed by a fresh episode policy by task matrix, then oracle projection diagnostics. Oracle reconstruction uses trained target parameters to pick coefficients. Practical held-out learning remains a separate experiment.

## Measurement validation without training

Use an existing repository development environment with NumPy, Torch and pytest. These checks do not require Gymnasium, MuJoCo, Meta-World or Stable-Baselines3, and do not launch training:

```bash
python -m pytest tests/test_metaworld_geometry.py \
  tests/test_metaworld_pilot_report.py tests/test_metaworld_checkpoint_audit.py
python -m scripts.metaworld_pilot_report \
  --manifest artifacts/metaworld-one-hour-pilot-r2/pilot_manifest.json \
  --output artifacts/metaworld-validation/pilot_report.json
python -m scripts.metaworld_checkpoint_audit \
  --manifest artifacts/metaworld-one-hour-pilot-r2/pilot_manifest.json \
  --output artifacts/metaworld-validation/checkpoint_audit.json
```

The report checks every planned policy/evaluation slot, complete episodes and initialization hashes. It groups outcomes by task and independent policy-library replicate; episodes and task folds do not increase the replication count. Checkpoint auditing compares saved actor vectors, parameter layouts and hashes to the saved SAC checkpoint on CPU. It verifies parameter integrity; competence comes from the recorded environment episodes.

The geometry tests include a known subspace and an unseen vector in its span, an out-of-span vector, exact actor restoration and function-preserving hidden-unit permutations. Permutation checks examine both the mean and log standard deviation of the action distribution and restore the original actor even if verification fails.

Interpretation depends on the information used: in-sample reconstruction checks implementation, oracle reconstruction uses target weights to choose coefficients, and practical adaptation must learn coefficients from target-task interactions while keeping the source-fitted actor basis fixed. Critic training and all interaction costs must be counted separately. This validation adds no practical-adaptation result to the pilot.

Future pilot manifests record measurement coverage and code hashes. Completion requires the expected source/evaluation slots and valid initialization metadata, rather than merely finishing before the runtime deadline. The runner rejects nonpositive evaluation counts/reserves and nonempty output directories to preserve previous runs. Existing pilot manifests and checkpoints are not rewritten by these checks.
