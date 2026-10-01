# One hour Meta World feasibility pilot

This isolated project supports the approved 18 source policy runs: reach, push and pick place; three independent library replicates; shared and independent actor initialization. Meta-World 3.1.1 requires Gymnasium 1.1 or newer and MuJoCo 3.3.0. Stable-Baselines3 2.6.0 is pinned because its changelog adds Gymnasium 1.1 support; 2.5.0 is incompatible with this benchmark release.

This pilot tests setup, competence and preliminary functional reuse within a one-hour local-use cap. It is exploratory. A partial or undertrained population cannot establish transfer or novelty. The first 25,000-step reach run is a throughput calibration, stored separately and excluded from the balanced cohort. The balanced cohort targets 4,000 environment steps per policy, with one evaluation episode per task; this shortened budget was chosen after measuring runtime and is not a confirmation test.

The laptop already has Python 3.10 and CUDA enabled PyTorch 2.2.2 in the repository environment. To avoid installing another multi gigabyte CUDA stack, the one time pilot reused that installed PyTorch from an isolated temporary environment. For a standalone reproduction, resolve and install this project's exact dependencies using uv. On this laptop, route the existing PyTorch site packages into the isolated interpreter with `PYTHONPATH` if CUDA is needed.

```bash
uv lock --project experiments/metaworld
uv sync --project experiments/metaworld --frozen
uv run --project experiments/metaworld --frozen python scripts/metaworld_hour_pilot.py \
  --output artifacts/metaworld-one-hour-pilot-r1
```

The script accepts a hard runtime budget and reserves time for evaluation. All derived artifacts go under the output directory. It records every planned policy and evaluation, including runs skipped or interrupted by the time cap. The realized command and manifests, rather than the example below, are authoritative for the pilot.

The public benchmark declares goal observations at indices 36 to 38 with zero width. The reset observation falls outside those declared bounds. The adapter broadens only those three goal bounds to infinity and leaves every observation value unchanged. It rejects observations outside the other declared bounds and checks shape and finiteness.

The measured plan is source policy training followed by a fresh episode policy by task matrix, then oracle projection diagnostics. Oracle reconstruction uses a fully trained target actor to pick coefficients and is not practical adaptation. The later held out learning experiment remains a separate issue.
