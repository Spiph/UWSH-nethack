# Pilot decisions and limitations

On 2026-10-01 the user approved the three-task × three-policy-library × two-initialization feasibility design and supplied a maximum one-hour local-use budget with no other compute. This is a runtime-limited pilot. If the time cap prevents all runs or competence, preserve that fact; do not change the fixed confirmation plan based on partial results.

The laptop has an RTX 4060 Laptop GPU with 8 GiB VRAM. It has Python 3.10 and Torch 2.2.2+cu121, but no preinstalled Gymnasium, MuJoCo or Meta-World in the project's environment. Runtime packages are installed in a separate environment. The pilot uses the existing Torch through a temporary `PYTHONPATH`, avoiding a duplicate CUDA package download.

The compatible direct dependency choices are Meta-World 3.1.1, Gymnasium 1.1.1, MuJoCo 3.3.0 and Stable-Baselines3 2.6.0. Meta-World 3.1.1 requires Gymnasium >=1.1; SB3 2.5.0 requires Gymnasium <1.1 and therefore fails resolution. SB3 2.6.0 added Gymnasium 1.1 support. Sources: [Meta-World's pinned dependency declaration](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/pyproject.toml), [SB3 changelog](https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html).

The reach reset/step preflight returns a 39-feature observation, with 4 actions. The three last features are visible goal coordinates; the pinned release advertises `[0,0]` for all three although the sampled goal is nonzero. The adapter widens only those space bounds and leaves observations unchanged. Bounds outside those goal coordinates remain enforced.

The proposed 18 jobs target 25,000 environment steps each, stopping the training phase before the overall run's ten-minute evaluation reserve. Initial actor weights are shared across tasks within a replicate in the shared condition; critic weights, environment seeds and training randomness remain separately initialized. The independent condition uses a distinct actor initialization seed per task. The script records initial-actor hashes to verify this claim.

After training, the script evaluates each completed source policy on every task with fresh episodes and computes leave-one-task-out rank-one reconstruction/random-basis diagnostics within each library. Target weights select oracle coefficients; this is not held-out task learning and cannot establish practical coefficient adaptation. Three library replicates are exploratory and cannot support a powered confirmatory conclusion.
