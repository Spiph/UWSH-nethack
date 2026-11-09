The offline reporter implements [developmental competence and complete cost reporting](https://github.com/Spiph/UWSH-nethack/issues/50). It reads staged JSON manifests without importing the plan, runner, Torch, NumPy, SB3 or an environment. This software was written after pilot outcomes began; it is descriptive reporting infrastructure and does not freeze a confirmatory analysis.

Use stdout or create a new file outside every input and declared artifact directory:

```bash
python -m scripts.metaworld_development_report \
  --manifest /path/to/run/development_manifest.json \
  --prior-manifest /path/to/retained-failed-run/development_manifest.json \
  --strict-coverage --format markdown \
  --output /tmp/development-evidence/summary.md
```

`--prior-manifest` is optional for the first attempt. For a replacement, the embedded retry record already contains cumulative costs. The reporter verifies a supplied prior manifest against that record, subtracts it to obtain attempt-only costs, and never adds it again. The source/development/preflight components determine total interactions; a stale raw aggregate is retained and identified. Constructor starts, retry counts, failed step calls and attempted/completed SAC `train()` calls remain distinct. Warmup is included in source interactions. These optimizer counters are not counts of every individual Adam operation.

Every planned task and milestone appears. A task passes only at its earliest scheduled checkpoint with the saved completion marker and the declared number of complete full-horizon episodes meeting the original threshold. A planned later visit after that pass is skipped. A failed first checkpoint with a missing or partial later visit remains incomplete. Full-budget failure requires the final milestone's completed evaluation. Protocol inconsistencies and runtime `ERROR` are distinguished from competence failure. Episodes and checkpoints do not create independent library replications or confidence intervals.

Declared episode order and seeds are always checked. `--strict-coverage` additionally checks source/development counts, disjoint goal identities, serialized payload commitments and each saved episode's actual goal and bank commitment. It hashes the payload bytes without deserializing their contents. It cannot establish physical runtime correctness from JSON alone.

`RUNNING` always produces an as-of report with lower-bound accounting. Its `finished_at_utc` is the last persistence timestamp, not a final finish. Optional `--progress /path/to/run/progress.json` supplies component maxima for monitoring; records with unknown tasks or timestamps outside that attempt are rejected. Progress never proves final completeness or replaces a terminal manifest. Ongoing training may lack its final measured duration. An explicit `--as-of-utc` constrains admissible observations; it does not manufacture measurements up to that time.

Timing separates recorded source/evaluation/checkpoint durations, current and combined attempt runtime, UTC versus monotonic spans, and inclusive session wall time. Pre-run wall time includes preparation, repairs and approval waits, as well as any prior attempt until separately subtracted. It is not attributed to algorithm compute. UTC/monotonic differences are reported without guessing their cause. Historical cohorts remain separate.

Optional `--artifact-root /path/to/run` inventories actual file bytes and SHA-256 hashes and verifies declared checkpoint digests. It streams file bytes without loading actors, arrays, replay buffers or RNG states. Symlinks are not read. A live inventory is non-atomic and can include partial files; final evidence requires a terminal stable inventory. Recorded storage counters precede the manifest write. Cumulative storage combines the current inventory with explicitly labeled recorded prior bytes, rather than calling both newly measured. Local paths are local-only evidence, not public downloads.

Existing output files, symlinks and writes inside any input or declared run directory are rejected. The input JSON is read once per file, so its recorded file hash identifies the bytes actually parsed. No input record is corrected or overwritten.

Synthetic verification:

```bash
PYTHONPATH=src python -m pytest tests/test_metaworld_development_report.py
```
