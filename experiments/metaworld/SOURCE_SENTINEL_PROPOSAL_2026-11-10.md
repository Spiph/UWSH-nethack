This records the [approved scripted physical sentinel](https://github.com/Spiph/UWSH-nethack/issues/56#issuecomment-6778019497), prepared before new interaction and executed on2026-11-10. It tests the physical manipulation path after the failed developmental source gate. It is not training, held-out adaptation, a competence estimate or confirmation.

The exact candidate is two complete500-step episodes: one pinned official scripted controller on source-goal index0 for push-v3 and pick-place-v3. Total new source-diagnostic interactions are1000; development, target and final-test interactions are zero. Maximums are two environment constructors, two scripted-controller instances, one CPU process, five inclusive minutes,8MiB of new artifacts, one attempt and no retry. No Torch, SAC, source-policy array loading, learned actor construction or optimizer update is permitted. The larger4000-step scripted/frozen comparison remains a separate later option without allocation.

Source goal banks, selected goal IDs, episode seeds, package versions, controller-file hashes and terminal source evidence are committed in `source_sentinel_candidate.json`. A fresh controller uses the original source streams and `goal_order=(0,)`; source configurations are loaded from the hashed final JSON, never regenerated. The original, extension and supervisor must be final; their frozen JSON hashes, known PIDs and a source-process scan are checked before native imports or output creation. The false approval file cannot run the harness.

The harness uses `metaworld_protocol.make_protocol_env` directly and does not import `metaworld_hour_pilot`. It validates the realized source goal and episode seed after reset, retains all39 observation coordinates, and delivers `clip(raw_action,-1,1)` cast to float32. The official movement controller can emit XYZ values outside the Box; clipping matches the native action boundary and adds no second spatial scale. The raw and delivered actions are both retained.

Each compact JSONL trace stores every transition, reward/success components, raw TCP and object positions, initial object/goal positions, object lift and gripper-aperture observation. A complete episode requires exactly500 steps, truncation at500 and no termination. Both successful and unsuccessful scripted episodes are retained. A completed sentinel means the prescribed diagnostic completed; it does not promise either controller will succeed. Failed steps retain attempted versus observed costs, partial traces and their hashes; interrupts/cutoffs do not create a retry.

Review the dry plan without importing NumPy or any native runtime:

```bash
python -m scripts.metaworld_source_sentinel --plan
```

After actual user allocation, code review and a frozen commit, a separate approval JSON must bind the exact plan/code hashes, commit, output and fixed timezone-aware300-second session. All setup/hash/import time counts against UTC and a latched monotonic allowance anchored at process entry. The new output directory exclusively consumes the attempt, including import failure. `CUDA_VISIBLE_DEVICES` is empty before imports; BLAS thread limits are one. The proposed launch would use an outer interrupt guard as well as inner checks:

```bash
timeout --signal=INT --kill-after=5s 300s env -u PYTHONPATH \
  /home/icannon/code/UWSH-nethack-worktrees/metaworld-measurement-validation-2026-10-05/experiments/metaworld/.venv/bin/python \
  -u -m scripts.metaworld_source_sentinel \
  --config experiments/metaworld/source_sentinel_candidate.json \
  --approval /path/to/ACTUALLY_APPROVED_FROZEN.json
```

This command is a proposal, not a launch instruction. Actual five-minute timestamps, current-head hashes and the approval reference must be frozen before execution. Blocking native calls cannot be interrupted by a Python check; the outer guard permits at most five seconds for cleanup. Synthetic/fake tests prepare the harness without executing this command or creating an environment. Scripted traces are reference diagnostics and must remain excluded from policy-library fitting or target adaptation data.


## Approved execution and result

The user replied “continue” to the exact1000-step/five-minute/8MiB request. The approved run used frozen commit `fceca5658bbac8fff410d37013a87510b3166c41`; subsequent reporting changes do not change that executable record. All hosted quality/container checks passed before launch.

| Source task | First successful transition (one-based) | Success steps /500 | Final-step success |
| --- | ---: | ---: | --- |
| push-v3 |63 |193 |false |
| pick-place-v3 |54 |447 |true |

Both episodes completed all500 steps and achieved the declared any-time criterion. Pick-place lifted the object131.97mm. Push later displaced the object away again (final object-goal distance484.76mm); it does not demonstrate stable final placement. Maximum native reward was10 for both. These are two physical positive controls, not success-rate estimates or learned-policy competence.

The process finished20:08:19UTC, within the fixed20:05:56–20:10:56UTC setup/interaction allowance. Inclusive setup+runtime was143.161seconds; the native process recorded3.476seconds. Attempted and observed source-diagnostic steps were both1000, environment/controller counts2each, resets2; errors, training updates, learned-weight loads, Torch/SAC work and development/target/final interactions were zero. Independent auditing verified trace hashes, goals/seeds, finite coordinates, adjacent/previous-frame continuity, all-four float32 action clipping and exact500-step truncations.

The local run root is `artifacts/metaworld-scripted-source-sentinel-2026-11-10`; manifest SHA-256 `bcbb391f0ea5f2cb04354eefe94ead5ebaa170283e475027470bf1caf406e06b`. Push/pick-place trace hashes are respectively `af9146dedc835e699ad910fbae3a31320188350ff0732eac0d80198a9ffed5ef` and `e52d7635e671d136221778cee2d8970091601b17b52057521efcbdc594b96b21`. Persistent control/approval/recovery records are under `artifacts/metaworld-sentinel-control-2026-11-10`. The temporary checkout and old control path had disappeared before setup; the exact prior supervisor JSON was recovered from the recorded tool output and matched its frozen SHA byte-for-byte. A persistent recovery receipt distinguishes that metadata recovery from scientific results. Code was restored from its signed commit into a persistent workspace checkout.

## Read-only comparison with retained learning

The final source histories contain2successes in40episodes on selected push source-goal0 and0in25 on selected pick-place source-goal0. These are adaptive stochastic training histories, not a matched frozen-policy experiment.

`compare_scripted_source_replay.py` additionally compares the saved scripted trace with the final50k source replay window, conditional on the same visible goal XYZ. In that window push goal0 has3000 transitions: median hand-object distance proxy274mm versus43mm for the scripted trajectory; closest object-goal distance65.57mm versus15.60mm. The replay uses different episode seeds and evolving stochastic policies; this is descriptive support for poor learned approach/transport, not a causal estimate. Pick-place goal0 is absent from the final retained window, so the tool reports no conditional replay summary rather than inventing a comparison.

```bash
python -m scripts.compare_scripted_source_replay \
  --source-manifest /home/icannon/code/UWSH-nethack/artifacts/metaworld-development-2026-11-09-r2-extension/development_manifest.json \
  --sentinel-manifest /home/icannon/code/UWSH-nethack/artifacts/metaworld-scripted-source-sentinel-2026-11-10/sentinel_manifest.json \
  --output /path/to/new-comparison.json
```

The comparison verifies input hashes and imports no native stack or environment. It performs zero new interactions. Its local output is `artifacts/metaworld-sentinel-control-2026-11-10/replay_comparison.json`.

The positive control narrows the explanation toward the learned source recipe and acquisition of manipulation behavior for these settings. It does not identify replay capacity, entropy, warmup or any other hyperparameter as the cause. A further controlled learning diagnostic needs a separately frozen design and resource decision; no additional training is approved by this run.
