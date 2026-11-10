This is the unapproved [scripted physical sentinel proposal](https://github.com/Spiph/UWSH-nethack/issues/56), prepared before any new interaction. It tests the physical manipulation path after the failed developmental source gate. It is not training, held-out adaptation, a competence estimate or confirmation.

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
