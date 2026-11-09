# Same-learner source-development continuation

This runner implements the user-authorized exploratory amendment in [Extend the same source learners through 1M steps within Thursday’s window](https://github.com/Spiph/UWSH-nethack/issues/51#issuecomment-6749201546). Evidence and later decisions belong in [the source-competence experiment](https://github.com/Spiph/UWSH-nethack/issues/14). It does not authorize target adaptation or native coefficient-adapter validation.

The original frozen c2cd process keeps its original cutoffs. A one-shot local supervisor waits for its recorded Linux PID/start identity to disappear and a valid final manifest. It binds the exact prospective frozen launch, failed-attempt inventory, final parent manifest and both artifact inventories before launching one continuation. An error, user interruption, identity change or reconstruction mismatch stops the handoff without an automatic retry. A completed source library is a no-op.

Only the same unfinished push/pick-place SAC learners reload. Reach remains stopped at its first100k pass. Remaining milestones are250k/500k/1M; each task stops at its first complete16/20 development pass. Goals, seeds, initialization identities, SAC configuration and evaluation order remain fixed. Cumulative limits include every earlier attempt:2.1M physical source steps,90k development evaluation steps,1k preflight/reconstruction steps,8 constructor starts and2GiB artifacts. Training stops November13 at02:30UTC and the absolute cutoff is03:00UTC (Thursday November12 at10p.m.Eastern). There is one training process and no paid service.

## Recovery contract

The old checkpoints lack a full simulator snapshot. This is a narrow deterministic reconstruction of the saved committed source episode, not general exact environment resume. Native SAC/replay/optimizer/entropy and learner RNG are loaded unchanged. Every replay observation, next observation, reward and final saved observation must match exactly before any learner resumes. Reconstruction transitions are charged before attempts. Historical Monitor timing/history remains segmented; old JSON and native episode buffers remain evidence.

Model timesteps N, committed replay transitions C and physical source steps S are recorded separately. At most one uncommitted transition or pending optimizer update is accepted when the saved evidence proves it. Lost steps are retained in the cumulative cost; no decrement, refund or compensating rollout is permitted. The source cap and milestones use physical steps. Every loaded learner must pass reconstruction and aggregate cost reconciliation before any pending update or new rollout. Tonight’s runner rejects absent or interrupted-warmup checkpoints before reload. Failures remain in the new manifest while parent artifacts stay immutable.

## Operation

Prepare without importing an RL runtime:

```bash
python -m scripts.metaworld_source_continuation --plan
```

The actual approval JSON must bind the accepted amendment hash, all executable hashes, reviewed Git commit, fixed original session/cutoffs, authorization URL and captured parent process identity. After checks pass, run from that frozen checkout with the pinned Meta-World interpreter:

```bash
python -u -m scripts.metaworld_source_supervisor \
  --approval /path/to/frozen-source-extension-approval.json \
  --old-pid ORIGINAL_PYTHON_PID \
  --control /path/outside/all/artifact/roots \
  --python /path/to/pinned/metaworld/.venv/bin/python
```

The supervisor holds a fixed-output exclusive lock for the whole child lifetime. Its launch marker prevents a second attempt even from another control directory. The original absolute host monotonic cutoff includes waiting and child startup. A `STOP` file in the control directory stops future work; the supervisor interrupts and then kills a child that fails to exit within30seconds. Its bounded status/log files consume the reserved16MiB within the combined2GiB allowance. It does not restart failures or extend the allocation. The machine must remain running.

Pure fake/synthetic tests cover counter edge cases, episode reconstruction, parent validation, clock changes, duplicate supervisors, evaluation completion and resource limits. Native recovery itself is validated only during the separately logged authorized handoff, before learning resumes. Completed development results do not establish policy-subspace utility or independent library replication.

The existing `metaworld_development_report` CLI describes the original100k/250k phase only. Extension evidence must include500k/1M, physical/model counter differences and inherited artifact roots; do not interpret original-reporter protocol errors on the extension as scientific failures.
