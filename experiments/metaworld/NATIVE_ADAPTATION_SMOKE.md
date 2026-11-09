This is the **unapproved** mechanical validation proposed in [decision #49](https://github.com/Spiph/UWSH-nethack/issues/49).
Preparing this code does not authorize native construction or rollout. The source
extension remains a separate allocation. No native smoke has run during preparation.

The candidate is fixed in `adaptation_native_smoke_candidate.json`. Ordinary tests
inject fake runtimes. `--plan` prints the candidate and code hashes without importing
NumPy, Torch, Gymnasium or SB3; false approval also refuses before output creation.

| Resource | Maximum or exact completed-check count |
| --- | --- |
| Inclusive window |300 seconds from first approved setup; fixed UTC and monotonic guards |
| Device | One CPU process; Torch one thread; empty `CUDA_VISIBLE_DEVICES` before imports |
| SAC initializations |8: four initial learners and four checkpoint reloads |
| Standalone frozen actors |2: reference prediction and verified-source prediction |
| Synthetic transitions |120: 4 × (20 initial + 5 original continuation + 5 restored continuation) |
| Native SAC update calls |104: 4 × (16 initial + 5 original + 5 restored) |
| Adam step calls |312: actor, twin-critic optimizer and entropy optimizer per update |
| Benchmark source/target/final steps |0 |
| New artifacts |32 MiB, including partial and failure records |
| Attempts |1; no automatic retry |

The environment is a deterministic synthetic 39-dimensional Box with four actions
bounded exactly by minus one and one, and timeout 7. It uses an explicit DummyVecEnv.
The small 39→8→8→4 actor has two raw heads. Three deterministic NumPy source fixtures
produce a rank-two source-anchored PCA family and its dimension/scale-matched random
control. They are mechanical fixtures; they are not trained source policies or
independent library replicates. The four learning arms use coefficients, random
coefficients, full actor updates and scratch, with matched fresh critics and SAC
settings. No Meta-World, MiniHack or MuJoCo environment is constructed.

All four initial learners must pass shared preflight before any learning. Check
reference behavior, critic/target hashes, seeds, configuration and precision. The
native squashed Gaussian owns action sampling and log probabilities. Restricted
actors must have coefficient-only optimizers, nonzero coefficient gradients and
unchanged frozen map buffers. Source reuse requires a declared actual source anchor.

After 20 steps, each arm saves a native model ZIP, replay buffer and a serialized
Python/NumPy/Torch CPU RNG plus environment capsule. The capsule is read back from
disk. Restore the same capsule immediately before both five-step continuations;
reload uses `force_reset=False`. Compare actual policy/critic/target values, all
three optimizer states, entropy temperature, replay arrays/pointer/timeouts,
counters, last observations, global RNG and synthetic environment state. Timeout
transitions must retain truncation flags while their sampled bootstrap dones are
zero. Retain final branch states and comparison hashes.

This validates explicit synthetic DummyVecEnv restoration. It does not validate
automatic Monitor restoration or the real benchmark controller. HOSVD map behavior
has separate component tests; this native fixture uses the declared flat family.
A mechanical pass does not establish competent policies, transfer utility or a
paper result.

Before output or native imports, a future approved launch must bind the exact
plan/code commit/hashes, fixed five-minute session, actual decision49 approval
comment, and source-lifecycle evidence file/SHA. The lifecycle gate requires the
original source phase and extension supervisor/child to be terminal, all relevant
recorded PIDs to be gone and no active or waiting known source job. It rejects a
pending handoff; the old r2 process exiting alone is insufficient. The current
approval template stays false with no invented terminal lifecycle evidence.

The process captures UTC and monotonic entry anchors. Its immutable monotonic
allowance is the fixed cutoff minus entry UTC, capped at 300 seconds; setup, hashing
and imports consume it. Later UTC rollback cannot renew the window. Every
constructor, environment step and optimizer entry is guarded, and the output
directory claims the single attempt before runtime imports. Interruptions retain
partial learner state and costs where cleanup can finish. Use a 300-second outer
timeout with `--signal=INT --kill-after=5s`; the additional five seconds permit
cleanup only. A failure or interrupted attempt requires a new explicit decision
before another native run.
