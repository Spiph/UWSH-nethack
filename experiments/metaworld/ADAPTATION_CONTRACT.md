The actor adapter in `scripts/metaworld_sac_adapter.py` targets the pinned SB3
2.6.0 stack. Its pure Torch and fake-interface checks do not validate real SB3
integration. No real SAC construction, target rollout or adaptation smoke has
been executed for this implementation. Such validation needs its own allowance;
the source-development allocation authorizes zero target interactions.

`build_matched_sac` constructs a fresh learner from explicit `SeedStreams`, a
verified affine family, activation and `MatchedSACConfig`. Every config field is
required; none selects an implicit development design. This config is an API
value, not an approved experiment design or budget. The learning arms are
`learned`, `random`, `full` and `scratch`. Learned and random families retain
their distinct estimator provenance. The structured HOSVD random comparator
retains its parameter-factor structure; it is not a uniformly random flat
subspace. Full adaptation starts at the reference cast to float32, exactly as
zero coefficients do. Scratch uses a fresh actor seed and different initial
behavior. The caller must establish the same source-only reference across the
learned, random and full arms. A source anchor that translates a truncated
mean-centered family is a different affine family and must have its own oracle.

The coefficient actor replaces all native actor layers before Adam is created.
Only its coefficient vector is optimized; both raw heads and every actor bias
are decoded. SB3 still owns its squashed diagonal Gaussian, log-standard-deviation
clamp, tanh actions and corrected log probabilities. The actor uses explicit
float32 materialization, retains the original float64 family ID and records the
executed-coordinate hash. Its precision cannot change through `.to(dtype=...)`.
The original `BasisArtifact.oracle_project` operates on the float64 family. For
an oracle diagnostic of the float32 adapter, export the core actor's actual map
with `export_executed_map()` and pass it to the separate privileged
`privileged_oracle_project_executed_map` API. Report coefficient/materialization
rounding tolerances. Privileged target weights never enter the learner builder.

All learning arms use the same declared actor/critic architecture, fresh twin
critics, fresh target critics, empty replay, Adam configuration, update schedule,
entropy temperature initialized at one, and target entropy minus four. Critics
have an isolated initialization seed, independent of the actor's representation.
The training and action-space seeds are applied after construction. Warmup is
SB3's ordinary training interaction stream and counts toward the target budget.
No source critic, optimizer state or replay is reused. Call
`validate_matched_preflight` on all four construction records before enabling
learning; it checks common critics, configuration, streams, activation and the
cast reference of learned/random/full.
The helper also checks random-parent identities, matched coefficient counts and
the full actor's learned-family binding. Zero-dimensional coefficient families
are rejected as learning arms. A `FrozenReferencePolicy` has a nonlearning
prediction interface and no critic or optimizer. `FrozenSourcePolicy` additionally
requires explicit source-policy-anchor provenance; a population mean is not
labeled source reuse.
The builder and frozen interfaces require a 39-dimensional Box observation and
a four-dimensional Box action space bounded exactly by minus one and one.
This keeps SB3's physical-action scaling identical to the source actor's action
coordinates; environment reward, goal and observation semantics remain the
protocol caller's responsibility.

The caller owns protocol goal banks, evaluation RNGs, run deadlines, the
`TargetInteractionLedger`, and all actual resource accounting. Source selection
using a target rollout must charge every probe. Such probes do not silently
advance SAC's warmup counter or populate replay. Record this choice explicitly.
Charge failures, warmup, tuning and evaluation as well as learner transitions;
report source-library construction and storage before claiming amortized gains.
Count fresh critic and entropy parameters separately from actor coefficients.
Construction metadata also counts stored target critics, frozen actor buffers,
state-dictionary tensor bytes and initially empty optimizer state. Measure actual
checkpoint bytes after writing: serialized shared artifacts and metadata also
occupy space. Scratch metadata declares independent native initial behavior and
does not claim its source reference is the initial actor.
Small actor coordinates alone imply neither a small learner nor faster updates.

Persist native SAC checkpoints together with the verified shared family and the
returned construction metadata. `state_sha256` includes frozen tensor buffers
and non-tensor extra-state metadata. For native `SAC.load`, replace the saved
`policy_kwargs` with `verified_policy_kwargs(saved_policy_kwargs, family=...)`
through `custom_objects` before reconstruction. This verifies the supplied
external artifact identity; the coefficient actor additionally validates its
executed map when loading the prefixed state dictionary. Checkpoint reload and
native replay/entropy continuation still require the separately budgeted real
runtime validation.
