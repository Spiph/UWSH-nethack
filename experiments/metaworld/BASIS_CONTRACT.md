# Offline source-only basis artifacts

This API implements [the fitting and information-access contract](https://github.com/Spiph/UWSH-nethack/issues/34). It does not select a primary estimator, task panel, rank, effect size or run allocation. Those choices remain prospective research decisions. The tests fit synthetic arrays and evaluate tiny CPU networks; they do not read experiment outcomes or train policies.

`scripts.metaworld_basis.SourcePolicy` binds an actor to its task class, independent library, initialization condition, training run, checkpoint, source-goal bank, eight effective RNG streams, complete parameter layout, saved initial array and hashes. `from_actor_npz` accepts the existing `vector`/`layout_json` schema for both final and initial actors, with their original parameter dtype. Named-parameter hashes match `metaworld_geometry.hash_actor`. Source records and fitting specifications have content commitments rechecked before fitting, including nested dictionaries and arrays.

`fit_source_basis` requires the following explicit inputs:

- Source-policy records from exactly one library and initialization condition. Shared initialization requires equal saved arrays/hashes and effective actor seeds; independent initialization requires distinct initial states per training run. Repeated checkpoints need an explicit flag and remain rows from the same training/library replicates.
- An allowed source/development `GoalManifest`, with no actual final-test goal payloads. The artifact binds the entire allowed manifest hash and the source training-bank hashes.
- A `TaskSplitManifest`, separately declaring source classes, development-target classes and named final-target classes or an opaque sealed-final commitment. Within-class goal transfer and held-out task-class transfer have different membership checks. Source actors' own development-goal banks remain allowed in both regimes; those are not held-out task classes. Sealed or unassigned final-class overlap cannot be locally verified and remains a confirmation-freeze requirement.
- A `BasisFitSpec` naming the estimator/revision, representation, rank rule, normalization, raw/aligned coordinates and affine reference. Optional rank/reference selection evidence can reference declared source weights or source/development metric content hashes, never teacher actions, target weights or final data.
- A `TaskRelations` declaration separating benchmark classes from broader skill groups and explicit near/far pairs. Empty relationships mean no grouping claim. Any nonempty mapping needs a prospective decision reference; the implementation supplies no project-specific grouping.

Explicit fold exclusions are checked before alignment, centering, normalization and rank selection. Unknown exclusion IDs fail rather than silently fitting an intended holdout. Source-class eligibility comes from the task-role allowlist. Final-goal payloads and target/teacher weight records are rejected even when they have convenient shapes.

## Estimators and affine references

`flat_population_pca` centers each parameter coordinate over the N retained source rows. Its rank is bounded by both numerical source rank and N−1. Optional source-coordinate standard deviations are folded into the deployed directions. Oracle projection uses Euclidean least squares against those actual directions, rather than silently projecting in whitened coordinates.

`uwsh_v3_parameter_hosvd` follows the parameter-axis and feature-centering conventions of [UWSH v3, Algorithm 1](https://arxiv.org/html/2512.05117v3#alg1). For each tensor, the mean reduces the model axis and all **undecomposed** parameter axes. Each selected parameter axis gets an orthonormal factor estimated from the centered stacked fibers. The model axis estimates the factors and is never compressed. Source explained-variance rank selection follows the paper; fixed declared ranks are an explicitly recorded variant. This object differs from flattened population PCA. A parameter-mode rank need not be bounded by N−1 because its samples include fibers within models.

All uncompressed dimensions remain coefficients, including complete bias/head tensors when their modes are not decomposed. They are not copied from a target actor for free. The coefficient count is the sum of per-tensor core sizes. The implementation covers the complete actor layout.

Both estimators support final weights or deltas from **one stored common base**. Per-initialization deltas are rejected as a deployable family. With elementwise population centering, common-base subtraction cancels algebraically in flat PCA, up to numerical roundoff; it is bookkeeping rather than a separate initialization intervention. Parameter-mode feature centering can retain other base components, so that identity does not automatically extend to HOSVD.

The deployed reference is either the actual estimator center or an exact retained source actor. Choosing a source anchor can translate a truncated centered affine family; it is not promised to be a coordinate change of the original centered family. Zero coefficients reproduce the stored reference exactly. `BasisArtifact.decode` is the sole coefficient-to-policy map used by the separately labeled `oracle_project` diagnostic and future practical adaptation. Oracle target weights do not update the artifact.

## Alignment, persistence and costs

`metaworld_alignment` supports the fixed two-hidden-layer SAC actor with both `mu` and `log_std` heads. Bounded source-only weight matching uses a retained source anchor. Its permutations compensate adjacent layers and both output heads, and check preservation on fixed synthetic feature probes. This verifies the applied symmetry, not the quality of the matching heuristic or downstream transfer. Raw and aligned fits are separate artifacts; raw arrays remain intact.

`artifact.save(path)` preserves existing files and returns logical artifact/reference-map hashes, the actual archive file hash and byte size, and numeric storage accounting. `BasisArtifact.load(path)` checks hashes, complete layouts, task/library provenance, affine reference identity, factor dimensions and storage counts. Archives use NumPy without pickle.

The storage report distinguishes all persisted research arrays from the shared deployment reference/factors and each target's coefficients. Archive bytes include compression and container overhead; array counts alone do not. Critics, optimizer state, source checkpoints and data remain separate costs for the adaptation runner. Tiny coefficients do not imply tiny total storage.

`family_sha256` identifies the **exact stored reference and coefficient map**. Unequal identifiers do not prove that two parameterizations span different affine spaces. Matching identifiers let oracle and practical consumers verify that they use the same map.

Focused synthetic verification:

```bash
PYTHONPATH=src python -m pytest tests/test_metaworld_basis.py
```
