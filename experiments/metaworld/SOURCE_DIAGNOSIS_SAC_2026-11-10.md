# Offline SAC audit, 2026-11-10

For [the source-competence investigation](https://github.com/Spiph/UWSH-nethack/issues/55), this audit
found active, finite native SAC updates and no concrete actor/optimizer, RNG,
action-transport or bootstrap implementation defect that explains the failed
push/pick-place competence gate. This narrows the investigation; it does not prove
that the environment is physically correct or that this SAC configuration should
solve those tasks.

Scope: frozen original `c2cd64fd5fa697618f72d34ea3bf46a553ca8f4b`, extension
`52d8124`, installed SB3 2.6.0/Torch 2.3.0 sources, retained ZIP JSON and CPU-only
reads of native optimizer tensors/actor NPZ arrays. No SAC/native actor/env
constructor, rollout, physics, training, benchmark interaction or native49 run
was performed. All artifacts were opened read-only.

## Supported exclusions and limits

| Candidate explanation | Evidence | Conclusion and limit |
| --- | --- | --- |
| Actor overwrite disconnected Adam | SB3 constructs Adam over the live actor. `build_seeded_sac` copies a newly seeded actor's state into that existing object; it never replaces `policy.actor`. Torch's default `assign=False` copies into existing parameters. | No optimizer-disconnection bug in this path. Final actor values and every actor Adam slot also show updates. |
| Actors/critics reset between visits | Original training uses `learn(increment, reset_num_timesteps=False)`. The extension loads checkpoint/replay, verifies serialized policy/optimizer/entropy hashes, and reconstructs the committed environment before learning. | No repeated fresh initialization is indicated. Extension constructor costs remain continuations of the same library. |
| Other tasks/evaluation corrupt learner RNG | Per-task capsules save Python, legacy NumPy, Torch CPU and all CUDA generator states, then restore the caller. Evaluation runs inside RNG preservation; goal/episode/action generators belong to each controller. | The code isolates the intended streams on this single-GPU sequential run. This is not a claim of universal bitwise GPU determinism. |
| Exploration disabled or warmup repeated | Native collection uses uniform actions for its first1,000 transitions and stochastic policy predictions afterward. `use_sde=False` selects ordinary Gaussian SAC exploration. | There is no deterministic-training flag or repeated warmup introduced at100k/250k stage boundaries. |
| Action order/scaling changes in the trainer | Native collection stores scaled actions and executes their native unscaled values. The environment checks four action dimensions with exact[-1,1] bounds. | No coordinate permutation or extra clipping/scale transform appears in this Python path. Physical gripper/controller correctness still needs separate evidence. |
| Timeouts incorrectly zero the value target | Native replay stores terminal observations and timeout flags, and samples `done × (1−timeout)`. The project requires full500-step truncations and enables timeout handling. | The implemented bootstrap path is correct. Existing replay audits supply the empirical timeout check; metadata alone would not prove physical trajectories. |

Primary code for these conclusions:

- Actor copy and seeding: [original protocol lines532–569](https://github.com/Spiph/UWSH-nethack/blob/c2cd64fd5fa697618f72d34ea3bf46a553ca8f4b/scripts/metaworld_protocol.py#L532-L569), [SB3 policy/optimizer construction lines280–310](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/stable_baselines3/sac/policies.py#L280-L310), and [Torch copy/default assignment lines2070–2124](https://github.com/pytorch/pytorch/blob/v2.3.0/torch/nn/modules/module.py#L2070-L2124).
- RNG and training-call guard: [original runner lines97–129](https://github.com/Spiph/UWSH-nethack/blob/c2cd64fd5fa697618f72d34ea3bf46a553ca8f4b/scripts/metaworld_development.py#L97-L129), [lines262–281](https://github.com/Spiph/UWSH-nethack/blob/c2cd64fd5fa697618f72d34ea3bf46a553ca8f4b/scripts/metaworld_development.py#L262-L281), and [lines448–466](https://github.com/Spiph/UWSH-nethack/blob/c2cd64fd5fa697618f72d34ea3bf46a553ca8f4b/scripts/metaworld_development.py#L448-L466).
- Warmup, action transport and one update per transition after warmup: [SB3 off-policy lines327–347](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/stable_baselines3/common/off_policy_algorithm.py#L327-L347), [lines381–407](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/stable_baselines3/common/off_policy_algorithm.py#L381-L407). Terminal-observation storage is at [lines472–501](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/stable_baselines3/common/off_policy_algorithm.py#L472-L501), and timeout masking at [replay lines307–325](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/stable_baselines3/common/buffers.py#L307-L325).
- Extension verification and guard locations in the reviewed tree: `scripts/metaworld_source_continuation.py:159` loads with `force_reset=False`; `:187` verifies checkpoint hashes; `:202` reconstructs the environment; `:210` rechecks learner hashes. The original/extension diff in `metaworld_development.py` changes progress-ledger reporting, not the native SAC loss or update rule; the protocol addition exposes the controller for recovery.

## Retained checkpoint evidence

ZIP JSON confirms unchanged `[256,256]` architecture, batch256, replay50,000,
warmup1,000, one gradient step per collection step, learning rate0.0003,
gamma0.99, tau0.005, `ent_coef="auto"`, target entropy−4 and no SDE. Native SAC
optimizes actor, critic and log-temperature on each update; the frozen config does
not replace that loss. [Native SAC implementation](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/stable_baselines3/sac/sac.py#L199-L290).

| Task/checkpoint | Native updates | Actor/critic/entropy Adam slots | Step in every slot | Alpha | Actor L2 change from initial |
| --- | ---: | --- | ---: | ---: | ---: |
| reach100k |99,000 |8 /12 /1 |99,000 |0.0293603 |80.3045 |
| push1M |999,000 |8 /12 /1 |999,000 |0.296277 |173.0620 |
| pick-place1M |999,000 |8 /12 /1 |999,000 |0.0343373 |157.2819 |

Every inspected optimizer moment tensor is finite; every parameter's second
moment is nonzero. All eight actor tensors, including both heads/biases, changed
from initialization and remain finite. These are evidence of updates, not proof
of useful gradients, calibrated critics or competent behavior.

The pick-place200,215 checkpoint has199,215 updates. Actual recovery reports
show `S=N=C=200215`, no lost transition and no pending optimizer update. Push
resumed with `S=N=C=250000`, also with no pending/lost work. Thus the retained
boundary does not exhibit the one-transition edge cases supported defensively
by recovery code. Poor100k performance also predates extension recovery.

Local roots are
`artifacts/metaworld-development-2026-11-09-r2` and
`artifacts/metaworld-development-2026-11-09-r2-extension`.
The inspected final ZIP byte hashes are:

- reach100k, original root: `41be6cd7b199424cb82bbe3cf6d67ef679613251473a2e7907903006ec85170e`;
- push1M, extension root: `187a786ddd18543dafe76f6d0a7dfaf5529e33d7708afeae3fd6292157d35c10`;
- pick-place1M, extension root: `1f55de63f3ca665673c93fc4028b8a302f25e5a37b50536b8c90fde17fc02273`.

## Unresolved explanations

The50k replay retains roughly100 full episodes, so increasing lifetime steps does
not retain early successful trajectories indefinitely. The coordinator's offline
audit found rare successful source episodes and disappearing high-reward
pick-place transitions between saved buffers. Together these support rare-event
retention/exploration or forgetting as hypotheses. They do not establish that
replay capacity caused failure. With batch256 and one update per transition,
increasing capacity changes retention/diversity, not automatically the expected
steady-state number of gradient exposures per transition.

Deterministic evaluation uses the squashed Gaussian mean; stochastic collection
samples actions. Consequently source successes and development mean-action
success can differ because of action stochasticity, goal support or both.
[Native distribution modes/samples](https://github.com/DLR-RM/stable-baselines3/blob/v2.6.0/stable_baselines3/common/distributions.py#L247-L259).
The observed finite alpha values exclude neither a poor reward/entropy tradeoff
nor optimization failure; they provide no temperature-tuning prescription.

The smallest remaining offline discriminator is to condition saved actor outputs
and replay reward/action statistics on the already identified rare-success
episodes/configurations, keeping checkpoint time and source/development roles
explicit. This can expose support loss or a mean-versus-sampled-action mismatch,
but cannot identify counterfactual returns. Before another SAC campaign, a
separately authorized physical reference-controller check would test an important
explanation this static audit cannot exclude. Any new rollout, seed or training
comparison needs its own recorded numerical allowance.

One unrelated provenance defect was observed: retained reach checkpoint rows in
the extension manifest omit `artifact_root`, while copied unfinished-task rows
include it. Resolving those reach paths against the extension directory fails;
they belong to the original root. This should be corrected in downstream artifact
resolution, without changing the frozen experiment. It does not explain source
learning failure.

## Offline float32 functional parity

A separate authorized offline check compared two arithmetic implementations on
2,048 retained replay observations at each of 10 unique source checkpoints,
including the partial 200,215 checkpoint: 20,480 observations total. All saved
parameters and features were materialized as float32. The comparison used NumPy
matrix operations and CPU `torch.nn.functional.linear`/ReLU/clamp/tanh, with no
native Actor/SAC/env constructor, simulator, sampling or rollout.

| Output | Largest absolute NumPy versus CPU Torch difference |
| --- | ---: |
| Raw mu |0.000004291534423828125 |
| Clamped log standard deviation |0.000001430511474609375 |
| tanh(mu) |0.0000031888484954833984 |

All compared outputs were finite float32. This characterizes rounding on the
retained observations. It does not certify GPU bitwise equality, every possible
observation, action-unscaling arithmetic or trajectory equivalence.

Execution used Python3.10.16, NumPy1.26.4, Torch2.3.0+cu121, CPU Torch threads1,
and `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`
before imports. The original execution receipt is retained under
`/tmp/ups-source-diagnosis-evidence-20261110/`:

- recipe `sac_float32_functional_parity.py`, SHA256
  `222b5a3d58492fec2ea39785aabe93c103e4de3ee8b7b4288358a313d6443eb4`;
- full per-checkpoint result `sac_float32_functional_parity.json`, SHA256
  `8854de72db369bd767ba92978867afd18e7a36da14e954a6fbcff65b2bef29a3`;
- `sac_float32_functional_parity_receipt.json` records those paths/hashes, sample
  counts and maxima. Each result row binds both source file byte hashes and the
  deterministic replay-slot sample hash.

The following portable recipe reproduces the numerical comparison from the
repository root with the retained artifacts. Run it in the pinned interpreter
with the environment variables above. It prints diagnostics and writes no source
artifact. `load_replay` substitutes plain records for native ReplayBuffer/Box
classes; it does not construct them.

```python
import hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from scripts.metaworld_source_diagnosis import load_replay

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
original = Path("artifacts/metaworld-development-2026-11-09-r2")
extended = Path(str(original) + "-extension")
manifest = json.loads((extended / "development_manifest.json").read_text())
max_error = dict.fromkeys(("raw_mu", "clamped_log_std", "tanh_mu"), 0.0)
seen, checkpoints, samples = set(), 0, 0
for task in ("reach-v3", "push-v3", "pick-place-v3"):
    for row in manifest["tasks"][task]["checkpoints"]:
        root = Path(row.get("artifact_root", original))
        actor = root / row["artifacts"]["actor"]["path"]
        if actor.resolve() in seen:
            continue
        seen.add(actor.resolve())
        replay = root / row["artifacts"][".replay.pkl"]["path"]
        for path, key in ((actor, "actor"), (replay, ".replay.pkl")):
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["artifacts"][key]["sha256"]
        with np.load(actor, allow_pickle=False) as archive:
            vector = archive["vector"]
            layout = json.loads(str(archive["layout_json"].item()))
            p = {r["name"]: vector[r["start"]:r["end"]].reshape(r["shape"]).astype(np.float32)
                 for r in layout}
        with replay.open("rb") as reader:
            buffer = load_replay(reader)
        n = buffer.buffer_size if buffer.full else buffer.pos
        indices = np.linspace(0, n - 1, min(2048, n), dtype=np.int64)
        x = buffer.observations[indices, 0].astype(np.float32)
        t = torch.from_numpy(x)
        weights = {name: torch.from_numpy(value) for name, value in p.items()}
        with torch.no_grad():
            for layer in (0, 2):
                w, b = f"latent_pi.{layer}.weight", f"latent_pi.{layer}.bias"
                x = np.maximum(x @ p[w].T + p[b], np.float32(0))
                t = torch.relu(F.linear(t, weights[w], weights[b]))
            mu = x @ p["mu.weight"].T + p["mu.bias"]
            logs = np.clip(x @ p["log_std.weight"].T + p["log_std.bias"], -20, 2)
            tm = F.linear(t, weights["mu.weight"], weights["mu.bias"])
            tl = F.linear(t, weights["log_std.weight"], weights["log_std.bias"]).clamp(-20, 2)
            pairs = ((mu, tm.numpy()), (logs, tl.numpy()), (np.tanh(mu), torch.tanh(tm).numpy()))
        for name, (left, right) in zip(max_error, pairs):
            assert left.dtype == right.dtype == np.float32
            assert np.isfinite(left).all() and np.isfinite(right).all()
            error = np.abs(left.astype(np.float64) - right.astype(np.float64)).max()
            max_error[name] = max(max_error[name], float(error))
        checkpoints, samples = checkpoints + 1, samples + len(indices)
print(json.dumps(dict(checkpoints=checkpoints, samples=samples, max_error=max_error)))
```
