1# Universal Policy Subspaces

A publication research plan for testing the Universal Weight Subspace Hypothesis in reinforcement learning

*MiniHack → behaviorally grounded subspaces → NetHack*

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Recommended paper in one sentence</strong></p>
<p>Show that independently learned policies with a fixed architecture occupy a reusable, layer-wise low-dimensional update subspace—and that its coordinates are causally tied to reward-defined behavior—then use that subspace for coefficient-only transfer and continual adaptation.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

**Research synthesis and experimental blueprint**

Prepared 27 July 2026

Scope: online RL, offline sequence policies, hierarchical behavior, and NetHack

# Executive decision

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Verdict</strong></p>
<p>Your line of thinking is directionally strong, but the publishable claim must move one level deeper. Options are temporal abstractions; LoRA is a parameterization of a weight update. They are not substitutes. The novel bridge is to test whether discrete options or continuous HBS behavior coordinates map to shared weight-space coordinates.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## The recommended claim

**Core claim.** For a fixed policy architecture, competent RL solutions learned across tasks, rewards, seeds, and selected algorithms concentrate in a low-dimensional, layer-wise parameter-update subspace beyond what is explained by low-rank parameterization, shared initialization, symmetry, or common state visitation. That subspace supports held-out behavior reconstruction, coefficient-only adaptation, and continual learning.

**Why this is stronger than “LoRA works in RL.”** Recent work already shows that low-rank adapters can fine-tune robotic policies, regularize critics, and store policy libraries efficiently. The scientific novelty is universality, behavioral grounding, and transfer—not parameter efficiency alone.

## Recommended experimental route

1.  Establish the phenomenon under controlled task composition in MiniHack.

2.  Deconfound shared initialization and neural symmetries with independent-from-scratch policy cohorts and symmetry-aware alignment.

3.  Use HBS as a causal coordinate system: compare known reward-mixture coordinates with learned subspace coefficients.

4.  Replicate outside roguelikes in Meta-World+ / Continual World; use Craftax as a scalable open-ended bridge.

5.  Use full NLE, the revised interface, scout and milestone metrics, and Dungeons & Data only after the earlier stages pass.

6.  Treat Decision Transformers as a secondary offline-transformer replication, not the primary evidence.

## Minimum viable paper

- MiniHack: controlled existence, composition, and held-out skill transfer.

- Meta-World+ or Continual World: non-roguelike replication and direct comparison with PaCo / continual baselines.

- NLE or Craftax: open-ended capstone with behavior-space grounding.

- At least one independent-initialization cohort and one shared-base LoRA cohort.

- Functional validation: return, success, action-distribution divergence, and behavior interventions—not weight variance alone.

## The headline contribution

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Best framing</strong></p>
<p>Universal Policy Subspaces: reinforcement-learned behaviors share an architecture-specific parameter geometry that can be measured, interpreted through behavior coordinates, and reused for fast adaptation.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## Contents

- Scientific thesis and falsifiable hypotheses

- How LoRA, options, HBS, Decision Transformers, and behavior distillation relate

- What the UWSH authors’ publication trajectory signals

- Benchmark ladder and selection decisions

- Experimental program, controls, metrics, and statistics

- A concrete LoRA-to-RL / Share-RL design

- Novel NetHack lines and publication strategy

- Execution timeline, risks, and references

# Scientific thesis and hypotheses

The Universal Weight Subspace Hypothesis (UWSH) reports layer-wise, architecture-specific joint subspaces across more than 1,100 trained models and adapters, using zero-centered SVD/HOSVD and held-out reconstruction [<u>\[1\]</u>](https://arxiv.org/abs/2512.05117). Its strongest RL extension is not a modality swap. RL introduces non-stationary state distributions, policy-induced data, actor–critic coupling, partial observability, and behavior that can be intervened on. Those properties turn UWSH into a causal and falsifiable study of learned behavior.

## Primary hypotheses

| **Hypothesis**              | **Falsifiable statement**                                                                                                                                                                         | **Primary evidence**                                                                                                |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------|
| H1 — Spectral concentration | After symmetry-aware alignment, layer-wise policy updates across tasks and seeds have lower cross-validated effective rank and smaller held-out reconstruction error than matched null ensembles. | Rank/variance curves; principal angles; normalized excess compression; held-out weight and function reconstruction. |
| H2 — Reusable adaptation    | A basis learned from training tasks supports coefficient-only adaptation on held-out tasks with performance close to full fine-tuning or task-specific LoRA, at lower parameter and sample cost.  | Return or success ratio; steps-to-threshold; trainable parameters; wall-clock and memory.                           |
| H3 — Behavioral grounding   | Subspace coefficients predict reward-defined behaviors, and coefficient interventions produce systematic changes in milestone visitation or option use.                                           | Cross-validated R² / CCA; zero-shot behavior mixtures; causal coefficient sweeps; ablation of directions.           |
| H4 — Continual saturation   | An online shared basis expands rapidly at first, then saturates as tasks accrue, enabling positive forward transfer with bounded forgetting.                                                      | Residual rank per task; basis-angle drift; forward/backward transfer; forgetting; memory growth.                    |
| H5 — Boundary conditions    | Universality is stronger in shared perception/trunk or actor modules than in task-specific heads and critics, and degrades predictably under architecture, action-interface, or dynamics shifts.  | Module-level rank; actor/critic contrast; within- vs cross-algorithm and cross-dynamics generalization.             |

## What would falsify the paper’s central claim?

- Aligned independent-initialization policies are no more compressible than norm- and spectrum-matched random controls.

- Low weight reconstruction error does not preserve action distributions or return on common evaluation states.

- Held-out coefficient-only adaptation performs no better than an equally sized random basis.

- Apparent behavior alignment disappears when controlling for task labels, performance, or state-visitation distribution.

- All shared structure is confined to a common frozen base or to LoRA’s imposed rank.

## Claims ladder

| **Level**     | **Claim**                                                          | **Publication value**                                 |
|---------------|--------------------------------------------------------------------|-------------------------------------------------------|
| A. Existence  | Policies show non-trivial shared geometry after controls.          | Necessary, not sufficient for a strong paper.         |
| B. Utility    | The geometry supports held-out reconstruction and fast adaptation. | Turns measurement into a method.                      |
| C. Meaning    | Coordinates predict and control behaviors.                         | Distinctive RL contribution; strongest novelty.       |
| D. Continuity | The basis can grow online with transfer and little forgetting.     | Connects directly to Share and continual RL.          |
| E. Scope      | Findings replicate across at least two environment families.       | Supports “policy” rather than “NetHack-only” framing. |

# The conceptual bridge: weights, adapters, and behavior

## Do not equate options with LoRA

**LoRA** restricts a layer update to ΔW = BA while freezing a base weight W₀. An option is a temporally extended policy with an initiation rule, intra-option policy, and termination rule. SOL scales the joint learning of option policies and a controller; HBS replaces a discrete option choice with a continuous combination of reward functions [<u>\[15\]</u>](https://arxiv.org/abs/2509.00338) [<u>\[16\]</u>](https://arxiv.org/abs/2604.24558).

**The bridge.** Use one fixed backbone and express option-, task-, role-, or reward-mixture policies with adapters or shared basis coefficients. Then ask whether the controller’s behavior coordinate can predict which parameter coordinate should be activated.

| **Object**            | **Level**                 | **Mechanism**                                                 | **Use in this paper**                                            | **Main caveat**                                                    |
|-----------------------|---------------------------|---------------------------------------------------------------|------------------------------------------------------------------|--------------------------------------------------------------------|
| LoRA                  | Parameter update          | Low-rank ΔW around a frozen W₀                                | Creates comparable task deltas and efficient adaptation.         | Rank is imposed; raw A/B factors are non-identifiable.             |
| UWSH / EigenLoRAx     | Population geometry       | Principal directions shared across trained weights/adapters   | Defines measurement and a frozen shared basis.                   | Same architecture only; shared bases can confound the claim.       |
| Options / SOL         | Temporal behavior         | Discrete controller selects a sub-policy and duration         | Supplies behavior specialists and a scalable NLE implementation. | Not itself a weight subspace.                                      |
| HBS                   | Continuous behavior       | Controller selects reward-mixture coordinates ρ               | Provides known semantic coordinates for causal alignment.        | Benefits may be exploration rather than long-horizon reasoning.    |
| Decision Transformer  | Offline sequence policy   | Transformer conditioned on return, history, and action tokens | Clean LoRA/UWSH replication with fixed architecture and data.    | May reproduce transformer PEFT without testing online RL dynamics. |
| Behavior distillation | Synthetic training signal | Tiny evolved state–action datasets train policies by BC       | Cheap scratch-policy and initialization diagnostic.              | Current evidence is small continuous control, not NLE scale.       |

## The most novel experiment

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Behavior simplex → weight manifold</strong></p>
<p>Train policies for HBS reward vectors ρ. Recover their layer-wise subspace coefficients c. Fit c = f(ρ) on anchor behaviors; test held-out mixtures and extrapolations. If f is smooth or approximately linear and coefficient interpolation produces the intended behaviors, the study explains what shared directions mean—an open limitation in UWSH.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## LoRA factor identifiability

**Never compare raw LoRA factors across runs.** For any invertible R, BA = (BR)(R⁻¹A), so A and B can rotate and rescale without changing ΔW. Compare the composed update ΔW, its canonical SVD, projection matrices, or Grassmannian principal angles. The same principle applies to permutation, sign, and scale symmetries in independently trained networks.

# What the UWSH authors appear to be building toward

The authors’ publication sequence is unusually coherent: EigenLoRAx recycles existing adapters into a principal basis [<u>\[2\]</u>](https://arxiv.org/abs/2502.04700); UWSH generalizes the empirical claim across model populations [<u>\[1\]</u>](https://arxiv.org/abs/2512.05117); and Share makes the basis evolve continually while reprojecting prior task coefficients [<u>\[3\]</u>](https://arxiv.org/abs/2602.06043). The first author’s publication page groups these under “Efficient & Continual Learning” [<u>\[4\]</u>](https://toshi2k2.github.io/publications/).

| **Work**          | **Scientific move**                                                                                  | **Best RL extension**                                                                          |
|-------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
| EigenLoRAx (2025) | Existing LoRAs share principal directions; learn only coefficients for new tasks.                    | RL analogue: recycle a policy-adapter library into an option/task basis.                       |
| UWSH (2025)       | Architecture-specific weights converge to low-rank joint subspaces across tasks and initializations. | RL test: non-stationarity, actors/critics, state visitation, and independent scratch policies. |
| Share (ECCV 2026) | One shared LoRA basis grows online, reprojects old tasks, and reduces adapter storage.               | Share-RL: sequential skills/rewards/roles with forward transfer and forgetting metrics.        |

## Explicit openings left by the authors

- Interpretability of principal directions: RL supplies reward-defined behaviors and causal rollouts.

- Learning a subspace directly from data rather than requiring many trained predictors: NLD, behavior distillation, or gradient subspaces can test this.

- Task arithmetic inside a universal subspace: compositional MiniHack skills and HBS mixtures create clean tests.

- Cross-architecture comparison: treat as a boundary study, not a promised positive result.

- Whether shared convergence also induces shared biases and failures: rare NetHack milestones and failure modes are a compelling probe.

- Share cannot yet integrate multiple backbone types or perform broad cross-task continual learning; sequential RL tasks directly stress that limitation.

**Signal, not announcement.** I found no author post or released project announcing an RL extension as of 27 July 2026. The evidence instead supports an inference: their program is moving from observation → reusable basis → continual basis. An RL paper should claim the unexplored causal/behavioral territory, not race them on another adapter-compression application.

## Closest RL precedents that narrow the novelty

| **Prior line**                | **What it already shows**                                                                | **How the proposed paper differs**                                                                                 |
|-------------------------------|------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| PaCo                          | Directly learns a policy parameter subspace and task coefficients on Meta-World.         | Must be a central baseline; distinguish discovered universal geometry from jointly designed parameter composition. |
| Policy-gradient subspaces     | Shows low-dimensional, slowly changing high-curvature/gradient structure in PPO and SAC. | Motivates trajectory-level analysis; gradient subspace ≠ population weight subspace.                               |
| Subspace of policies          | Learns a parameter subspace whose members adapt to unseen dynamics.                      | Baseline for online adaptation and diversity.                                                                      |
| Merging Decision Transformers | Averages task-specific DTs; common initialization and Fisher weighting matter.           | Direct evidence that initialization is a confound and model merging is a valid RL comparator.                      |
| Git Re-Basin                  | Aligns permutation-equivalent units before weight-space comparison or interpolation.     | Mandatory for independently initialized MLP/CNN policies.                                                          |

# Benchmark strategy

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Selection principle</strong></p>
<p>Every benchmark must isolate a distinct scientific axis: controlled composition, cross-domain generality, memory, offline sequence modeling, continual transfer, or open-ended scale. More environments are not automatically stronger evidence.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

| **Benchmark**   | **Decision**          | **Scientific axis**                                                                           | **Proposed slice**                                                                               | **Judgment**                                       |
|-----------------|-----------------------|-----------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|----------------------------------------------------|
| MiniHack        | Core / Tier 1         | Controlled skills, composition, procedural variation, NLE-compatible interface.               | 24 tasks across navigation, key–door, combat, inventory/use, lava, and compound skills; 5 seeds. | Best discovery benchmark.                          |
| Meta-World+     | Core / Tier 2         | 50 structured manipulation tasks; direct PaCo and policy-LoRA comparison; continuous control. | MT10 pilot, MT50/selected 20-task publication cohort; use standardized Meta-World+ protocol.     | Best cross-domain replication.                     |
| Continual World | Core if Share-RL      | Sequential tasks, forward transfer, forgetting, capacity constraints.                         | CW20 with randomized task orders and fixed step budgets.                                         | Best continual-learning test.                      |
| Craftax         | Recommended bridge    | Open-ended, procedural, NetHack-like complexity with high-throughput JAX.                     | Classic for iteration; full Craftax for achievement and held-out reward variants.                | Best compute-efficient capstone fallback.          |
| Full NLE        | Capstone / Tier 3     | Long horizon, partial observability, large action interface, roles, rare milestones.          | Revised interface; scout + milestones + dungeon depth; role/race stratification; 3–5 seeds.      | High value, high compute; gate it.                 |
| POPGym / Arcade | Diagnostic            | Isolates memory and partial observability with controlled difficulty.                         | Only if claiming recurrent/transformer subspaces; compare GRU/LSTM/attention modules.            | Use one suite, not both.                           |
| Atari + MGDT    | Secondary             | Offline transformer, multi-game transfer, same architecture, abundant trajectories.           | 8–12 games for LoRA/UWSH pilot; 46-game result only if resources allow.                          | Clean replication, weaker online-RL novelty.       |
| Procgen         | Optional              | Unseen-level generalization under shared visual architecture.                                 | 4–6 games or within-game level distributions.                                                    | Useful, but overlaps Craftax/generalization.       |
| D4RL / Minari   | Optional data control | Offline mixtures, locomotion, antmaze, manipulation.                                          | Use only to compare DT/BC/CQL-style policy deltas under fixed data.                              | Mature and cheap; limited novelty.                 |
| XLand-100B      | Defer                 | 30k tasks and complete learning histories for in-context RL.                                  | Small precomputed slice only; no need for initial paper.                                         | Scale is attractive but distracts from NLE thesis. |

## Recommended benchmark bundle

1.  MiniHack + Meta-World+: minimum evidence for a universal-policy claim.

2.  Add Continual World if the method contribution is Share-RL.

3.  Add Craftax as the compute-efficient open-ended bridge.

4.  Use full NLE as the capstone only after coefficient-only transfer beats random-basis controls.

5.  Use Atari/MGDT only as a secondary architecture/algorithm validation.

## Where Dungeons & Data fits

Dungeons & Data’s NetHack Learning Dataset contains 10 billion state transitions from 1.5 million human games and 3 billion state–action–score transitions from 100,000 AutoAscend trajectories [<u>\[14\]</u>](https://arxiv.org/abs/2211.00539). It is valuable, but it is a data source—not by itself a multi-policy benchmark.

- NLD-NAO is state-only, so it cannot directly train standard BC or Decision Transformer policies without action recovery.

- NLD-AA has action labels, but its trajectories come from one symbolic bot; slicing by role or milestone does not magically create independent policy diversity.

- Use NLD-AA for offline pretraining, action/milestone supervision, and state-buffer evaluation; generate your own policy population for the universality claim.

- Stratify by role, race, alignment, depth, achievement, death cause, and strategy label; report distribution shift explicitly.

## Benchmarks to avoid as primary evidence

- Single-task MuJoCo only: too easy to explain by shared dynamics and architecture.

- Only task-specific LoRAs from one base: shows adapter reuse, not independence from the base.

- Only full NLE score: expensive, noisy, gameable, and hard to diagnose.

- A very broad benchmark sweep with one or two seeds: breadth cannot substitute for geometry controls.

# Experimental program

## Policy population design

| **Cohort**                              | **Construction**                                                    | **Question answered**                                                              |
|-----------------------------------------|---------------------------------------------------------------------|------------------------------------------------------------------------------------|
| A — Shared base / LoRA                  | One competent W₀; task adapters at ranks 1, 2, 4, 8; ≥5 seeds.      | Tests whether independently trained deltas share more structure than imposed rank. |
| B — Shared initialization / full update | Same θ₀; all weights train; task and seed diversity.                | Separates low-rank adapter effects from shared optimization trajectory.            |
| C — Independent initialization          | Different θ₀; same architecture; full training; symmetry alignment. | Most important universality deconfound.                                            |
| D — Algorithm shift                     | PPO vs SAC where applicable; PPO vs BC/DT in offline settings.      | Tests whether subspace is optimizer/objective specific.                            |
| E — Behavior coordinates                | SOL options and HBS anchor/mixed reward vectors.                    | Grounds directions in known behavior variables.                                    |
| F — Continual stream                    | Sequential tasks/rewards/roles; randomized orders.                  | Tests online basis expansion, transfer, and forgetting.                            |

## Phase 0 — Measurement harness and nulls

1.  Reproduce UWSH’s zero-centered layer-wise SVD/HOSVD on a small public adapter set and one RL policy family.

2.  Implement canonical LoRA comparison on ΔW = BA; never stack raw A/B across runs.

3.  Implement weight matching / Git Re-Basin for MLP and CNN policies; record unresolved symmetries for recurrent modules.

4.  Build a common evaluation-state buffer and compute policy KL, action agreement, feature CKA, and return after reconstruction.

5.  Validate against random, shuffled, untrained, and independently oriented low-rank null ensembles.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Stage gate 0</strong></p>
<p>Proceed only if the analysis distinguishes a genuinely shared population subspace from trivial single-matrix low rank and preserves policy function after reconstruction.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## Phase 1 — MiniHack discovery study

- Tasks: navigation, key–door, monster avoidance/combat, potion or wand use, lava crossing, inventory identification, and compound tasks.

- Population: 24 tasks × 5 seeds for shared-init and independent-init cohorts; begin with 12 × 3 for the pilot.

- Architecture: one fixed encoder + recurrent core + policy/value heads; analyze modules separately.

- Holdouts: individual tasks, entire skill families, and compositions not seen during basis extraction.

- Primary test: frozen basis + learned coefficients versus full fine-tuning, task LoRA, random basis, and PaCo-style composition.

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Stage gate 1</strong></p>
<p>Advance when held-out coefficient-only adaptation reaches a pre-registered fraction of full fine-tuning (for example ≥90% return/success) and materially exceeds a size-matched random basis with confidence intervals excluding zero.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

## Phase 2 — Cross-domain replication

- Meta-World+ MT10 pilot, then selected MT50/20-task cohort; compare PaCo, multi-head sharing, full fine-tuning, and LoRA.

- If continual learning is central, run CW20 with several task orders and report forward transfer, backward transfer, forgetting, and parameter growth.

- Use Craftax Classic to iterate rapidly; add full Craftax for achievement-space and open-ended behavior tests.

- Optional POPGym study isolates whether recurrent-memory weights share a subspace across memory demands.

## Phase 3 — Behavioral grounding with SOL/HBS

1.  Choose 4–6 semantically distinct reward components, including exploration/scout and progression proxies.

2.  Train anchor policies at simplex vertices and mixed policies in the interior.

3.  Recover subspace coefficients c for each policy; fit linear, kernel, and small hypernetwork maps f: ρ → c.

4.  Test held-out mixtures, extrapolations, and zero-shot coefficient interpolation.

5.  Intervene on individual directions and measure changes in visitation, risk, inventory behavior, depth, and milestones.

6.  Compare discrete option adapters, continuous HBS-conditioned adapters, and a flat coefficient-conditioned policy.

## Phase 4 — NLE capstone

Use the revised NLE interface: richer tokenization, explicit menu and inventory interaction, full action support, and progress metrics centered on scout rather than raw score [<u>\[17\]</u>](https://iclr-blogposts.github.io/2026/blog/2026/revisiting-the-nle/). HBS already demonstrates that reward-mixture hierarchy can improve NLE exploration, while its own analysis cautions that the gain is not necessarily long-horizon reasoning [<u>\[16\]</u>](https://arxiv.org/abs/2604.24558).

- Report scout, unique milestone visitation, dungeon depth, experience, branch visitation, survival, and ascension; score is secondary.

- Stratify results by role/race/alignment and include median plus tail metrics.

- Use 3 seeds for expensive screening and 5 seeds for final selected configurations.

- Pretrain from NLD-AA or symbolic strategy labels, then compare full online adaptation, LoRA, universal-basis coefficients, and HBS-conditioned coefficients.

- Publish checkpoints, adapters, basis factors, alignment maps, and state buffers so others can test the geometry without retraining billions of steps.

# Measurement, controls, and statistics

## Weight-space construction

For layer ℓ and policy i, define *ΔWᵢˡ = Wᵢˡ − W₀ˡ* when a common base exists. For scratch policies, first align symmetries to a reference or compute symmetry-invariant subspace measures. Stack centered layer updates across tasks and seeds, then extract row/column or tensor-mode bases on training policies only. Select rank k by cross-validated explained variance and functional reconstruction—not by a single in-sample threshold.

**Coefficient model.** *ΔWˡ(c) = Σⱼ cⱼ Qⱼˡ*, where Qⱼˡ are shared basis matrices. A stricter LoRA-form alternative shares layer factors and learns task coefficients, but the unconstrained basis-matrix form is cleaner for testing existence.

## Primary metrics

| **Axis**                 | **Measures**                                                                                                                                   |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| Geometry                 | Cross-validated effective rank; explained-variance curve; principal angles; projection-matrix distance; subspace stability vs number of tasks. |
| Null-normalized evidence | Observed reconstruction or rank minus matched random / shuffled baseline, normalized by null variance.                                         |
| Function preservation    | Action-distribution KL; action agreement; feature CKA; value error on a common state buffer; rollout return.                                   |
| Adaptation               | Area under learning curve; steps to threshold; final return/success; trainable parameters; memory; wall-clock.                                 |
| Behavior                 | Milestone visitation vector; scout; depth; option occupancy; reward-component returns; coefficient–behavior R² / CCA.                          |
| Continual                | Forward transfer, backward transfer, forgetting, average performance, residual basis growth, and order sensitivity.                            |

## Mandatory negative controls

| **Control**                         | **Purpose**                                                                    |
|-------------------------------------|--------------------------------------------------------------------------------|
| Random Gaussian                     | Match layer shape and Frobenius norm.                                          |
| Spectrum-matched random orientation | Preserve each update’s singular values but randomize singular vectors.         |
| Independent low-rank matrices       | Same rank budget as LoRA, independent orientations.                            |
| Untrained networks                  | Same architecture and initialization distribution.                             |
| Element/layer shuffle               | Destroy cross-model orientation while preserving marginals.                    |
| Task-label permutation              | Tests coefficient–behavior associations.                                       |
| Shared-base vs independent-base     | Quantifies how much geometry comes from W₀.                                    |
| Aligned vs unaligned scratch models | Measures permutation-symmetry artifacts.                                       |
| Performance-matched subsets         | Avoids competent policies clustering merely because failed policies are noisy. |
| State-buffer controls               | Evaluate on common, per-policy, and out-of-distribution states.                |

## Critical confounds and remedies

| **Confound**                      | **Failure mode**                                                       | **Remedy**                                                                                       |
|-----------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| LoRA is low-rank by construction  | A joint basis can look compact even when orientations are independent. | Compare against independent spectrum-matched low-rank nulls; evaluate held-out cross-task reuse. |
| Shared W₀                         | All deltas share the same tangent neighborhood.                        | Include full-update and independent-init cohorts; report both absolute weights and deltas.       |
| Permutation/sign/scale symmetries | Functionally identical networks can appear far apart.                  | Weight matching, canonical SVD, invariant projection metrics, and function-space checks.         |
| Non-stationary visitation         | Similar states can induce similar features/weights.                    | Common state buffers, cross-policy rollouts, and controlled MiniHack distributions.              |
| Actor–critic coupling             | Critic/reward structure can dominate shared trunks.                    | Analyze encoder, actor, critic, recurrent core, and heads separately; detach when feasible.      |
| Head/action-interface mismatch    | Different actions or heads invalidate weight comparison.               | Fixed canonical action space; revised NLE interface; compare trunk-only when tasks differ.       |
| Task quality                      | Poor adapters enlarge residual rank.                                   | Performance filtering plus sensitivity analyses across quality quantiles.                        |
| Multiple comparisons              | Many layers/ranks/metrics invite post-hoc claims.                      | Pre-register primary layers, thresholds, rank rule, and aggregate hypothesis tests.              |

## Statistical plan

- Use task as the main generalization unit; do not treat every state or parameter as an independent sample.

- Report bootstrap confidence intervals over tasks and seeds, with hierarchical bootstrap when both vary.

- Use permutation tests for subspace overlap and coefficient–behavior association.

- Fit saturation curves versus number/diversity of source tasks; evaluate on completely held-out task families.

- Choose one primary variance threshold (for example 95%) and one pre-registered performance-retention threshold; show full curves in appendices.

- Correct layer-wise confirmatory tests or aggregate layer evidence before testing.

# A concrete LoRA-to-RL and Share-RL design

## Policy adapters

1.  Start from a competent multi-task or curriculum policy W₀, then freeze it.

2.  Insert LoRA into linear layers of the visual/glyph encoder, recurrent or transformer core, and policy/value MLPs; keep output heads fixed when action spaces match.

3.  Train task-, option-, role-, or reward-vector-specific adapters with identical budgets.

4.  Compose ΔW = BA and extract a layer-wise shared basis from training adapters.

5.  For held-out tasks, freeze W₀ and the basis; optimize only coefficients c and, in a controlled ablation, a small residual rank.

6.  Measure performance and geometry separately for actor and critic; recent work suggests low-rank critic constraints may behave differently from actor adaptation.

## Universal Option Adapter

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Method sketch</strong></p>
<p>Replace a bank of independent option networks with one frozen backbone and a shared set of basis updates. A controller selects coefficients c (and duration) instead of only an option ID. Discrete c recovers option selection; continuous c creates an adapter keyboard analogous to HBS.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

- Discrete baseline: one LoRA per SOL option.

- Shared-basis baseline: reconstruct each option LoRA from common Q₁…Qₖ plus coefficients.

- Continuous method: controller emits c directly, with norm/simplex constraints and temporal persistence.

- HBS-aligned method: map reward vector ρ to c and test unseen mixtures.

- Residual expansion: add a basis direction only when gradient or reconstruction residual exceeds a threshold.

## Share-RL continual update

1.  Maintain a basis Q and stored coefficients for prior tasks, without storing prior trajectories.

2.  On a new task, train coefficients plus a temporary residual adapter of small rank.

3.  Merge the residual into Q with incremental SVD/HOSVD and reproject old coefficients analytically.

4.  Optionally fine-tune only the new coefficients after reprojection.

5.  Evaluate task-order robustness, basis growth, forgetting, forward transfer, and recovery from rare-task outliers.

## Baselines

| **Baseline**                    | **Role**                                                                         |
|---------------------------------|----------------------------------------------------------------------------------|
| Full fine-tuning                | Upper performance reference; highest trainable/storage cost.                     |
| Task-specific LoRA              | Primary PEFT baseline at matched ranks.                                          |
| Random frozen basis             | Essential test of learned basis value.                                           |
| PaCo                            | Directly learned policy-parameter subspace and task compositions.                |
| Joint multi-task policy         | Tests whether one conditioned policy removes the need for parameter composition. |
| EigenLoRAx-style basis          | Offline shared basis, coefficient-only new-task adaptation.                      |
| Share-style incremental basis   | Continual adapter compression and transfer.                                      |
| SOL / HBS                       | Behavior hierarchy baselines with discrete or continuous reward coordinates.     |
| Weight averaging / task vectors | Training-free composition; especially relevant for DT branch.                    |
| Distillation / BC               | Tests whether function transfer explains weight-space effects.                   |

## Recent evidence that LoRA is viable in RL

The engineering premise is now credible: policy-library experiments report similar success with large storage reductions [<u>\[27\]</u>](https://arxiv.org/abs/2606.25700); a 2026 critic study freezes randomly initialized dense backbones and trains low-rank updates as structural regularization [<u>\[28\]</u>](https://arxiv.org/abs/2604.18978); and SLowRL reports rank-1 sim-to-real policy adaptation on a quadruped with safety constraints [<u>\[29\]</u>](https://arxiv.org/abs/2603.17092). These works strengthen feasibility while raising the novelty bar.

# Decision Transformers, behavior distillation, and other alternatives

## Decision Transformer: valuable secondary thread

Decision Transformer casts RL as return-conditioned sequence modeling [<u>\[10\]</u>](https://arxiv.org/abs/2106.01345). Multi-Game Decision Transformer trains one model across up to 46 Atari games and adapts to held-out games [<u>\[11\]</u>](https://arxiv.org/abs/2205.15241), while prior work has already merged task-specific Decision Transformers in parameter space [<u>\[9\]</u>](https://arxiv.org/abs/2303.07551).

- Why include it: fixed transformer architecture, large offline policy populations, exact LoRA analogue, and easy state-distribution control.

- Best experiment: one base DT + game/task LoRAs; recover a shared adapter basis; adapt to held-out games by coefficients only.

- Strong control: independent DT initializations plus Git Re-Basin/representation comparison.

- Why it should not lead: it can be dismissed as another transformer PEFT result and does not test policy-induced online data.

- Position: cross-architecture/algorithm appendix or second benchmark family after the online-policy result.

## Behavior distillation: a diagnostic, not the NLE engine

HaDES evolves tiny synthetic state–action datasets that train competitive continuous-control policies from scratch and generalize across architectures and hyperparameters [<u>\[12\]</u>](https://arxiv.org/abs/2406.15042).

- Use it to cheaply produce many scratch policies under different initializations and test whether data-induced behavior recovers the same subspace.

- Use synthetic datasets as a counterfactual: same behavior signal, different optimizer/architecture.

- A useful future method is to distill a task family into examples that directly identify subspace coefficients.

- Do not rely on current HaDES evidence to claim scalability to NetHack; the gap is substantial.

## Other methods that strengthen the claim

| **Method**                      | **Experiment**                                                                                     | **Why it helps**                                                           |
|---------------------------------|----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| Gradient-subspace tracking      | Compare learned weight basis with leading policy-gradient/Hessian directions over training.        | Links population geometry to optimization dynamics.                        |
| Function-space distillation     | Distill many experts into coefficient-conditioned policy; compare function and weight compression. | Tests whether weight universality adds value beyond ordinary distillation. |
| Task arithmetic / model merging | Add, subtract, or interpolate policy update vectors after alignment.                               | Causal compositional test on MiniHack skills and HBS mixtures.             |
| Hypernetwork from behavior code | Map task/reward/milestone code to shared coefficients.                                             | Model-independent/data-derived direction that UWSH leaves open.            |
| World-model module analysis     | Analyze encoder, dynamics, reward, actor, and critic subspaces separately.                         | Tests whether universality lives in dynamics rather than policy.           |
| Symbolic strategy supervision   | Use AutoAscend internal strategies as option labels or teachers.                                   | Separates action hierarchy from low-level control; grounded NLE skills.    |

# Novel NetHack directions

## Priority 1 — Behavioral simplex to weight manifold

**Question.** Does the continuous reward simplex learned by HBS induce a smooth, low-dimensional manifold of policy updates? This is the cleanest combination of your HBS idea and UWSH.

- Train anchor and interior reward vectors; recover parameter coordinates.

- Predict held-out adapters from reward vectors without environment interaction.

- Interpolate coefficients during a rollout and test behavior continuity and temporal stability.

- Compare linear maps, small hypernetworks, and direct coefficient optimization.

## Priority 2 — Universal Option Adapter

**Question.** Can a shared parameter basis replace a discrete option-policy bank while retaining SOL’s throughput and HBS’s expressivity?

- Treat option-specific adapters as the policy population used to discover the basis.

- Make the high-level controller choose coefficients and duration.

- Evaluate storage, throughput, option diversity, exploration, and compositional held-outs.

## Priority 3 — Offline-to-online NLD basis

- Pretrain a shared backbone on NLD-AA with action and symbolic strategy supervision.

- Build adapters for roles, achievements, branches, or policy-quality strata.

- Extract the offline basis, then perform coefficient-only online adaptation with the revised NLE interface.

- Compare an offline basis, an online basis, and a hybrid basis; quantify state-distribution shift.

## Additional publishable extensions

| **Direction**                   | **Experiment**                                                                                | **Value**                                                                                 |
|---------------------------------|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Milestone task arithmetic       | Learn skill/branch directions and compose them for held-out multi-stage goals.                | Strong if composition works; negative results still diagnose nonlinearity.                |
| Outlier-triggered rank growth   | Expand the basis only when rare milestones produce large gradient/reconstruction residuals.   | Connects universal structure with rare-skill exceptions and compute allocation.           |
| Actor–critic asymmetry          | Shared actor basis with task-specific or low-rank-regularized critics.                        | May reveal a principled boundary of universality.                                         |
| Hybrid symbolic–neural adapters | AutoAscend strategy tree selects neural adapter coefficients for low-level control.           | Pragmatic NLE progress; tests whether symbolic abstractions align with weight directions. |
| Failure-direction analysis      | Identify directions associated with starvation, menu loops, risky combat, or shallow farming. | Addresses UWSH’s question about shared biases/failures.                                   |
| Interface subspaces             | Compare old action/observation interface with revised full interface.                         | Tests whether ‘universal’ geometry is actually an interface artifact.                     |

## What a novel NLE approach should not do

- Optimize raw in-game score as the only objective; it rewards shallow farming.

- Use a truncated interface that makes inventory, menus, or role-specific behavior impossible.

- Assume hierarchy solves long-term credit assignment; HBS’s evidence points primarily to exploration.

- Scale to tens of billions of frames before controlled MiniHack evidence establishes the mechanism.

- Hide symbolic priors or action macros; report exactly where environment knowledge enters.

# Publication strategy and execution plan

## Recommended contribution package

1.  A symmetry-aware protocol for measuring shared RL weight subspaces across tasks, seeds, and initialization regimes.

2.  Evidence of non-trivial universal policy subspaces in MiniHack and a non-roguelike family.

3.  Behavioral grounding via HBS reward coordinates and coefficient interventions.

4.  A coefficient-only transfer or Share-RL method with strong random-basis, LoRA, PaCo, and continual baselines.

5.  An NLE case study using the revised interface and progress metrics.

## Ablation matrix

| **Axis**         | **Levels**                                       | **Scientific purpose**                  |
|------------------|--------------------------------------------------|-----------------------------------------|
| Initialization   | Common W₀; common θ₀; independent θ₀             | Separates tangent-neighborhood effects. |
| Parameterization | Full update; LoRA ranks 1/2/4/8; random basis    | Separates imposed from discovered rank. |
| Alignment        | None; weight matching; function-only             | Quantifies symmetry artifacts.          |
| Module           | Encoder; recurrent core; actor; critic; heads    | Locates universality.                   |
| Task shift       | Reward; dynamics; observations; action interface | Defines scope and failure boundaries.   |
| Algorithm        | PPO; SAC; BC/DT where feasible                   | Tests objective/optimizer dependence.   |
| Basis size       | Fixed k; EV threshold; adaptive residual growth  | Tests capacity and saturation.          |
| Behavior code    | Discrete option; HBS ρ; learned task embedding   | Tests semantic grounding.               |

## Go / no-go milestones

| **Gate**           | **Pass criterion**                                                          | **Failure response**                                                                       |
|--------------------|-----------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| Pilot geometry     | Shared subspace beats spectrum-matched null and survives alignment.         | If not, narrow to shared-base adapters or write a methodological negative result.          |
| Held-out utility   | Coefficient-only adaptation beats random basis and approaches LoRA/full FT. | If not, do not claim reuse; focus on descriptive geometry.                                 |
| Behavior grounding | ρ predicts c and interventions shift behavior.                              | If nonlinear, report manifold/hypernetwork result; if absent, drop interpretability claim. |
| Cross-domain       | Effect replicates in Meta-World+/Craftax.                                   | If not, frame as architecture/environment-specific boundary.                               |
| NLE scaling        | Earlier gates pass and capstone shows transfer or exploration value.        | If compute-limited, use Craftax capstone and NLE as targeted case study.                   |

## Indicative 10-month schedule

| **Time**   | **Deliverable**                                                                           |
|------------|-------------------------------------------------------------------------------------------|
| Month 1    | Reproduce UWSH analysis; implement canonical ΔW, nulls, alignment, and function metrics.  |
| Months 2–3 | MiniHack pilot; choose architecture, task families, ranks, and pre-registered thresholds. |
| Months 4–5 | Full MiniHack populations; held-out basis adaptation; PaCo/LoRA/random-basis baselines.   |
| Month 6    | Meta-World+ / Continual World replication; actor–critic and algorithm ablations.          |
| Month 7    | SOL/HBS behavior-simplex experiment and coefficient interventions.                        |
| Months 8–9 | Craftax and/or gated NLE capstone; NLD offline-to-online study.                           |
| Month 10   | Final statistics, release artifacts, paper writing, and reproducibility audit.            |

## Compute discipline

- Use MiniHack and Craftax for rank/architecture sweeps; reserve NLE for selected configurations.

- Store checkpoints sparsely but consistently across training so trajectory subspaces can be studied.

- Analyze layer samples before full-tensor decompositions; use randomized SVD where appropriate.

- Release policy populations incrementally; checkpoint generation is a durable research asset.

- Precompute common state buffers and cache action distributions for functional comparisons.

## Candidate titles

- Universal Policy Subspaces: Shared Weight Geometry Across Reinforcement-Learned Behaviors

- From Behavior Simplices to Weight Subspaces in Reinforcement Learning

- Share-RL: Continual Reinforcement Learning in a Common Policy Subspace

- Universal Option Adapters for Open-Ended Reinforcement Learning

## Likely venue fit

**ICLR / NeurIPS / ICML:** appropriate if the paper includes symmetry-aware evidence, causal behavior grounding, cross-domain replication, and a useful method. RLC is a strong fit if the result is primarily an RL mechanism or NLE advance. A MiniHack-only descriptive study is more likely workshop-scale.

## The first three experiments to run

<table>
<colgroup>
<col style="width: 100%" />
</colgroup>
<thead>
<tr class="header">
<th><p><strong>Immediate sequence</strong></p>
<p>1) MiniHack: 12 tasks × 3 seeds, shared-base LoRA and independent-init full policies, with spectrum-matched nulls. 2) Held-out coefficient-only adaptation versus task LoRA and random basis. 3) A small HBS-style 3-reward simplex where reward coordinates are hidden from the subspace analysis and predicted afterward.</p></th>
</tr>
</thead>
<tbody>
</tbody>
</table>

# References and primary sources

\[1\] Kaushik et al. The Universal Weight Subspace Hypothesis. arXiv:2512.05117 (2025). [<u>https://arxiv.org/abs/2512.05117</u>](https://arxiv.org/abs/2512.05117)

\[2\] Kaushik et al. EigenLoRAx: Recycling Adapters to Find Principal Subspaces for Resource-Efficient Adaptation and Inference. arXiv:2502.04700 (2025). [<u>https://arxiv.org/abs/2502.04700</u>](https://arxiv.org/abs/2502.04700)

\[3\] Kaushik et al. Shared LoRA Subspaces for almost Strict Continual Learning. arXiv:2602.06043; ECCV 2026. [<u>https://arxiv.org/abs/2602.06043</u>](https://arxiv.org/abs/2602.06043)

\[4\] Prakhar Kaushik. Publications and research themes. [<u>https://toshi2k2.github.io/publications/</u>](https://toshi2k2.github.io/publications/)

\[5\] Hu et al. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685 (2021). [<u>https://arxiv.org/abs/2106.09685</u>](https://arxiv.org/abs/2106.09685)

\[6\] Sun et al. PaCo: Parameter-Compositional Multi-Task Reinforcement Learning. NeurIPS 2022. [<u>https://arxiv.org/abs/2210.11653</u>](https://arxiv.org/abs/2210.11653)

\[7\] Schneider et al. Identifying Policy Gradient Subspaces. ICLR 2024. [<u>https://proceedings.iclr.cc/paper_files/paper/2024/hash/1292cf2ff215e3c857c34c32336413a5-Abstract-Conference.html</u>](https://proceedings.iclr.cc/paper_files/paper/2024/hash/1292cf2ff215e3c857c34c32336413a5-Abstract-Conference.html)

\[8\] Gaya et al. Learning a Subspace of Policies for Online Adaptation in Reinforcement Learning. NeurIPS 2021. [<u>https://arxiv.org/abs/2110.05169</u>](https://arxiv.org/abs/2110.05169)

\[9\] Lawson and Qureshi. Merging Decision Transformers: Weight Averaging for Forming Multi-Task Policies. arXiv:2303.07551. [<u>https://arxiv.org/abs/2303.07551</u>](https://arxiv.org/abs/2303.07551)

\[10\] Chen et al. Decision Transformer: Reinforcement Learning via Sequence Modeling. arXiv:2106.01345. [<u>https://arxiv.org/abs/2106.01345</u>](https://arxiv.org/abs/2106.01345)

\[11\] Lee et al. Multi-Game Decision Transformers. NeurIPS 2022. [<u>https://arxiv.org/abs/2205.15241</u>](https://arxiv.org/abs/2205.15241)

\[12\] Lupu et al. Behaviour Distillation. ICLR 2024. [<u>https://arxiv.org/abs/2406.15042</u>](https://arxiv.org/abs/2406.15042)

\[13\] Samvelyan et al. MiniHack the Planet: A Sandbox for Open-Ended Reinforcement Learning Research. NeurIPS Datasets & Benchmarks 2021. [<u>https://arxiv.org/abs/2109.13202</u>](https://arxiv.org/abs/2109.13202)

\[14\] Hambro et al. Dungeons and Data: A Large-Scale NetHack Dataset. NeurIPS 2022. [<u>https://arxiv.org/abs/2211.00539</u>](https://arxiv.org/abs/2211.00539)

\[15\] Henaff et al. Scalable Option Learning in High-Throughput Environments. arXiv:2509.00338; ICML 2026. [<u>https://arxiv.org/abs/2509.00338</u>](https://arxiv.org/abs/2509.00338)

\[16\] Matthews et al. Hierarchical Behaviour Spaces. arXiv:2604.24558 (2026). [<u>https://arxiv.org/abs/2604.24558</u>](https://arxiv.org/abs/2604.24558)

\[17\] Matthews et al. Revisiting the NetHack Learning Environment. ICLR Blogposts 2026. [<u>https://iclr-blogposts.github.io/2026/blog/2026/revisiting-the-nle/</u>](https://iclr-blogposts.github.io/2026/blog/2026/revisiting-the-nle/)

\[18\] Küttler et al. The NetHack Learning Environment. NeurIPS 2020. [<u>https://arxiv.org/abs/2006.13760</u>](https://arxiv.org/abs/2006.13760)

\[19\] Piterbarg et al. NetHack is Hard to Hack. NeurIPS 2023. [<u>https://proceedings.neurips.cc/paper_files/paper/2023/hash/764ba7236fb63743014fafbd87dd4f0e-Abstract-Conference.html</u>](https://proceedings.neurips.cc/paper_files/paper/2023/hash/764ba7236fb63743014fafbd87dd4f0e-Abstract-Conference.html)

\[20\] Yu et al. Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning. CoRL 2019/2020. [<u>https://proceedings.mlr.press/v100/yu20a.html</u>](https://proceedings.mlr.press/v100/yu20a.html)

\[21\] McLean et al. Meta-World+: An Improved, Standardized RL Benchmark. arXiv:2505.11289 (2025). [<u>https://arxiv.org/abs/2505.11289</u>](https://arxiv.org/abs/2505.11289)

\[22\] Wołczyk et al. Continual World: A Robotic Benchmark for Continual Reinforcement Learning. NeurIPS 2021. [<u>https://arxiv.org/abs/2105.10919</u>](https://arxiv.org/abs/2105.10919)

\[23\] Matthews et al. Craftax: A Lightning-Fast Benchmark for Open-Ended Reinforcement Learning. ICML 2024. [<u>https://proceedings.mlr.press/v235/matthews24a.html</u>](https://proceedings.mlr.press/v235/matthews24a.html)

\[24\] Cobbe et al. Leveraging Procedural Generation to Benchmark Reinforcement Learning. arXiv:1912.01588. [<u>https://arxiv.org/abs/1912.01588</u>](https://arxiv.org/abs/1912.01588)

\[25\] Morad et al. POPGym: Benchmarking Partially Observable Reinforcement Learning. ICLR 2023. [<u>https://arxiv.org/abs/2303.01859</u>](https://arxiv.org/abs/2303.01859)

\[26\] Fu et al. D4RL: Datasets for Deep Data-Driven Reinforcement Learning. arXiv:2004.07219. [<u>https://arxiv.org/abs/2004.07219</u>](https://arxiv.org/abs/2004.07219)

\[27\] Lyngset et al. Memory-Efficient Policy Libraries with Low-Rank Adaptation. arXiv:2606.25700 (2026). [<u>https://arxiv.org/abs/2606.25700</u>](https://arxiv.org/abs/2606.25700)

\[28\] Zhuang et al. Low-Rank Adaptation for Critic Learning in Off-Policy Reinforcement Learning. arXiv:2604.18978 (2026). [<u>https://arxiv.org/abs/2604.18978</u>](https://arxiv.org/abs/2604.18978)

\[29\] Daneshmand et al. SLowRL: Safe Low-Rank Adaptation Reinforcement Learning. arXiv:2603.17092 (2026). [<u>https://arxiv.org/abs/2603.17092</u>](https://arxiv.org/abs/2603.17092)

\[30\] Ainsworth et al. Git Re-Basin: Merging Models modulo Permutation Symmetries. ICLR 2023. [<u>https://arxiv.org/abs/2209.04836</u>](https://arxiv.org/abs/2209.04836)

\[31\] Nikulin et al. XLand-100B: A Large-Scale Multi-Task Dataset for In-Context Reinforcement Learning. arXiv:2406.08973. [<u>https://arxiv.org/abs/2406.08973</u>](https://arxiv.org/abs/2406.08973)

\[32\] Paglieri et al. BALROG: Benchmarking Agentic LLM and VLM Reasoning on Games. ICLR 2025. [<u>https://arxiv.org/abs/2411.13543</u>](https://arxiv.org/abs/2411.13543)

\[33\] Hambro et al. Insights from the NeurIPS 2021 NetHack Challenge. NeurIPS Competition Track 2022. [<u>https://proceedings.mlr.press/v176/hambro22a.html</u>](https://proceedings.mlr.press/v176/hambro22a.html)

\[34\] Facebook Research. Official SOL implementation, including HBS and revised NLE interface code. [<u>https://github.com/facebookresearch/sol</u>](https://github.com/facebookresearch/sol)

## Source assessment note

The plan prioritizes papers, official project pages, conference proceedings, and official repositories. Blogposts are used where they introduce concrete NLE interface changes or reveal an author’s research direction. Claims about future work are labeled as explicit paper limitations or as inference; no unreleased RL paper by the UWSH authors was found in the sources reviewed as of the preparation date.
