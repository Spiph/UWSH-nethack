# Source-competence diagnosis: environment and protocol audit

Evidence for [the source-competence investigation](https://github.com/Spiph/UWSH-nethack/issues/55), against repository commit `7c88f1b` and the installed measurement runtime. This is a static source/retained-JSON investigation: **no model/environment construction, simulation, rollout, training, or native issue-49 check was run**. Existing artifacts were not modified.

Comparison with original-run commit `c2cd64fd5fa697618f72d34ea3bf46a553ca8f4b` showed no change to `metaworld_hour_pilot.py`; the only `metaworld_protocol.py` change is the later public controller getter used for continuation. The audited task/reset/action behavior therefore also covers the original phase.

## Finding

**No concrete project protocol defect was identified that explains push/pick-place failing while reach succeeds.** Frozen task installation, visible goals, observation stacking, action coordinates, reward selection, reset behavior, full-length episodes and success aggregation are consistent with the pinned upstream implementation. This is evidence against several specific interface mistakes, not proof that the physical environment or learned behavior is correct.

Reach does not exercise the same success mechanism as the manipulation tasks: its reward measures gripper TCP-to-target distance; push/pick-place require moving the object. Consequently, a working reach learner alone cannot validate object contact, grasping, or learned gripper behavior. See the pinned [reach reward](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/envs/sawyer_reach_v3.py#L140-L162), [push reward](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/envs/sawyer_push_v3.py#L171-L213) and [pick-place reward](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/envs/sawyer_pick_place_v3.py#L251-L303).

## Exact source identity

Installed source root:

`/home/icannon/code/UWSH-nethack-worktrees/metaworld-measurement-validation-2026-10-05/experiments/metaworld/.venv/lib/python3.10/site-packages`

The following installed files were compared byte-for-byte with their official tagged raw GitHub versions on 2026-11-10. Every comparison matched; no imported runtime object was needed.

| Installed file, relative to site-packages | Official tag | SHA256 |
| --- | --- | --- |
| `metaworld/sawyer_xyz_env.py` | `v3.1.1` | `4282580b80befc557f67d25acffffec737ad0dae2c440601be76fc09139f351d` |
| `metaworld/envs/sawyer_push_v3.py` | `v3.1.1` | `17aa389f1e77f2edce2122260ed4df28a6870f152986b4c3cfebf958068b81c4` |
| `metaworld/envs/sawyer_pick_place_v3.py` | `v3.1.1` | `1504b7b5bbfae3719d65bd965d841eedc7ff0c5f755e8c9f3c96a353f5806779` |
| `metaworld/envs/sawyer_reach_v3.py` | `v3.1.1` | `6403bb039f0627c376be11f2157ff48b8747154024effd8c16f19904762c7c47` |
| `gymnasium/envs/mujoco/mujoco_env.py` | `v1.1.1` | `1aef55fa320260143da79ed5ceb170f539de30606187520a7a51a4f704c95924` |
| `gymnasium/wrappers/common.py` | `v1.1.1` | `bc0ebcd8d1cb65286fc2d610c43bd262accfef2f06869460861a9ec0b1d86824` |

Repository references below are relative to `/tmp/ups-source-diagnosis-20261110`; installed-source references use the root above. Official links pin the same versions rather than mutable `main`.

## Interface checks

### Task selection, goals and resets

`scripts/metaworld_protocol.py:281–307` generates MT1 task payloads using a split-derived seed. `TaskController.reset`, lines 345–387, selects a frozen source task or declared evaluation order, draws an independent episode seed, explicitly calls `raw.seed`, installs the task, and verifies its random vector before and after reset. `make_protocol_env`, lines 423–488, uses a raw environment class and a 500-step TimeLimit; it does not install another task-selection wrapper.

Upstream MT1 sets `partially_observable=False`, whereas ML benchmarks hide the goal. `set_task` freezes the supplied random vector and updates observability. Thus this is goal-visible, within-family development; it is not a hidden-goal ML1 adaptation experiment. The serialized random vector includes initial object and target configuration, so “held-out development goal” means a held-out configuration, not only a new target XYZ. Sources: [benchmark task generation/MT overrides](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/__init__.py#L97-L180), [task installation](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/sawyer_xyz_env.py#L275-L319).

The unusual double reset is **upstream behavior**, not an extra project reset. `SawyerXYZEnv.reset` first calls `reset_model`, then Gymnasium's Mujoco reset clears simulator data and calls `reset_model` again. Finally Sawyer reset copies the returned current frame into both observation-frame slots. The project does not call `_get_obs` while recording protocol metadata. Static evidence therefore does not support cross-episode previous-frame leakage. Sources: [Sawyer reset, lines 664–681](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/sawyer_xyz_env.py#L664-L681), [Gymnasium reset, lines 170–185](https://github.com/Farama-Foundation/Gymnasium/blob/v1.1.1/gymnasium/envs/mujoco/mujoco_env.py#L170-L185).

### Observation meaning and visible-goal bounds

The 39-vector is current 18D state, previous 18D state, then target XYZ at `[36:39]`. Current-state `[0:3]` is `data.body("hand").xpos`, `[3]` is normalized finger separation, `[4:7]` is object position, and `[7:11]` is its quaternion for these tasks. **Observed hand position is not the reward's exact TCP**: `tcp_center` averages the two end-effector site positions. Any offline distance using `[0:3]` should be called *hand-object distance*, not TCP distance. Sources: [hand/TCP definitions, lines 67–81](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/sawyer_xyz_env.py#L67-L81), [observation assembly, lines 475–527](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/sawyer_xyz_env.py#L475-L527).

`CorrectedGoalBounds`, `scripts/metaworld_hour_pilot.py:94–121`, changes the declared Box bounds for `[36:39]`; it does not zero, normalize, reorder or clip these returned values. This addresses the initial cached `observation_space` metadata, while upstream `set_task` invalidates its separate cached `sawyer_observation_space`; upstream `step` clips against that recomputed visible-goal space. Source: [cached bounds and step clipping](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/sawyer_xyz_env.py#L537-L640). No project goal-masking operation was found.

### Actions and manipulation reward

Native actions are four floats in `[-1,1]`: XYZ mocap increments use scale `0.01`, and the last component controls opposing gripper actuators. The project does not negate or reorder them. Positive gripper effort is rewarded as closing/caging; the official scripted pick-place policy also chooses positive effort near the object. The scalar observation `[3]` instead measures *opening distance*. Confusing these quantities in a diagnostic would invert its interpretation, but no such project action inversion was found. Sources: [XYZ action, lines 321–335](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/sawyer_xyz_env.py#L321-L335), [gripper actuation, lines 580–595](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/sawyer_xyz_env.py#L580-L595), [official pick-place controller](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/policies/sawyer_pick_place_v3_policy.py#L25-L69).

The project explicitly requests `reward_function_version="v2"`. The `-v3` environment name does not imply another reward version: the pinned task constructors default to v2 and dispatch on that field. One important distinction is that pick-place reports success within **0.07 m**, while its v2 terminal reward plateau uses **0.05 m**; push uses **0.05 m** for success and its final plateau. Rewards also depend on caging/gripping and staged proximity/lift conditions, so return alone is not a competence measure. Sources: [pick-place success, lines 84–114](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/envs/sawyer_pick_place_v3.py#L84-L114), [pick-place shaping, lines 238–303](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/envs/sawyer_pick_place_v3.py#L238-L303), [push shaping/success](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/envs/sawyer_push_v3.py#L84-L213).

### Horizon and success aggregation

Raw Sawyer tasks never terminate on success and truncate at step 500. The outer TimeLimit uses the same horizon, rather than shortening it. `TaskController.step`, lines 389–410, rejects unexpected termination/truncation and ORs the binary success signal across the episode. `scripts/metaworld_development.py:342–383` evaluates deterministic actions, retains complete episodes and uses the 16/20 gate. The extension retains those conventions. The evaluation is not accidentally terminal-step-only success. Sources: [raw return semantics, lines 580–640](https://github.com/Farama-Foundation/Metaworld/blob/v3.1.1/metaworld/sawyer_xyz_env.py#L580-L640), [Gymnasium TimeLimit, lines 112–145](https://github.com/Farama-Foundation/Gymnasium/blob/v1.1.1/gymnasium/wrappers/common.py#L112-L145).

## Retained JSON checks

Read-only inspection of `artifacts/metaworld-development-2026-11-09-r2-extension/development_manifest.json` found, for every task: 50 source and 20 development six-dimensional configurations, no shared configuration IDs between those splits, and identical development goal/episode-seed order across checkpoints. Minimum source/development object-to-target XY separation was respectively: reach **0.1713/0.1578 m**, push **0.1519/0.1620 m**, pick-place **0.1561/0.1510 m**. These satisfy the upstream reset rejection condition of at least 0.15 m; no frozen configuration was caught in that loop. Pick-place development target heights spanned **0.0584–0.2594 m**, within its declared 0.05–0.30 m range. Source: retained JSON plus pinned task `reset_model` methods linked above. These are configuration checks, not evidence that the robot achieved the configurations physically.

## What remains unresolved

The retained JSON contains returns and success histories but not complete per-step contact, caging, lift and TCP diagnostics. The existing one-step preflight and exact 215-transition continuation reconstruction establish narrower facts than “push and grasp mechanics work.” A policy can also optimize shaped reward without achieving object transport; that is a **hypothesis**, not a demonstrated cause of this cohort's failure.

The next read-only discriminator is source replay analysis: verify goal coordinates, hand/object motion, gripper action/separation and reward distributions, while labeling hand-based distances as proxies. Do not reconstruct exact contact/caging rewards from the 39-vector alone.

If new interaction becomes necessary, a small prospective mechanics check is the official scripted push and pick-place controllers on the first frozen source configuration for each task, at most **1,000 transitions total**, logging TCP/object/goal distances, gripper separation/effort, reward components and success. That design is **proposed, not authorized or executed**; it would require a recorded interaction allowance. Passing it would validate those two concrete settings only and would not establish competent learned policies, explain optimization failure, or authorize another training campaign.
