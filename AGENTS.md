# Research workflow

GitHub is the canonical tracker for this repository's research planning, decisions, experiments and evidence discussions. Start with [the research map](https://github.com/Spiph/UWSH-nethack/issues/1), including its comments and native sub-issues, using `gh`.

## Before working

- Read the map and the relevant issue, comments, native parent and blocking dependencies. Claim an unassigned, unblocked issue before starting. Do not duplicate an existing issue or another agent's claim.
- Use the Wayfinder skill when available at `skills/skills/engineering/wayfinder/SKILL.md`, with its GitHub tracker adapter. Follow the map's explicit execution scope: decisions use Wayfinder; implementations and experiments use linked issues and focused pull requests.
- The accepted direction includes Meta-World, initialization controls, functional reconstruction, held-out actor adaptation and a focused MiniHack replication. The objective is novel, defensible results, not a promised positive finding.
- Local `.scratch/research-refresh` and `docs/research/2026-10-01-*` records are historical snapshots after migration. Do not maintain a second local backlog. The original broad research plan is background, not the current execution contract.
- Preserve existing uncommitted changes and experiment artifacts. Use an isolated branch/worktree when changes overlap unrelated work.

## Evidence and completion

- Record design decisions and their rationale in issue comments. Human decisions require actual user input; do not manufacture acceptance or treat proposed sample sizes as approved budgets.
- Freeze confirmatory design before inspecting outcomes. Keep pilot and historical cohorts distinct. Fit bases and all data-dependent transforms on source/development data only.
- Distinguish geometric reconstruction, oracle behavioral retention and practical held-out adaptation. Use fresh rollouts and independent policy-library/training replicates; episodes and checkpoints are not independent replications.
- Retain failures and report source training, target interactions, tuning, storage and evaluation costs. An implementation PR does not complete an experiment or prove novelty.
- Link each implementation PR to its issue. Experiment issues close only after their completion criteria and evidence record are satisfied; use `Refs` rather than `Closes` when runs remain outstanding.
- Store reproducible code/configs in commits and large artifacts in an explicitly designated location with hashes/manifests. Put durable planning and result summaries in GitHub issues, not only chat.
- Do not launch paid services, unbudgeted training campaigns, optional Jev/curriculum work, submissions or public artifact releases without the relevant authorization.

## Code quality checks

- Fix actionable lint, formatting, type-check, test, and pipeline-check failures instead of ignoring or bypassing them. If a check cannot run because of an explicit resource or environment constraint, say so and run the strongest relevant local alternative.
- Avoid linter suppressions such as `noqa` for unused imports. For intentional import side effects, make the dependency explicit in executable code (for example, call the registration function) so the import is genuinely used. Add a narrow, explained suppression only when no sound code-level fix exists.

See [the GitHub research workflow](.github/RESEARCH_WORKFLOW.md) for tracker commands and reporting conventions.
