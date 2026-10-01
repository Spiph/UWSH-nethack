# GitHub research workflow

[The research map](https://github.com/Spiph/UWSH-nethack/issues/1) is the entry point and canonical index. Decisions, experiment specifications, progress, failure analyses and result summaries live in issues. Pull requests hold implementation changes. Historical local notes remain provenance, not a second active plan.

## Structure

- The map carries `wayfinder:map` and `epic` labels. Native sub-issues express ownership; the experiment-sequence epic groups execution issues.
- `wayfinder:research` records sourced investigations. `wayfinder:grilling` records decisions requiring human input. `wayfinder:task` records prerequisites or explicitly authorized execution.
- `experiment` identifies experiment/runtime work, `optional` marks secondary branches, and `historical` marks archived plans. `epic` is a repository convention, not a special issue-type requirement.
- Milestones are evidence gates, not promised dates: measurement readiness, Meta-World pilot, confirmatory evidence, and replication/publication.
- Native issue dependencies are authoritative. Do not maintain a competing manual blocked/unblocked label. A closed prerequisite no longer blocks its dependent.

## Find and claim work

Use `gh` for all GitHub operations. The repository is `Spiph/UWSH-nethack`.

```bash
gh issue view 1 --repo Spiph/UWSH-nethack --comments
gh api repos/Spiph/UWSH-nethack/issues/1/sub_issues
gh issue list --repo Spiph/UWSH-nethack --state open --limit 100 \
  --json number,title,labels,assignees,milestone
```

For a candidate issue, inspect its native blockers and comments, then claim it if it is open, unassigned and has no open blockers. Also check prerequisite decisions inherited from its parent epic. Descend into the sequence epic to find its execution children.

```bash
gh api repos/Spiph/UWSH-nethack/issues/ISSUE_NUMBER/dependencies/blocked_by
gh issue view ISSUE_NUMBER --repo Spiph/UWSH-nethack --comments
gh issue edit ISSUE_NUMBER --repo Spiph/UWSH-nethack --add-assignee @me
```

Create children before wiring their relationships. The API requires database IDs, not displayed issue numbers, for relationship payloads:

```bash
gh api repos/Spiph/UWSH-nethack/issues/CHILD_NUMBER --jq .id
gh api --method POST repos/Spiph/UWSH-nethack/issues/PARENT_NUMBER/sub_issues \
  -F sub_issue_id=CHILD_DATABASE_ID
gh api --method POST repos/Spiph/UWSH-nethack/issues/CHILD_NUMBER/dependencies/blocked_by \
  -F issue_id=BLOCKER_DATABASE_ID
```

Use linked titles in prose so the plan is readable without memorizing numbers. Resolve decisions in comments and add a brief linked context pointer to the map; do not copy the entire decision into multiple places. New unresolved decisions become child issues, not silently accepted assumptions.

## Experiment record

Before a run, the issue must specify the hypothesis, closest prior work, estimand, representation, intervention, comparison arms, task/seed split, interaction budget, meaningful effect, analysis/stopping rule and resource authorization. Pilot values remain exploratory until confirmation is frozen.

After a run, add a comment with:

1. Code commit, environment/reward versions, config and initialization hashes.
2. Exact command, hardware, resource use, run/artifact manifest and durable evidence location.
3. Every planned run's outcome, including failures, aborts and deviations.
4. Primary/secondary results, uncertainty at the correct replication unit, and controls.
5. What the result supports, what it does not, and the next decision or blocking issue.

Do not paste credentials, private data or enormous logs into the public tracker. Link appropriately published artifacts; do not label local-only paths as downloadable evidence.

## Pull requests and completion

Use small branches and focused PRs. Separate implementation, configuration freeze, analysis and manuscript changes when that makes review clearer. A PR must identify its issue and validation. Use `Refs #...` for experiment infrastructure; use `Closes #...` only if merging genuinely satisfies the issue's full completion criteria.

A merged trainer does not mean training ran. A completed run does not mean its hypothesis was supported. A new result is not necessarily novel. At the novelty checkpoint, compare the precise claim against closest prior work and competing explanations. If the evidence duplicates prior work or remains confounded, open one focused follow-up rather than escalating the claim.

Keep optional LLM curriculum and Jev selection work separate unless a recorded decision promotes it. GitHub workflow permission does not itself authorize paid compute, campaign launches, submission or publication.
