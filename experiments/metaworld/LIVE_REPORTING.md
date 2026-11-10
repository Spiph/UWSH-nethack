# Live research reports

Future approved development, technical retry, continuation and scripted sentinel
CLI runs start an independent Python reporting process before executing the
runner. The earlier experiments had progress logging and deadline supervision;
their report generation was offline. This observer fills that monitoring gap.
Implementation and validation are tracked in
[the live reporting issue](https://github.com/Spiph/UWSH-nethack/issues/60).

The observer reads the manifest and progress JSON every 60 seconds and refreshes
once at shutdown. Reports appear in the run's `live_reports/` directory:

- `report.md` and `report.json`: current outcomes, checkpoint completeness,
  source collection success, costs, protocol findings and artifact verification.
- `monitor_status.json`: observer readiness, freshness, process identity and alerts.
- `sidecar_status.json`: launcher health and shutdown coverage.
- `events.json`: bounded recent observation changes.

The launcher requires readiness before starting scientific work. After startup,
observer failure is recorded separately and does not change the runner's result.
The observer exits after a terminal manifest, a requested final snapshot, or
parent exit. A parent exit without a terminal manifest is flagged as incomplete.
Polling observes new checkpoints within the reporting interval; it does not
establish that an unresponsive training process is healthy.

Alerts cover 300 seconds without observed progress, malformed or inconsistent
evidence, nonfinite diagnostics and failed artifact checks. A competence failure
remains a scientific outcome. Source collection success uses complete 500-step
episodes, with lifetime and recent-100 denominators; it is adaptive training
telemetry rather than frozen-policy evaluation or independent replication.
Historical records without that telemetry are explicitly marked missing or
checkpoint-only. Reports preserve partial checkpoints and distinguish the
original budget from continuation stages and inherited cumulative costs.

The process imports only the Python standard library. It streams published
artifact bytes to verify SHA-256 commitments, caches unchanged sealed files, and
does not deserialize policies or replay buffers. It performs no environment
interactions, tuning, retries or changes to stopping rules. Output is bounded by
the smaller of 1 MiB and one tenth of the run's artifact allowance; that reserved
space is included in the same allowance as scientific artifacts. Fixed output
names, atomic writes and ownership locks protect raw evidence.

For an existing completed run, use a separate output directory so the historical
cohort stays unchanged:

```bash
python -m scripts.metaworld_run_monitor \
  --manifest /absolute/run/development_manifest.json \
  --output /absolute/new-report-directory \
  --once
```

The November 10 validation reports are stored locally under
`artifacts/metaworld-live-reports-2026-11-10/`, with separate directories for the
failed attempt, original phase, continuation and scripted sentinel. These are
read-only observations of saved evidence, not new scientific runs. The issue
records the inventory and validation hashes. Independent AI interpretation is a
separate review activity; this module does not start an unattended AI service or
send notifications to external channels.
