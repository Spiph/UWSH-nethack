# Reproducing the retained-source diagnosis

The canonical findings and next decision live in [the source-competence investigation](https://github.com/Spiph/UWSH-nethack/issues/55). The companion protocol, SAC and benchmark notes provide source-level evidence. These commands only inspect retained artifacts; they do not run an environment, update a policy, fit a basis or perform transfer.

## NumPy-only replay and history report

Run with Python3.10 and the project's NumPy dependency, from this checkout. Pick a new output directory outside every source artifact root:

```bash
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m scripts.metaworld_source_diagnosis \
  --manifest /home/icannon/code/UWSH-nethack/artifacts/metaworld-development-2026-11-09-r2-extension/development_manifest.json \
  --output /path/to/new-offline-report
```

The tool resolves inherited artifact roots, verifies every referenced replay/actor digest, substitutes data-only records for native replay/space classes, and only permits the required NumPy pickle globals. This is for trusted, hash-verified project artifacts, not a general untrusted-pickle sandbox. A fresh CLI asserts that Torch, SB3, Gymnasium, Meta-World and MuJoCo were never imported. It preserves existing outputs and all input directories.

The report includes all ten checkpoint inventories, including the retained partial checkpoint, nine completed development summaries, full source episode histories, and overlapping replay windows. Replay windows are never summed as new interaction costs. It checks finite coordinates, adjacent transitions, current/previous-frame consistency, episode-reset frames, goal continuity, native timeout masks, reward/action distributions and hand/object proxies. Observation[0:3] is hand position rather than exact TCP; saved replay actions are normalized buffer values rather than a bitwise physical-action trace.

The recorded local report is `artifacts/metaworld-source-diagnosis-2026-11-10/offline-replay/source_diagnosis.json`, SHA-256 `35a7a3b169871ad1ce12599506d08f43a7a710affe52413db532cc8fe5236bd8`. Its input extension-manifest SHA-256 is `2b751ea1d9c4d051c4edb89237987d360924d1b49d75f325ae4adcc472d94b74`. Paths are local-only and do not publish policy artifacts.

## Descriptive figure

The plotting command additionally needs Matplotlib, available in the pinned measurement virtual environment. It imports no model or simulator:

```bash
python scripts/plot_metaworld_source_diagnosis.py \
  --manifest /home/icannon/code/UWSH-nethack/artifacts/metaworld-development-2026-11-09-r2-extension/development_manifest.json \
  --output-prefix /path/to/new-source-learning-figure
```

This writes a standalone PNG, SVG and source-data JSON. The recorded local figure is `artifacts/metaworld-source-diagnosis-2026-11-10/source_learning.png`. Source success blocks contain200 complete episodes per100k steps; development points use the same20 fixed configurations at each checkpoint. Alpha curves show logged snapshots without smoothing. These are descriptive views of one library, not independent replications or causal estimates.

## Validation and scope

Eight fake/subprocess tests cover forbidden native pickle globals, array/frame/timeout checks, exclusion of partial episodes from success denominators, inherited paths, hashes and output preservation. No native constructor or physics is exercised by these tests. The separate CPU-only optimizer and arithmetic checks are documented in the SAC note.

The retained data support poor source learning and sparse high-reward replay support. They do not establish whether exploration, replay capacity, optimization or a physical control problem caused that failure. The separately proposed scripted positive control remains gated by [its actual interaction allowance](https://github.com/Spiph/UWSH-nethack/issues/56).
