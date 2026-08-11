# ridgeline build report

Producer-agnostic label-localization QA plus a gated ridge snapper for Vesuvius
Challenge segmentation labels (Dataset059). This report is appended as work lands,
with real command output pasted verbatim. No number here is hand-written.

Head commit at start of this session: `f4c9440` (Ridge engine, NRD metric, snapper,
synthetic harness). Environment: villa venv, numpy 2.4.6, scipy 1.18.0, skimage
0.26.0, sklearn 1.9.0.

## What was already verified clean (inherited, re-checked)

Layer-0 synthetic backbone. Re-ran the k=2 planted-shift cell and the null controls
on entry to confirm the inherited core still works before adding to it:

```
k=2 cell: {'kind': 'tube', 'planted': 2, 'pre_snap_error': 2.001, 'nrd_drift': 2.125,
 'ridge_recovery': 0.138, 'brightest_recovery': 1.582, 'random_recovery': 2.527,
 'support_fraction': 1.0, 'half_thickness': 1.0}
null_controls: {'idempotence_pre_error': 0.053, 'idempotence_median_move': 0.157,
 'idempotence_post_error': 0.162}
```

Read: a 2-voxel planted offset (pre-snap error 2.001) is recovered by the ridge
snapper to 0.138 voxels, while snap-to-brightest lands at 1.582 and a
random-direction move of equal magnitude lands at 2.527. Idempotence holds: snapping
an already-clean label moves it 0.157 voxels and leaves it at 0.162.

<!-- APPEND-BELOW -->
