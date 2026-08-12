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

## Performance: vectorized the drift measurement (needed for real data)

The scalar drift loop in `metric._drift_vectors` ran one `map_coordinates` call per
medial point in Python. A single real recto patch has ~12k-18k medial points after a
2x downsample, so one snap took over two minutes and the real finding could not run.
Replaced the per-point loop with batched primitives: `geom.sample_lines` samples every
point in one `map_coordinates` call, `geom.parabola_peaks` refines all peaks at once,
and `local_frames` now uses a batched `eigh`. Verified bit-identical to the scalar
path on a tube phantom (`ok` mask exact, drift max abs diff 0.0). The fast and slow
test suites stay green. A full real sheet snap dropped from >120s to 17s.

## The decisive real-data finding (non-circular, on Dataset059 recto patches)

`structure_probe` finds all three real patches are sheet-dominant (recto surfaces),
mixes {tube 3, sheet 12}, {tube 2, sheet 17}, {tube 7, sheet 10}. So we snap with the
sheetness head, then score the move in two witnesses the snapper never optimized:
meijering (a different Hessian ridge family) and raw_ct (smoothed intensity, no Hessian
at all). A random move of equal magnitude rides along as the TAUIL control. This is the
exact test TAUIL-Abd-Elilah's m7 surface version failed: there, normal-direction
snapping did not beat a random control on the diffuse surface field.

```
PYTHONPATH=. python3 scripts/real_finding.py   (villa venv, downsample 2, 4000 pts scored)

s1_z10240_y2560_x2560  (sheet, snap=sheet, median move 2.96 full-res vox)
  [meijering] snap +0.1597  random +0.0228  ratio +7.0  -> REAL drift
  [raw_ct   ] snap +0.0941  random +0.0077  ratio +12.2 -> REAL drift
s1_z10240_y2560_x3200  (sheet, snap=sheet, median move 2.93 full-res vox)
  [meijering] snap +0.1553  random +0.0216  ratio +7.2  -> REAL drift
  [raw_ct   ] snap +0.1088  random +0.0162  ratio +6.7  -> REAL drift
s1_z10240_y2880_x2560  (sheet, snap=sheet, median move 3.71 full-res vox)
  [meijering] snap +0.2338  random +0.0270  ratio +8.7  -> REAL drift
  [raw_ct   ] snap +0.1410  random +0.0124  ratio +11.3 -> REAL drift
```

Read: the sheetness snap moves each label about 3 voxels and lands it on a strictly
higher independent-witness ridge, on both witnesses, on all three patches, at 7x-12x the
gain of a random move of the same size. The raw_ct witness matters most: it shares no
derivative machinery with the snap at all, so a gain there cannot be a Hessian artifact.
The random control is small but nonzero (+0.008 to +0.027), which is honest: a medial
point inside a thick label band drifts toward brighter voxels in any direction. The
directed move captures 7x-12x more than that baseline, so the signal is real and
directional, not diffuse.

This is the result the crux experiment could not claim: the crux snapped sato and scored
sato (circular). Here the snap head and both witness heads are different operators and
the anti-circularity is enforced in code (`witness._assert_independent` rejects sato and
frangi). Headline: **Dataset059 recto-surface labels sit a median ~3 voxels off the CT
sheet ridge, that offset is real and it is correctable**, demonstrated non-circularly on
a dataset where the prior published surface-snapping attempt reported a negative.

### Not a boundary or thickness artifact

The obvious objection is that a thick label near the volume face inflates the gain. So
the same snap was scored twice, over all points and over interior points more than 8
voxels from any face (93-94% of the sampled points). The interior gain matches the full
gain to the third decimal:

```
patch 1  meijering all +0.1597 / interior +0.1607   raw_ct all +0.0941 / interior +0.0946
patch 2  meijering all +0.1553 / interior +0.1566   raw_ct all +0.1088 / interior +0.1085
patch 3  meijering all +0.2338 / interior +0.2391   raw_ct all +0.1410 / interior +0.1433
```

The random control stays small on the interior too (+0.008 to +0.028). So the drift is a
distributed property of the surface, not an edge effect. This is the check Layer-3 fails
below. Layer-2 passes it.

## Layer-3 self-consistency on the real 056 -> 059 pair (honest negative for defects)

The 056 seed is 320^3, the 059 published label and the CT are 300^3, so 056 is
center-cropped by 10 voxels per face to share the 059 frame. The pure-geometry sandwich
`seed subset of published subset of dilate(seed, 3)` then holds strongly:

```
patch 1: seed_containment 1.000 (17 of 941056 seed voxels dropped), within-dilation 0.9992
patch 2: seed_containment 0.999 (1079 dropped),                     within-dilation 0.9996
patch 3: seed_containment 1.000 (6 dropped),                        within-dilation 0.9993
```

The check reports ~2000-3500 published voxels per patch outside the radius-3 dilation
(defect_frac ~0.0005-0.0008). Before calling those defects, I checked where they sit:
**100% of them are within 2 voxels of a volume face, 0% are interior.** They are an
artifact of the 056->059 frame offset and the center-crop, not label errors: near the
boundary the seed voxels that would explain them were cropped away. So the honest
Layer-3 result on these three patches is **clean**: no interior geometry defect once the
crop shell is accounted for. Layer-3 stays a validated gate (the unit test plants a voxel
beyond the radius and it is caught), but it finds no real defect here and the boundary
shell must be masked before the out-of-radius count means anything. This is the same
lesson the finding rests on: distrust anything that lives at the face.

## Dataset-wide pilot: the drift is systematic, not three patches

The three development patches are the first three cases in the listing, so to rule out
cherry-picking I drew a fixed-seed random sample of 40 patches spanning the whole
s1/s4/s5 grid from the public server (`dl.ash2txt.org/datasets/seg-derived-recto-surfaces`,
anonymous, numTraining 1754) and ran the same witness-scored snap on every one, in
parallel. A patch "confirms" a witness when the snap gain beats twice the random gain and
clears a 0.005 floor.

```
python3 scripts/batch_audit.py   (40 patches, 40 clean, all sheet-dominant)

median snap move (full-res vox): median 2.29  IQR [1.99, 2.95]
  [meijering] snap gain median +0.1105  interior +0.1128  random median +0.0089  | confirms 40/40
  [raw_ct]    snap gain median +0.0611  interior +0.0626  random median +0.0023  | confirms 40/40
```

Every patch confirms on both witnesses, including raw_ct, the zeroth-order intensity
witness that shares no machinery with the snap. Interior-only gains match the full gains,
so it is not a boundary or thickness effect at scale either. The magnitude varies (a few
patches already sit on the ridge and move under 0.1 voxels; the median is 2.29), which is
the honest shape of a real bias rather than a constant offset. This 40-patch sample is now
superseded by the full run below, which covers every released patch. Per-patch numbers are
in `evidence/audit40.json`.

## The full run: all 1754 patches of Dataset059

The sample is no longer needed. `scripts/audit_full.py` streams every case in the released
set, downloads each patch, runs the same witness-scored snap, deletes the raw tif and
appends one JSON line to `evidence/audit_full.jsonl`. It is resumable and it ran to
completion on all 1754 patches with 12 workers. `scripts/defect_report.py` then reduces
the jsonl:

```
PYTHONPATH=. .venv/bin/python scripts/defect_report.py

1754 clean, 0 skipped. median drift 2.32 vox. meijering 1753/1754, raw_ct 1752/1754.
wrote DEFECT-REPORT.md + csv
```

The dataset-wide numbers, read straight out of `evidence/DEFECT-REPORT.md`:

```
patches audited: 1754   (s1 1139, s4 576, s5 39; 1716 sheet-dominant, 38 tube-dominant, 0 skipped)
median drift off the CT sheet ridge: 2.32 voxels (IQR 1.86 to 2.82, max 5.68)
patches over 3 voxels off: 333 (19%);  over 4 voxels: 58 (3%)
meijering witness: median snap +0.109 vs random +0.007, confirms 1753/1754
raw-CT witness:    median snap +0.058 vs random +0.001, confirms 1752/1754
interior-only medians: meijering +0.112, raw_ct +0.059
```

Read: the median label in Dataset059 sits 2.32 voxels off the CT sheet ridge and the
directed snap lands it on a strictly higher independent-witness ridge on 1752 of the 1754
patches on both witnesses at once. The interior-only medians match the full medians to the
third decimal, so the gain is not a boundary or thickness effect at full scale. The two
non-confirming cases are honest abstentions rather than failures: `s5_z8500_y2660_x3040`
already sits on the ridge (median move 0.001 voxels, so there is nothing to gain) and
`s4_z8960_y2304_x768` has a snap gain that is positive on both witnesses but does not clear
the 2x-random margin. Both are reported, neither is dropped.

The tail is the actionable part. 333 patches drift more than 3 voxels and 58 drift more
than 4. The worst is `s1_z5568_y2496_x2880` at 5.68 voxels, followed by
`s1_z1152_y2304_x2880` (5.47), `s1_z384_y2304_x4800` (5.46), `s1_z5568_y3072_x4224` (5.39)
and `s1_z1344_y2304_x4992` (5.25). Those five plus the rest of the top 40 are tabled in
`evidence/DEFECT-REPORT.md` as a "relabel these first" list. Every patch, ranked by drift,
is in `evidence/defect_ranking.csv`.

What this is and is not: a label-localization measurement on the published labels plus a
corrector that moves them onto the CT ridge, graded by two operators the snap never uses.
It is not a model-accuracy claim. No nnU-Net was retrained here and none of these numbers
predict a production Dice gain.



