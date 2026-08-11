# ridgeline

Producer-agnostic label-localization QA and a gated ridge snapper for Vesuvius
Challenge segmentation labels (Dataset059, the seg-derived recto-surface patches).

The label geometry in Dataset059 is produced by a deterministic pipeline that
binarizes a hand-traced seed, dilates it by a fixed radius and runs a Frangi ridge
over the dilated mask. It never reads CT grey values. So a structure estimate
computed from the CT is genuinely independent of how the label was placed and "how
far does the label sit from the CT structure" is a fair, non-circular question.
ridgeline measures that drift, snaps the label onto the CT ridge where a crisp ridge
exists and proves the correction on synthetic data with an exact answer.

## The method

Normal Ridge Drift (NRD) is measured in Hessian ridge-response space, not raw
intensity. The CT under a fiber label is only a few grey levels above the shell, so
an intensity argmax lands on noise while the second-derivative ridge is strong and
sign stable. Per medial point the tool samples the ridge field along the
cross-structure normal, refines the peak to sub-voxel with a parabola fit, rejects
flat or weak peaks and records the signed offset. The snapper moves accepted points
to the sub-voxel peak, caps the displacement, smooths the displacement field over the
medial graph so no point jumps to a neighbour structure, then re-voxelizes.

### The separation that keeps it honest

The engine picks a head from the structure of each label component: sato vesselness
for rod-like tubes, a Descoteaux/Antiga sheetness for plate-like surfaces. Dataset059's
recto patches are sheet-dominant, so the snap there runs on the sheetness head. If the
drift were then scored with that same head, the snap would be graded on the exact field
it chased. So every real-data claim is scored with a different operator:

- The snapper moves labels with the structure's own head (sheetness for the recto
  surfaces, sato for tubes), bright ridges, `black_ridges=False`.
- The witness scores the result with meijering (an independent ridge filter from a
  different family) and with a raw-CT intensity crest that shares no derivative
  machinery with the snap at all.
- Neither witness is frangi, the label producer's own operator.

The separation is enforced in code, not just in comments: `witness.witness_field` raises
if you pass it sato or frangi. `witness_scored_snap` raises if the witness head equals
the snap head. The lift probe applies the same rule to its features and its scoring
target.

### Four layers of evidence

- Layer 0 (synthetic, exact truth): plant a known defect on a synthetic tube, recover
  it, score against the planted geometry. This calibrates the instrument.
- Layer 2 (real data, the drift signal): score the label against an independent CT
  witness (meijering / raw-CT), never sato and never frangi.
- Layer 3 (free provable defects): re-derive the deterministic label geometry from its
  056 seed and gate the published label against it, with no truth model.
- Downstream lift (D-lift Regime A): a small logistic-regression probe on local CT
  features shows a learner trained on snapped labels recovers most of the accuracy a
  planted corruption cost, with a null control and a random-motion control.

## Numbers, all produced by running the code

Synthetic planted-offset recovery, 2-voxel normal shift, recovery in voxels (lower is
better), from `ridgeline validate`:

| method | recovery (vox) |
| --- | --- |
| pre-snap error | 2.001 |
| ridge snap (this tool) | 0.138 |
| snap-to-brightest baseline | 1.582 |
| random-direction control | 2.527 |
| frangi arm (producer's own operator) | 1.161 |

The ridge snap recovers the offset to 0.14 voxels while every baseline stays near or
above 1 voxel. The frangi arm recovers to 1.16 and diverges from the ridge snap by
1.27 voxels, so the sato witness is doing something the producer's frangi does not, it
is not re-running their idea.

Controls, on an already-clean label:

- idempotence: snapping a clean label moves it 0.157 voxels and leaves it at 0.162.
- random-direction: a 2-voxel random move takes the same clean label to 1.617 voxels
  of error. A random move breaks a good label, the snap preserves it.

Downstream lift probe (Regime A, held-out Dice of the probe prediction vs the clean
planted reference, features are sato-free):

| probe trained on | held-out Dice |
| --- | --- |
| clean reference | 0.825 |
| corrupted label (2-vox shift) | 0.641 |
| snapped label | 0.810 |
| random-move label | 0.637 |

Recovery fraction 0.92: the snap undid 92% of the planted damage as a learner sees it.
Null control delta -0.005 (snapping a clean label barely changes the probe). Random
control recovery -0.02 (a random move does not recover). All three D-lift gates hold.

Real Dataset059, the dataset-wide drift finding. All recto patches are sheet-dominant,
so the snap runs on the sheetness head and the move is scored on two witnesses the snap
never used. On a fixed-seed random sample of 40 patches spanning scrolls s1, s4 and s5
(`scripts/batch_audit.py`, 2x audit downsample):

| witness | median snap gain | median random gain | patches confirming |
| --- | --- | --- | --- |
| meijering (independent Hessian) | +0.110 | +0.009 | 40 / 40 |
| raw-CT crest (no Hessian at all) | +0.061 | +0.002 | 40 / 40 |

Median move onto the ridge is 2.29 voxels (IQR 1.99 to 2.95). Every patch confirms on
both witnesses, including the raw-CT crest that shares no machinery with the snap, so the
gain cannot be an artifact of one filter. Restricting to interior medial points more than
8 voxels from any face gives the same gain, so it is not a boundary or thickness effect.
The labels sit systematically about 2 to 3 voxels off the CT sheet ridge, measured with
operators the snapper never used and the label generator never used. This is not circular,
and it is not three cherry-picked patches. Per-patch numbers are in `evidence/audit40.json`.

Layer-3 self-consistency (seed 056 vs published 059): the seed is contained in the
published label (containment 1.0) and 99.9% of the published label lies within EDT radius
3 of the seed, the pipeline's stated dilation distance. The check reports a few thousand
published voxels outside that radius, but 100% of them sit within 2 voxels of the volume
face: they are an artifact of the 056 to 059 frame offset and the center-crop, not label
errors. So Layer-3 finds no interior geometry defect on these patches. It stays a
validated gate (the unit test plants a voxel beyond the radius and it is caught). The
honest result here is clean once the crop shell is masked.

## Usage

```bash
pip install -e .

ridgeline demo                         # whole pipeline on synthetic tubes, no download
ridgeline validate --json              # calibration matrix + controls + lift, with gates
ridgeline measure IMAGE.tif LABEL.tif  # independent-witness drift QA on a real patch
ridgeline snap IMAGE.tif LABEL.tif -o corrected.tif
```

`demo` and `validate` need no data. To reproduce the dataset-wide finding, pull the
sampled patches and run the audit:

```bash
python3 scripts/download_sample.py       # 40 patches from dl.ash2txt.org (anonymous)
python3 scripts/batch_audit.py           # the 40/40 witness-scored result above
```

## Honest limitations

- The lift probe is a hand-feature sklearn proxy for learnability. It is orders of
  magnitude smaller than a production 3D nnU-Net. Its numbers do NOT predict a
  production model gain and the tool never claims a "+X% nnU-Net Dice".
- On real published labels there is no ground truth. The witness shows the label is
  closer or further from an independent CT signal, never that it is "correct".
- The planted recovery fraction calibrates the snapper's operating range. It does not
  measure how wrong the real labels are. The real drift magnitude is estimated
  separately from the Layer-2 distance distribution, with wider error bars.
- The snapper targets small offsets. It is honest at k <= 2 to 3 voxels and abstains
  where local CT contrast is too low to form a peak (reported as coverage).
- Layer 3 re-derives the exact dilation geometry. The ridge-threshold step of the
  producer's pipeline is scale-sensitive without the original script, so the
  geometric sandwich carries the claim, not a byte-exact ridge reconstruction.

## AI disclosure

AI assistance (Claude, Anthropic) was used in developing this tool. The design, review
and verification were done by the author. Every number in this README and in
BUILD-REPORT.md was produced by running the code in this repository, not written by
hand. Verified locally: `pytest` green (6 tests), `ridgeline validate` passes its gates,
and the dataset-wide finding reproduces from `scripts/download_sample.py` then
`scripts/batch_audit.py` (40 of 40 patches confirm on both independent witnesses).
