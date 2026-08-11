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

The snapper optimizes sato. If the drift were then scored with sato, the snap would
be graded on the exact field it chased. So every real-data claim is scored with a
different operator:

- The snapper moves labels using sato (bright ridges, `black_ridges=False`).
- The witness scores the result with meijering, an independent ridge filter and with
  a raw-CT intensity crest that shares no derivative machinery with sato at all.
- Neither witness is frangi, the label producer's own operator.

The separation is enforced in code, not just in comments: `witness.witness_field`
raises if you pass it sato or frangi. The lift probe applies the same rule to its
features and its scoring target.

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

Real Dataset059 patch, independent-witness drift (patch `s1_z10240_y2560_x2560`,
witness at 2x audit downsample):

| witness | contrast at label | higher ridge within 4 vox | move-toward vs random | offset to crest (vox) |
| --- | --- | --- | --- | --- |
| meijering | 1.49x | 97.3% | 10.3x | 2.83 |
| raw-CT crest | 1.19x | 93.4% | 8.3x | 3.46 |

The label sits about 3 voxels off the CT ridge and moving toward it beats a
random-direction move by 8x to 10x. The drift shows up in two independent operators,
so it is not an artifact of one filter. This is measured with operators the snapper
never used and the label generator never used, so it is not circular.

Layer-3 self-consistency (seed 056 vs published 059): the seed is fully contained in
the published label (containment 1.0) and 99.9% of the published label lies within EDT
radius 3 of the seed, exactly the pipeline's stated dilation distance. The few
thousand published voxels beyond that radius are flagged as provable pipeline
inconsistencies with no truth model needed.

## Usage

```bash
pip install -e .

ridgeline demo                         # whole pipeline on synthetic tubes, no download
ridgeline validate --json              # calibration matrix + controls + lift, with gates
ridgeline measure IMAGE.tif LABEL.tif  # independent-witness drift QA on a real patch
ridgeline snap IMAGE.tif LABEL.tif -o corrected.tif
```

`demo` and `validate` need no data. `scripts/fetch_subset.sh` pulls a small dev subset
of Dataset059 for `measure` and `snap`.

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
hand. Verified locally: `pytest` green, `ridgeline validate` passes its gates and the
real-patch witness numbers reproduce on the downloaded Dataset059 patches.
