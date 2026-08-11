"""Real-data independent-witness scorer (Layer 2) and Layer-3 self-consistency.

The snapper optimizes SATO. Scoring a real-data drift claim with sato would be
circular: the snap chases the exact field it is then graded on. So the witness
here scores with a DIFFERENT operator, never sato and never a re-run of the label
producer's own frangi generator (detect_ridges_3d / edt_frangi). Two independent
witnesses are offered:

  - meijering: a Hessian ridge filter from a different family than sato, the
    engine's declared independent secondary.
  - raw_ct: a smoothed raw-CT intensity ridge crest. Zeroth-order intensity, so it
    shares no derivative machinery with sato at all. Weak (the fiber core is only a
    few grey levels above the shell) but genuinely orthogonal, used to confirm the
    drift survives an operator change.

The separation is enforced in code (`_assert_independent`), not just asserted in a
comment: passing the snapper's own head raises.

The finding this module carries: snap the label with the snapper (sato), then score
the move in the INDEPENDENT witness field. If the sato-snap lands the label on a
higher independent ridge and a random move of equal magnitude does not, the drift is
real and is not an artifact of the sato operator or of the label's frangi generator.
A random-direction control (TAUIL discipline) rides along on every call.
"""
import numpy as np
import scipy.ndimage as ndi
from skimage.filters import meijering, frangi
from .engine import normalize
from .snapper import snap

SNAP_HEAD = "sato"                                  # what the shipped snapper optimizes
WITNESS_HEADS = ("meijering", "raw_ct")             # allowed independent witnesses
PRODUCER_HEAD = "frangi"                             # the label generator's own operator


def _assert_independent(witness_head, snap_head=SNAP_HEAD):
    """Enforce the anti-circularity contract in code. The witness may not be the
    snapper's own operator and may not be the label producer's frangi generator."""
    if witness_head == snap_head:
        raise ValueError(
            f"witness head {witness_head!r} is the snapper's own operator; "
            "scoring drift with it is circular")
    if witness_head == PRODUCER_HEAD:
        raise ValueError(
            "frangi is the label producer's own operator, not an independent witness")
    if witness_head not in WITNESS_HEADS:
        raise ValueError(f"witness head must be one of {WITNESS_HEADS}, got {witness_head!r}")


def witness_field(ct, witness_head="meijering", sigmas=(1, 2, 3, 4), smooth=1.5):
    """Build the independent witness ridge field, scaled to [0, 1]."""
    _assert_independent(witness_head)
    ct = ct.astype(np.float32)
    if witness_head == "meijering":
        w = meijering(ct, sigmas=tuple(float(s) for s in sigmas), black_ridges=False)
    elif witness_head == "raw_ct":
        # smoothed raw intensity: a bright fiber core is a local intensity maximum at
        # the fiber scale. No Hessian, no vesselness, so fully orthogonal to sato.
        w = ndi.gaussian_filter(ct, smooth)
    return normalize(w)


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def _sample(field, coords):
    shape = np.array(field.shape)
    c = np.clip(np.round(coords).astype(int), 0, shape - 1)
    return field[c[:, 0], c[:, 1], c[:, 2]]


def _downsample(ct, mask, factor):
    """2x-audit downsample (DESIGN section 3): filter cheaply at low res. Returns the
    downsampled CT and mask; distances measured here are scaled back by `factor`."""
    if factor <= 1:
        return ct, mask
    z = 1.0 / factor
    ct_d = ndi.zoom(ct.astype(np.float32), z, order=1)
    mask_d = ndi.zoom(mask.astype(np.float32), z, order=0) > 0.5
    return ct_d, mask_d


def witness_drift(ct, mask, witness_head="meijering", sigmas=(1, 2, 3, 4),
                  step=2.0, r_peak=4, n_sample=4000, epsilon=0.2, seed=0, downsample=2):
    """Layer-2 drift QA on a real label, scored with an independent operator.

    Reports, all in the witness field W (distances scaled to full-res CT voxels):
      - contrast of W at the label vs the cropped volume mean,
      - fraction of label voxels with a strictly higher W within r_peak voxels,
      - gain moving `step` voxels along the W gradient (toward the independent ridge)
        vs a random move of equal magnitude (the TAUIL control), with the ratio,
      - median unsigned offset from the label to the thresholded W crest (the drift
        magnitude), gated against the Layer-0 calibrated epsilon.

    This is descriptive drift, computed WITHOUT the snapper, so it cannot be circular
    with the sato snap. `witness_head` may not be sato or frangi (enforced)."""
    _assert_independent(witness_head)
    rng = np.random.default_rng(seed)
    n_label_full = int(mask.astype(bool).sum())
    ct, lbl = _downsample(ct.astype(np.float32), mask.astype(bool), downsample)
    W = witness_field(ct, witness_head=witness_head, sigmas=sigmas)

    gz, gy, gx = np.gradient(ndi.gaussian_filter(W, 1.0))
    grad = np.stack([gz, gy, gx], axis=-1)

    idx = np.array(np.where(lbl)).T
    n_label = len(idx)
    if n_label > n_sample:
        idx = idx[rng.choice(n_label, n_sample, replace=False)]

    W_here = _sample(W, idx)
    g = grad[idx[:, 0], idx[:, 1], idx[:, 2]]
    W_snap = _sample(W, idx + step * _unit(g))
    W_rand = _sample(W, idx + step * _unit(rng.normal(size=g.shape)))

    Wmax = ndi.maximum_filter(W, size=2 * r_peak + 1)
    higher_nearby = float((Wmax[idx[:, 0], idx[:, 1], idx[:, 2]] > W_here + 0.05).mean())

    # drift magnitude: distance from each label voxel to the nearest witness crest,
    # scaled from downsampled voxels back to full-res CT voxels.
    crest = W > np.percentile(W[lbl], 75)
    dist_to_crest = ndi.distance_transform_edt(~crest)
    offs = dist_to_crest[idx[:, 0], idx[:, 1], idx[:, 2]] * downsample
    median_offset = float(np.median(offs))

    snap_gain = float((W_snap - W_here).mean())
    rand_gain = float((W_rand - W_here).mean())
    ratio = snap_gain / (abs(rand_gain) + 1e-6)
    return {
        "witness_head": witness_head,
        "snap_head": SNAP_HEAD,
        "downsample": downsample,
        "label_voxels": n_label_full,
        "sampled": int(len(idx)),
        "witness_at_label": round(float(W_here.mean()), 4),
        "witness_volume_mean": round(float(W.mean()), 4),
        "contrast_x": round(float(W_here.mean() / (W.mean() + 1e-9)), 3),
        "higher_within_%dvox_frac" % r_peak: round(higher_nearby, 4),
        "grad_gain": round(snap_gain, 4),
        "random_gain": round(rand_gain, 4),
        "grad_vs_random_ratio": round(ratio, 2),
        "median_offset_to_crest": round(median_offset, 3),
        "epsilon": epsilon,
        "exceeds_epsilon": bool(median_offset > epsilon),
    }


def witness_scored_snap(ct, mask, witness_head="meijering", wit_sigmas=(1, 2, 3, 4),
                        snap_sigmas=(1, 2, 3), radius=6.0, n_sample=4000, seed=0,
                        downsample=2):
    """The real-data finding, non-circular by construction. Snap the label with the
    SATO snapper, then score the move in an INDEPENDENT witness field (default
    meijering). If the sato-snap lands the label on a higher independent ridge and a
    random move of equal magnitude does not, the drift is real and not a sato artifact.

    Returns witness gain at the snapped vs original label plus the random control."""
    _assert_independent(witness_head)
    rng = np.random.default_rng(seed)
    ct, msk = _downsample(ct.astype(np.float32), mask.astype(bool), downsample)
    W = witness_field(ct, witness_head=witness_head, sigmas=wit_sigmas)

    corrected, snapped_pts, info, det = snap(ct, msk, sigmas=snap_sigmas, radius=radius)
    orig_pts = det["points"]
    move = snapped_pts - orig_pts
    mag = np.linalg.norm(move, axis=1)

    # subsample medial points for a bounded score
    n = len(orig_pts)
    sel = np.arange(n)
    if n > n_sample:
        sel = rng.choice(n, n_sample, replace=False)

    W_orig = _sample(W, orig_pts[sel])
    W_snap = _sample(W, snapped_pts[sel])
    rand = _unit(rng.normal(size=(len(sel), 3))) * mag[sel][:, None]
    W_rand = _sample(W, orig_pts[sel] + rand)

    snap_gain = float((W_snap - W_orig).mean())
    rand_gain = float((W_rand - W_orig).mean())
    ratio = snap_gain / (abs(rand_gain) + 1e-6)
    return {
        "witness_head": witness_head,
        "snap_head": SNAP_HEAD,
        "downsample": downsample,
        "medial_points": int(n),
        "scored": int(len(sel)),
        "median_snap_move": round(float(np.median(mag)) * downsample, 3),
        "witness_at_orig": round(float(W_orig.mean()), 4),
        "witness_at_snapped": round(float(W_snap.mean()), 4),
        "witness_gain_snap": round(snap_gain, 4),
        "witness_gain_random": round(rand_gain, 4),
        "snap_vs_random_ratio": round(ratio, 2),
        "verdict": ("sato-snap moves label onto the independent ridge, beats random"
                    if snap_gain > 2 * abs(rand_gain) and snap_gain > 0.005
                    else "no independent-witness gain over random"),
    }


# --- Layer 3: self-consistency re-derivation (FREE provable structure, no truth) ----
#
# Dataset059's label geometry is produced by a deterministic, CT-blind pipeline
# (edt_frangi_label.py): threshold_0.5(detect_ridges_3d(dilate_by_inverse_edt(seed, 3)))
# with dilation_distance=3, gauss_sigma=2, sigma=6, ridge_threshold=0.5. The 056 patch
# is the raw seed, 059 the dilated+ridged expansion.
#
# The robust re-derivation is the geometry, not the ridge threshold. The first step,
# dilate_by_inverse_edt(seed, 3), is exact and reproducible with no free constant, so
# the published label must satisfy a pure-geometry sandwich:
#     seed  subset of  published  subset of  dilate(seed, 3)
# Both bounds are CT-blind and truth-free. A seed voxel missing from the published
# label, or a published voxel beyond EDT radius 3 of the seed, is a provable pipeline
# inconsistency with no truth model needed. The ridge-threshold step is deliberately
# not used as a gate here: normalize+threshold=0.5 is sensitive to the original
# script's exact scaling, which is not in hand, so it under-reproduces and would only
# add noise. `derive_edt_frangi` is kept for a best-effort ridge reconstruction, but
# the sandwich carries the Layer-3 claim.

def dilate_by_inverse_edt(seed, distance=3):
    """EDT dilation: every voxel within `distance` of the seed. This is exactly the
    producer's dilate_by_inverse_edt step, no free constant."""
    seed = seed.astype(bool)
    return ndi.distance_transform_edt(~seed) <= distance


def derive_edt_frangi(seed, dilation=3, gauss_sigma=2.0, frangi_sigma=6.0, thresh=0.5):
    """Best-effort reconstruction of the full producer label from a binary seed with
    the stated constants: dilate by EDT radius, gaussian pre-smooth, frangi at scale
    (bright ridges), normalize, threshold. The threshold step is scale-sensitive
    without the original script, so this under-reproduces; the sandwich in
    layer3_selfconsistency is the robust check."""
    dilated = dilate_by_inverse_edt(seed, dilation).astype(np.float32)
    smoothed = ndi.gaussian_filter(dilated, gauss_sigma)
    ridge = frangi(smoothed, sigmas=(frangi_sigma,), black_ridges=False)
    ridge = normalize(ridge)
    return ridge >= thresh


def layer3_selfconsistency(seed, published, dilation=3):
    """Re-derive the deterministic geometry from the 056 seed and gate the published
    059 label against it with no CT and no truth model. Reports the pure-geometry
    sandwich (seed subset of published subset of dilate(seed, dilation)) and counts the
    provable inconsistencies: seed voxels dropped by the label, and published voxels
    beyond the stated dilation radius."""
    seed = seed.astype(bool)
    published = published.astype(bool)
    n_seed = int(seed.sum())
    n_pub = int(published.sum())

    seed_kept = int((seed & published).sum())
    seed_containment = seed_kept / (n_seed + 1e-9)
    seed_dropped = n_seed - seed_kept

    dil = dilate_by_inverse_edt(seed, dilation)
    pub_in_dil = int((published & dil).sum())
    within = pub_in_dil / (n_pub + 1e-9)
    outside = n_pub - pub_in_dil            # published voxels the geometry cannot explain
    return {
        "seed_voxels": n_seed,
        "published_voxels": n_pub,
        "dilation_radius": dilation,
        "seed_containment_in_published": round(float(seed_containment), 4),
        "seed_voxels_dropped": seed_dropped,
        "published_within_dilation_frac": round(float(within), 4),
        "defect_voxels_outside_dilation": outside,
        "defect_frac": round(float(outside / (n_pub + 1e-9)), 5),
    }

