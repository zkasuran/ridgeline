"""Planted-defect harness. Runs the whole measure + snap over the offset matrix on
synthetic phantoms with exact truth, and scores recovery against both baselines.
This is the claim carrier: shift a label by k, recover it within tolerance, and
beat snap-to-brightest and random-direction while doing it.
"""
import numpy as np
from . import synth, geom
from .metric import measure
from .snapper import snap, snap_brightest, snap_random


def _tube_reference(sample, dense=400):
    cl = sample["true_centerline"]
    if len(cl) >= dense:
        return cl
    t = np.linspace(0, len(cl) - 1, dense)
    return np.stack([np.interp(t, np.arange(len(cl)), cl[:, a]) for a in range(3)], axis=1)


def _reference_points(sample):
    if sample["kind"] == "sheet":
        s = sample["vol"].shape
        zz, yy = np.mgrid[0:s[0]:4, 0:s[1]:4]
        xx = np.full(zz.size, sample["true_plane_x"])
        return np.stack([zz.ravel(), yy.ravel(), xx], axis=1)
    return _tube_reference(sample)


def run_cell(sample, radius=6.0, sigmas=(1, 2, 3), seed=0):
    """One planted-defect cell. Returns recovery numbers for ridge/brightest/random."""
    vol, mask = sample["vol"], sample["label"]
    ref = _reference_points(sample)
    res, det = measure(vol, mask, sigmas=sigmas, radius=radius)
    pts = det["points"]
    pre = geom.nearest_distance(pts, ref)

    corrected, snapped_ridge, info, det2 = snap(vol, mask, sigmas=sigmas, radius=radius)
    ridge_disp_mag = np.linalg.norm(snapped_ridge - det2["points"], axis=1)
    rec_ridge = geom.nearest_distance(snapped_ridge, ref)

    snapped_bright = snap_brightest(vol, mask, det2, radius=radius)
    rec_bright = geom.nearest_distance(snapped_bright, ref)

    snapped_rand = snap_random(det2, ridge_disp_mag, seed=seed)
    rec_rand = geom.nearest_distance(snapped_rand, ref)

    # frangi-independence arm: snap to the producer's own operator. If our sato snap
    # only matched this, the witness would just be re-running their idea. Reporting
    # the frangi recovery and how far the two snaps diverge answers that objection.
    _, snapped_frangi, _, det_f = snap(vol, mask, head="frangi", sigmas=sigmas, radius=radius)
    rec_frangi = geom.nearest_distance(snapped_frangi, ref)
    frangi_div = float(np.median(np.linalg.norm(snapped_ridge - snapped_frangi, axis=1))) \
        if len(snapped_ridge) == len(snapped_frangi) else float("nan")

    row = {
        "kind": sample["kind"],
        "planted": sample.get("k", sample.get("extra", sample.get("amp"))),
        "pre_snap_error": round(pre, 3),
        "nrd_drift": round(res["median_abs_drift"], 3),
        "ridge_recovery": round(rec_ridge, 3),
        "brightest_recovery": round(rec_bright, 3),
        "random_recovery": round(rec_rand, 3),
        "frangi_recovery": round(rec_frangi, 3),
        "ridge_vs_frangi_div": round(frangi_div, 3),
        "support_fraction": round(res["support_fraction"], 3),
        "half_thickness": round(res["half_thickness"], 3),
    }
    if "distractor_y" in sample:
        dist_ref = np.array([[z, sample["distractor_y"], sample["vol"].shape[2] / 2]
                             for z in range(sample["vol"].shape[0])])
        row["dist_to_distractor"] = round(geom.nearest_distance(snapped_ridge, dist_ref), 3)
    return row


def run_matrix(radius=6.0, sigmas=(1, 2, 3), size=64, contrast=8.0, noise=2.0):
    """The full planted-defect matrix from the design table."""
    rows = []
    base = dict(size=size, contrast=contrast, noise=noise)
    for k in (0.5, 1, 2, 3, 5):
        rows.append(("shift_normal_k%.1f" % k,
                     run_cell(synth.plant_tube_shift(k, **base), radius)))
    for k in (1, 2, 3):
        d = np.array([0.0, 0.7, 0.7])
        rows.append(("shift_random_k%d" % k,
                     run_cell(synth.plant_tube_shift(k, direction=d, **base), radius)))
    for e in (1, 2, 3):
        rows.append(("dilation_+%d" % e,
                     run_cell(synth.plant_tube_dilation(e, **base), radius)))
    for a in (1, 2):
        rows.append(("wander_amp%d" % a,
                     run_cell(synth.plant_tube_wander(a, **base), radius)))
    rows.append(("distractor_k2_gap3",
                 run_cell(synth.plant_tube_distractor(2, 3, **base), radius)))
    for c in (8, 6, 4, 2):
        rows.append(("contrast_%dgl_k2" % c,
                     run_cell(synth.plant_tube_shift(2, size=size, contrast=c, noise=noise), radius)))
    return rows


def null_controls(radius=6.0, sigmas=(1, 2, 3), size=64, random_mag=2.0, seed=0):
    """Two first-class controls on an already-clean label, not flags.

    Idempotence: snap a clean label and require it barely moves and stays correct.
    Random-direction: move the same clean label by `random_mag` voxels in random
    directions and show it does NOT stay correct. Together these prove the snapper
    preserves a good label while an arbitrary move of comparable size damages it, so
    a recovery seen elsewhere is signal-seeking and not an artifact of moving labels.
    """
    clean = synth.plant_tube_shift(0.0, size=size)
    vol, mask = clean["vol"], clean["label"]
    ref = _reference_points(clean)
    _, det = measure(vol, mask, radius=radius)
    pre = geom.nearest_distance(det["points"], ref)

    corrected, snapped, info, det2 = snap(vol, mask, radius=radius)
    move = float(np.median(np.linalg.norm(snapped - det2["points"], axis=1)))
    post = geom.nearest_distance(snapped, ref)

    # random-direction control on the SAME clean label, fixed magnitude
    n = len(det2["points"])
    mag = np.full(n, random_mag)
    rand_pts = snap_random(det2, mag, seed=seed)
    rand_post = geom.nearest_distance(rand_pts, ref)
    return {
        "idempotence_pre_error": round(pre, 3),
        "idempotence_median_move": round(move, 3),
        "idempotence_post_error": round(post, 3),
        "random_control_mag": random_mag,
        "random_control_post_error": round(rand_post, 3),
    }
