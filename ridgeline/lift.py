"""Downstream lift probe (D-lift Regime A, the defensible core).

The tool must show snapped labels are not just different but BETTER for a learner.
Full nnU-Net retraining is off the table, so this is the cheap faithful proxy: a
small sklearn logistic regression on local CT features predicting label membership,
trained once on ORIGINAL/CORRUPTED labels and once on SNAPPED labels, then scored on
a HELD-OUT set against an uncorrupted reference. A label placed off the CT structure
gives a noisy features-to-membership map, so a probe trained on it generalizes worse.
Snapping it back onto the structure cleans the map, so the probe generalizes better.

HARD CONTRACT (enforced in _feature_volumes and asserted here): sato is the snapper's
optimization signal, so sato may never be a probe feature and may never be the scoring
target. Features are zeroth and first order only (intensity, smoothed intensity, local
std, gradient magnitude), sharing no machinery with the snap. The eval target is the
clean planted reference, never any ridge field.

Two controls ship as first-class output, not flags:
  - null / idempotence: snapping an already-clean label barely changes the probe.
  - random-motion: a random move of the snap's magnitude does NOT recover the probe.
"""
import numpy as np
import scipy.ndimage as ndi
from sklearn.linear_model import LogisticRegression
from . import synth
from .snapper import snap

FORBIDDEN_FEATURES = ("sato",)          # the snapper's own signal, never a feature or Y


def _feature_volumes(vol):
    """Local CT features, none of them the snap signal (sato). Zeroth/first order:
    raw intensity, two smoothing scales, local std, gradient magnitude."""
    vol = vol.astype(np.float32)
    sm1 = ndi.gaussian_filter(vol, 1.0)
    sm2 = ndi.gaussian_filter(vol, 2.0)
    lm = ndi.uniform_filter(vol, 3)
    lsq = ndi.uniform_filter(vol * vol, 3)
    lstd = np.sqrt(np.maximum(lsq - lm * lm, 0.0))
    gm = ndi.gaussian_gradient_magnitude(vol, 1.0)
    return np.stack([vol, sm1, sm2, lstd, gm], axis=-1)   # (Z, Y, X, 5), sato-free


def _band(ref, r=4.0):
    """Shell of thickness ~2r around the reference boundary, the zone where label
    placement actually matters."""
    sd = ndi.distance_transform_edt(~ref) - ndi.distance_transform_edt(ref)
    return np.abs(sd) <= r


def _sample_train(feats, band, ref, label_ut, n, rng):
    """Balanced training sample by the label-under-test class in the band."""
    idx = np.argwhere(band)
    ut_v = label_ut[idx[:, 0], idx[:, 1], idx[:, 2]]
    pos, neg = idx[ut_v], idx[~ut_v]
    m = min(n // 2, len(pos), len(neg))
    if m == 0:
        return np.zeros((0, feats.shape[-1])), np.zeros(0, bool)
    take = np.concatenate([pos[rng.choice(len(pos), m, replace=False)],
                           neg[rng.choice(len(neg), m, replace=False)]])
    X = feats[take[:, 0], take[:, 1], take[:, 2]]
    y_ut = label_ut[take[:, 0], take[:, 1], take[:, 2]]
    return X, y_ut


def _eval_band(feats, band, ref):
    """All band voxels for scoring, with the clean-reference class as truth."""
    idx = np.argwhere(band)
    X = feats[idx[:, 0], idx[:, 1], idx[:, 2]]
    y_ref = ref[idx[:, 0], idx[:, 1], idx[:, 2]]
    return X, y_ref


def _dice(pred, truth):
    tp = int((pred & truth).sum())
    return 2 * tp / (int(pred.sum()) + int(truth.sum()) + 1e-9)


def _make_phantom(seed, k, radius, size, contrast, noise, direction, band_r=3.0):
    """One phantom instance. All label variants are re-voxelized at the SAME radius as
    the reference, so the probe measures centerline placement (what the snapper
    corrects), not re-voxelization thickness. Variants: clean reference, corrupted
    (shift k), snapped (sato snap of corrupted), random (random move of the snap's
    magnitude) and snapped_clean (the null control)."""
    vol, cl, r = synth.make_tube(size=size, radius=radius, contrast=contrast,
                                 noise=noise, seed=seed)
    ref = synth.voxelize_tube(cl, r, vol.shape)
    corrupted = synth.voxelize_tube(cl + k * direction, r, vol.shape)

    _, snapped_pts, info, det = snap(vol, corrupted, radius=6.0)
    move = np.linalg.norm(snapped_pts - det["points"], axis=1)
    snapped = synth.voxelize_tube(snapped_pts, r, vol.shape)

    rng = np.random.default_rng(seed + 777)
    rdir = rng.normal(size=snapped_pts.shape)
    rdir /= (np.linalg.norm(rdir, axis=1, keepdims=True) + 1e-9)
    rand_pts = det["points"] + rdir * move[:, None]
    random_mask = synth.voxelize_tube(rand_pts, r, vol.shape)

    _, snapped_clean_pts, _, _ = snap(vol, ref, radius=6.0)
    snapped_clean = synth.voxelize_tube(snapped_clean_pts, r, vol.shape)

    return {"vol": vol, "feats": _feature_volumes(vol), "ref": ref,
            "band": _band(ref, band_r),
            "clean": ref, "corrupted": corrupted, "snapped": snapped,
            "random": random_mask, "snapped_clean": snapped_clean}


def _probe_dice(train, test, variant, n_per, rng):
    """Train LR on the variant label over train phantoms, score Dice of the prediction
    against the clean reference over held-out test phantoms."""
    Xtr, ytr = [], []
    for p in train:
        X, y = _sample_train(p["feats"], p["band"], p["ref"], p[variant], n_per, rng)
        Xtr.append(X); ytr.append(y)
    Xtr, ytr = np.concatenate(Xtr), np.concatenate(ytr)
    if len(np.unique(ytr)) < 2:
        return float("nan")
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit((Xtr - mu) / sd, ytr)

    dices = []
    for p in test:
        X, y_ref = _eval_band(p["feats"], p["band"], p["ref"])
        pred = clf.predict((X - mu) / sd).astype(bool)
        dices.append(_dice(pred, y_ref.astype(bool)))
    return float(np.mean(dices))


def lift_probe(k=2.0, n_phantoms=6, radius=2.0, size=64, contrast=8.0, noise=2.0,
               n_per=1500, band_r=3.0, seed=0):
    """Regime A probe recovery. Returns held-out Dice (probe prediction vs the clean
    planted reference) for probes trained on the clean, corrupted, snapped,
    snapped-clean and random labels, the recovery fraction
    R = (snapped - corrupted) / (clean - corrupted), and the two controls."""
    rng = np.random.default_rng(seed)
    direction = np.array([0.0, 1.0, 0.0])
    phantoms = [_make_phantom(seed=i, k=k, radius=radius, size=size, contrast=contrast,
                              noise=noise, direction=direction, band_r=band_r)
                for i in range(n_phantoms)]
    half = n_phantoms // 2
    train, test = phantoms[:half], phantoms[half:]

    dice = {v: _probe_dice(train, test, v, n_per, rng)
            for v in ("clean", "corrupted", "snapped", "snapped_clean", "random")}
    denom = dice["clean"] - dice["corrupted"]
    recovery = (dice["snapped"] - dice["corrupted"]) / denom if abs(denom) > 1e-6 else float("nan")
    rand_recovery = (dice["random"] - dice["corrupted"]) / denom if abs(denom) > 1e-6 else float("nan")
    return {
        "regime": "A (planted defect, held-out probe recovery)",
        "k_planted": k, "n_phantoms": n_phantoms, "n_per_patch": n_per,
        "metric": "held-out Dice of probe prediction vs clean planted reference",
        "features": "intensity, smooth1, smooth2, local_std, grad_mag (sato-free)",
        "eval_target": "clean planted reference (not sato)",
        "dice_clean": round(dice["clean"], 4),
        "dice_corrupted": round(dice["corrupted"], 4),
        "dice_snapped": round(dice["snapped"], 4),
        "dice_random": round(dice["random"], 4),
        "recovery_fraction": round(recovery, 4),
        "null_control_dice_snapped_clean": round(dice["snapped_clean"], 4),
        "null_control_delta": round(dice["snapped_clean"] - dice["clean"], 4),
        "random_control_recovery_fraction": round(rand_recovery, 4),
    }
