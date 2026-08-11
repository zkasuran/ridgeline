"""Ridge engine: one Hessian core, two heads (vesselness / sheetness), plus the
structure-type probe that decides which head a component wants.

Vesuvius fibers and recto surfaces are BRIGHT on dark shell, so black_ridges=False
is mandatory. The skimage default hunts dark ridges and silently returns zeros.
We use sato as the primary vesselness head and meijering as the independent
secondary. frangi is kept as an ablation arm only, never the primary, because the
label producer used a frangi ridge and auditing with frangi is self-referential.
"""
import numpy as np
from skimage.filters import sato, meijering, frangi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.measure import label as cc_label

EPS = 1e-9


def normalize(field):
    """Scale a field to [0, 1]. numpy 2.x: use np.ptp, not ndarray.ptp."""
    field = field.astype(np.float32)
    return (field - field.min()) / (np.ptp(field) + EPS)


def _mag_sorted_eigs(vol, sigma):
    """Hessian eigenvalues sorted by magnitude: |l1| <= |l2| <= |l3|."""
    H = hessian_matrix(vol, sigma=sigma, use_gaussian_derivatives=True)
    e = np.stack(hessian_matrix_eigvals(H))          # (3, Z, Y, X), value-descending
    order = np.argsort(np.abs(e), axis=0)
    l1 = np.take_along_axis(e, order[0:1], axis=0)[0]
    l2 = np.take_along_axis(e, order[1:2], axis=0)[0]
    l3 = np.take_along_axis(e, order[2:3], axis=0)[0]
    return l1, l2, l3


def sheetness(vol, sigmas, black_ridges=False, alpha=0.5, beta=0.5, gamma=None):
    """Descoteaux/Antiga sheetness built from the Hessian eigenvalues. A plate has
    |l1|, |l2| << |l3|. Bright sheet requires l3 < 0. Multiscale max over sigmas."""
    vol = vol.astype(np.float32)
    out = np.zeros_like(vol)
    for s in np.atleast_1d(sigmas):
        l1, l2, l3 = _mag_sorted_eigs(vol, float(s))
        a1, a2, a3 = np.abs(l1), np.abs(l2), np.abs(l3)
        r_sheet = np.sqrt(a1 * a2) / (a3 + EPS)
        r_blob = a3 / (np.sqrt(a1 * a2) + EPS)
        struct = np.sqrt(l1 ** 2 + l2 ** 2 + l3 ** 2)
        g = gamma if gamma is not None else 0.5 * struct.max()
        resp = (np.exp(-r_sheet ** 2 / (2 * alpha ** 2))
                * (1 - np.exp(-r_blob ** 2 / (2 * beta ** 2)))
                * (1 - np.exp(-struct ** 2 / (2 * (g + EPS) ** 2))))
        if not black_ridges:
            resp = np.where(l3 < 0, resp, 0.0)          # keep bright plates only
        else:
            resp = np.where(l3 > 0, resp, 0.0)
        out = np.maximum(out, resp)
    return out.astype(np.float32)


def ridge_field(vol, sigmas=(1, 2, 3), head="sato", black_ridges=False, normalized=True):
    """Return the ridge-response field for the chosen head, scaled to [0, 1]."""
    vol = vol.astype(np.float32)
    sig = tuple(float(s) for s in np.atleast_1d(sigmas))
    if head == "sato":
        r = sato(vol, sigmas=sig, black_ridges=black_ridges)
    elif head == "meijering":
        r = meijering(vol, sigmas=sig, black_ridges=black_ridges)
    elif head == "frangi":
        r = frangi(vol, sigmas=sig, black_ridges=black_ridges)
    elif head == "sheet":
        r = sheetness(vol, sig, black_ridges=black_ridges)
    else:
        raise ValueError(f"unknown head {head!r}")
    return normalize(r) if normalized else r.astype(np.float32)


def structure_probe(mask, min_voxels=30):
    """Per connected-component PCA of the label voxels. Rod-like (one large
    eigenvalue, two small) => tube => vesselness. Plate-like (two large, one small)
    => sheet => sheetness. Returns (components, mix) where mix counts each type."""
    lab = cc_label(mask.astype(bool))
    comps = []
    counts = {"tube": 0, "sheet": 0, "blob": 0}
    for i in range(1, int(lab.max()) + 1):
        pts = np.argwhere(lab == i)
        if len(pts) < min_voxels:
            continue
        c = pts.mean(0)
        cov = np.cov((pts - c).T)
        w = np.sort(np.linalg.eigvalsh(cov))[::-1]      # descending w0>=w1>=w2
        r1 = w[1] / (w[0] + EPS)
        r2 = w[2] / (w[0] + EPS)
        if r1 < 0.25:
            kind = "tube"
        elif r2 < 0.25:
            kind = "sheet"
        else:
            kind = "blob"
        counts[kind] += 1
        comps.append({"cc": i, "kind": kind, "eigs": w.tolist(), "voxels": int(len(pts))})
    total = sum(counts.values()) or 1
    dominant = max(counts, key=counts.get) if comps else "tube"
    head = "sheet" if dominant == "sheet" else "sato"
    return {"components": comps, "mix": counts, "dominant": dominant,
            "head": head, "n": total}
