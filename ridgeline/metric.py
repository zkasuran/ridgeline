"""Normal Ridge Drift (NRD): the drift metric. Measured in Hessian ridge-response
space, never in raw intensity (the CT under a label is only a few grey levels
above the shell, so an intensity argmax lands on noise while the second-derivative
ridge is strong and sign stable).

Per medial point we sample the ridge field along the cross-structure direction(s),
refine the peak to sub-voxel, and record the signed offset from the label to the
ridge peak. A tube gets a 2D offset in its cross-section plane, a sheet gets a 1D
signed offset along its normal.
"""
import numpy as np
from . import geom
from .engine import ridge_field, structure_probe


def _drift_vectors(R, points, frames, kind, radius, step, min_height, min_curv):
    """Per-point drift vector (label -> ridge peak) and an ok mask."""
    dirs = geom.cross_directions(frames, kind)
    n = len(points)
    drift = np.zeros((n, 3))
    offs = np.zeros((n, len(dirs)))
    ok = np.zeros(n, dtype=bool)
    heights = np.zeros(n)
    for i in range(n):
        good = True
        vec = np.zeros(3)
        for d, dvec in enumerate(dirs):
            ts, vals, st = geom.sample_line(R, points[i], dvec[i], radius, step)
            t, h, curv, okp = geom.parabola_peak(ts, vals, st, min_height, min_curv)
            if not okp:
                good = False
                break
            vec = vec + t * dvec[i]
            offs[i, d] = t
            heights[i] = max(heights[i], h)
        if good:
            drift[i] = vec
            ok[i] = True
    return drift, offs, ok, heights


def measure(vol, mask, head=None, sigmas=(1, 2, 3), radius=6.0, step=0.5,
            black_ridges=False, min_curv=1e-4, probe=None):
    """Full NRD measurement. Returns a dict with the drift field and aggregates."""
    vol = vol.astype(np.float32)
    if probe is None:
        probe = structure_probe(mask)
    kind = "sheet" if probe["dominant"] == "sheet" else "tube"
    if head is None:
        head = probe["head"]
    R = ridge_field(vol, sigmas=sigmas, head=head, black_ridges=black_ridges)
    floor = geom.noise_floor(R, mask)
    pts = geom.medial_points(mask)
    frames = geom.local_frames(pts)
    drift, offs, ok, heights = _drift_vectors(
        R, pts, frames, kind, radius, step, floor, min_curv)
    mag = np.linalg.norm(drift[ok], axis=1) if ok.any() else np.array([])
    signed = offs[ok, 0] if (ok.any() and kind == "sheet") else mag
    # thickness bias: label half-thickness (EDT interior median) reported so a
    # symmetric dilation, which does not move the medial axis, is still caught.
    import scipy.ndimage as ndi
    edt = ndi.distance_transform_edt(mask.astype(bool))
    half_thick = float(np.median(edt[mask.astype(bool)])) if mask.any() else 0.0
    res = {
        "head": head, "kind": kind, "n_points": int(len(pts)),
        "n_ok": int(ok.sum()),
        "support_fraction": float(ok.mean()) if len(pts) else 0.0,
        "median_abs_drift": float(np.median(mag)) if mag.size else float("nan"),
        "iqr_drift": float(np.subtract(*np.percentile(mag, [75, 25]))) if mag.size else float("nan"),
        "signed_median": float(np.median(signed)) if signed.size else float("nan"),
        "peak_height_median": float(np.median(heights[ok])) if ok.any() else float("nan"),
        "noise_floor": floor,
        "half_thickness": half_thick,
        "mix": probe["mix"],
    }
    return res, {"points": pts, "drift": drift, "ok": ok, "frames": frames,
                 "R": R, "kind": kind, "offsets": offs}


def worst_regions(detail, top=10):
    """Rank medial points by drift magnitude for active-learning triage."""
    pts, drift, ok = detail["points"], detail["drift"], detail["ok"]
    mag = np.linalg.norm(drift, axis=1)
    mag[~ok] = -1
    order = np.argsort(mag)[::-1][:top]
    return [{"point": pts[i].tolist(), "drift": float(mag[i])}
            for i in order if mag[i] >= 0]
