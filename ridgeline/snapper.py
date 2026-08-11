"""The snapper. Move each medial point toward the sub-voxel ridge peak along the
cross-structure direction, cap the per-point displacement, Laplacian-smooth the
displacement field over the medial graph so no single point jumps to a neighbour
structure, then re-voxelize a corrected mask.

Two baselines it must beat, both reported by run_snappers:
- snap-to-brightest: raw CT intensity argmax in the window at voxel resolution.
  The few-grey-level contrast makes this noise dominated.
- random-direction: a random move of equal magnitude (the TAUIL control). It must
  NOT recover, else the ridge signal is diffuse and the claim is dead.
The independence baseline (snap to the producer's own frangi ridge) is exposed by
choosing head='frangi'.
"""
import numpy as np
import scipy.ndimage as ndi
from sklearn.neighbors import NearestNeighbors
from . import geom
from .metric import measure


def _smooth_field(points, vectors, k=8, lam=0.5, iters=5):
    """Laplacian smoothing of a per-point vector field over the medial knn graph."""
    n = len(points)
    if n < 3:
        return vectors
    kk = min(k + 1, n)
    nn = NearestNeighbors(n_neighbors=kk).fit(points)
    _, idx = nn.kneighbors(points)
    idx = idx[:, 1:]                                    # drop self
    out = vectors.copy()
    for _ in range(iters):
        neigh_mean = out[idx].mean(axis=1)
        out = out + lam * (neigh_mean - out)
    return out


def _revoxelize(points, shape, radius):
    """Rasterize medial points and dilate by a ball of the label half-thickness."""
    seed = np.zeros(shape, dtype=bool)
    ic = np.round(points).astype(int)
    inb = np.all((ic >= 0) & (ic < np.array(shape)), axis=1)
    ic = ic[inb]
    seed[ic[:, 0], ic[:, 1], ic[:, 2]] = True
    r = max(1, int(round(radius)))
    ball = np.linalg.norm(np.stack(np.mgrid[-r:r + 1, -r:r + 1, -r:r + 1]), axis=0) <= radius
    return ndi.binary_dilation(seed, structure=ball)


def snap(vol, mask, head=None, sigmas=(1, 2, 3), radius=6.0, step=0.5,
         cap=None, smooth_iter=5, black_ridges=False):
    """Ridge snap. Returns (corrected_mask, snapped_points, info)."""
    res, det = measure(vol, mask, head=head, sigmas=sigmas, radius=radius,
                       step=step, black_ridges=black_ridges)
    pts, drift, ok = det["points"], det["drift"], det["ok"]
    cap = radius if cap is None else cap
    disp = drift.copy()
    disp[~ok] = 0.0                                     # rejected points stay put
    mag = np.linalg.norm(disp, axis=1)
    scale = np.where(mag > cap, cap / (mag + 1e-9), 1.0)
    disp = disp * scale[:, None]
    disp = _smooth_field(pts, disp, iters=smooth_iter)
    snapped = pts + disp
    corrected = _revoxelize(snapped, mask.shape, res["half_thickness"])
    info = {"metric": res, "n_moved": int(ok.sum()),
            "median_move": float(np.median(np.linalg.norm(disp[ok], axis=1))) if ok.any() else 0.0}
    return corrected, snapped, info, det


def snap_brightest(vol, mask, det, radius=6.0, step=0.5):
    """Baseline: move each medial point to the raw-CT intensity argmax in the
    window, at voxel resolution (order=0)."""
    pts, frames, kind = det["points"], det["frames"], det["kind"]
    dirs = geom.cross_directions(frames, kind)
    ts = np.arange(-radius, radius + 1e-6, step)
    snapped = pts.copy()
    for i in range(len(pts)):
        move = np.zeros(3)
        for dvec in dirs:
            coords = pts[i][:, None] + dvec[i][:, None] * ts[None, :]
            vals = ndi.map_coordinates(vol.astype(np.float32), coords, order=0, mode="nearest")
            move = move + ts[int(np.argmax(vals))] * dvec[i]
        snapped[i] = pts[i] + move
    return snapped


def snap_random(det, ridge_disp_mag, seed=0):
    """Control: random unit move of equal magnitude to the ridge snap."""
    rng = np.random.default_rng(seed)
    pts = det["points"]
    r = rng.normal(size=pts.shape)
    r /= (np.linalg.norm(r, axis=1, keepdims=True) + 1e-9)
    return pts + r * ridge_disp_mag[:, None]
