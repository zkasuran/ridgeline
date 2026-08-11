"""Geometry helpers: medial extraction, local structure frames, sub-voxel ridge
sampling and parabola peak refinement. Shared by the metric and the snapper.

The measurement primitive is deliberately medial, not boundary. A label mask is a
fattened structure, so its marching-cubes surface sits half a thickness off the
CT ridge. The medial axis (tube) or medial surface (sheet) is where the CT crest
should coincide, so drift is measured there.
"""
import numpy as np
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
from sklearn.neighbors import NearestNeighbors

EPS = 1e-9


def medial_points(mask):
    """1-voxel medial skeleton (tube -> line, sheet -> surface)."""
    skel = skeletonize(mask.astype(bool))
    return np.argwhere(skel).astype(np.float64)


def local_frames(points, k=12):
    """Per-point orthonormal frame from PCA of the k nearest medial neighbours.
    Columns are eigenvectors in ASCENDING eigenvalue order, so for a tube the last
    column is the axis tangent and the first two span the cross-section plane, and
    for a sheet the first column is the surface normal. Batched: one knn query and
    one stacked eigendecomposition over all points, no Python per-point loop."""
    n = len(points)
    if n == 0:
        return np.zeros((0, 3, 3))
    kk = min(k, n)
    nn = NearestNeighbors(n_neighbors=kk).fit(points)
    _, idx = nn.kneighbors(points)                      # (n, kk)
    neigh = points[idx]                                 # (n, kk, 3)
    neigh = neigh - neigh.mean(axis=1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", neigh, neigh)       # (n, 3, 3)
    _, V = np.linalg.eigh(cov)                          # batched, ascending eigenvalues
    return V


def cross_directions(frames, kind):
    """The directions to search per point. Tube: the two smallest-eigenvalue
    eigenvectors (cross-section plane). Sheet: the smallest-eigenvalue eigenvector
    (surface normal)."""
    if kind == "sheet":
        return [frames[:, :, 0]]
    return [frames[:, :, 0], frames[:, :, 1]]


def sample_line(R, p, direction, radius, step=0.5):
    """Sample R along p + t*direction for t in [-radius, radius]. order=3."""
    ts = np.arange(-radius, radius + 1e-6, step)
    coords = p[:, None] + direction[:, None] * ts[None, :]
    vals = ndi.map_coordinates(R, coords, order=3, mode="nearest")
    return ts, vals, step


def parabola_peak(ts, vals, step, min_height, min_curv):
    """Refine the discrete argmax to sub-voxel with a 3-point parabola. Returns
    (t_peak, height, curvature, ok). ok is False for edge, non-concave, weak or
    low peaks, which is the rejection gate that beats blind argmax."""
    j = int(np.argmax(vals))
    if j == 0 or j == len(vals) - 1:
        return ts[j], vals[j], 0.0, False
    ym1, y0, yp1 = vals[j - 1], vals[j], vals[j + 1]
    denom = ym1 - 2 * y0 + yp1
    curv = -denom
    if denom >= 0 or curv < min_curv or y0 < min_height:
        return ts[j], y0, curv, False
    delta = 0.5 * (ym1 - yp1) / denom
    delta = float(np.clip(delta, -1.0, 1.0))
    return ts[j] + delta * step, y0, curv, True


def sample_lines(R, points, directions, radius, step=0.5):
    """Vectorized `sample_line` for every point along one per-point direction. This
    is the batched form used by the drift measurement: one `map_coordinates` call
    instead of a Python loop over points. `points` (n,3), `directions` (n,3).
    Returns ts (m,), vals (n,m), step. Bit-identical to looping sample_line."""
    ts = np.arange(-radius, radius + 1e-6, step)
    pts = np.asarray(points, dtype=np.float64)
    dirs = np.asarray(directions, dtype=np.float64)
    coords = pts.T[:, :, None] + dirs.T[:, :, None] * ts[None, None, :]   # (3, n, m)
    n, m = len(pts), len(ts)
    vals = ndi.map_coordinates(R, coords.reshape(3, n * m), order=3, mode="nearest")
    return ts, vals.reshape(n, m), step


def parabola_peaks(ts, vals, step, min_height, min_curv):
    """Vectorized `parabola_peak` over vals (n, m). Same gates as the scalar form:
    a point is rejected at an edge argmax, a non-concave fit, a curvature below
    min_curv or a peak height below min_height. Returns (t (n,), height (n,),
    curv (n,), ok (n,)). Non-ok points keep the un-refined discrete argmax offset,
    matching the scalar function's early-return behavior."""
    vals = np.asarray(vals)
    n, m = vals.shape
    j = np.argmax(vals, axis=1)
    ii = np.arange(n)
    edge = (j == 0) | (j == m - 1)
    jc = np.clip(j, 1, m - 2)                            # safe interior index for neighbours
    y0 = vals[ii, jc]
    ym1 = vals[ii, jc - 1]
    yp1 = vals[ii, jc + 1]
    denom = ym1 - 2.0 * y0 + yp1
    curv = -denom
    ok = (~edge) & (denom < 0) & (curv >= min_curv) & (y0 >= min_height)
    delta = np.zeros(n)
    nz = denom != 0
    delta[nz] = 0.5 * (ym1[nz] - yp1[nz]) / denom[nz]
    delta = np.clip(delta, -1.0, 1.0)
    t = ts[j] + np.where(ok, delta * step, 0.0)          # refine only accepted peaks
    curv = np.where(edge, 0.0, curv)
    return t, y0, curv, ok


def noise_floor(R, mask, dilate=6):
    """Estimate a ridge noise floor away from the label: mean + std of R outside a
    dilated label band. Used as the peak-height rejection threshold."""
    band = ndi.binary_dilation(mask.astype(bool), iterations=dilate)
    outside = R[~band]
    if outside.size == 0:
        return float(R.mean())
    return float(outside.mean() + outside.std())


def nearest_distance(a, b):
    """Median nearest-neighbour distance from point set a to point set b."""
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    nn = NearestNeighbors(n_neighbors=1).fit(b)
    d, _ = nn.kneighbors(a)
    return float(np.median(d[:, 0]))
