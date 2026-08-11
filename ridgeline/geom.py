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
    for a sheet the first column is the surface normal."""
    n = len(points)
    if n == 0:
        return np.zeros((0, 3, 3))
    kk = min(k, n)
    nn = NearestNeighbors(n_neighbors=kk).fit(points)
    _, idx = nn.kneighbors(points)
    frames = np.zeros((n, 3, 3))
    for i in range(n):
        P = points[idx[i]] - points[idx[i]].mean(0)
        cov = P.T @ P
        w, V = np.linalg.eigh(cov)                      # ascending eigenvalues
        frames[i] = V
    return frames


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
