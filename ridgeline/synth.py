"""Synthetic phantoms with a KNOWN answer. This is Layer 0, the backbone that
calibrates the instrument before any real data is touched.

A tube is a bright Gaussian ridge on a textured background at a chosen grey-level
contrast, so the ridge peak sits exactly on a known centerline. The label is that
centerline dilated by a fixed radius (the real pipeline dilates a seed by an EDT
radius), and a defect is planted by shifting, fattening or wandering the label's
centerline by a known amount while the CT stays put. Recovery is then measured
against the exact truth.
"""
import numpy as np
import scipy.ndimage as ndi


def _rasterize(points, shape):
    seed = np.zeros(shape, dtype=bool)
    ic = np.round(points).astype(int)
    inb = np.all((ic >= 0) & (ic < np.array(shape)), axis=1)
    ic = ic[inb]
    seed[ic[:, 0], ic[:, 1], ic[:, 2]] = True
    return seed


def voxelize_tube(centerline, radius, shape):
    """Dilate a centerline by an EDT radius, mirroring the real label pipeline."""
    seed = _rasterize(centerline, shape)
    dist = ndi.distance_transform_edt(~seed)
    return dist <= radius


def voxelize_sheet(plane_x, thickness, shape):
    xs = np.arange(shape[2])[None, None, :]
    return np.abs(xs - plane_x) <= thickness + np.zeros(shape)


def straight_centerline(size, cy, cx, axis=0):
    z = np.arange(size, dtype=float)
    pts = np.zeros((size, 3))
    pts[:, axis] = z
    other = [a for a in (0, 1, 2) if a != axis]
    pts[:, other[0]] = cy
    pts[:, other[1]] = cx
    return pts


def wander_centerline(size, cy, cx, amp, wavelength, axis=0, wander_axis=1):
    pts = straight_centerline(size, cy, cx, axis)
    z = pts[:, axis]
    pts[:, wander_axis] += amp * np.sin(2 * np.pi * z / wavelength)
    return pts


def make_tube(size=64, radius=3.0, contrast=8.0, noise=2.0, blur=1.0,
              bg=0.4, centerline=None, seed=0):
    """CT volume with a bright Gaussian tube. contrast and noise are in grey
    levels out of 255. Returns (vol[0,1], centerline, radius)."""
    rng = np.random.default_rng(seed)
    shape = (size, size, size)
    if centerline is None:
        centerline = straight_centerline(size, size / 2, size / 2)
    seedv = _rasterize(centerline, shape)
    dist = ndi.distance_transform_edt(~seedv)
    sigma_prof = max(radius / 2.0, 0.8)
    profile = np.exp(-(dist ** 2) / (2 * sigma_prof ** 2))
    vol = bg + (contrast / 255.0) * profile
    vol = ndi.gaussian_filter(vol, blur)
    vol = vol + rng.normal(0, noise / 255.0, shape)
    return np.clip(vol, 0, 1).astype(np.float32), centerline, radius


def make_sheet(size=64, thickness=3.0, contrast=8.0, noise=2.0, blur=1.0,
               bg=0.4, plane_x=None, seed=0):
    rng = np.random.default_rng(seed)
    shape = (size, size, size)
    plane_x = size / 2 if plane_x is None else plane_x
    xs = np.arange(size)[None, None, :] + np.zeros(shape)
    sigma_prof = max(thickness / 2.0, 0.8)
    profile = np.exp(-((xs - plane_x) ** 2) / (2 * sigma_prof ** 2))
    vol = bg + (contrast / 255.0) * profile
    vol = ndi.gaussian_filter(vol, blur)
    vol = vol + rng.normal(0, noise / 255.0, shape)
    return np.clip(vol, 0, 1).astype(np.float32), plane_x, thickness


def plant_tube_shift(k, direction=None, size=64, radius=3.0, **kw):
    """CT at the true centerline, label shifted by k voxels along direction."""
    vol, cl, r = make_tube(size=size, radius=radius, **kw)
    if direction is None:
        direction = np.array([0.0, 1.0, 0.0])           # perpendicular to z-axis tube
    direction = np.asarray(direction, float)
    direction /= (np.linalg.norm(direction) + 1e-9)
    shifted = cl + k * direction
    label = voxelize_tube(shifted, r, vol.shape)
    return {"vol": vol, "label": label, "true_centerline": cl,
            "shifted_centerline": shifted, "radius": r, "k": k, "kind": "tube"}


def plant_tube_dilation(extra, size=64, radius=3.0, **kw):
    """Label fattened by `extra` voxels; the medial axis does not move."""
    vol, cl, r = make_tube(size=size, radius=radius, **kw)
    label = voxelize_tube(cl, r + extra, vol.shape)
    return {"vol": vol, "label": label, "true_centerline": cl,
            "shifted_centerline": cl, "radius": r, "extra": extra, "kind": "tube"}


def plant_tube_wander(amp, wavelength=24, size=64, radius=3.0, **kw):
    """CT straight, label wanders sinusoidally by amp voxels."""
    vol, cl, r = make_tube(size=size, radius=radius, **kw)
    wl = wander_centerline(size, size / 2, size / 2, amp, wavelength)
    label = voxelize_tube(wl, r, vol.shape)
    return {"vol": vol, "label": label, "true_centerline": cl,
            "shifted_centerline": wl, "radius": r, "amp": amp, "kind": "tube"}


def plant_tube_distractor(k, gap, size=64, radius=3.0, **kw):
    """A second parallel tube `gap` voxels away. Label sits on the true tube,
    shifted by k. The snap must not jump to the distractor."""
    seed = kw.pop("seed", 0)
    vol, cl, r = make_tube(size=size, radius=radius, seed=seed, **kw)
    d2 = make_tube(size=size, radius=radius, seed=seed + 1,
                   centerline=straight_centerline(size, size / 2 + gap, size / 2), **kw)[0]
    vol = np.maximum(vol, d2)
    shifted = cl + k * np.array([0.0, 1.0, 0.0])
    label = voxelize_tube(shifted, r, vol.shape)
    return {"vol": vol, "label": label, "true_centerline": cl,
            "shifted_centerline": shifted, "radius": r, "k": k, "gap": gap,
            "kind": "tube", "distractor_y": size / 2 + gap}


def plant_sheet_shift(k, size=64, thickness=3.0, **kw):
    vol, px, th = make_sheet(size=size, thickness=thickness, **kw)
    label = voxelize_sheet(px + k, th, vol.shape)
    return {"vol": vol, "label": label, "true_plane_x": px,
            "thickness": th, "k": k, "kind": "sheet"}
