"""Loading and aligning real Dataset059 patches.

Image and lbl059 are 300^3 and voxel-aligned. lbl056 (the raw seed used by Layer 3)
is 320^3 and needs a 10-voxel center-crop to line up, confirmed by containment: the
cropped 056 sits entirely inside 059 (a pure geometric expansion), while any other
crop offset falls apart. This module reads a patch stem and returns aligned arrays.
"""
import glob
import os
import numpy as np
import tifffile


def list_stems(data_dir):
    """Patch stems that have both an image and a 059 label present."""
    stems = []
    for p in sorted(glob.glob(os.path.join(data_dir, "*_img.tif"))):
        stem = os.path.basename(p)[:-len("_img.tif")]
        if os.path.exists(os.path.join(data_dir, stem + "_lbl059.tif")):
            stems.append(stem)
    return stems


def center_crop(vol, target):
    """Center-crop a cube to `target` on each axis (e.g. 320 -> 300)."""
    out = vol
    for ax, (s, t) in enumerate(zip(vol.shape, (target,) * vol.ndim)):
        if s > t:
            lo = (s - t) // 2
            out = out.take(range(lo, lo + t), axis=ax)
    return out


def load_patch(data_dir, stem, want_056=False):
    """Return (img float32, lbl059 bool[, seed056 bool]). 056 is center-cropped to
    the image size so it aligns with 059."""
    img = tifffile.imread(os.path.join(data_dir, stem + "_img.tif")).astype(np.float32)
    l59 = tifffile.imread(os.path.join(data_dir, stem + "_lbl059.tif")) > 0
    if not want_056:
        return img, l59
    p56 = os.path.join(data_dir, stem + "_lbl056.tif")
    seed = None
    if os.path.exists(p56):
        raw = tifffile.imread(p56) > 0
        seed = center_crop(raw, img.shape[0])
    return img, l59, seed


def crop_to_label(img, mask, margin=12):
    """Crop image and mask to the label bounding box plus a margin. The heavy ridge
    filters run on the crop, not the full cube (DESIGN CPU discipline)."""
    idx = np.array(np.where(mask))
    if idx.size == 0:
        return img, mask
    sl = tuple(slice(max(0, a.min() - margin), min(s, a.max() + margin + 1))
               for a, s in zip(idx, mask.shape))
    return img[sl], mask[sl]
