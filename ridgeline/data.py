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


def _is_zarr(path):
    """A path is a zarr store if it ends .zarr or is a directory holding zarr metadata."""
    p = str(path)
    if p.rstrip("/").endswith(".zarr"):
        return True
    return os.path.isdir(p) and (os.path.exists(os.path.join(p, "zarr.json"))
                                 or os.path.exists(os.path.join(p, ".zgroup"))
                                 or os.path.exists(os.path.join(p, ".zarray")))


def load_array(path):
    """Read a volume from a `.tif` or a zarr store into a numpy array. For an OME-Zarr
    multiscale group the full-resolution level (dataset "0", else the first array) is
    read, so the community formats the pipeline emits load without conversion."""
    if not _is_zarr(path):
        return tifffile.imread(path)
    try:
        import zarr
    except ImportError as exc:
        raise ImportError("reading a zarr store needs the 'zarr' package: pip install zarr") from exc
    node = zarr.open(str(path), mode="r")
    if hasattr(node, "shape"):                          # already an array
        return np.asarray(node[:])
    keys = list(node.array_keys()) if hasattr(node, "array_keys") else list(node.keys())
    if not keys:
        raise ValueError(f"{path}: zarr group has no arrays")
    name = "0" if "0" in keys else keys[0]              # OME-Zarr full-res is dataset 0
    return np.asarray(node[name][:])


def write_mask(path, mask, chunk=128):
    """Write a boolean/uint8 mask to `.tif` or a zarr array. A `.zarr` path is written
    as a plain zarr v2 uint8 array (a community-standard Zarr array), so the corrected
    label drops straight back into the pipeline."""
    mask = mask.astype(np.uint8)
    if not _is_zarr(path):
        tifffile.imwrite(path, mask)
        return
    import zarr
    chunks = tuple(min(chunk, s) for s in mask.shape)
    arr = zarr.open_array(str(path), mode="w", shape=mask.shape, chunks=chunks,
                          dtype="uint8", zarr_format=2)
    arr[:] = mask
