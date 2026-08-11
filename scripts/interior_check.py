"""Robustness supplement to the real-data finding: is the witness-scored drift a
boundary or thickness artifact, or does it survive on interior medial points?

Layer-3 taught us to distrust anything that lives at the volume face: 100% of its
apparent out-of-radius voxels sat within 2 voxels of a crop boundary (a 056->059
frame-offset artifact). So here we recompute the same sheetness snap and report the
independent-witness gain twice: over all scored medial points, and restricted to
interior points more than 8 voxels from any face. If the interior gain matches the
full gain, the drift is a real property of the surface, not an edge effect.
"""
import sys, glob, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tifffile
import scipy.ndimage as ndi
from ridgeline import engine, snapper
from ridgeline.witness import witness_field, _sample, _unit, WITNESS_HEADS

DATA = "/home/asuran/Downloads/hackathon-hq/work/vesuvius/labelqc/crux/data"
DOWNSAMPLE, N_SCORE, RADIUS = 2, 4000, 6.0
MARGIN = 8                                               # interior = >8 full-res vox from any face

pairs = sorted(set(p[:-8] for p in glob.glob(f"{DATA}/*_img.tif")))
rng = np.random.default_rng(0)
for b in pairs:
    name = os.path.basename(b)
    ct = tifffile.imread(f"{b}_img.tif").astype(np.float32) / 255.0
    lbl = tifffile.imread(f"{b}_lbl059.tif") > 0
    head = engine.structure_probe(lbl)["head"]
    z = 1.0 / DOWNSAMPLE
    ct_d = ndi.zoom(ct, z, order=1)
    msk_d = ndi.zoom(lbl.astype(np.float32), z, order=0) > 0.5
    _, snapped_pts, _, det = snapper.snap(ct_d, msk_d, head=head, sigmas=(1, 2, 3), radius=RADIUS)
    orig = det["points"]
    mag = np.linalg.norm(snapped_pts - orig, axis=1)

    n = len(orig)
    sel = rng.choice(n, N_SCORE, replace=False) if n > N_SCORE else np.arange(n)
    S = np.array(ct_d.shape)
    border = np.minimum.reduce([orig[sel, 0], orig[sel, 1], orig[sel, 2],
                                S[0] - 1 - orig[sel, 0], S[1] - 1 - orig[sel, 1],
                                S[2] - 1 - orig[sel, 2]]) * DOWNSAMPLE
    interior = border > MARGIN

    row = {"patch": name, "head": head, "scored": int(len(sel)),
           "interior_frac": round(float(interior.mean()), 3)}
    for wit in WITNESS_HEADS:
        W = witness_field(ct_d, witness_head=wit)
        Wo, Ws = _sample(W, orig[sel]), _sample(W, snapped_pts[sel])
        Wr = _sample(W, orig[sel] + _unit(rng.normal(size=(len(sel), 3))) * mag[sel][:, None])
        gain = Ws - Wo
        rgain = Wr - Wo
        row[wit] = {
            "gain_all": round(float(gain.mean()), 4),
            "gain_interior": round(float(gain[interior].mean()), 4),
            "random_all": round(float(rgain.mean()), 4),
            "random_interior": round(float(rgain[interior].mean()), 4),
        }
    print(json.dumps(row), flush=True)
