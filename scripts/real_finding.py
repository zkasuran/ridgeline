"""The decisive non-circular test on the real recto-surface patches of Dataset059.

These patches are sheet-dominant (confirmed by structure_probe), so we snap with the
sheetness head, then SCORE the move in a witness field that is NOT the snap signal:
meijering (a different Hessian ridge family) and raw_ct (zeroth-order intensity, no
Hessian at all). If the sheetness snap lands labels on a higher independent-witness
ridge and a random move of equal magnitude does not, the drift is real, not an artifact
of the snap operator. This is the exact test TAUIL-Abd-Elilah's m7 surface version
failed (normal snapping did not beat a random control on the diffuse surface field).

The snap is computed once per patch and scored against every witness, so the snapper
never sees the field it is graded on. A random-direction control of equal magnitude
rides along on every witness (TAUIL discipline).
"""
import sys, glob, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, tifffile
import scipy.ndimage as ndi
from ridgeline import engine, snapper
from ridgeline.witness import witness_field, _sample, _unit, WITNESS_HEADS

DATA = "/home/asuran/Downloads/hackathon-hq/work/vesuvius/labelqc/crux/data"
DOWNSAMPLE = 2
N_SCORE = 4000
SNAP_SIGMAS = (1, 2, 3)
WIT_SIGMAS = (1, 2, 3, 4)
RADIUS = 6.0

pairs = sorted(set(p[:-8] for p in glob.glob(f"{DATA}/*_img.tif")))
rng = np.random.default_rng(0)
out = []
for b in pairs:
    name = os.path.basename(b)
    ct = tifffile.imread(f"{b}_img.tif").astype(np.float32) / 255.0
    lbl = tifffile.imread(f"{b}_lbl059.tif") > 0
    probe = engine.structure_probe(lbl)
    snap_head = probe["head"]                            # "sheet" for recto patches

    # 2x-audit downsample, then snap ONCE with the structure's own head
    z = 1.0 / DOWNSAMPLE
    ct_d = ndi.zoom(ct, z, order=1)
    msk_d = ndi.zoom(lbl.astype(np.float32), z, order=0) > 0.5
    corrected, snapped_pts, info, det = snapper.snap(
        ct_d, msk_d, head=snap_head, sigmas=SNAP_SIGMAS, radius=RADIUS)
    orig_pts = det["points"]
    move = snapped_pts - orig_pts
    mag = np.linalg.norm(move, axis=1)

    n = len(orig_pts)
    sel = np.arange(n)
    if n > N_SCORE:
        sel = rng.choice(n, N_SCORE, replace=False)

    row = {"patch": name, "structure": probe["dominant"], "mix": probe["mix"],
           "snap_head": snap_head, "medial_points": int(n),
           "median_snap_move_fullres": round(float(np.median(mag)) * DOWNSAMPLE, 3)}

    # score the SAME snap against each independent witness (never the snap head)
    for wit in WITNESS_HEADS:
        W = witness_field(ct_d, witness_head=wit, sigmas=WIT_SIGMAS)
        W_orig = _sample(W, orig_pts[sel])
        W_snap = _sample(W, snapped_pts[sel])
        rand = _unit(rng.normal(size=(len(sel), 3))) * mag[sel][:, None]
        W_rand = _sample(W, orig_pts[sel] + rand)
        snap_gain = float((W_snap - W_orig).mean())
        rand_gain = float((W_rand - W_orig).mean())
        row[wit] = {
            "witness_gain_snap": round(snap_gain, 4),
            "witness_gain_random": round(rand_gain, 4),
            "ratio": round(snap_gain / (abs(rand_gain) + 1e-6), 2),
            "witness_at_orig": round(float(W_orig.mean()), 4),
            "witness_at_snapped": round(float(W_snap.mean()), 4),
        }
    out.append(row)
    print(json.dumps(row), flush=True)

print("\n=== READ ===", flush=True)
for r in out:
    print(f"\n{r['patch']}  ({r['structure']}, snap={r['snap_head']}, "
          f"median move {r['median_snap_move_fullres']} full-res vox)", flush=True)
    for wit in WITNESS_HEADS:
        d = r[wit]
        real = d["witness_gain_snap"] > 2 * abs(d["witness_gain_random"]) and d["witness_gain_snap"] > 0.005
        verdict = ("REAL drift: snap beats random on this independent witness"
                   if real else "DIFFUSE: snap not distinguishable from random here")
        print(f"  [{wit:9s}] snap {d['witness_gain_snap']:+.4f}  "
              f"random {d['witness_gain_random']:+.4f}  ratio {d['ratio']:+.1f}  -> {verdict}",
              flush=True)
