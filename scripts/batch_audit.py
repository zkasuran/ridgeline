"""Dataset-wide non-circular audit: run the witness-scored snap on the sampled
Dataset059 patches and report the distribution.

For each patch: pick the head from the structure probe (sheet for the recto surfaces,
sato for the few tube-dominant ones), 2x-audit downsample, snap once, then score the
move on two witnesses the snapper never optimized (meijering, raw_ct) against a random
move of equal magnitude. A patch "confirms" a witness when the snap gain beats twice the
random gain and clears a small floor. The sample (evidence/sample40.txt) is a fixed-seed
random draw across the s1/s4/s5 grid, so it is representative, not cherry-picked; the
three patches the tool was developed on are just the first three cases in the listing.

Run `python3 scripts/download_sample.py` first. Writes evidence/audit40.json.
"""
import os
import sys
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tifffile
import scipy.ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from ridgeline import engine, snapper
from ridgeline.witness import witness_field, _sample, _unit, WITNESS_HEADS

DATADIR = os.path.join(REPO, "data", "audit")
SAMPLE = os.path.join(REPO, "evidence", "sample40.txt")
OUTJSON = os.path.join(REPO, "evidence", "audit40.json")
DOWNSAMPLE, N_SCORE, RADIUS, MARGIN = 2, 4000, 6.0, 8


def audit_one(case):
    img = os.path.join(DATADIR, f"{case}_0000.tif")
    lblp = os.path.join(DATADIR, f"{case}.tif")
    if not (os.path.exists(img) and os.path.exists(lblp)):
        return {"patch": case, "error": "missing files, run download_sample.py"}
    try:
        ct = tifffile.imread(img).astype(np.float32) / 255.0
        lbl = tifffile.imread(lblp) > 0
        if lbl.sum() < 500:
            return {"patch": case, "error": f"tiny label {int(lbl.sum())}"}
        probe = engine.structure_probe(lbl)
        head = probe["head"]
        z = 1.0 / DOWNSAMPLE
        ct_d = ndi.zoom(ct, z, order=1)
        msk_d = ndi.zoom(lbl.astype(np.float32), z, order=0) > 0.5
        _, snapped, _, det = snapper.snap(ct_d, msk_d, head=head, sigmas=(1, 2, 3), radius=RADIUS)
        orig = det["points"]
        mag = np.linalg.norm(snapped - orig, axis=1)
        n = len(orig)
        if n < 50:
            return {"patch": case, "error": f"few medial points {n}"}
        rng = np.random.default_rng(0)
        sel = rng.choice(n, N_SCORE, replace=False) if n > N_SCORE else np.arange(n)
        S = np.array(ct_d.shape)
        border = np.minimum.reduce([orig[sel, 0], orig[sel, 1], orig[sel, 2],
                                    S[0] - 1 - orig[sel, 0], S[1] - 1 - orig[sel, 1],
                                    S[2] - 1 - orig[sel, 2]]) * DOWNSAMPLE
        interior = border > MARGIN
        row = {"patch": case, "structure": probe["dominant"], "head": head,
               "medial_points": int(n),
               "median_move_fullres": round(float(np.median(mag)) * DOWNSAMPLE, 3)}
        for wit in WITNESS_HEADS:
            W = witness_field(ct_d, witness_head=wit)
            Wo, Ws = _sample(W, orig[sel]), _sample(W, snapped[sel])
            Wr = _sample(W, orig[sel] + _unit(rng.normal(size=(len(sel), 3))) * mag[sel][:, None])
            g, r = float((Ws - Wo).mean()), float((Wr - Wo).mean())
            gi = float((Ws - Wo)[interior].mean())
            row[wit] = {"snap": round(g, 4), "random": round(r, 4),
                        "snap_interior": round(gi, 4),
                        "ratio": round(g / (abs(r) + 1e-6), 2),
                        "confirms": bool(g > 2 * abs(r) and g > 0.005)}
        return row
    except Exception as e:
        return {"patch": case, "error": f"{type(e).__name__}: {e}"}


def main():
    cases = [l.strip() for l in open(SAMPLE) if l.strip()]
    rows = []
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(audit_one, c): c for c in cases}
        for f in as_completed(futs):
            r = f.result()
            rows.append(r)
            print(json.dumps(r), flush=True)
    rows.sort(key=lambda r: r["patch"])
    json.dump(rows, open(OUTJSON, "w"), indent=1)

    ok = [r for r in rows if "error" not in r]
    print("\n=== DATASET-WIDE READ (%d patches, %d clean) ===" % (len(rows), len(ok)), flush=True)
    if not ok:
        print("no clean patches, did you run scripts/download_sample.py?", flush=True)
        return 1
    moves = np.array([r["median_move_fullres"] for r in ok])
    struct = {}
    for r in ok:
        struct[r["structure"]] = struct.get(r["structure"], 0) + 1
    print("structure mix across patches:", struct, flush=True)
    print("median snap move (full-res vox): median %.2f  IQR [%.2f, %.2f]" % (
        np.median(moves), np.percentile(moves, 25), np.percentile(moves, 75)), flush=True)
    for wit in WITNESS_HEADS:
        g = np.array([r[wit]["snap"] for r in ok])
        gi = np.array([r[wit]["snap_interior"] for r in ok])
        rd = np.array([r[wit]["random"] for r in ok])
        conf = sum(r[wit]["confirms"] for r in ok)
        print("  [%s] snap gain median %+.4f  interior %+.4f  random median %+.4f  | "
              "confirms %d/%d patches" % (wit, np.median(g), np.median(gi), np.median(rd), conf, len(ok)),
              flush=True)
    errs = [r for r in rows if "error" in r]
    if errs:
        print("errors:", [(r["patch"], r["error"]) for r in errs], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
