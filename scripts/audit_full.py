"""Full-dataset non-circular audit: every patch in Dataset059, streamed.

Downloads each patch, runs the same witness-scored snap as scripts/batch_audit.py,
deletes the raw tif, and appends one JSON line to evidence/audit_full.jsonl. Resumable:
rerunning skips cases already in the jsonl, so a stop or a crash never loses progress.
This is the dataset-wide version over all 1754 patches, so the claim is the whole
released set rather than a 40-patch sample. CPU-only, one BLAS thread per worker.
"""
import os
import sys
import json
import tempfile
import urllib.request

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import tifffile
import scipy.ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
from ridgeline import engine, snapper
from ridgeline.witness import witness_field, _sample, _unit, WITNESS_HEADS

BASE = "https://dl.ash2txt.org/datasets/seg-derived-recto-surfaces"
CASES = os.path.join(REPO, "evidence", "all_cases.txt")
OUT = os.path.join(REPO, "evidence", "audit_full.jsonl")
SCRATCH = os.environ.get("RIDGELINE_SCRATCH", os.path.join(tempfile.gettempdir(), "ridgeline_scratch"))
DOWNSAMPLE, N_SCORE, RADIUS, MARGIN = 2, 4000, 6.0, 8


def _dl(url, dest):
    urllib.request.urlretrieve(url, dest)


def audit_one(case):
    os.makedirs(SCRATCH, exist_ok=True)
    img = os.path.join(SCRATCH, f"{case}_0000.tif")
    lbl = os.path.join(SCRATCH, f"{case}.tif")
    try:
        _dl(f"{BASE}/imagesTr/{case}_0000.tif", img)
        _dl(f"{BASE}/labelsTr/{case}.tif", lbl)
        ct = tifffile.imread(img).astype(np.float32) / 255.0
        mask = tifffile.imread(lbl) > 0
        if mask.sum() < 500:
            return {"patch": case, "error": f"tiny label {int(mask.sum())}"}
        probe = engine.structure_probe(mask)
        head = probe["head"]
        z = 1.0 / DOWNSAMPLE
        ct_d = ndi.zoom(ct, z, order=1)
        msk_d = ndi.zoom(mask.astype(np.float32), z, order=0) > 0.5
        _, snapped, _, det = snapper.snap(ct_d, msk_d, head=head, sigmas=(1, 2, 3), radius=RADIUS)
        orig = det["points"]
        magv = np.linalg.norm(snapped - orig, axis=1)
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
               "median_move_fullres": round(float(np.median(magv)) * DOWNSAMPLE, 3)}
        for wit in WITNESS_HEADS:
            W = witness_field(ct_d, witness_head=wit)
            Wo, Ws = _sample(W, orig[sel]), _sample(W, snapped[sel])
            Wr = _sample(W, orig[sel] + _unit(rng.normal(size=(len(sel), 3))) * magv[sel][:, None])
            g, r = float((Ws - Wo).mean()), float((Wr - Wo).mean())
            gi = float((Ws - Wo)[interior].mean())
            row[wit] = {"snap": round(g, 4), "random": round(r, 4),
                        "snap_interior": round(gi, 4),
                        "ratio": round(g / (abs(r) + 1e-6), 2),
                        "confirms": bool(g > 2 * abs(r) and g > 0.005)}
        return row
    except Exception as e:
        return {"patch": case, "error": f"{type(e).__name__}: {e}"}
    finally:
        for p in (img, lbl):
            try:
                os.remove(p)
            except OSError:
                pass


# PLACEHOLDER_MAIN
def main():
    cases = [l.strip() for l in open(CASES) if l.strip()]
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT):
            try:
                done.add(json.loads(line)["patch"])
            except Exception:
                pass
    todo = [c for c in cases if c not in done]
    print(f"{len(cases)} total, {len(done)} already done, {len(todo)} to go", flush=True)
    workers = int(os.environ.get("WORKERS", "12"))
    with open(OUT, "a") as sink, ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(audit_one, c): c for c in todo}
        n = 0
        for f in as_completed(futs):
            r = f.result()
            sink.write(json.dumps(r) + "\n")
            sink.flush()
            n += 1
            if n % 25 == 0 or "error" in r:
                tag = r.get("error", f"{r.get('median_move_fullres')}vox")
                print(f"  {n}/{len(todo)}  {r['patch']}  {tag}", flush=True)
    print("audit_full complete", flush=True)


if __name__ == "__main__":
    sys.exit(main())
