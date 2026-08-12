"""Turn the full-dataset audit into an actionable defect report.

Reads evidence/audit_full.jsonl (from scripts/audit_full.py) and writes:
  - evidence/DEFECT-REPORT.md : the dataset-wide read plus a ranked worst-drift table,
    the "relabel these first" list a maintainer can act on.
  - evidence/defect_ranking.csv : every clean patch, sorted by drift, machine-readable.

Runs on partial results too, so it can be checked while the audit is still going.
"""
import os
import csv
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
JSONL = os.path.join(REPO, "evidence", "audit_full.jsonl")
MD = os.path.join(REPO, "evidence", "DEFECT-REPORT.md")
CSV = os.path.join(REPO, "evidence", "defect_ranking.csv")
TOP = 40


def load():
    rows, errors = [], []
    with open(JSONL) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            (errors if "error" in r else rows).append(r)
    return rows, errors


def main():
    rows, errors = load()
    if not rows:
        print("no clean rows yet")
        return 1
    rows.sort(key=lambda r: r["median_move_fullres"], reverse=True)
    moves = np.array([r["median_move_fullres"] for r in rows])
    n = len(rows)
    conf = {w: sum(r[w]["confirms"] for r in rows) for w in ("meijering", "raw_ct")}
    med = {w: float(np.median([r[w]["snap"] for r in rows])) for w in ("meijering", "raw_ct")}
    rnd = {w: float(np.median([r[w]["random"] for r in rows])) for w in ("meijering", "raw_ct")}
    over3 = int((moves > 3.0).sum())
    over4 = int((moves > 4.0).sum())

    with open(CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["patch", "median_move_vox", "meijering_snap", "meijering_random",
                    "raw_ct_snap", "raw_ct_random", "medial_points"])
        for r in rows:
            w.writerow([r["patch"], r["median_move_fullres"], r["meijering"]["snap"],
                        r["meijering"]["random"], r["raw_ct"]["snap"], r["raw_ct"]["random"],
                        r["medial_points"]])

    lines = []
    lines.append("# Dataset059 label-drift defect report")
    lines.append("")
    lines.append(f"Produced by `scripts/audit_full.py` then `scripts/defect_report.py` over "
                 f"{n} patches of Dataset059 (seg-derived recto surfaces)"
                 + (f", plus {len(errors)} skipped." if errors else "."))
    lines.append("Drift is the median offset from the label medial surface to the CT sheet ridge, "
                 "in full-resolution voxels. It is scored non-circularly: the label was built "
                 "without reading the CT. The snap is graded on two operators it never uses.")
    lines.append("")
    lines.append("## Dataset-wide")
    lines.append("")
    lines.append(f"- patches audited: {n}")
    lines.append(f"- median drift off the CT sheet ridge: {np.median(moves):.2f} voxels "
                 f"(IQR {np.percentile(moves,25):.2f} to {np.percentile(moves,75):.2f}, "
                 f"max {moves.max():.2f})")
    lines.append(f"- patches over 3 voxels off: {over3} ({100*over3/n:.0f}%); over 4 voxels: {over4} "
                 f"({100*over4/n:.0f}%)")
    lines.append(f"- meijering witness: median snap {med['meijering']:+.3f} vs random "
                 f"{rnd['meijering']:+.3f}, confirms {conf['meijering']}/{n}")
    lines.append(f"- raw-CT witness: median snap {med['raw_ct']:+.3f} vs random "
                 f"{rnd['raw_ct']:+.3f}, confirms {conf['raw_ct']}/{n}")
    lines.append("")
    lines.append(f"## Relabel these first: the {TOP} worst-drift patches")
    lines.append("")
    lines.append("| rank | patch | drift (vox) | meijering snap/rand | raw_ct snap/rand |")
    lines.append("| --- | --- | --- | --- | --- |")
    for i, r in enumerate(rows[:TOP], 1):
        m, rc = r["meijering"], r["raw_ct"]
        lines.append(f"| {i} | `{r['patch']}` | {r['median_move_fullres']:.2f} | "
                     f"{m['snap']:+.3f} / {m['random']:+.3f} | {rc['snap']:+.3f} / {rc['random']:+.3f} |")
    lines.append("")
    lines.append("Full ranking of every patch is in `evidence/defect_ranking.csv`.")
    with open(MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"{n} clean, {len(errors)} skipped. median drift {np.median(moves):.2f} vox. "
          f"meijering {conf['meijering']}/{n}, raw_ct {conf['raw_ct']}/{n}. wrote DEFECT-REPORT.md + csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
