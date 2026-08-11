"""Download the exact 40-patch sample the dataset-wide audit ran on.

The sample is a fixed-seed random draw across the whole s1/s4/s5 grid of Dataset059
(seg-derived-recto-surfaces, 1754 patches). The case ids are pinned in
evidence/sample40.txt so the download is reproducible and matches evidence/audit40.json.
The server is public and anonymous, no credentials. Each patch is a CT image
(`{case}_0000.tif`) and its frangiedt label (`{case}.tif`), both 300^3.
"""
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://dl.ash2txt.org/datasets/seg-derived-recto-surfaces"
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SAMPLE = os.path.join(REPO, "evidence", "sample40.txt")
OUT = os.path.join(REPO, "data", "audit")


def fetch(url, dest):
    if os.path.exists(dest) and os.path.getsize(dest) > 1000:
        return f"skip {os.path.basename(dest)}"
    urllib.request.urlretrieve(url, dest)
    return f"got  {os.path.basename(dest)} ({os.path.getsize(dest) // 1024} KiB)"


def one(case):
    a = fetch(f"{BASE}/imagesTr/{case}_0000.tif", os.path.join(OUT, f"{case}_0000.tif"))
    b = fetch(f"{BASE}/labelsTr/{case}.tif", os.path.join(OUT, f"{case}.tif"))
    return f"{a} | {b}"


def main():
    os.makedirs(OUT, exist_ok=True)
    cases = [l.strip() for l in open(SAMPLE) if l.strip()]
    print(f"downloading {len(cases)} patches to {OUT}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for line in ex.map(one, cases):
            print(line, flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
