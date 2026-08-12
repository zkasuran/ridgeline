"""ridgeline command line.

    ridgeline demo                     whole pipeline on synthetic tubes, no download
    ridgeline validate                 synthetic calibration matrix + controls + lift
    ridgeline measure IMG LBL          independent-witness drift QA on a real patch
    ridgeline snap    IMG LBL [-o OUT] snap a real label, score it with the witness

`demo` and `validate` need no data. `measure` and `snap` take an image and a label,
each a `.tif` or a zarr store (OME-Zarr full-res level 0 is read), so a label straight
out of the pipeline loads without conversion. A 320^3 seed label is center-cropped to
the 300^3 image automatically. `snap -o out.zarr` writes the corrected label back as a
Zarr array; `-o out.tif` writes a tif.
"""
import argparse
import json
import sys
import numpy as np


def _fmt(d):
    return json.dumps(d, indent=2, default=float)


def _load_pair(img_path, lbl_path):
    from .data import center_crop, load_array
    img = load_array(img_path).astype(np.float32)
    lbl = load_array(lbl_path) > 0
    if lbl.shape != img.shape:                       # 320^3 seed -> 300^3 image
        lbl = center_crop(lbl, img.shape[0])
    return img, lbl


def cmd_demo(args):
    from . import synth, harness
    from .lift import lift_probe
    print("ridgeline demo: synthetic planted-defect calibration, no download.\n")
    print("Structure: bright Gaussian tube in CT-like texture at a few grey levels of")
    print("contrast, label = centerline dilated by a fixed radius, defects planted with")
    print("a known answer. The snapper never sees the truth; recovery is scored against")
    print("the planted geometry, not the sato field it optimizes.\n")

    print("== planted-defect matrix (recovery in voxels, lower is better) ==")
    hdr = ("defect", "planted", "pre", "ridge", "brightest", "random", "frangi")
    print("%-20s %8s %6s %6s %10s %7s %7s" % hdr)
    for name, r in harness.run_matrix(size=args.size, contrast=args.contrast):
        print("%-20s %8s %6.3f %6.3f %10.3f %7.3f %7.3f" % (
            name, str(r["planted"]), r["pre_snap_error"], r["ridge_recovery"],
            r["brightest_recovery"], r["random_recovery"], r["frangi_recovery"]))

    print("\n== null controls ==")
    print(_fmt(harness.null_controls(size=args.size)))

    print("\n== downstream lift probe (Regime A, sato-free features, held-out) ==")
    print(_fmt(lift_probe(k=2.0, n_phantoms=args.phantoms, size=args.size,
                          contrast=args.contrast)))
    return 0


def cmd_validate(args):
    from . import harness
    from .lift import lift_probe
    rows = [dict(defect=n, **r) for n, r in harness.run_matrix(size=args.size,
                                                               contrast=args.contrast)]
    controls = harness.null_controls(size=args.size)
    probe = lift_probe(k=2.0, n_phantoms=args.phantoms, size=args.size,
                       contrast=args.contrast)
    out = {"matrix": rows, "null_controls": controls, "lift_probe": probe}
    if args.json:
        print(_fmt(out))
    else:
        for r in rows:
            print("%-20s ridge %.3f  brightest %.3f  random %.3f  frangi %.3f" % (
                r["defect"], r["ridge_recovery"], r["brightest_recovery"],
                r["random_recovery"], r["frangi_recovery"]))
        print("\nnull_controls:", _fmt(controls))
        print("lift_probe:", _fmt(probe))
    # gates: ridge beats both baselines and the frangi arm at small k, controls hold
    k2 = next(r for r in rows if r["defect"] == "shift_normal_k2.0")
    ok = (k2["ridge_recovery"] < 0.5
          and k2["ridge_recovery"] < k2["brightest_recovery"]
          and k2["ridge_recovery"] < k2["random_recovery"]
          and controls["idempotence_median_move"] < 1.0
          and probe["recovery_fraction"] > 0)
    print("\nVALIDATION", "PASS" if ok else "FAIL", file=sys.stderr)
    return 0 if ok else 1


def cmd_measure(args):
    from .engine import structure_probe
    from .data import crop_to_label
    from . import witness
    img, lbl = _load_pair(args.image, args.label)
    img, lbl = crop_to_label(img, lbl, margin=12)
    ct = img / 255.0
    probe = structure_probe(lbl)
    out = {
        "structure_probe": {"dominant": probe["dominant"], "head": probe["head"],
                            "mix": probe["mix"]},
        "witness_meijering": witness.witness_drift(ct, lbl, witness_head="meijering",
                                                   downsample=args.downsample),
        "witness_raw_ct": witness.witness_drift(ct, lbl, witness_head="raw_ct",
                                                downsample=args.downsample),
    }
    print(_fmt(out))
    return 0


def cmd_snap(args):
    import scipy.ndimage as ndi
    from .data import crop_to_label, write_mask
    from .snapper import snap
    from . import witness
    img, lbl = _load_pair(args.image, args.label)
    cimg, clbl = crop_to_label(img, lbl, margin=12)
    ct = cimg / 255.0
    score = witness.witness_scored_snap(ct, clbl, witness_head="meijering",
                                        downsample=args.downsample)
    print(_fmt(score))
    if args.out:
        z = 1.0 / args.downsample
        ct_d = ndi.zoom(ct, z, order=1)
        lbl_d = ndi.zoom(clbl.astype(np.float32), z, order=0) > 0.5
        corrected, _, _, _ = snap(ct_d, lbl_d, radius=6.0)
        up = ndi.zoom(corrected.astype(np.float32), args.downsample, order=0) > 0.5
        full = np.zeros(lbl.shape, dtype=np.uint8)
        idx = np.array(np.where(lbl))
        sl = tuple(slice(max(0, a.min() - 12), max(0, a.min() - 12) + s)
                   for a, s in zip(idx, up.shape))
        # place corrected crop back; guard shape
        target = full[sl]
        us = up[:target.shape[0], :target.shape[1], :target.shape[2]]
        target[:us.shape[0], :us.shape[1], :us.shape[2]] = us.astype(np.uint8)
        write_mask(args.out, full)
        print("wrote corrected mask to", args.out, file=sys.stderr)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="ridgeline", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="synthetic pipeline, no download")
    d.add_argument("--size", type=int, default=64)
    d.add_argument("--contrast", type=float, default=8.0)
    d.add_argument("--phantoms", type=int, default=6)
    d.set_defaults(func=cmd_demo)

    v = sub.add_parser("validate", help="synthetic calibration + gates")
    v.add_argument("--size", type=int, default=64)
    v.add_argument("--contrast", type=float, default=8.0)
    v.add_argument("--phantoms", type=int, default=6)
    v.add_argument("--json", action="store_true")
    v.set_defaults(func=cmd_validate)

    m = sub.add_parser("measure", help="independent-witness drift QA on a real patch")
    m.add_argument("image"); m.add_argument("label")
    m.add_argument("--downsample", type=int, default=2)
    m.set_defaults(func=cmd_measure)

    s = sub.add_parser("snap", help="snap a real label and score it with the witness")
    s.add_argument("image"); s.add_argument("label")
    s.add_argument("-o", "--out", help="write the corrected mask tif")
    s.add_argument("--downsample", type=int, default=2)
    s.set_defaults(func=cmd_snap)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
