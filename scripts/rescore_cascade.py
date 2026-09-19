"""Offline: does the refiner's own mask confidence improve the instance ranking?

For every OOF crop record the instance mask (thr) is built once and its features recorded:
detector conf, mean refiner prob inside the mask, max prob, area. Then PQ is swept over cuts
of a combined score  s = conf^1 * meanprob^b  (b = 0: detector conf only, the v5/v6 recipe).

    python3 scripts/rescore_cascade.py runs/<stamp> --b 0 1 2 4 --cuts 0.15 0.2 0.25 0.3 0.35 0.4
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cascade  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402
from metric import PQAccumulator  # noqa: E402
from refiner import seed_mask  # noqa: E402
from rle import masks_to_rles  # noqa: E402

THR = 0.5


def features(crops):
    """Per crop: mean prob of the thresholded seed-touching mask (0 if empty), max prob, area."""
    out = []
    for c in crops:
        x0, y0, side = int(c["x0"]), int(c["y0"]), int(c["side"])
        p = c["prob"].astype(np.float32) / 255.0
        if p.shape[0] != side:
            p = cv2.resize(p, (side, side), interpolation=cv2.INTER_LINEAR)
        b = (p >= THR).astype(np.uint8)
        mp, area = 0.0, 0
        if b.any():
            n, labels = cv2.connectedComponents(b, connectivity=8)
            bx0, by0, bx1, by1 = c["box"]
            seed = seed_mask([bx0, by0, bx1 - bx0, by1 - by0], x0, y0, side).astype(bool)
            keep = np.unique(labels[seed & (labels > 0)])
            if len(keep):
                m = np.isin(labels, keep)
                mp, area = float(p[m].mean()), int(m.sum())
        out.append((c["conf"], mp, float(p.max()), area))
    return np.array(out, np.float32).reshape(-1, 4)


_G = {}


def _init(oof, feats, meta, gt):
    _G.update(oof=oof, feats=feats, meta=meta, gt=gt)


def _score_file(args):
    f, b, cut = args
    crops, ft = _G["oof"][f], _G["feats"][f]
    s = ft[:, 0] * np.power(np.maximum(ft[:, 1], 1e-3), b)
    scored = [dict(c, conf=float(v)) for c, v in zip(crops, s)]
    preds = masks_to_rles(cascade.crop_instances(scored, cut, THR, 100))
    acc = PQAccumulator()
    for rid in _G["meta"]["records_by_file"][f]:
        acc.add(_G["gt"][rid], preds)
    return acc.tp_ious, acc.fp, acc.fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--b", type=float, nargs="+", default=[0, 1, 2, 4])
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4])
    ap.add_argument("--procs", type=int, default=3)
    a = ap.parse_args()
    meta, gt = load_meta(), load_gt_rles()
    oof = cascade.load_crops(Path(a.run, "oof_crops"))
    feats = {f: features(v) for f, v in oof.items()}
    allf = np.concatenate(list(feats.values()))
    print(f"{len(oof)} files, {len(allf)} crops; meanprob quantiles 10/50/90: "
          f"{np.percentile(allf[:, 1], [10, 50, 90]).round(3)}; corr(conf, meanprob) {np.corrcoef(allf[:, 0], allf[:, 1])[0, 1]:.3f}")
    files = sorted(oof)
    with Pool(a.procs, initializer=_init, initargs=(oof, feats, meta, gt)) as pool:
        for b in a.b:
            for cut in a.cuts:
                if b == 0 and cut < 0.2:
                    continue
                tp, fp, fn = [], 0, 0
                for ious, p, n in pool.map(_score_file, [(f, b, cut) for f in files], chunksize=8):
                    tp.extend(ious); fp += p; fn += n
                d = len(tp) + 0.5 * fp + 0.5 * fn
                print(f"b={b:g} cut={cut:.2f}: PQ {sum(tp) / d:.4f} SQ {sum(tp) / max(len(tp), 1):.3f} "
                      f"RQ {len(tp) / d:.3f} tp {len(tp)} fp {fp} fn {fn}", flush=True)


if __name__ == "__main__":
    main()
