"""Decisive test of the sequence idea under the metric: does temporal persistence, added as
a feature to the metric-true rescorer, raise pooled OOF PQ on the long-baseline files?

Joins probe candidates (persistence_probe_long_xy.csv: conf >= 0.05 boxes on the labelled
frame with per-baseline persistence) to the v9 OOF candidates (rescore_learned_tp.csv:
features + metric-true label tp_frac + crop index) by box IoU >= 0.85 within the file.
Trains three fold-grouped GBMs on the joined files: conf only is the baseline ranking;
A = conf + image/refiner features; B = A + persistence features. Sweeps each score's cut
into pooled PQ over the joined files' records.

    python scripts/persistence_pq_test.py runs/20260927-130912
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cascade  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402
from detector import box_iou  # noqa: E402
from metric import PQAccumulator  # noqa: E402
from rle import masks_to_rles  # noqa: E402

IMG_FEATS = ["long", "short", "aspect", "area", "rho", "contrast", "core_dark", "inside_rel", "ring_rel",
             "plage_frac", "dark_frac", "ring_std", "ref_meanprob", "ref_maxprob", "ref_area", "dens150", "nn_dist"]
_G = {}


def _init(crops, meta, gt, thr, min_area):
    _G.update(crops=crops, meta=meta, gt=gt, thr=thr, min_area=min_area)


def _score_file(args):
    f, cut = args
    preds = masks_to_rles(cascade.crop_instances(_G["crops"][f], cut, _G["thr"], _G["min_area"]))
    acc = PQAccumulator()
    for rid in _G["meta"]["records_by_file"][f]:
        acc.add(_G["gt"][rid], preds)
    return acc.tp_ious, acc.fp, acc.fn


def pooled_pq(crops_by_file, cut, meta, gt, thr, min_area, procs):
    with Pool(procs, initializer=_init, initargs=(crops_by_file, meta, gt, thr, min_area)) as pool:
        parts = pool.map(_score_file, [(f, cut) for f in crops_by_file])
    acc = PQAccumulator()
    for ious, fp, fn in parts:
        acc.tp_ious.extend(ious); acc.fp += fp; acc.fn += fn
    return acc.summary()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--probe", default="runs/persistence_probe_long_xy.csv")
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    ap.add_argument("--base-cuts", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35])
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    run = Path(a.run)
    meta, gt = load_meta(), load_gt_rles()
    cand = pd.read_csv(run / "rescore_learned_tp.csv")
    probe = pd.read_csv(a.probe)
    pcols = [c for c in probe.columns if c.startswith("pers|")]
    crops = cascade.load_crops(run / "oof_crops")

    # join probe rows -> OOF candidates by box IoU within file
    files = sorted(set(probe.file) & set(cand.file))
    rows = []
    for f in files:
        c = cand[cand.file == f]
        p = probe[probe.file == f]
        cb = np.stack([c.x0, c.y0, c.x1, c.y1], 1) if {"x0", "x1"} <= set(c.columns) else None
        if cb is None:   # candidate boxes from the json (same order as crop index)
            import json
            allb = {}
            for k in range(5):
                allb.update(json.load(open(run / f"fold{k}_oof_boxes.json")))
            cb = np.array(allb[f], np.float32).reshape(-1, 5)[:, :4][c.crop.astype(int).values]
        pb = np.stack([p.x0, p.y0, p.x1, p.y1], 1)
        m = box_iou(cb.astype(np.float32), pb.astype(np.float32))
        for i in range(len(c)):
            j = int(m[i].argmax())
            if m[i, j] >= 0.85:
                r = c.iloc[i].to_dict()
                r.update({k: float(p.iloc[j][k]) for k in pcols if not np.isnan(p.iloc[j][k])})
                rows.append(r)
    df = pd.DataFrame(rows)
    for k in pcols:
        if k not in df.columns:
            df[k] = np.nan
    df["pers_mean"] = df[pcols].mean(axis=1)
    df["pers_long"] = df[[c for c in pcols if "3h" in c or "6h" in c]].mean(axis=1)
    print(f"{len(df)} OOF candidates joined to probe boxes on {df.file.nunique()} files "
          f"(of {len(cand[cand.file.isin(files)])} candidates on those files)")

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    y = (df.tp_frac >= 0.5).values
    folds = df.file.map(meta["folds"]).values
    pers_feats = pcols + ["pers_mean", "pers_long"]

    def fit(feats):
        X = df[feats].values; oof = np.zeros(len(df))
        for k in range(5):
            tr, te = folds != k, folds == k
            if te.sum() == 0 or tr.sum() == 0:
                continue
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.04, max_leaf_nodes=15,
                                                 l2_regularization=1.0, min_samples_leaf=30)
            clf.fit(X[tr], y[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        return oof

    df["A"] = fit(["conf"] + IMG_FEATS)
    df["B"] = fit(["conf"] + IMG_FEATS + pers_feats)
    df["P"] = fit(["conf"] + pers_feats)
    print(f"AUC(metric TP >= half): conf {roc_auc_score(y, df.conf):.3f} | A conf+features {roc_auc_score(y, df.A):.3f} "
          f"| B A+persistence {roc_auc_score(y, df.B):.3f} | P conf+persistence {roc_auc_score(y, df.P):.3f}")
    m = df.conf >= 0.2
    print(f"  conf >= 0.2: conf {roc_auc_score(y[m], df.conf[m]):.3f} | A {roc_auc_score(y[m], df.A[m]):.3f} | B {roc_auc_score(y[m], df.B[m]):.3f} | P {roc_auc_score(y[m], df.P[m]):.3f}")
    df.to_csv(run / "persistence_pq_test.csv", index=False)

    # PQ on the joined files: unjoined candidates keep their detector conf (they are the ones the
    # probe never saw at conf >= 0.05 - rare); scores substituted for conf in the crop records
    print(f"\npooled OOF PQ on {df.file.nunique()} files, thr {a.thr} min_area {a.min_area}:")
    base = {f: crops[f] for f in df.file.unique()}
    for cut in a.base_cuts:
        s = pooled_pq(base, cut, meta, gt, a.thr, a.min_area, a.procs)
        print(f"  baseline conf >= {cut:.2f}: PQ {s['pq']:.4f} RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}", flush=True)
    for name in ("A", "B", "P"):
        sc = {(r.file, int(r.crop)): getattr(r, name) for r in df.itertuples()}
        rescored = {f: [dict(c, conf=float(sc.get((f, i), c["conf"]))) for i, c in enumerate(crops[f])] for f in base}
        for cut in a.cuts:
            s = pooled_pq(rescored, cut, meta, gt, a.thr, a.min_area, a.procs)
            print(f"  {name} >= {cut:.2f}: PQ {s['pq']:.4f} RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}", flush=True)


if __name__ == "__main__":
    main()
