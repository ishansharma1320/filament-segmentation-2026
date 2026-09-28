"""Learned instance rescoring: detector conf + single-image box features + refiner mask
features -> P(annotators draw it), trained with MASK-matched labels under the grouped folds,
then swept into pooled OOF PQ in place of the detector confidence.

Inputs (runs/<stamp>/): fold*_oof_boxes.json, oof_crops/*.npz, skipped_candidates.csv
(features + support_frac from scripts/skipped_candidates.py and the mask-support step).

    python scripts/rescore_learned.py runs/20260927-130912 --target half --cuts 0.3 0.4 0.5 0.6
"""
import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cascade  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402
from metric import PQAccumulator  # noqa: E402
from rle import masks_to_rles  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rescore_cascade import features as refiner_features  # noqa: E402

IMG_FEATS = ["long", "short", "aspect", "area", "rho", "contrast", "core_dark", "inside_rel", "ring_rel",
             "plage_frac", "dark_frac", "ring_std"]
_G = {}


def _init(crops, meta, gt, thr, min_area):
    _G.update(crops=crops, meta=meta, gt=gt, thr=thr, min_area=min_area)


def _score_file(args):
    f, cut = args
    crops = _G["crops"][f]
    preds = masks_to_rles(cascade.crop_instances(crops, cut, _G["thr"], _G["min_area"]))
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
    ap.add_argument("--target", default="tp", choices=["tp", "half", "any"],
                    help="tp: the candidate's refined mask is a metric TP (IoU > 0.5) in >= 1/2 of the "
                         "file's records (needs tp_frac in --csv); half: >= 1/2 of annotators drew a "
                         "filament mostly inside the box (rewards big boxes - biased); any: >= 1")
    ap.add_argument("--csv", default="", help="reuse a saved rescore_learned.csv (features + labels) instead of rebuilding")
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    ap.add_argument("--base-cuts", type=float, nargs="+", default=[0.2, 0.25, 0.3])
    ap.add_argument("--thr", type=float, default=0.6)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--density", action="store_true", help="add neighbour-density features")
    a = ap.parse_args()
    run = Path(a.run)
    meta, gt = load_meta(), load_gt_rles()
    crops = cascade.load_crops(run / "oof_crops")
    boxes = {}
    for k in range(5):
        boxes.update(json.load(open(run / f"fold{k}_oof_boxes.json")))

    if a.csv:
        df = pd.read_csv(a.csv)
        print(f"{len(df)} candidates from {a.csv}")
    else:
        df = pd.read_csv(run / "skipped_candidates.csv")
        # join candidates (json order) <-> crop records: verified identical order (conf sequence and
        # boxes match 8821/8821 on v9), so join by index; add refiner mask features
        rows = []
        for f, sub in df.groupby("file", sort=False):
            cb = np.array(boxes[f], np.float32).reshape(-1, 5)
            cr = crops.get(f, [])
            assert len(cr) == len(cb) and np.allclose(cb[:, 4], [c["conf"] for c in cr], atol=1e-4), f
            rf = refiner_features(cr) if cr else np.zeros((0, 5), np.float32)
            for i in range(len(cb)):
                rows.append({"file": f, "idx": i, "crop": i, "ref_meanprob": float(rf[i, 1]),
                             "ref_maxprob": float(rf[i, 2]), "ref_area": float(rf[i, 3])})
        join = pd.DataFrame(rows)
        df = pd.concat([df.reset_index(drop=True), join.drop(columns=["file"]).reset_index(drop=True)], axis=1)
        df = df[df.crop.notna()].copy()
        print(f"{len(df)} candidates joined to crop records ({df.file.nunique()} files)")

    if a.density and "dens150" not in df.columns:
        d150, nn = [], []
        for f, sub in df.groupby("file", sort=False):
            cb = np.array(boxes[f], np.float32).reshape(-1, 5)
            cx, cy = (cb[:, 0] + cb[:, 2]) / 2, (cb[:, 1] + cb[:, 3]) / 2
            for i in sub.idx.astype(int):
                d = np.hypot(cx - cx[i], cy - cy[i]); d = d[d > 1]
                d150.append(int((d < 150).sum())); nn.append(float(d.min()) if len(d) else 2000.0)
        df["dens150"], df["nn_dist"] = d150, nn

    feats = ["conf"] + IMG_FEATS + ["ref_meanprob", "ref_maxprob", "ref_area"] + (["dens150", "nn_dist"] if a.density else [])
    if a.target == "tp":
        y = (df.tp_frac >= 0.5).values
    elif a.target == "half":
        y = (df.support_frac >= 0.5).values
    else:
        y = (df.support_mask >= 1).values
    fold_of = meta["folds"]
    folds = df.file.map(fold_of).values
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    X = df[feats].values
    oof = np.zeros(len(df))
    for k in range(5):
        tr, te = folds != k, folds == k
        clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.04, max_leaf_nodes=15,
                                             l2_regularization=1.0, min_samples_leaf=40)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
    df["learned"] = oof
    print(f"target={a.target}: positives {int(y.sum())}/{len(y)} | AUC learned (fold-OOF) {roc_auc_score(y, oof):.3f} "
          f"vs conf {roc_auc_score(y, df.conf):.3f}")
    m = df.conf >= 0.2
    print(f"  among conf >= 0.2: learned {roc_auc_score(y[m], oof[m]):.3f} vs conf {roc_auc_score(y[m], df.conf[m]):.3f}")
    df.to_csv(run / f"rescore_learned_{a.target}.csv", index=False)

    # PQ: baseline (detector conf) vs learned score substituted for conf in the crop records
    print(f"\npooled OOF PQ, thr {a.thr} min_area {a.min_area} ({df.file.nunique()} files):")
    files = sorted(df.file.unique())
    base = {f: crops[f] for f in files}
    for cut in a.base_cuts:
        s = pooled_pq(base, cut, meta, gt, a.thr, a.min_area, a.procs)
        print(f"  baseline conf >= {cut:.2f}: PQ {s['pq']:.4f} SQ {s['sq']:.3f} RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}", flush=True)
    learned = {}
    for f, sub in df.groupby("file", sort=False):
        cr = crops[f]
        sc = dict(zip(sub.crop.astype(int), sub.learned))
        learned[f] = [dict(c, conf=float(sc.get(i, 0.0))) for i, c in enumerate(cr)]
    for cut in a.cuts:
        s = pooled_pq(learned, cut, meta, gt, a.thr, a.min_area, a.procs)
        print(f"  learned  score >= {cut:.2f}: PQ {s['pq']:.4f} SQ {s['sq']:.3f} RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}", flush=True)


if __name__ == "__main__":
    main()
