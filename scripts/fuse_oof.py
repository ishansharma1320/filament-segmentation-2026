"""Offline: OOF value of fusing the detectors of two runs (fold k of run A + fold k of run B,
both trained without fold k), refined by run A's fold refiners on CPU. Compares single-run
boxes vs fused boxes under the same refiner (no TTA, to keep it tractable on CPU).

    python3 scripts/fuse_oof.py runs/<A> runs/<B> [--eval-files N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cascade  # noqa: E402
from dataset import load_gt_rles, load_meta  # noqa: E402
from detector import fuse_boxes  # noqa: E402
from refiner import load_refiner, read_native, refine_boxes  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--eval-files", type=int, default=0)
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35])
    args = ap.parse_args()
    torch.set_num_threads(2)
    meta, gt = load_meta(), load_gt_rles()
    dev = torch.device("cpu")
    single, fused = {}, {}
    for k in range(5):
        ba = {f: np.array(v, np.float32).reshape(-1, 5) for f, v in json.load(open(f"{args.a}/fold{k}_oof_boxes.json")).items()}
        bb = {f: np.array(v, np.float32).reshape(-1, 5) for f, v in json.load(open(f"{args.b}/fold{k}_oof_boxes.json")).items()}
        files = sorted(ba)
        if args.eval_files:
            files = files[:args.eval_files]
        ref, _ = load_refiner(f"{args.a}/fold{k}_refiner.pt", dev)
        for i, f in enumerate(files):
            img = read_native(f)
            fb = fuse_boxes([ba[f], bb[f]])
            single[f] = refine_boxes([ref], img, ba[f][ba[f][:, 4] >= 0.1], dev, tta=False)
            fused[f] = refine_boxes([ref], img, fb[fb[:, 4] >= 0.1], dev, tta=False)
            if i % 25 == 0:
                print(f"fold {k} {i}/{len(files)}", flush=True)
    for name, crops in (("single(A)", single), ("fused(A+B)", fused)):
        for cut in args.cuts:
            s = cascade.score_crops(crops, meta, gt, cut, 0.5, 100)
            print(f"{name} cut={cut}: PQ {s['pq']:.4f} SQ {s['sq']:.3f} RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}", flush=True)
