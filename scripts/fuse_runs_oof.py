"""Offline: OOF value of fusing the fold-k detectors of SEVERAL runs (every run's
fold{k}_oof_boxes.json was produced without fold k), refined by the LAST run's fold
refiner on CPU (no TTA). Reports single (last run's boxes) vs fused, same refiner.

    python3 scripts/fuse_runs_oof.py runs/A runs/B runs/C ... [--eval-files N]
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
from detector import box_pr, fuse_boxes  # noqa: E402
from refiner import load_refiner, read_native, refine_boxes  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--eval-files", type=int, default=0)
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.2, 0.25, 0.3, 0.35])
    args = ap.parse_args()
    torch.set_num_threads(2)
    meta, gt = load_meta(), load_gt_rles()
    dev = torch.device("cpu")
    last = args.runs[-1]
    single, fused, sb, fb_all = {}, {}, {}, {}
    for k in range(5):
        sets = [{f: np.array(v, np.float32).reshape(-1, 5) for f, v in json.load(open(f"{r}/fold{k}_oof_boxes.json")).items()}
                for r in args.runs]
        files = sorted(sets[-1])
        if args.eval_files:
            files = files[:args.eval_files]
        ref, _ = load_refiner(f"{last}/fold{k}_refiner.pt", dev)
        for i, f in enumerate(files):
            img = read_native(f)
            fb = fuse_boxes([s.get(f, np.zeros((0, 5), np.float32)) for s in sets])
            sb[f], fb_all[f] = sets[-1][f], fb
            single[f] = refine_boxes([ref], img, sets[-1][f][sets[-1][f][:, 4] >= 0.1], dev, tta=False)
            fused[f] = refine_boxes([ref], img, fb[fb[:, 4] >= 0.1], dev, tta=False)
            if i % 50 == 0:
                print(f"fold {k} {i}/{len(files)}", flush=True)
    for name, bx in (("single", sb), ("fused", fb_all)):
        for c in (0.2, 0.3):
            r = box_pr(bx, meta, gt, c)
            print(f"boxes {name} conf>={c}: P {r['precision']:.3f} R {r['recall']:.3f}", flush=True)
    for name, crops in (("single(last)", single), (f"fused({len(args.runs)})", fused)):
        for cut in args.cuts:
            s = cascade.score_crops(crops, meta, gt, cut, 0.5, 100)
            print(f"{name} cut={cut}: PQ {s['pq']:.4f} SQ {s['sq']:.3f} RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}", flush=True)
