"""Build a test submission by fusing the fold detectors of several pulled runs (25 YOLO
models) with weighted boxes fusion, refined by the LAST run's fold refiners (with flip
TTA, as the kernel does). CPU-only; no kernel run. Per-run detector settings are given as
run:imgsz:source (source = cache | native).

    python3 scripts/fuse_runs_test.py runs/A:1024:cache ... runs/E:1280:native --cut 0.25 --out sub.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cascade  # noqa: E402
from config import CACHE, DET_CONF_MIN, DET_MAX_DET, DET_NMS_IOU, H  # noqa: E402
from data import test_files, test_img_dir  # noqa: E402
from dataset import load_meta  # noqa: E402
from detector import fuse_boxes  # noqa: E402
from refiner import load_refiner, read_native, refine_boxes  # noqa: E402


def predict(model, files, imgsz, source, size):
    if source == "native":
        paths = [str(test_img_dir() / f) for f in files]; scale = 1.0
    else:
        paths = [str(CACHE / f"test{size}" / (f.rsplit(".", 1)[0] + ".png")) for f in files]; scale = H / size
    out = {}
    for i in range(0, len(paths), 8):
        res = model.predict(source=paths[i:i + 8], imgsz=imgsz, conf=DET_CONF_MIN, iou=DET_NMS_IOU,
                            max_det=DET_MAX_DET, verbose=False, device="cpu")
        for f, r in zip(files[i:i + 8], res):
            if r.boxes is None or len(r.boxes) == 0:
                out[f] = np.zeros((0, 5), np.float32); continue
            b = np.concatenate([r.boxes.xyxy.float().cpu().numpy() * scale,
                                r.boxes.conf.float().cpu().numpy()[:, None]], 1).astype(np.float32)
            out[f] = b[np.argsort(-b[:, 4])]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run:imgsz:source")
    ap.add_argument("--cut", type=float, default=0.25); ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=100); ap.add_argument("--no-tta", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    from ultralytics import YOLO
    meta = load_meta(); files = test_files(); dev = torch.device("cpu")
    box_sets = []
    for spec in a.runs:
        run, imgsz, source = spec.split(":")
        for k in range(5):
            box_sets.append(predict(YOLO(f"{run}/fold{k}_yolo.pt"), files, int(imgsz), source, meta["size"]))
            print(f"{run} fold {k} boxes done ({len(box_sets)} detectors)", flush=True)
    fused = {f: fuse_boxes([s[f] for s in box_sets]) for f in files}
    np.savez_compressed(Path(a.out).with_suffix(".boxes.npz"), **{f: fused[f] for f in files})
    last = a.runs[-1].split(":")[0]
    refiners = [load_refiner(f"{last}/fold{k}_refiner.pt", dev)[0] for k in range(5)]
    crops = {}
    for i, f in enumerate(files):
        b = fused[f]
        crops[f] = refine_boxes(refiners, read_native(f, "test"), b[b[:, 4] >= DET_CONF_MIN], dev, tta=not a.no_tta)
        if i % 20 == 0:
            print(f"refine {i}/{len(files)}", flush=True)
    cascade.save_crops(crops, Path(a.out).parent / "test_crops_fused")
    df = cascade.write_cascade_submission(crops, a.cut, a.thr, a.min_area, Path(a.out).parent)
    df.to_csv(a.out, index=False)
    print("wrote", a.out, len(df), "rows")
