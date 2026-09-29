"""Rung 4 orchestration: YOLO detector -> crop refiner, per grouped fold.

Per fold: train the detector on the fold's train files -> boxes for the val files (OOF) and
the test files; train the refiner on every (record, filament) of the train files -> refine
the OOF boxes into crop probability records. The OOF sweep then picks (detector conf cut,
refiner threshold, min_area) by pooled PQ over all annotator records. Test: boxes of the
fold detectors are fused (WBF), refined by the refiner ensemble, thresholded with the
selected knobs. Crop records (uint8 probs + bounds) are saved for OOF and test so every
post-processing knob can be re-swept offline (scripts/sweep_cascade.py).

Instances are assembled confidence-first at native resolution: a refined mask that mostly
lies on pixels already taken is a duplicate detection and is dropped; otherwise it is
clipped, so no two instances ever share a pixel.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import (DET_CONF_MIN, DET_CONFS, DUP_FRACTION, EVAL_FILES, H, MAX_FOLDS, MAX_STEPS, N_FOLDS,
                    ON_KAGGLE, REF_BATCH, REF_CROP, REF_ENCODER, REF_EPOCHS, REF_LR, REF_MIN_AREAS, REF_NEG_CONF,
                    REF_NEG_FRAC, REF_NEG_IOU, REF_OBJ_HEAD, REF_THRS, START_FOLD, T0, TIME_BUDGET, TTA, W, WEIGHT_DECAY,
                    WORKERS, DEBUG, log)
from data import test_files
from detector import box_iou, box_of_rle, box_pr, build_yolo_labels, fuse_boxes, predict_boxes, train_detector
from metric import PQAccumulator
from model import get_device
from refiner import (RefinerDataset, build_refiner, load_refiner, read_native, refine_boxes, save_refiner,
                     seed_mask, split_out)
from rle import mask_to_rle, masks_to_rles


# --- crop records <-> instances ---------------------------------------------------------------
def crop_instances(crops: list[dict], conf_thr: float, thr: float, min_area: int,
                   dup_fraction: float = DUP_FRACTION) -> list[np.ndarray]:
    """Crop records of one image -> non-overlapping uint8 (H, W) masks, confidence order."""
    taken = np.zeros((H, W), bool)
    out = []
    for c in sorted(crops, key=lambda c: -c["conf"]):
        if c["conf"] < conf_thr:
            continue
        x0, y0, side = int(c["x0"]), int(c["y0"]), int(c["side"])
        p = c["prob"].astype(np.float32) / 255.0
        if p.shape[0] != side:
            p = cv2.resize(p, (side, side), interpolation=cv2.INTER_LINEAR)
        b = (p >= thr).astype(np.uint8)
        if not b.any():
            continue
        # keep only the components that touch the seed box: the crop may contain neighbours
        n, labels = cv2.connectedComponents(b, connectivity=8)
        bx0, by0, bx1, by1 = c["box"]
        seed = seed_mask([bx0, by0, bx1 - bx0, by1 - by0], x0, y0, side).astype(bool)
        keep = np.unique(labels[seed & (labels > 0)])
        if len(keep) == 0:
            continue
        m = np.isin(labels, keep)
        area = int(m.sum())
        if area < min_area:
            continue
        t = taken[y0:y0 + side, x0:x0 + side]
        if (m & t).sum() > dup_fraction * area:
            continue
        m &= ~t
        if m.sum() < min_area:
            continue
        t |= m
        full = np.zeros((H, W), np.uint8)
        full[y0:y0 + side, x0:x0 + side] = m
        out.append(full)
    return out


def crop_instances_never_empty(crops: list[dict], conf_thr: float, thr: float, min_area: int) -> list[np.ndarray]:
    inst = crop_instances(crops, conf_thr, thr, min_area)
    if not inst and crops:
        inst = crop_instances(crops, 0.0, thr * 0.5, max(min_area // 3, 10))[:1]
    if not inst:
        m = np.zeros((H, W), np.uint8)
        m[1020:1024, 1020:1024] = 1
        inst = [m]
    return inst


def rescored(crops: list[dict], b: float) -> list[dict]:
    """Instance score = detector conf x objectness^b (b = 0: detector conf alone)."""
    if b == 0:
        return crops
    return [dict(c, conf=c["conf"] * max(c.get("obj", 1.0), 1e-3) ** b) for c in crops]


def score_crops(crops_by_file: dict, meta: dict, gt_rles: dict, conf_thr: float, thr: float,
                min_area: int) -> dict:
    acc = PQAccumulator()
    for f, crops in crops_by_file.items():
        preds = masks_to_rles(crop_instances(crops, conf_thr, thr, min_area))
        for rid in meta["records_by_file"][f]:
            acc.add(gt_rles[rid], preds)
    return acc.summary()


def sweep_cascade(oof: dict, meta: dict, gt_rles: dict) -> tuple[float, float, int, list]:
    table = []
    for conf in DET_CONFS:
        for t in REF_THRS:
            for a in REF_MIN_AREAS:
                s = score_crops(oof, meta, gt_rles, conf, t, a)
                table.append({"conf": conf, "thr": t, "min_area": a, **s})
                log(f"sweep conf={conf} thr={t} min_area={a}: PQ {s['pq']:.4f} SQ {s['sq']:.3f} "
                    f"RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}")
    best = max(table, key=lambda r: r["pq"])
    return best["conf"], best["thr"], best["min_area"], table


def save_crops(crops_by_file: dict, d: Path):
    """One compressed npz per file: probs (n, crop, crop) uint8 + meta (n, 9) float32
    [x0, y0, side, conf, bx0, by0, bx1, by1, obj]."""
    d.mkdir(exist_ok=True, parents=True)
    for f, crops in crops_by_file.items():
        stem = f.rsplit(".", 1)[0]
        probs = np.stack([c["prob"] for c in crops]) if crops else np.zeros((0, REF_CROP, REF_CROP), np.uint8)
        meta = np.array([[c["x0"], c["y0"], c["side"], c["conf"], *c["box"], c.get("obj", 1.0)] for c in crops],
                        np.float32).reshape(-1, 9)
        np.savez_compressed(d / f"{stem}.npz", probs=probs, meta=meta)


def load_crops(d: Path) -> dict:
    out = {}
    for p in sorted(Path(d).glob("*.npz")):
        z = np.load(p)
        out[p.stem + ".jpeg"] = [{"prob": pr, "x0": m[0], "y0": m[1], "side": m[2], "conf": float(m[3]),
                                  "box": list(m[4:8]), "obj": float(m[8]) if len(m) > 8 else 1.0}
                                 for pr, m in zip(z["probs"], z["meta"])]
    return out


def write_cascade_submission(test_crops: dict, conf: float, thr: float, min_area: int, out_dir: Path):
    rows = []
    for f, crops in test_crops.items():
        stem = f.rsplit(".", 1)[0]
        for k, m in enumerate(crop_instances_never_empty(crops, conf, thr, min_area), 1):
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": mask_to_rle(m)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    assert df.filament_id.is_unique and len(df) >= len(test_crops)
    df.to_csv(out_dir / "submission.csv", index=False)
    log(f"submission.csv: {len(df)} rows over {len(test_crops)} images (conf={conf}, thr={thr}, min_area={min_area})")
    return df


# --- hard negatives ---------------------------------------------------------------------------
def hard_negatives(oof_boxes: dict, meta: dict, gt_rles: dict, conf: float = REF_NEG_CONF,
                   iou: float = REF_NEG_IOU) -> dict:
    """{file: (n, 4) xyxy} detector OOF boxes above `conf` that overlap no GT box of any
    annotator record of the file by more than `iou`: false positives by every annotator."""
    out = {}
    for f, b in oof_boxes.items():
        b = b[b[:, 4] >= conf]
        if not len(b):
            continue
        g = np.array([box_of_rle(r) for rid in meta["records_by_file"][f] for r in gt_rles[rid]],
                     np.float32).reshape(-1, 4)
        g[:, 2:] += g[:, :2]
        m = box_iou(b[:, :4], g)
        keep = (m.max(1) < iou) if len(g) else np.ones(len(b), bool)
        if keep.any():
            out[f] = b[keep, :4]
    return out


@torch.no_grad()
def obj_separation(model, ds, device, n: int = 400) -> dict:
    """Objectness head on val crops: mean sigmoid on positives / negatives, fraction of
    negatives rejected (obj < 0.5) and of negatives with a < 100 px mask at thr 0.5."""
    model.eval()
    out = {}
    for name, idx in (("pos", list(range(0, min(ds.n_pos, n)))),
                      ("neg", list(range(ds.n_pos, min(len(ds.items), ds.n_pos + n))))):
        if not idx:
            continue
        objs, small = [], 0
        for i in range(0, len(idx), 32):
            x = torch.stack([ds[j][0] for j in idx[i:i + 32]]).to(device)
            mk, o = split_out(model(x))
            objs.append(torch.sigmoid(o[:, 0]).cpu() if o is not None else torch.ones(x.shape[0]))
            small += int(((torch.sigmoid(mk) >= 0.5).sum(dim=(1, 2, 3)) < 100).sum())
        o = torch.cat(objs)
        out[f"{name}_obj"] = float(o.mean())
        out[f"{name}_reject"] = float((o < 0.5).float().mean())
        out[f"{name}_small"] = small / len(idx)
    return out


# --- refiner training -------------------------------------------------------------------------
def refiner_loss(logits, obj, y, is_pos):
    """Positives: pixel BCE + soft Dice; negatives (all-zero target): pixel BCE only — Dice on
    an empty target has a ~1/area^2 gradient and taught nothing in v7. Plus BCE on the
    objectness logit vs the positive/negative label."""
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = torch.sigmoid(logits)
    inter = (p * y).sum(dim=(1, 2, 3))
    den = p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (den + 1)
    dice = (dice * is_pos).sum() / is_pos.sum().clamp(min=1)
    if obj is None:
        return bce + dice
    return bce + dice + F.binary_cross_entropy_with_logits(obj[:, 0], is_pos)


@torch.no_grad()
def val_dice(model, dl, device) -> float:
    model.eval()
    num = den = 0.0
    for x, y, _ in dl:
        x, y = x.to(device), y.to(device)
        p = (torch.sigmoid(split_out(model(x))[0]) >= 0.5).float()
        num += float((2 * (p * y).sum(dim=(1, 2, 3))).sum())
        den += float((p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1e-6).sum())
    return num / max(den, 1e-6)


def train_refiner(fold: int, meta: dict, gt_rles: dict, device, out_dir: Path, epochs: int = REF_EPOCHS,
                  max_steps: int = MAX_STEPS, eval_files: int = EVAL_FILES, negatives: dict | None = None) -> Path:
    folds = meta["folds"]
    tr = sorted(f for f, k in folds.items() if k != fold)
    va = sorted(f for f, k in folds.items() if k == fold)
    if eval_files:
        va = va[:eval_files]
        if DEBUG:
            tr = tr[:max(eval_files * 2, 4)]
    ds_tr = RefinerDataset(tr, meta, gt_rles, augment=True, negatives=negatives)
    ds_va = RefinerDataset(va, meta, gt_rles, augment=False)
    ds_va_neg = RefinerDataset(va, meta, gt_rles, augment=False, negatives=negatives, neg_frac=0.5) if negatives else None
    log(f"fold {fold}: refiner {REF_ENCODER} crops {ds_tr.n_pos} pos + {ds_tr.n_neg} neg train / "
        f"{len(ds_va)} val (+{ds_va_neg.n_neg if ds_va_neg else 0} val neg), {epochs} epochs")
    dl = DataLoader(ds_tr, batch_size=REF_BATCH, shuffle=True, num_workers=WORKERS, drop_last=True,
                    persistent_workers=WORKERS > 0)
    dl_va = DataLoader(ds_va, batch_size=REF_BATCH, shuffle=False, num_workers=min(WORKERS, 2))
    model = build_refiner(REF_ENCODER, pretrained=not DEBUG).to(device)
    steps = min(len(dl), max_steps) if max_steps else len(dl)
    opt = torch.optim.AdamW(model.parameters(), lr=REF_LR, weight_decay=WEIGHT_DECAY)
    # OneCycleLR divides by (warm-up steps - 1): with a 10-step debug epoch pct_start 0.1 is 0/0
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=REF_LR, epochs=epochs, steps_per_epoch=steps,
                                                pct_start=0.1 if epochs * steps >= 30 else 0.3)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt, best, history = out_dir / f"fold{fold}_refiner.pt", -1.0, []
    for epoch in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for step, (x, y, is_pos) in enumerate(dl):
            if step >= steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            is_pos = is_pos.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                logits, obj = split_out(model(x))
                loss = refiner_loss(logits.float(), obj.float() if obj is not None else None, y, is_pos)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item(); n += 1
        d = val_dice(model, dl_va, device)
        sep = obj_separation(model, ds_va_neg, device) if ds_va_neg else {}
        history.append({"epoch": epoch, "loss": tot / max(n, 1), "val_dice": d, **sep,
                        "time": time.time() - t0})
        log(f"fold {fold} refiner epoch {epoch:2d} loss {tot / max(n, 1):.4f} | val dice {d:.4f} | "
            f"obj pos {sep.get('pos_obj', float('nan')):.3f} neg {sep.get('neg_obj', float('nan')):.3f} | "
            f"neg reject {sep.get('neg_reject', float('nan')):.3f} small {sep.get('neg_small', float('nan')):.3f} | "
            f"{time.time() - t0:.0f}s")
        if d > best:
            best = d
            save_refiner(ckpt, model, encoder=REF_ENCODER, crop=REF_CROP, fold=fold, epoch=epoch, dice=best,
                         obj_head=REF_OBJ_HEAD)
        json.dump(history, open(out_dir / f"fold{fold}_refiner_history.json", "w"), indent=1)
    return ckpt


# --- inference helpers ------------------------------------------------------------------------
def refine_files(refiners: list, boxes_by_file: dict, files: list[str], split: str, device,
                 tta: bool) -> dict:
    out = {}
    for f in files:
        b = boxes_by_file.get(f, np.zeros((0, 5), np.float32))
        b = b[b[:, 4] >= DET_CONF_MIN]
        out[f] = refine_boxes(refiners, read_native(f, split), b, device, tta=tta)
    return out


def fused_test_boxes(test_boxes: list[dict]) -> dict:
    files = test_files()
    return {f: fuse_boxes([tb.get(f, np.zeros((0, 5), np.float32)) for tb in test_boxes]) for f in files}


def find_attached_cascade() -> tuple[list[Path], list[Path]]:
    base = Path("/kaggle/input")
    if not base.is_dir():
        return [], []
    y = sorted(set(base.glob("*/fold*_yolo.pt")) | set(base.glob("*/*/fold*_yolo.pt")) |
               set(base.glob("*/fold*_rfdetr.pth")) | set(base.glob("*/*/fold*_rfdetr.pth")))
    r = sorted(set(base.glob("*/fold*_refiner.pt")) | set(base.glob("*/*/fold*_refiner.pt")))
    return y, r


# --- entry ------------------------------------------------------------------------------------
def main_cascade(meta: dict, gt_rles: dict, out_dir: Path):
    device = get_device()
    build_yolo_labels(meta, gt_rles)
    tfiles = test_files()
    if EVAL_FILES and DEBUG:
        tfiles = tfiles[:EVAL_FILES]

    yolos, refs = find_attached_cascade()
    if yolos and refs:
        log(f"inference mode: {len(yolos)} detectors + {len(refs)} refiners attached")
        refiners, cks = [], []
        for p in refs:
            m, ck = load_refiner(p, device)
            refiners.append(m); cks.append(ck)
        conf = float(np.median([ck.get("conf", 0.3) for ck in cks]))
        thr = float(np.median([ck.get("thr", 0.5) for ck in cks]))
        area = int(np.median([ck.get("min_area", 100) for ck in cks]))
        tb = [predict_boxes(p, tfiles, meta["size"], split="test") for p in yolos]
        crops = refine_files(refiners, fused_test_boxes(tb), tfiles, "test", device, TTA)
        save_crops(crops, out_dir / "test_crops")
        write_cascade_submission(crops, conf, thr, area, out_dir)
        return

    max_folds = N_FOLDS if MAX_FOLDS == "auto" else int(MAX_FOLDS)
    train_limit = max(EVAL_FILES * 2, 4) if (DEBUG and EVAL_FILES) else 0
    val_files = {}
    for fold in range(N_FOLDS):
        va = sorted(f for f, k in meta["folds"].items() if k == fold)
        val_files[fold] = va[:EVAL_FILES] if EVAL_FILES else va

    # phase 1: detectors. Every fold's detector first, because the refiners (phase 2) train on
    # the detectors' OOF false positives as hard negatives (REF_NEG_FRAC > 0).
    det_folds, test_boxes, oof_boxes_all, box_prs, det_time = [], [], {}, {}, 0.0
    for fold in range(START_FOLD, N_FOLDS):
        if len(det_folds) >= max_folds:
            break
        elapsed = time.time() - T0
        if det_folds and elapsed + 1.15 * det_time > TIME_BUDGET / 2:
            log(f"time budget (detectors): {elapsed / 60:.0f} min used -> stop after {len(det_folds)} folds")
            break
        t0 = time.time()
        det = train_detector(fold, meta, out_dir, train_limit=train_limit, val_limit=EVAL_FILES)
        ob = predict_boxes(det, val_files[fold], meta["size"])
        pr = {c: box_pr(ob, meta, gt_rles, c) for c in (0.2, 0.3, 0.5)}
        for c, r in pr.items():
            log(f"fold {fold} boxes conf>={c}: P {r['precision']:.3f} R {r['recall']:.3f} "
                f"(tp {r['tp']} fp {r['fp']} fn {r['fn']})")
        test_boxes.append(predict_boxes(det, tfiles, meta["size"], split="test"))
        json.dump({f: b.tolist() for f, b in ob.items()}, open(out_dir / f"fold{fold}_oof_boxes.json", "w"))
        oof_boxes_all.update(ob)
        box_prs[fold] = pr
        det_folds.append(fold)
        det_time = time.time() - t0
    negatives = hard_negatives(oof_boxes_all, meta, gt_rles) if REF_NEG_FRAC > 0 else None
    if negatives:
        log(f"hard negatives: {sum(len(v) for v in negatives.values())} detector FP boxes over "
            f"{len(negatives)} files (conf >= {REF_NEG_CONF}, IoU < {REF_NEG_IOU} to every annotator)")

    # phase 2: refiners, OOF crops, scoreable submission after every fold
    results, ref_ckpts, fold_time = [], [], 0.0
    oof = {}
    for fold in det_folds:
        elapsed = time.time() - T0
        if results and elapsed + 1.15 * fold_time > TIME_BUDGET:
            log(f"time budget: {elapsed / 60:.0f} min used, fold takes {fold_time / 60:.0f} min -> stop")
            break
        t0 = time.time()
        va = val_files[fold]
        oof_boxes = {f: oof_boxes_all[f] for f in va}
        ck = train_refiner(fold, meta, gt_rles, device, out_dir, negatives=negatives)
        ref_ckpts.append(ck)
        refiner, _ = load_refiner(ck, device)
        crops = refine_files([refiner], oof_boxes, va, "train", device, TTA)
        save_crops(crops, out_dir / "oof_crops")
        oof.update(crops)
        s = score_crops(crops, meta, gt_rles, 0.3, 0.5, 100)
        log(f"fold {fold} OOF cascade PQ {s['pq']:.4f} SQ {s['sq']:.3f} RQ {s['rq']:.3f} "
            f"tp {s['tp']} fp {s['fp']} fn {s['fn']} (conf 0.3 / thr 0.5 / min_area 100)")
        results.append({"fold": fold, "box_pr": box_prs[fold], "oof_pq": s["pq"], "refiner": str(ck)})
        del refiner
        torch.cuda.empty_cache() if device.type == "cuda" else None
        fold_time = time.time() - t0
        # keep the run scoreable: fused boxes of ALL trained detectors + refiners so far, default knobs
        refiners = [load_refiner(c, device)[0] for c in ref_ckpts]
        tcrops = refine_files(refiners, fused_test_boxes(test_boxes), tfiles, "test", device, tta=False)
        write_cascade_submission(tcrops, 0.3, 0.5, 100, out_dir)
        del refiners, tcrops
        torch.cuda.empty_cache() if device.type == "cuda" else None

    log(f"OOF sweep over {len(oof)} files from {len(results)} folds")
    conf, thr, area, table = sweep_cascade(oof, meta, gt_rles)
    best = max(table, key=lambda r: r["pq"])
    summary = {"pipeline": "cascade", "folds": results, "sweep": table, "best_conf": conf, "best_thr": thr,
               "best_min_area": area, "best_pq": best["pq"]}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    for c in ref_ckpts:  # stamp the selected knobs into the refiner checkpoints (inference mode reads them)
        ck = torch.load(c, map_location="cpu", weights_only=False)
        ck.update(conf=conf, thr=thr, min_area=area)
        torch.save(ck, c)
    refiners = [load_refiner(c, device)[0] for c in ref_ckpts]
    tcrops = refine_files(refiners, fused_test_boxes(test_boxes), tfiles, "test", device, TTA)
    save_crops(tcrops, out_dir / "test_crops")
    write_cascade_submission(tcrops, conf, thr, area, out_dir)
    log(f"OOF PQ at selected (conf={conf}, thr={thr}, min_area={area}): {best['pq']:.4f}")
