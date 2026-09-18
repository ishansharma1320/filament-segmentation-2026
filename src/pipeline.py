"""Orchestrator. Same entry point locally and in the Kaggle kernel.

main():  placeholder submission -> cache -> grouped folds
         -> [PIPELINE == "cascade": hand over to cascade.main_cascade (rung 4) and return]
         -> [inference mode if fold*_best.pt checkpoints are attached under /kaggle/input]
         -> else train folds until TIME_BUDGET, saving fold{k}_best.pt + OOF prob maps,
            rewriting submission.csv after every fold so the run is always scoreable
         -> OOF sweep of (threshold, min_area) on all trained folds
         -> fold-ensemble test prediction -> submission.csv
run():   main() inside a crash guard.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import (ARCH, BATCH, CACHE, CLOSE_KERNEL, CROP, ENCODER, EPOCH_THR, EPOCHS, EVAL_FILES,
                    H, IMG_SIZE, LR, MAX_FOLDS, MAX_STEPS, MIN_AREA, MIN_AREAS, MIN_SCORES, N_FOLDS,
                    ON_KAGGLE, PIPELINE, POS_WEIGHT, RUNS, SAMPLES_PER_FILE, SEED, START_FOLD, T0,
                    THRESHOLDS, TIME_BUDGET, TTA, W, WEIGHT_DECAY, WORKERS, log, seed_everything)
from cascade import main_cascade
from data import test_files
from dataset import CropDataset, build_cache, load_gt_rles, load_meta
from evaluate import evaluate, predict_files, score_probs
from model import bce_dice_loss, build_model, get_device, load_checkpoint, save_checkpoint
from postprocess import instances_never_empty
from rle import mask_to_rle


# --- always-scoreable ---------------------------------------------------------------------
def write_placeholder(out_dir: Path):
    rows = []
    for f in test_files():
        m = np.zeros((H, W), np.uint8)
        m[1020:1028, 1020:1028] = 1
        rows.append({"filament_id": f"{f.rsplit('.', 1)[0]}_1", "segmentation_rle": mask_to_rle(m)})
    pd.DataFrame(rows).to_csv(out_dir / "submission.csv", index=False)
    log(f"placeholder submission written: {len(rows)} rows")


def write_submission(test_probs: dict, threshold: float, min_area: int, out_dir: Path,
                     close_kernel: int = 0, min_score: float = 0.0) -> pd.DataFrame:
    rows = []
    for name, prob in test_probs.items():
        stem = name.rsplit(".", 1)[0]
        inst = instances_never_empty(prob, threshold, min_area, close_kernel, min_score)
        for k, m in enumerate(inst, 1):
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": mask_to_rle(m)})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    assert df.filament_id.is_unique and len(df) >= len(test_probs)
    df.to_csv(out_dir / "submission.csv", index=False)
    log(f"submission.csv: {len(df)} rows over {len(test_probs)} images "
        f"(thr={threshold}, min_area={min_area}, min_score={min_score})")
    return df


def save_test_probs(test_probs: dict, out_dir: Path):
    """Ensemble-mean test prob maps as uint8 PNGs: lets post-processing be re-done
    locally without another kernel run."""
    d = out_dir / "test"
    d.mkdir(exist_ok=True)
    for name, p in test_probs.items():
        cv2.imwrite(str(d / (name.rsplit(".", 1)[0] + ".png")), np.round(p * 255).astype(np.uint8))


# --- training -----------------------------------------------------------------------------
def train_fold(fold: int, meta: dict, device, out_dir: Path, gt_rles: dict, epochs: int = EPOCHS,
               max_steps: int = MAX_STEPS, eval_files: int = EVAL_FILES, arch: str = ARCH,
               encoder: str = ENCODER) -> dict:
    folds = meta["folds"]
    tr = sorted(f for f, k in folds.items() if k != fold)
    va = sorted(f for f, k in folds.items() if k == fold)
    if eval_files:
        va = va[:eval_files]
    log(f"fold {fold}: train {len(tr)} files, val {len(va)} files, {arch}/{encoder}")

    model = build_model(arch, encoder).to(device)
    ds = CropDataset(tr, meta, crop=CROP, samples_per_file=SAMPLES_PER_FILE)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=WORKERS, drop_last=True,
                    persistent_workers=WORKERS > 0)
    steps = min(len(dl), max_steps) if max_steps else len(dl)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, epochs=epochs, steps_per_epoch=steps,
                                                pct_start=0.1)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ckpt = out_dir / f"fold{fold}_best.pt"
    best, history = -1.0, []
    for epoch in range(epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for step, (x, y) in enumerate(dl):
            if step >= steps:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=use_amp):
                loss = bce_dice_loss(model(x), y, pos_weight=POS_WEIGHT)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item(); n += 1
        res, _ = evaluate(model, va, meta, device, gt_rles, (EPOCH_THR,), MIN_AREA, tta=False)
        r = res[EPOCH_THR]
        rec = {"epoch": epoch, "loss": tot / max(n, 1), **r, "time": time.time() - t0}
        history.append(rec)
        log(f"fold {fold} epoch {epoch:2d} loss {rec['loss']:.4f} | PQ {r['pq']:.4f} SQ {r['sq']:.3f} "
            f"RQ {r['rq']:.3f} tp {r['tp']} fp {r['fp']} fn {r['fn']} | {rec['time']:.0f}s")
        if r["pq"] > best:
            best = r["pq"]
            save_checkpoint(ckpt, model, arch=arch, encoder=encoder, img_size=meta["size"], fold=fold,
                            epoch=epoch, pq=best)
        json.dump(history, open(out_dir / f"fold{fold}_history.json", "w"), indent=1)

    # OOF prob maps with the best epoch + TTA, saved as uint8 PNGs for offline sweeps
    model, _ = load_checkpoint(ckpt, device)
    res, probs = evaluate(model, va, meta, device, gt_rles, (EPOCH_THR,), MIN_AREA, tta=TTA,
                          keep_probs=True)
    oof_dir = out_dir / "oof"
    oof_dir.mkdir(exist_ok=True)
    for name, p in probs.items():
        cv2.imwrite(str(oof_dir / (name.rsplit(".", 1)[0] + ".png")), p)
    log(f"fold {fold} done: best epoch PQ {best:.4f}, TTA PQ {res[EPOCH_THR]['pq']:.4f}")
    return {"fold": fold, "ckpt": ckpt, "best_pq": best, "tta_pq": res[EPOCH_THR]["pq"],
            "oof": probs, "val_files": va, "model": model}


def sweep_postprocess(oof: dict, meta: dict, gt_rles: dict) -> tuple[float, int, float, list]:
    table = []
    for t in THRESHOLDS:
        for a in MIN_AREAS:
            for ms in MIN_SCORES:
                s = score_probs(oof, meta, gt_rles, t, a, CLOSE_KERNEL, ms)
                table.append({"thr": t, "min_area": a, "min_score": ms, "close": CLOSE_KERNEL, **s})
                log(f"sweep thr={t} min_area={a} min_score={ms} close={CLOSE_KERNEL}: PQ {s['pq']:.4f} "
                    f"SQ {s['sq']:.3f} RQ {s['rq']:.3f} tp {s['tp']} fp {s['fp']} fn {s['fn']}")
    best = max(table, key=lambda r: r["pq"])
    return best["thr"], best["min_area"], best["min_score"], table


def predict_test(models: list, size: int, device) -> dict:
    """Average prob maps of several models over the test set. {file: float32 (size,size)}."""
    files = test_files()
    acc = {}
    # float16: 180 maps at 2048^2 are 3 GB in float32; several such dicts coexist
    for m in models:
        for name, p in predict_files(m, files, size, device, split="test", tta=TTA):
            acc[name] = (acc[name] + p.astype(np.float16)) if name in acc else p.astype(np.float16)
    return {k: (v / len(models)).astype(np.float16) for k, v in acc.items()}


def find_attached_checkpoints() -> list[Path]:
    base = Path("/kaggle/input")
    if not base.is_dir():
        return []
    return sorted(set(base.glob("*/fold*_best.pt")) | set(base.glob("*/*/fold*_best.pt")))


# --- entry ---------------------------------------------------------------------------------
# local runs get a "local-" prefix so they can never be confused with pulled kernel runs
OUT_DIR = RUNS if ON_KAGGLE else RUNS / time.strftime("local-%Y%m%d-%H%M%S")


def main():
    seed_everything(SEED)
    device = get_device()
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"device={device} on_kaggle={ON_KAGGLE} out={out_dir} cache={CACHE}")
    write_placeholder(out_dir)

    meta = build_cache(IMG_SIZE, N_FOLDS)
    gt_rles = load_gt_rles()

    if PIPELINE == "cascade":          # rung 4: detector -> crop refiner (cascade.py)
        main_cascade(meta, gt_rles, out_dir)
        return

    attached = find_attached_checkpoints()
    if attached:
        log(f"inference mode: {len(attached)} checkpoints attached")
        models, cks = [], []
        for p in attached:
            m, ck = load_checkpoint(p, device)
            models.append(m); cks.append(ck)
            log(f"  {p.name}: fold {ck.get('fold')} epoch {ck.get('epoch')} PQ {ck.get('pq', 0):.4f}")
        thr = float(np.median([ck.get("thr", EPOCH_THR) for ck in cks]))
        area = int(np.median([ck.get("min_area", MIN_AREA) for ck in cks]))
        ms = float(np.median([ck.get("min_score", 0.0) for ck in cks]))
        probs = predict_test(models, meta["size"], device)
        save_test_probs(probs, out_dir)
        write_submission(probs, thr, area, out_dir, CLOSE_KERNEL, ms)
        return

    # training mode
    max_folds = N_FOLDS if MAX_FOLDS == "auto" else int(MAX_FOLDS)
    results, test_sum, n_models, fold_time = [], None, 0, 0.0
    for fold in range(START_FOLD, N_FOLDS):
        if len(results) >= max_folds:
            break
        elapsed = time.time() - T0
        if results and elapsed + 1.15 * fold_time > TIME_BUDGET:
            log(f"time budget: {elapsed / 60:.0f} min used, fold takes {fold_time / 60:.0f} min -> stop")
            break
        t0 = time.time()
        r = train_fold(fold, meta, device, out_dir, gt_rles)
        fold_time = time.time() - t0
        results.append(r)
        # keep the run scoreable: ensemble of folds so far
        p = predict_test([r["model"]], meta["size"], device)
        test_sum = p if test_sum is None else {k: (test_sum[k] + p[k]).astype(np.float16) for k in p}
        del p
        n_models += 1
        write_submission({k: v / n_models for k, v in test_sum.items()}, EPOCH_THR, MIN_AREA, out_dir,
                         CLOSE_KERNEL)
        del r["model"]
        torch.cuda.empty_cache() if device.type == "cuda" else None

    oof = {}
    for r in results:
        oof.update(r["oof"])
    log(f"OOF sweep over {len(oof)} files from {len(results)} folds")
    thr, area, ms, table = sweep_postprocess(oof, meta, gt_rles)
    summary = {"folds": [{k: v for k, v in r.items() if k in ("fold", "best_pq", "tta_pq")} for r in results],
               "sweep": table, "best_thr": thr, "best_min_area": area, "best_min_score": ms}
    json.dump(summary, open(out_dir / "summary.json", "w"), indent=1)
    for r in results:  # stamp the selected post-processing into the checkpoints
        ck = torch.load(r["ckpt"], map_location="cpu", weights_only=False)
        ck.update(thr=thr, min_area=area, min_score=ms)
        torch.save(ck, r["ckpt"])
    test_mean = {k: v / n_models for k, v in test_sum.items()}
    save_test_probs(test_mean, out_dir)
    write_submission(test_mean, thr, area, out_dir, CLOSE_KERNEL, ms)
    log(f"OOF PQ at selected (thr={thr}, min_area={area}, min_score={ms}): "
        f"{max(table, key=lambda x: x['pq'])['pq']:.4f}")


def run():
    """Entry point with the crash guard: a placeholder submission exists from the first
    seconds of main(), so a crash still leaves a scoreable file."""
    try:
        main()
    except Exception:
        traceback.print_exc()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        if not (OUT_DIR / "submission.csv").exists():
            write_placeholder(OUT_DIR)
        log("run FAILED; see traceback above")
    log("done")
