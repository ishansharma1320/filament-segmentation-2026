"""Model -> probability maps -> instances -> Panoptic Quality."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import WORKERS
from dataset import FullImageDataset
from metric import PQAccumulator
from model import predict_probs
from postprocess import probs_to_instances
from rle import masks_to_rles


@torch.no_grad()
def predict_files(model, files, size, device, split="train", tta=True, batch_size=2):
    """Yield (file_name, prob map float32 at model resolution (size x size))."""
    model.eval()
    dl = DataLoader(FullImageDataset(files, size, split), batch_size=batch_size,
                    num_workers=min(WORKERS, 2), shuffle=False)
    for x, names in dl:
        p = predict_probs(model, x.to(device), tta=tta).float().cpu().numpy()[:, 0]
        for prob, name in zip(p, names):
            yield name, prob


def score_probs(probs: dict, meta: dict, gt_rles: dict, threshold: float, min_area: int,
                close_kernel: int = 0) -> dict:
    """probs: {file_name: prob map (any size, float or uint8)} -> PQ summary."""
    acc = PQAccumulator()
    for name, prob in probs.items():
        preds = masks_to_rles(probs_to_instances(prob, threshold, min_area, close_kernel))
        for rid in meta["records_by_file"][name]:
            acc.add(gt_rles[rid], preds)
    return acc.summary()


def evaluate(model, files, meta, device, gt_rles, thresholds=(0.5,), min_area=150, tta=False,
             keep_probs=False):
    """Run the model on `files` and score PQ at each threshold.
    Returns ({thr: summary}, {file: uint8 prob map at model size} if keep_probs else None)."""
    accs = {t: PQAccumulator() for t in thresholds}
    kept = {} if keep_probs else None
    for name, prob in predict_files(model, files, meta["size"], device, tta=tta):
        if keep_probs:
            kept[name] = np.round(prob * 255).astype(np.uint8)
        for t in thresholds:
            preds = masks_to_rles(probs_to_instances(prob, t, min_area))
            for rid in meta["records_by_file"][name]:
                accs[t].add(gt_rles[rid], preds)
    return {t: a.summary() for t, a in accs.items()}, kept
