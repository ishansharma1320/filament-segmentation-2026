"""Panoptic Quality exactly as in the organiser's self-evaluation notebook.

For every annotator record of an image, an (n_gt, n_pred) IoU matrix is
computed. A pair with IoU > 0.5 is a hit; every hit contributes its IoU to
the TP sum. A prediction with no hit is an FP, a GT filament with no hit an
FN. Counts are pooled over *all records of all images*:

    PQ = sum(IoU of hits) / (TP + 0.5 * FP + 0.5 * FN)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pycocotools import mask as mask_util

IOU_THRESHOLD = 0.5


def iou_matrix(gt_rles: list, pred_rles: list) -> np.ndarray:
    """gt_rles / pred_rles: lists of pycocotools RLE dicts. Returns (n_gt, n_pred)."""
    if not gt_rles or not pred_rles:
        return np.zeros((len(gt_rles), len(pred_rles)), dtype=np.float64)
    # iscrowd flags for the *second* argument; all zero -> plain IoU
    m = mask_util.iou(pred_rles, gt_rles, [0] * len(gt_rles))  # (n_pred, n_gt)
    return np.asarray(m, dtype=np.float64).T


@dataclass
class PQAccumulator:
    tp_ious: list[float] = field(default_factory=list)
    fp: int = 0
    fn: int = 0
    n_records: int = 0

    def add(self, gt_rles: list, pred_rles: list) -> dict:
        self.n_records += 1
        n_gt, n_pred = len(gt_rles), len(pred_rles)
        if n_gt == 0:
            self.fp += n_pred
            return {"tp": 0, "fp": n_pred, "fn": 0}
        if n_pred == 0:
            self.fn += n_gt
            return {"tp": 0, "fp": 0, "fn": n_gt}
        m = iou_matrix(gt_rles, pred_rles)
        hit = m > IOU_THRESHOLD
        ious = m[hit].tolist()
        fp = int((hit.sum(0) == 0).sum())
        fn = int((hit.sum(1) == 0).sum())
        self.tp_ious.extend(ious)
        self.fp += fp
        self.fn += fn
        return {"tp": len(ious), "fp": fp, "fn": fn, "sum_iou": float(sum(ious))}

    @property
    def tp(self) -> int:
        return len(self.tp_ious)

    def summary(self) -> dict:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        sum_iou = float(sum(self.tp_ious))
        pq = sum_iou / denom if denom > 0 else 0.0
        sq = sum_iou / self.tp if self.tp else 0.0
        rq = self.tp / denom if denom > 0 else 0.0
        return {
            "pq": pq, "sq": sq, "rq": rq,
            "tp": self.tp, "fp": self.fp, "fn": self.fn, "records": self.n_records,
        }


def score_predictions(gt_by_record: dict[str, list], preds_by_file: dict[str, list],
                      records_by_file: dict[str, list[str]], files: list[str]) -> dict:
    """gt_by_record: {record_id: [rle dict]}, preds_by_file: {file_stem: [rle dict]}."""
    acc = PQAccumulator()
    for f in files:
        stem = f.rsplit(".", 1)[0]
        preds = preds_by_file.get(stem, [])
        for rid in records_by_file[f]:
            acc.add(gt_by_record[rid], preds)
    return acc.summary()
