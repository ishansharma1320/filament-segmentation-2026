"""Submission RLE helpers (pycocotools compressed RLE, 2048x2048)."""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_util

from config import H, W


def mask_to_rle(mask: np.ndarray) -> str:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return rle["counts"].decode("ascii")


def rle_to_mask(counts: str, h: int = H, w: int = W) -> np.ndarray:
    return mask_util.decode({"size": [h, w], "counts": counts.encode("ascii")}).astype(np.uint8)


def masks_to_rles(masks) -> list:
    """Binary masks -> pycocotools RLE dicts (what the metric consumes)."""
    return [mask_util.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]
