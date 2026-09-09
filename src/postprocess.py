"""Turn a probability map into filament instances at native resolution."""
from __future__ import annotations

import cv2
import numpy as np

from config import H, W


def upsample(prob: np.ndarray) -> np.ndarray:
    """Any-size prob map -> float32 (H, W)."""
    if prob.dtype == np.uint8:
        prob = prob.astype(np.float32) / 255.0
    if prob.shape != (H, W):
        prob = cv2.resize(prob.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
    return prob


def probs_to_instances(prob: np.ndarray, threshold: float = 0.5, min_area: int = 150,
                       close_kernel: int = 0, fill_holes: bool = True) -> list[np.ndarray]:
    """Threshold -> optional morphological closing -> 8-connected components.

    Returns a list of uint8 (H, W) masks, largest first. Closing bridges tiny
    gaps so one filament is less likely to be emitted as several fragments
    (fragments fail the IoU > 0.5 match and each costs 0.5 FP).
    """
    prob = upsample(prob)
    binary = (prob >= threshold).astype(np.uint8)
    if close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = []
    order = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1
    for lab in order:
        if stats[lab, cv2.CC_STAT_AREA] < min_area:
            continue
        m = (labels == lab).astype(np.uint8)
        if fill_holes:
            x, y, w, h = stats[lab, :4]
            sub = m[y:y + h, x:x + w]
            contours, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(sub, contours, -1, 1, thickness=cv2.FILLED)
        out.append(m)
    return out


def instances_never_empty(prob: np.ndarray, threshold: float, min_area: int,
                          close_kernel: int = 0) -> list[np.ndarray]:
    """Like probs_to_instances but guarantees at least one instance so every
    test image has a submission row."""
    inst = probs_to_instances(prob, threshold, min_area, close_kernel)
    if not inst:
        inst = probs_to_instances(prob, threshold * 0.5, max(min_area // 3, 10), close_kernel)[:1]
    if not inst:
        p = upsample(prob)
        m = np.zeros(p.shape, np.uint8)
        y, x = np.unravel_index(int(p.argmax()), p.shape)
        m[y:y + 4, x:x + 4] = 1
        inst = [m]
    return inst
