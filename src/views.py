"""Training-time views from the public GONG archive (host ruling: training only).

An archive frame is NOT the competition's processing. Measured on 24 competition/archive
copies of the SAME frames (2026-09-28): no shift (< 0.1 px), no rotation (< 0.005 deg), no
scale (radius ratio 1.0015 +- 0.002, i.e. disk-fit noise), but the organisers clipped the
highlights (a monotone LUT: identity to ~150 then flat at ~152; archive p99 194 -> 156)
and their JPEGs carry ~1.5-2x more high-frequency content that the archive JPEGs do not
contain (no unsharp setting recovers it: it is a less-compressed source, not a filter).
A view is therefore (1) the archive frame through the LUT - no geometric warp - with a
residual mild blur gap that YOLO's 2048 -> 1280 downscale and blur augmentation cover,
and (2) labelled by projecting the competition boxes through solar differential rotation
for the elapsed time (identity for other-site frames of the same minute).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import H, W

# Snodgrass (1990) synodic differential rotation, deg/day
def omega_synodic_deg_per_day(lat: np.ndarray) -> np.ndarray:
    s2 = np.sin(lat) ** 2
    return 14.713 - 2.396 * s2 - 1.787 * s2 * s2 - 0.9856


def disk_fit(img: np.ndarray) -> tuple[float, float, float]:
    """(cx, cy, R) of the solar disk: Otsu on a 4x-downsampled frame, largest blob."""
    small = cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, stats, cents = cv2.connectedComponentsWithStats(b, connectivity=8)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cx, cy = cents[k] * 4 + 1.5
    return float(cx), float(cy), float(np.sqrt(stats[k, cv2.CC_STAT_AREA] / np.pi) * 4)


def align_geometry(view: np.ndarray, disk_view, disk_ref) -> np.ndarray:
    """Similarity transform: view disk (cx, cy, R) -> reference disk. Output HxW."""
    cxv, cyv, Rv = disk_view
    cxr, cyr, Rr = disk_ref
    s = Rr / Rv
    M = np.array([[s, 0, cxr - s * cxv], [0, s, cyr - s * cyv]], np.float32)
    return cv2.warpAffine(view, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)


def intensity_lut(pairs: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Monotone 256-entry LUT archive -> competition from (competition, aligned archive)
    frame pairs: for each archive grey level, the median competition grey level over all
    pixels of that level (on-disk and background alike), smoothed and made monotone."""
    sums = np.zeros(256); cnts = np.zeros(256)
    for comp, arch in pairs:
        a = arch.ravel(); c = comp.ravel().astype(np.float64)
        sums += np.bincount(a, weights=c, minlength=256); cnts += np.bincount(a, minlength=256)
    lut = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    idx = np.arange(256)
    ok = ~np.isnan(lut)
    lut = np.interp(idx, idx[ok], lut[ok])
    lut = np.maximum.accumulate(lut)               # monotone
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


def t_of(stem: str) -> datetime:
    return datetime.strptime(stem[:14], "%Y%m%d%H%M%S")


def project_boxes(boxes: np.ndarray, disk_ref, dt_days: float) -> tuple[np.ndarray, np.ndarray]:
    """Boxes (n, 4) xywh in the reference frame -> xywh in the same geometry after dt_days of
    rotation (the view has already been aligned to the reference disk). valid=False for
    boxes that leave the usable disk."""
    if len(boxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros(0, bool)
    cx, cy, R = disk_ref
    x = boxes[:, 0] + boxes[:, 2] / 2; y = boxes[:, 1] + boxes[:, 3] / 2
    X = (x - cx) / R; Y = (cy - y) / R
    valid = X * X + Y * Y < 0.94
    lat = np.arcsin(np.clip(Y, -1, 1)); coslat = np.maximum(np.cos(lat), 1e-6)
    lon = np.arcsin(np.clip(X / coslat, -1, 1))
    lon2 = lon + np.deg2rad(omega_synodic_deg_per_day(lat) * dt_days)
    valid &= np.abs(lon2) < np.deg2rad(75)
    fore = np.maximum(np.cos(lon2), 1e-3) / np.maximum(np.cos(lon), 1e-3)
    x2 = cx + coslat * np.sin(lon2) * R
    w2 = boxes[:, 2] * fore; h2 = boxes[:, 3]
    out = np.stack([x2 - w2 / 2, y - h2 / 2, w2, h2], 1).astype(np.float32)
    return out, valid


COMPETITION_SITES = "BCLMTU"   # archive sites S and Z are not in the competition and sit at
                               # different roll angles (up to 32 deg measured) - never used


def ecc_align(ref: np.ndarray, view: np.ndarray, max_roll_deg: float = 1.0, min_corr: float = 0.9):
    """Rigid (roll + shift) alignment of a same-minute OTHER-SITE frame onto the reference
    frame, estimated by ECC on 512 px copies (sites differ by 2-15 px of centring and
    < 0.5 deg of roll). Returns (warped view at full res, info dict) or (None, info) when
    the estimate is not trustworthy. Same-site frames must NOT go through this (they are
    already in the reference geometry; their labels are rotated instead)."""
    cs = cv2.resize(ref.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
    vs = cv2.resize(view.astype(np.float32), (512, 512), interpolation=cv2.INTER_AREA)
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(cs, vs, warp, cv2.MOTION_EUCLIDEAN,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-6), None, 7)
    except cv2.error:
        return None, {"ok": False, "reason": "ecc failed"}
    roll = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    info = {"ok": True, "roll": roll, "dx": float(warp[0, 2] * 4), "dy": float(warp[1, 2] * 4), "corr": float(cc)}
    if abs(roll) > max_roll_deg or cc < min_corr:
        info["ok"] = False; info["reason"] = "roll/corr out of range"
        return None, info
    full = warp.copy(); full[:, 2] *= 4
    # warp maps ref -> view; WARP_INVERSE_MAP brings the view into the ref frame
    out = cv2.warpAffine(view, full, (W, H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    return out, info


DEFAULT_LUT = None


def default_lut() -> np.ndarray:
    """The measured archive -> competition LUT (identity up to ~150, clipped above)."""
    global DEFAULT_LUT
    if DEFAULT_LUT is None:
        idx = np.arange(256, dtype=np.float32)
        DEFAULT_LUT = np.clip(np.minimum(idx * 0.995, 152), 0, 255).astype(np.uint8)
    return DEFAULT_LUT


def make_view(view_img: np.ndarray, lut: np.ndarray | None = None) -> np.ndarray:
    """Competition-processed view of an archive frame: LUT only (no geometric warp - none is
    needed, see module docstring)."""
    return (lut if lut is not None else default_lut())[view_img]
