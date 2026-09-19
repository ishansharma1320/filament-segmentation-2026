"""Rung 4, stage 2: per-box crop refiner.

Input  : 3 channels at native resolution, [image crop, CLAHE(image crop), seed], where the
         seed is the (padded) detector box filled with 1s - it tells the network WHICH
         filament to segment when several share the crop.
Target : the mask of that one annotation. Every (record, annotation) pair of the training
         files is a sample (all annotators: label augmentation, more data).
Crops  : side = max(REF_CROP, REF_CONTEXT x long side of the box) native px, centred on the
         box; only crops larger than REF_CROP are downscaled (INTER_AREA), and their
         probability map is upscaled back before thresholding - masks are never resampled.
Train-time box jitter (shift, scale, pad) mimics detector boxes; val uses clean GT boxes.
"""
from __future__ import annotations

import math
import random

import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
from pycocotools import mask as mask_util
from torch.utils.data import Dataset

from config import H, REF_CONTEXT, REF_CROP, REF_ENCODER, REF_NEG_FRAC, REF_PAD, W
from data import train_img_dir, test_img_dir
from detector import box_of_rle


def crop_bounds(box_xywh, crop: int = REF_CROP, context: float = REF_CONTEXT) -> tuple[int, int, int]:
    """(x0, y0, side): square crop inside the frame, centred on the box."""
    x, y, w, h = box_xywh
    side = int(min(max(crop, math.ceil(context * max(w, h))), min(H, W)))
    x0 = int(round(x + w / 2 - side / 2)); y0 = int(round(y + h / 2 - side / 2))
    x0 = int(np.clip(x0, 0, W - side)); y0 = int(np.clip(y0, 0, H - side))
    return x0, y0, side


def seed_mask(box_xywh, x0: int, y0: int, side: int, pad: float = REF_PAD) -> np.ndarray:
    """uint8 (side, side): the box grown by `pad` x its size (at least 4 px) per side, filled."""
    x, y, w, h = box_xywh
    px, py = max(4.0, w * pad), max(4.0, h * pad)
    m = np.zeros((side, side), np.uint8)
    ax0, ay0 = int(np.clip(math.floor(x - px) - x0, 0, side)), int(np.clip(math.floor(y - py) - y0, 0, side))
    ax1, ay1 = int(np.clip(math.ceil(x + w + px) - x0, 0, side)), int(np.clip(math.ceil(y + h + py) - y0, 0, side))
    m[ay0:ay1, ax0:ax1] = 1
    return m


_CLAHE = None


def clahe(img: np.ndarray) -> np.ndarray:
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return _CLAHE.apply(img)


def make_input(img_crop: np.ndarray, seed: np.ndarray, crop: int = REF_CROP) -> np.ndarray:
    """uint8 image crop (side, side) + uint8 seed -> float32 (3, crop, crop)."""
    if img_crop.shape[0] != crop:
        img_crop = cv2.resize(img_crop, (crop, crop), interpolation=cv2.INTER_AREA)
        seed = cv2.resize(seed, (crop, crop), interpolation=cv2.INTER_NEAREST)
    x = np.stack([img_crop, clahe(img_crop), seed * 255]).astype(np.float32) / 255.0
    return (x - 0.5) / 0.25


def jitter_box(box_xywh, rng: random.Random) -> tuple[list[float], float]:
    """Perturb a GT box so training seeds look like detector boxes: shift up to 15 % of the
    size, scale 0.8-1.25 per axis, pad 0.5-1.5 x REF_PAD."""
    x, y, w, h = box_xywh
    x += rng.uniform(-0.15, 0.15) * w; y += rng.uniform(-0.15, 0.15) * h
    sx, sy = rng.uniform(0.8, 1.25), rng.uniform(0.8, 1.25)
    cx, cy = x + w / 2, y + h / 2
    w, h = max(6.0, w * sx), max(6.0, h * sy)
    return [cx - w / 2, cy - h / 2, w, h], rng.uniform(0.5, 1.5) * REF_PAD


def read_native(file: str, split: str = "train") -> np.ndarray:
    d = train_img_dir() if split == "train" else test_img_dir()
    im = cv2.imread(str(d / file), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(d / file)
    if im.shape != (H, W):
        im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    return im


class RefinerDataset(Dataset):
    """One sample per (annotator record, filament) of the given files, plus - when
    `negatives` is given ({file: (n, 4) xyxy boxes of detector false positives}) - hard
    negative samples: a detector FP box as the seed and an all-zero target, drawn so that
    negatives are `neg_frac` of the samples (sampled with replacement if there are fewer)."""

    def __init__(self, files: list[str], meta: dict, gt_rles: dict, augment: bool, crop: int = REF_CROP,
                 negatives: dict | None = None, neg_frac: float = REF_NEG_FRAC):
        self.items = []
        for f in files:
            for rid in meta["records_by_file"][f]:
                for rle in gt_rles[rid]:
                    b = box_of_rle(rle)
                    if b[2] >= 1 and b[3] >= 1:
                        self.items.append((f, rle, b))
        self.n_pos = len(self.items)
        if negatives:
            pool = [(f, None, (float(x0), float(y0), float(x1 - x0), float(y1 - y0)))
                    for f in files for x0, y0, x1, y1 in negatives.get(f, [])]
            if pool and neg_frac > 0:
                n_neg = int(round(self.n_pos * neg_frac / (1 - neg_frac)))
                rng = random.Random(0)
                self.items += [pool[i] for i in (rng.choices(range(len(pool)), k=n_neg) if n_neg > len(pool)
                                                 else rng.sample(range(len(pool)), n_neg))]
        self.n_neg = len(self.items) - self.n_pos
        self.augment, self.crop = augment, crop

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        f, rle, box = self.items[i]
        rng = random.Random(random.getrandbits(32)) if self.augment else random.Random(i)
        img = read_native(f)
        gt = mask_util.decode(rle) if rle is not None else np.zeros((H, W), np.uint8)
        seed_box, pad = (jitter_box(box, rng) if self.augment else (list(box), REF_PAD))
        x0, y0, side = crop_bounds(seed_box, self.crop)
        im = img[y0:y0 + side, x0:x0 + side]
        tgt = gt[y0:y0 + side, x0:x0 + side]
        seed = seed_mask(seed_box, x0, y0, side, pad)
        if side != self.crop:
            tgt = (cv2.resize(tgt.astype(np.float32), (self.crop, self.crop), interpolation=cv2.INTER_AREA) >= 0.5)
        tgt = tgt.astype(np.uint8)
        if self.augment:
            # photometric on the raw crop (before CLAHE, which is part of the input transform)
            x = im.astype(np.float32) / 255.0
            if rng.random() < 0.8:
                x = np.clip(x, 0, 1) ** rng.uniform(0.7, 1.4)
            if rng.random() < 0.8:
                x = x * rng.uniform(0.8, 1.2) + rng.uniform(-0.08, 0.08)
            if rng.random() < 0.3:
                x = x + np.random.normal(0, rng.uniform(0.005, 0.03), x.shape).astype(np.float32)
            im = (np.clip(x, 0, 1) * 255).astype(np.uint8)
        x = make_input(im, seed, self.crop)
        y = tgt[None].astype(np.float32)
        if self.augment:  # same geometric op on all channels and the target
            if rng.random() < 0.5:
                x, y = x[:, :, ::-1], y[:, :, ::-1]
            if rng.random() < 0.5:
                x, y = x[:, ::-1], y[:, ::-1]
            k = rng.randrange(4)
            x, y = np.rot90(x, k, axes=(1, 2)), np.rot90(y, k, axes=(1, 2))
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(y))


def build_refiner(encoder: str = REF_ENCODER, pretrained: bool = True):
    return smp.Unet(encoder_name=encoder, encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def save_refiner(path, model, **meta):
    torch.save({"state_dict": model.state_dict(), "kind": "refiner", **meta}, path)


def load_refiner(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_refiner(ck.get("encoder", REF_ENCODER), pretrained=False).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


@torch.no_grad()
def refine_boxes(models: list, img: np.ndarray, boxes: np.ndarray, device, tta: bool = True,
                 crop: int = REF_CROP, batch: int = 32) -> list[dict]:
    """boxes (n, 5) xyxy+conf native -> one crop record per box:
    {"prob": uint8 (crop, crop), "x0", "y0", "side", "conf", "box": xyxy}. Probs are the mean
    over models (and 4 flips if tta). Nothing is thresholded here: the crop records are what
    the OOF sweep and the offline sweeps consume."""
    recs, xs = [], []
    for b in boxes:
        x0b, y0b, x1b, y1b, conf = [float(v) for v in b[:5]]
        box = [x0b, y0b, max(1.0, x1b - x0b), max(1.0, y1b - y0b)]
        x0, y0, side = crop_bounds(box, crop)
        seed = seed_mask(box, x0, y0, side)
        xs.append(make_input(img[y0:y0 + side, x0:x0 + side], seed, crop))
        recs.append({"x0": x0, "y0": y0, "side": side, "conf": conf, "box": [x0b, y0b, x1b, y1b]})
    if not recs:
        return recs
    probs = np.zeros((len(recs), crop, crop), np.float32)
    for i in range(0, len(xs), batch):
        x = torch.from_numpy(np.stack(xs[i:i + batch])).to(device)
        acc = torch.zeros((x.shape[0], 1, crop, crop), device=device)
        for m in models:
            acc += torch.sigmoid(m(x))
            if tta:
                acc += torch.sigmoid(m(x.flip(-1))).flip(-1)
                acc += torch.sigmoid(m(x.flip(-2))).flip(-2)
                acc += torch.sigmoid(m(x.flip(-1, -2))).flip(-1, -2)
        acc /= len(models) * (4 if tta else 1)
        probs[i:i + batch] = acc[:, 0].float().cpu().numpy()
    for r, p in zip(recs, probs):
        r["prob"] = np.round(p * 255).astype(np.uint8)
    return recs
