"""Cache building and torch datasets over the downscaled cache.

cache/img{S}/<stem>.png        grayscale image downscaled to S x S
cache/test{S}/<stem>.png       same for test images
cache/mask{S}/<record_id>.png  union mask of one annotator record at S x S
cache/gt_rles.pkl              {record_id: [compressed RLE dict, ...]} at 2048 (for PQ)
cache/meta.json                size, records_by_file, folds
"""
from __future__ import annotations

import json
import pickle
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from config import CACHE, IMG_SIZE, N_FOLDS, POSITIVE_BIAS, SEED, TARGET_MODE, log
from data import (annotation_rle, load_coco, make_folds, record_semantic_mask, test_files,
                  test_img_dir, train_files, train_img_dir)


def build_cache(size: int = IMG_SIZE, n_folds: int = N_FOLDS) -> dict:
    """Idempotent: skips files that already exist. Returns meta."""
    records, anns, by_file = load_coco()
    img_dir, tst_dir, msk_dir = CACHE / f"img{size}", CACHE / f"test{size}", CACHE / f"mask{size}"
    for d in (img_dir, tst_dir, msk_dir):
        d.mkdir(parents=True, exist_ok=True)
    gt_rles = {}
    for n, f in enumerate(train_files()):
        stem = f.rsplit(".", 1)[0]
        out = img_dir / f"{stem}.png"
        if not out.exists():
            im = cv2.imread(str(train_img_dir() / f), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(out), cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA))
        for rid in by_file[f]:
            a = anns.get(rid, [])
            gt_rles[rid] = [annotation_rle(x) for x in a if not int(x.get("iscrowd", 0))]
            mout = msk_dir / f"{rid}.png"
            if not mout.exists():
                m = record_semantic_mask(a).astype(np.float32)
                m = cv2.resize(m, (size, size), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(mout), ((m >= 0.5) * 255).astype(np.uint8))
        if n % 100 == 0:
            log(f"cache train {n}")
    for f in test_files():
        out = tst_dir / (f.rsplit(".", 1)[0] + ".png")
        if not out.exists():
            im = cv2.imread(str(test_img_dir() / f), cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(str(out), cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA))
    with open(CACHE / "gt_rles.pkl", "wb") as fh:
        pickle.dump(gt_rles, fh)
    meta = {"size": size, "records_by_file": by_file, "folds": make_folds(train_files(), n_folds, SEED),
            "medoid": medoid_records(by_file, msk_dir)}
    json.dump(meta, open(CACHE / "meta.json", "w"))
    log(f"cache ready: {len(by_file)} files, {len(gt_rles)} records, {len(test_files())} test")
    return meta


def medoid_records(by_file: dict, msk_dir) -> dict:
    """{file: record_id} — for multi-annotator files, the record whose union mask has
    the highest mean IoU with the other records' masks (the consensus annotator).
    Single-record files map to their only record."""
    out = {}
    for f, rids in by_file.items():
        if len(rids) == 1:
            out[f] = rids[0]
            continue
        masks = [_read(msk_dir / f"{r}.png") > 127 for r in rids]
        best, best_score = rids[0], -1.0
        for i, r in enumerate(rids):
            ious = []
            for j, m in enumerate(masks):
                if j == i:
                    continue
                u = (masks[i] | m).sum()
                ious.append((masks[i] & m).sum() / u if u else 1.0)
            s = float(np.mean(ious))
            if s > best_score:
                best, best_score = r, s
        out[f] = best
    return out


def load_meta() -> dict:
    return json.load(open(CACHE / "meta.json"))


def load_gt_rles() -> dict:
    return pickle.load(open(CACHE / "gt_rles.pkl", "rb"))


def _read(path) -> np.ndarray:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        raise FileNotFoundError(path)
    return im


def normalise(img: np.ndarray) -> np.ndarray:
    return (img.astype(np.float32) / 255.0 - 0.5) / 0.25


class CropDataset(Dataset):
    """Random crops from cached images. Each __getitem__ picks one annotator
    record at random for the file (label augmentation across annotators)."""

    def __init__(self, files: list[str], meta: dict, crop: int, samples_per_file: int,
                 positive_bias: float = POSITIVE_BIAS, target_mode: str = TARGET_MODE):
        self.files, self.meta = files, meta
        self.S, self.crop, self.spf, self.pos = meta["size"], crop, samples_per_file, positive_bias
        self.img_dir, self.msk_dir = CACHE / f"img{self.S}", CACHE / f"mask{self.S}"
        self.target_mode = target_mode

    def __len__(self):
        return len(self.files) * self.spf

    def __getitem__(self, i):
        f = self.files[i // self.spf]
        stem = f.rsplit(".", 1)[0]
        if self.target_mode == "medoid":
            rid = self.meta["medoid"][f]
        else:
            rid = random.choice(self.meta["records_by_file"][f])
        img = _read(self.img_dir / f"{stem}.png")
        msk = _read(self.msk_dir / f"{rid}.png") > 127

        c = self.crop
        if random.random() < self.pos and msk.any():
            ys, xs = np.nonzero(msk)
            k = random.randrange(len(ys))
            cy, cx = ys[k] + random.randint(-c // 3, c // 3), xs[k] + random.randint(-c // 3, c // 3)
            y0 = int(np.clip(cy - c // 2, 0, self.S - c))
            x0 = int(np.clip(cx - c // 2, 0, self.S - c))
        else:
            y0, x0 = random.randint(0, self.S - c), random.randint(0, self.S - c)
        img, msk = img[y0:y0 + c, x0:x0 + c], msk[y0:y0 + c, x0:x0 + c]

        # geometric augs: filaments have no canonical orientation on the disk
        if random.random() < 0.5:
            img, msk = img[:, ::-1], msk[:, ::-1]
        if random.random() < 0.5:
            img, msk = img[::-1], msk[::-1]
        k = random.randrange(4)
        img, msk = np.rot90(img, k), np.rot90(msk, k)

        # photometric augs: GONG sites differ in exposure/contrast
        x = img.astype(np.float32) / 255.0
        if random.random() < 0.8:
            x = np.clip(x, 0, 1) ** random.uniform(0.7, 1.4)
        if random.random() < 0.8:
            x = x * random.uniform(0.8, 1.2) + random.uniform(-0.08, 0.08)
        if random.random() < 0.3:
            x = x + np.random.normal(0, random.uniform(0.005, 0.03), x.shape).astype(np.float32)
        x = (np.clip(x, 0, 1) - 0.5) / 0.25
        return (torch.from_numpy(np.ascontiguousarray(x))[None],
                torch.from_numpy(np.ascontiguousarray(msk).astype(np.float32))[None])


class FullImageDataset(Dataset):
    """Whole cached images (S x S) for validation / test inference."""

    def __init__(self, files: list[str], size: int, split: str = "train"):
        self.files = files
        self.dir = CACHE / (f"img{size}" if split == "train" else f"test{size}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        f = self.files[i]
        img = _read(self.dir / (f.rsplit(".", 1)[0] + ".png"))
        return torch.from_numpy(normalise(img))[None], f
