"""COCO loading, polygon rasterisation and grouped folds.

Terminology
-----------
file   : a physical GONG H-alpha observation, e.g. ``20160213153054Bh.jpeg``.
record : one annotator's pass over a file. COCO ``images[].id`` is
         ``<annotator_id>-<file_stem>``; a file can have 1-3 records.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_util

from config import ANN_NAME, H, W, find_data_root


@lru_cache(maxsize=1)
def data_root() -> Path:
    return find_data_root()


def train_img_dir() -> Path:
    return data_root() / "train" / "train_images"


def test_img_dir() -> Path:
    return data_root() / "test" / "test_images"


def ann_path() -> Path:
    return data_root() / "train" / ANN_NAME


def _as_list(x):
    return json.loads(x) if isinstance(x, str) else x


def load_coco():
    """Return (records, anns_by_record, records_by_file).

    records         : {record_id: image dict}
    anns_by_record  : {record_id: [annotation, ...]}
    records_by_file : {file_name: [record_id, ...]} (sorted)
    """
    d = json.load(open(ann_path()))
    records = {i["id"]: i for i in d["images"]}
    anns_by_record: dict[str, list] = defaultdict(list)
    for a in d["annotations"]:
        anns_by_record[a["image_id"]].append(a)
    records_by_file: dict[str, list[str]] = defaultdict(list)
    for rid, i in records.items():
        records_by_file[i["file_name"]].append(rid)
    for k in records_by_file:
        records_by_file[k].sort()
    return records, dict(anns_by_record), dict(records_by_file)


def annotation_polygons(a) -> list[list[float]]:
    segs = _as_list(a["segmentation"])
    return [s for s in segs if len(s) >= 6]


def annotation_rle(a, h: int = H, w: int = W):
    """Compressed COCO RLE dict for one annotation (full resolution)."""
    polys = annotation_polygons(a)
    if not polys:
        return mask_util.encode(np.asfortranarray(np.zeros((h, w), np.uint8)))
    return mask_util.merge(mask_util.frPyObjects(polys, h, w))


def annotation_mask(a, h: int = H, w: int = W) -> np.ndarray:
    return mask_util.decode(annotation_rle(a, h, w)).astype(np.uint8)


def record_semantic_mask(anns: list, h: int = H, w: int = W) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    for a in anns:
        m |= annotation_mask(a, h, w)
    return m


def train_files() -> list[str]:
    return sorted(p.name for p in train_img_dir().glob("*.jpeg"))


def test_files() -> list[str]:
    return sorted(p.name for p in test_img_dir().glob("*.jpeg"))


def make_folds(files: list[str], n_folds: int = 5, seed: int = 2026) -> dict[str, int]:
    """Deterministic grouped folds: the unit is the physical file, so all
    annotator records of one observation land in the same fold."""
    files = sorted(files)
    rng = random.Random(seed)
    rng.shuffle(files)
    return {f: i % n_folds for i, f in enumerate(files)}
