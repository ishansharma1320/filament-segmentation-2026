"""Re-post-process an existing submission's masks without a model (LB probes).

    python scripts/transform_submission.py runs/<stamp>/submission.csv --dilate 2 --out submissions/v1_dil2.csv

Positive --dilate grows every mask by k px (elliptical SE), negative shrinks; masks that
vanish under erosion are kept as-is so every image keeps its rows.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from rle import mask_to_rle, rle_to_mask  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("csv")
ap.add_argument("--dilate", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

df = pd.read_csv(a.csv)
k = abs(a.dilate)
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
before, after = 0, 0
out = []
for r in df.segmentation_rle:
    m = rle_to_mask(r)
    before += int(m.sum())
    g = cv2.dilate(m, se) if a.dilate > 0 else cv2.erode(m, se)
    if not g.any():
        g = m
    after += int(g.sum())
    out.append(mask_to_rle(g))
df["segmentation_rle"] = out
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"{len(df)} rows, total area {before} -> {after} ({after / before:.3f}x), wrote {a.out}")
