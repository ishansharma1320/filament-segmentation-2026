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
df["img"] = df.filament_id.str.rsplit("_", n=1).str[0]
k = abs(a.dilate)
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
before, after, clipped = 0, 0, 0
out = {}
# The Kaggle evaluator errors on ANY overlap between predicted instances (2 px was enough),
# so grown masks are clipped against pixels already taken by earlier instances of the image.
for img, g in df.groupby("img", sort=False):
    taken = np.zeros((2048, 2048), bool)
    for idx, r in zip(g.index, g.segmentation_rle):
        m = rle_to_mask(r)
        before += int(m.sum())
        grown = cv2.dilate(m, se) if a.dilate > 0 else cv2.erode(m, se)
        if not grown.any():
            grown = m
        ov = grown.astype(bool) & taken
        clipped += int(ov.sum())
        grown[ov] = 0
        taken |= grown.astype(bool)
        after += int(grown.sum())
        out[idx] = mask_to_rle(grown)
df["segmentation_rle"] = pd.Series(out)
df = df.drop(columns="img")
print(f"clipped {clipped} overlapping px")
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print(f"{len(df)} rows, total area {before} -> {after} ({after / before:.3f}x), wrote {a.out}")
