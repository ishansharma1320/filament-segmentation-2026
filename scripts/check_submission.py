"""Validate a submission.csv: columns, unique ids, decodable RLEs, test coverage."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data import test_files  # noqa: E402
from rle import rle_to_mask  # noqa: E402


def main(path: str):
    df = pd.read_csv(path)
    assert list(df.columns) == ["filament_id", "segmentation_rle"], df.columns
    assert df.filament_id.is_unique, "duplicate filament_id"
    assert df.segmentation_rle.notna().all() and (df.segmentation_rle.str.len() > 0).all()
    stems = df.filament_id.str.rsplit("_", n=1).str[0]
    test = {f.rsplit(".", 1)[0] for f in test_files()}
    missing, extra = test - set(stems), set(stems) - test
    areas = np.array([rle_to_mask(r).sum() for r in df.segmentation_rle])
    print(f"rows={len(df)} images={stems.nunique()}/{len(test)} missing={len(missing)} extra={len(extra)}")
    print(f"per-image rows: mean {len(df) / stems.nunique():.2f} max {stems.value_counts().max()}")
    print(f"area px: min {areas.min()} median {np.median(areas):.0f} max {areas.max()}")
    if missing or extra:
        print("PROBLEMS FOUND", sorted(missing)[:5], sorted(extra)[:5])
        sys.exit(1)
    print("OK")


if __name__ == "__main__":
    main(sys.argv[1])
