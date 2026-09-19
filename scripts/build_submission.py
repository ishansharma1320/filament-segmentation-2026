"""Build a submission from saved test crop records with any instance scoring / knobs, so an
offline-selected recipe (e.g. conf x objectness^b) can be submitted without a kernel run.

    python3 scripts/build_submission.py runs/<stamp> --cut 0.3 --thr 0.5 --min-area 100 --b 0 --out sub.csv
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import cascade  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--cut", type=float, default=0.3)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--min-area", type=int, default=100)
    ap.add_argument("--b", type=float, default=0.0, help="score = conf * obj^b")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    crops = cascade.load_crops(Path(a.run, "test_crops"))
    crops = {f: cascade.rescored(v, a.b) for f, v in crops.items()}
    out_dir = Path(a.run)
    df = cascade.write_cascade_submission(crops, a.cut, a.thr, a.min_area, out_dir)
    if a.out:
        df.to_csv(a.out, index=False)
        print("wrote", a.out)
