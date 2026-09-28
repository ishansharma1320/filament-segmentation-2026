"""Download GONG frames around training images (training-time use only; the host ruled that
inference must use the single supplied test image).

Two modes:
  short  (default)  same-site frames within +-WINDOW min, up to --max-per-file
  long   (--long)   same-site frames at +-OFFSETS minutes (closest within --tol) plus the
                    same-minute frame of every OTHER site (within --tol): long baselines
                    separate day-lived filaments from minute-lived fibrils, which +-10 min
                    cannot (the 2026-09-27 probe).

Frames land in data/sequences[_long]/<stem>/<frame>.jpg. Resumable; day listings cached;
one request at a time with a pause.

    python scripts/fetch_sequences.py --long --sample 160 --seed 2026
    python scripts/fetch_sequences.py --window 10 --max-per-file 12 --limit 20
"""
import argparse
import random
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import REPO  # noqa: E402
from data import train_files  # noqa: E402

BASE = "https://gong2.nso.edu/HA/hag"
UA = {"User-Agent": "Mozilla/5.0 (filament-segmentation-2026 research; polite fetch)"}
_listing_cache = {}


def get(url, retries=3):
    for k in range(retries):
        try:
            with urlopen(Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def day_listing(day: str):
    if day not in _listing_cache:
        try:
            html = get(f"{BASE}/{day[:6]}/{day}/").decode(errors="ignore")
            _listing_cache[day] = sorted(set(re.findall(r'href="(\d{14}[A-Z]h\.jpg)"', html)))
        except Exception:  # noqa: BLE001
            _listing_cache[day] = []
    return _listing_cache[day]


def t_of(name: str) -> datetime:
    return datetime.strptime(name[:14], "%Y%m%d%H%M%S")


def frames_around(t0: datetime):
    """All frames of the days touching t0 +- 8 h."""
    days = {(t0 + timedelta(hours=h)).strftime("%Y%m%d") for h in (-8, 0, 8)}
    return sorted({f for d in sorted(days) for f in day_listing(d)})


def short_neighbours(stem, window_min, max_per_file):
    t0, site = t_of(stem), stem[14]
    cands = [(abs((t_of(f) - t0).total_seconds()) / 60, f) for f in day_listing(stem[:8])
             if f[14] == site and f[:14] != stem[:14]]
    return [f for dt, f in sorted(cands) if dt <= window_min][:max_per_file]


def long_neighbours(stem, offsets_min, tol_min):
    t0, site = t_of(stem), stem[14]
    allf = frames_around(t0)
    out = []
    for o in offsets_min:
        for sign in (-1, 1):
            target = t0 + timedelta(minutes=sign * o)
            same = [(abs((t_of(f) - target).total_seconds()) / 60, f) for f in allf if f[14] == site]
            same = [x for x in same if x[0] <= tol_min]
            if same:
                out.append(min(same)[1])
    for other in "BCLMTUSZ".replace(site, ""):
        near = [(abs((t_of(f) - t0).total_seconds()) / 60, f) for f in allf if f[14] == other]
        near = [x for x in near if x[0] <= tol_min]
        if near:
            out.append(min(near)[1])
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long", action="store_true")
    ap.add_argument("--offsets", default="60,180,360", help="minutes, +- each (long mode)")
    ap.add_argument("--tol", type=float, default=5.0, help="minutes tolerance (long mode)")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--max-per-file", type=int, default=12)
    ap.add_argument("--sample", type=int, default=0, help="random subset of train files")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.3)
    a = ap.parse_args()
    out_root = REPO / "data" / ("sequences_long" if a.long else "sequences")
    files = train_files()
    if a.sample:
        files = sorted(random.Random(a.seed).sample(files, a.sample))
    if a.limit:
        files = files[:a.limit]
    offsets = [int(x) for x in a.offsets.split(",")]
    n_ok = n_skip = n_fail = 0
    t_start = time.time()
    for i, f in enumerate(files, 1):
        stem = f.rsplit(".", 1)[0]
        d = out_root / stem
        d.mkdir(parents=True, exist_ok=True)
        try:
            names = long_neighbours(stem, offsets, a.tol) if a.long else short_neighbours(stem, a.window, a.max_per_file)
        except Exception as e:  # noqa: BLE001
            print(f"{stem}: listing failed ({e})", flush=True)
            n_fail += 1
            continue
        for name in names:
            p = d / name
            if p.exists() and p.stat().st_size > 10000:
                n_skip += 1
                continue
            try:
                p.write_bytes(get(f"{BASE}/{name[:6]}/{name[:8]}/{name}"))
                n_ok += 1
                time.sleep(a.pause)
            except Exception as e:  # noqa: BLE001
                print(f"{stem}/{name}: {e}", flush=True)
                n_fail += 1
        if i % 10 == 0 or i == len(files):
            print(f"[{time.time() - t_start:6.0f}s] {i}/{len(files)} | downloaded {n_ok} skipped {n_skip} "
                  f"failed {n_fail} | {stem} -> {len(names)} frames", flush=True)


if __name__ == "__main__":
    main()
