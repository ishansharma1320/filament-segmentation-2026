"""Download same-site GONG frames around each training image (training-time use only;
the host ruled that inference must use the single supplied test image).

For train file <YYYYMMDDhhmmss><S>h.jpeg the archive day listing
https://gong2.nso.edu/HA/hag/<YYYYMM>/<YYYYMMDD>/ is scanned for frames of site S within
+-WINDOW_MIN minutes (the labelled frame itself excluded). Files land in
data/sequences/<stem>/<frame>.jpg. Resumable: existing files are skipped; a day listing is
cached. Polite: one request at a time, small pause.

    python scripts/fetch_sequences.py                 # all 707 train files
    python scripts/fetch_sequences.py --window 10 --max-per-file 12 --limit 20   # smoke
"""
import argparse
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
OUT = REPO / "data" / "sequences"
UA = {"User-Agent": "Mozilla/5.0 (filament-segmentation-2026 research; polite fetch)"}
_listing_cache = {}


def get(url, retries=3):
    for k in range(retries):
        try:
            with urlopen(Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            time.sleep(2 * (k + 1))


def day_listing(day: str):
    if day not in _listing_cache:
        html = get(f"{BASE}/{day[:6]}/{day}/").decode(errors="ignore")
        _listing_cache[day] = sorted(set(re.findall(r'href="(\d{14}[A-Z]h\.jpg)"', html)))
    return _listing_cache[day]


def neighbours(stem: str, window_min: int, max_per_file: int):
    ts, site = stem[:14], stem[14]
    t0 = datetime.strptime(ts, "%Y%m%d%H%M%S")
    cands = []
    for f in day_listing(ts[:8]):
        if f[14] != site or f[:14] == ts:
            continue
        dt = abs((datetime.strptime(f[:14], "%Y%m%d%H%M%S") - t0).total_seconds()) / 60
        if dt <= window_min:
            cands.append((dt, f))
    cands.sort()
    return [f for _, f in cands[:max_per_file]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=10, help="minutes either side")
    ap.add_argument("--max-per-file", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--pause", type=float, default=0.3)
    a = ap.parse_args()
    files = train_files()
    if a.limit:
        files = files[:a.limit]
    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for i, f in enumerate(files, 1):
        stem = f.rsplit(".", 1)[0]
        d = OUT / stem
        d.mkdir(parents=True, exist_ok=True)
        try:
            names = neighbours(stem, a.window, a.max_per_file)
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
                p.write_bytes(get(f"{BASE}/{stem[:6]}/{stem[:8]}/{name}"))
                n_ok += 1
                time.sleep(a.pause)
            except Exception as e:  # noqa: BLE001
                print(f"{stem}/{name}: {e}", flush=True)
                n_fail += 1
        if i % 25 == 0 or i == len(files):
            print(f"[{time.time() - t0:6.0f}s] {i}/{len(files)} files | downloaded {n_ok} "
                  f"skipped {n_skip} failed {n_fail} | last: {stem} -> {len(names)} frames", flush=True)


if __name__ == "__main__":
    main()
