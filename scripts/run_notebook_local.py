#!/usr/bin/env python3
"""Execute notebooks/submission.ipynb cell by cell in ONE namespace, like Kaggle does.

Catches assembly bugs (a stripped import that was actually needed, name collisions
between modules) that importing src/ as separate files would hide. Run from repo root:

    FILAMENT_DEBUG=1 conda run -n filament python scripts/run_notebook_local.py
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
nb = json.load(open(REPO / "notebooks" / "submission.ipynb"))
ns = {}
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] != "code":
        continue
    src = "".join(c["source"])
    t = time.time()
    try:
        exec(compile(src, f"cell{i}", "exec"), ns)
    except Exception:
        print(f"### cell {i} FAILED: {src.splitlines()[0][:60]}", flush=True)
        raise
    print(f"### cell {i} ok ({time.time() - t:.1f}s): {src.splitlines()[0][:60]}", flush=True)
sys.exit(0)
