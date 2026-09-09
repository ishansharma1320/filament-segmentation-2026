"""Build the local 1024 px cache, GT RLEs and grouped folds (see src/dataset.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset import build_cache  # noqa: E402

if __name__ == "__main__":
    build_cache()
