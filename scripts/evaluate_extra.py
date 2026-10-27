"""scripts/evaluate.py (unchanged) with the extra raw-tree datasets of evaluation/extra_datasets.py registered."""
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
import hift_research.evaluation.extra_datasets  # noqa: E402,F401  (registers dataset counts)

if __name__ == "__main__":
    runpy.run_path(str(ROOT/"scripts"/"evaluate.py"), run_name="__main__")
