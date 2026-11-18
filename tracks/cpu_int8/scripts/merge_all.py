"""Merge every split condition listed in plan/merges.json whose parts have all finished.
Safe to run any time (it skips incomplete or already-merged conditions)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(results="results"):
    mf = ROOT / "plan/merges.json"
    if not mf.exists():
        return
    for m in json.loads(mf.read_text()):
        out = ROOT / results / m["name"]
        parts = [ROOT / results / p for p in m["parts"]]
        if (out / "summary.json").exists():
            continue
        if not all((p / "summary.json").exists() for p in parts):
            print(f"[wait] {m['name']}: {sum((p / 'summary.json').exists() for p in parts)}/{len(parts)} parts done")
            continue
        cmd = [sys.executable, str(ROOT / "scripts/merge_parts.py"), "--out", str(out)] + [str(p) for p in parts]
        if m.get("expect"):
            cmd += ["--expect-seqs-file", str(ROOT / m["expect"])]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
