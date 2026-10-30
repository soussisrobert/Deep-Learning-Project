"""Print the run_track arguments of the int8 recipe chosen on DEV (plan/round4/int8_selected.json).
Fails (non-zero exit, prints nothing) until scripts/r4_gate.py has made the choice."""
import json
import sys
from pathlib import Path

f = Path(__file__).resolve().parents[1] / "plan/round4/int8_selected.json"
if not f.exists():
    sys.stderr.write("plan/round4/int8_selected.json missing: run scripts/r4_gate.py after the Q0 pilot\n")
    sys.exit(1)
print(json.loads(f.read_text())["args"])
