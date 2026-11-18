"""Print the adaptive attacker's lambda chosen on DEV (plan/round5/attacker_selected.json).
Fails (non-zero exit, prints nothing) until scripts/r5_gate.py has made the choice."""
import json
import sys
from pathlib import Path

f = Path(__file__).resolve().parents[1] / "plan/round5/attacker_selected.json"
if not f.exists():
    sys.stderr.write("plan/round5/attacker_selected.json missing: run scripts/r5_gate.py after the P1 pilot\n")
    sys.exit(1)
lam = json.loads(f.read_text()).get("shift_lambda")
if lam is None:
    sys.exit(1)
print(lam)
