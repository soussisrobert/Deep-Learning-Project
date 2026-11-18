"""Print the selected defence name for queue commands: `python scripts/r3_selected.py jpeg` -> jpeg90.
Fails (non-zero exit, nothing printed) if scripts/r3_select.py has not been run, so stage-2 runs
cannot start with a guessed quality."""
import json
import sys
from pathlib import Path

f = Path(__file__).resolve().parents[1] / "plan/round3/selected.json"
if not f.exists():
    sys.stderr.write("plan/round3/selected.json missing: run scripts/r3_select.py after stage 1\n")
    sys.exit(1)
print(json.loads(f.read_text())[sys.argv[1] if len(sys.argv) > 1 else "jpeg"])
