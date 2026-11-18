"""Check the domain evidence bundle against its own SHA256SUMS.

Weights listed in SHA256SUMS are not stored in git; they are reported as 'not present' unless you
have downloaded them into place (see checkpoints/README.md), in which case they are checked too.
"""
import hashlib, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
B = ROOT / "tracks/domain_ft/evidence/stage7"
ok = bad = absent = 0
for line in (B / "SHA256SUMS").read_text().splitlines():
    if not line.strip():
        continue
    want, rel = line.split(None, 1)
    rel = rel.lstrip("*")
    f = B / rel
    if not f.exists():
        if rel.endswith((".pt", ".pth")):
            absent += 1
            continue
        print("MISSING", rel); bad += 1; continue
    # normalise nothing: hashes bind the exact bytes; .gitattributes keeps these files binary-safe
    got = hashlib.sha256(f.read_bytes()).hexdigest()
    if got == want:
        ok += 1
    else:
        print("CHANGED", rel); bad += 1
print(f"{ok} files match, {bad} problems, {absent} weight files not present (download separately)")
sys.exit(1 if bad else 0)
