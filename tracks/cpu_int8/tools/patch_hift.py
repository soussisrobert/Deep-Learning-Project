"""Make the official HiFT code run on CPU, Apple MPS or CUDA.

The upstream code calls .cuda() in two places inside the model. This script
replaces those calls with device-neutral versions. On a CUDA machine the
patched model behaves exactly like the original.

Usage:  python tools/patch_hift.py third_party/HiFT
It is safe to run twice: already-patched files are left alone.
"""
import sys
from pathlib import Path

PIN = "992aa7c1605360d607af84cb804b93cd8ee81caf"

EDITS = [
    ("pysot/models/model_builder.py",
     "self.backbone = AlexNet().cuda()", "self.backbone = AlexNet()"),
    ("pysot/models/model_builder.py",
     "self.grader=HiFT(cfg).cuda()", "self.grader=HiFT(cfg)"),
    ("pysot/models/utile/utile.py",
     "i = t.arange(w).cuda()", "i = t.arange(w, device=res3.device)"),
    ("pysot/models/utile/utile.py",
     "j = t.arange(h).cuda()", "j = t.arange(h, device=res3.device)"),
]


def main(root):
    root = Path(root)
    for rel, old, new in EDITS:
        path = root / rel
        text = path.read_text()
        if new in text and old not in text:
            print(f"already patched: {rel}: {new}")
            continue
        count = text.count(old)
        if count != 1:
            sys.exit(f"expected exactly 1 occurrence of {old!r} in {rel}, found {count}. "
                     f"Is the HiFT checkout at commit {PIN}?")
        path.write_text(text.replace(old, new))
        print(f"patched: {rel}: {old}  ->  {new}")
    print("HiFT is now device-neutral.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "third_party/HiFT")
