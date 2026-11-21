"""Exact-duplicate frame check between training images and evaluation images (leakage diagnostic).

Hashes every .jpg under each root (sha256 of the file bytes) and reports how many training frames are
byte-identical to an evaluation frame. Only exact copies are detected: re-encoded, resized or cropped copies of
the same footage are not. A training side can be a root or a hash list written earlier by --hashes-out, so a
TC2-side list can be compared locally against sets that never leave this machine (sealed test).

    python scripts/frame_overlap.py --train data/.../train_part2 --eval V1=data/.../val --out overlap.json
    python scripts/frame_overlap.py --train-hashes SOT2-frames.sha256 --eval test-dev=data/.../test-dev --out x.json
"""
import argparse
import hashlib
import json
from pathlib import Path

CHUNK = 1 << 20


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(root):
    """{relative posix path: sha256} for every .jpg under root (sorted, deterministic)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"not a directory: {root}")
    return {p.relative_to(root).as_posix(): file_sha256(p) for p in sorted(root.rglob("*.jpg"))}


def read_hashes(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        out[name] = digest
    return out


def overlap(train, evaluation):
    """Training frames whose bytes equal some evaluation frame: {train path: [eval paths]}."""
    by_digest = {}
    for name, digest in evaluation.items():
        by_digest.setdefault(digest, []).append(name)
    return {name: by_digest[digest] for name, digest in train.items() if digest in by_digest}


def main():
    parser = argparse.ArgumentParser()
    side = parser.add_mutually_exclusive_group(required=True)
    side.add_argument("--train", type=Path)
    side.add_argument("--train-hashes", type=Path)
    parser.add_argument("--eval", nargs="+", required=True, metavar="NAME=ROOT")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hashes-out", type=Path)
    args = parser.parse_args()
    train = hash_tree(args.train) if args.train else read_hashes(args.train_hashes)
    if not train:
        raise ValueError("no training frames found")
    if args.hashes_out:
        args.hashes_out.write_text("".join(f"{d}  {n}\n" for n, d in train.items()), encoding="utf-8")
    report = {"method": "sha256 of file bytes; exact copies only", "train_frames": len(train),
              "train_unique": len(set(train.values())), "eval": {}}
    for item in args.eval:
        name, _, root = item.partition("=")
        if not name or not root:
            raise ValueError(f"--eval expects NAME=ROOT, got {item!r}")
        hashes = hash_tree(root)
        dup = overlap(train, hashes)
        report["eval"][name] = {"frames": len(hashes), "duplicate_train_frames": len(dup),
                                "examples": dict(list(dup.items())[:10])}
        print(f"{name}: {len(hashes)} eval frames, {len(dup)} byte-identical training frames", flush=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
