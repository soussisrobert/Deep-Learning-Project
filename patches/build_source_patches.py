"""Regenerate explicit audit patches without changing the upstream worktree.

Run from any directory with Python. Apply/check patches separately; these are B2
research candidates, not a request to mutate the pinned submodule.
"""
from difflib import unified_diff
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "third_party" / "HiFT"
PATCHES = Path(__file__).resolve().parent


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError("pinned source fragment missing or ambiguous")
    return source.replace(old, new, 1)


def patch(filename, changes):
    chunks = []
    for relative, transform in changes:
        source = (UPSTREAM / relative).read_text(encoding="utf-8")
        corrected = transform(source)
        chunks.extend(unified_diff(source.splitlines(True), corrected.splitlines(True),
                                   fromfile="a/" + relative, tofile="b/" + relative))
    (PATCHES / filename).write_text("".join(chunks), encoding="utf-8", newline="\n")


def layout(source):
    start = source.index("        src=self.cross_attn(src.view(b,c,int(s**0.5),int(s**0.5))")
    stop = source.index("\n", source.index(",srcc.contiguous().view", start))
    return source[:start] + (
        "        spatial_src = src.permute(1, 2, 0).contiguous().view(b, c, int(s**0.5), int(s**0.5))\n"
        "        spatial_srcc = srcc.permute(1, 2, 0).contiguous().view(b, c, int(s**0.5), int(s**0.5))\n"
        "        src = self.cross_attn(spatial_src, spatial_srcc).view(b,c,-1).permute(2, 0, 1)"
    ) + source[stop:]


def negative_targets(source):
    return replace_once(source, "    def get(self,bbox,size):\n", (
        "    def get(self,bbox,size,neg=False):\n"
        "        if neg:\n"
        "            # Explicit B2 policy: dense background and no box supervision.\n"
        "            return (np.zeros((1,size,size)),\n"
        "                    np.zeros((4,size,size), dtype=np.float32),\n"
        "                    np.zeros((1,size,size)), np.zeros((1,size,size)))\n"
    ))


def negative_dataset(source):
    return replace_once(source, "self.anchor_target.get(bbox, cfg.TRAIN.OUTPUT_SIZE)",
                        "self.anchor_target.get(bbox, cfg.TRAIN.OUTPUT_SIZE, neg=neg)")


def distributed_guard(source):
    return replace_once(source, "def dist_init():\n", (
        "def dist_init():\n"
        "    if (int(os.environ.get('WORLD_SIZE', '1')) != 1 or\n"
        "            int(os.environ.get('RANK', '0')) != 0 or\n"
        "            int(os.environ.get('LOCAL_RANK', '0')) != 0):\n"
        "        raise RuntimeError('Pinned HiFT supports one DataParallel process only; '\n"
        "                           'multi-process launch requires an explicit DDP implementation')\n"
    ))


def singleton_loss(source):
    old = ".nonzero(as_tuple =False).squeeze().cuda()"
    if source.count(old) != 2:
        raise RuntimeError("expected exactly two pinned class selectors")
    return source.replace(old, ".nonzero(as_tuple=False).flatten().to(pred.device)")


if __name__ == "__main__":
    patch("0001-transformer-layout-b2.patch", [("pysot/models/utile/tran.py", layout)])
    patch("0002-negative-pair-supervision-b2.patch", [
        ("pysot/datasets/anchortarget.py", negative_targets),
        ("pysot/datasets/dataset.py", negative_dataset),
    ])
    patch("0003-reject-unsupported-distributed-launch.patch", [
        ("pysot/utils/distributed.py", distributed_guard),
    ])
    patch("0004-singleton-class-selector-b2.patch", [("pysot/models/loss.py", singleton_loss)])
