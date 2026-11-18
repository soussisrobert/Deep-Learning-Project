"""Generate Wave2 mask-before-arithmetic patch; never apply to upstream."""
import ast
from difflib import unified_diff
from pathlib import Path
from textwrap import indent

ROOT = Path(__file__).resolve().parents[1]
RELATIVE = "pysot/models/loss.py"


def corrected_source(source):
    # Patch 0004 already fixes singleton selectors. Retain that separate change.
    source = source.replace("        return 0\n", "        return pred.reshape(-1)[:0].sum()\n", 1)
    old_class = next(item for item in ast.parse(source).body if getattr(item, "name", None) == "IOULoss")
    old_lines = source.splitlines(True)
    adapter = (ROOT / "src/hift_research/corrections/masked_losses.py").read_text(encoding="utf-8")
    function = next(item for item in ast.parse(adapter).body if getattr(item, "name", None) == "masked_iou_loss")
    implementation = ast.get_source_segment(adapter, function)
    implementation = implementation.replace("def masked_iou_loss(pred, target, weight):",
                                          "def forward(self, pred, target, weight=None):", 1)
    implementation = implementation.replace("return _empty_zero(pred)", "return pred.reshape(-1)[:0].sum()")
    replacement = "class IOULoss(nn.Module):\n" + indent(implementation, "    ") + "\n"
    return "".join(old_lines[:old_class.lineno - 1]) + replacement + "".join(old_lines[old_class.end_lineno:])


if __name__ == "__main__":
    original = (ROOT / "third_party/HiFT" / RELATIVE).read_text(encoding="utf-8")
    diff = unified_diff(original.splitlines(True), corrected_source(original).splitlines(True),
                        fromfile="a/" + RELATIVE, tofile="b/" + RELATIVE, n=1)
    (ROOT / "patches/0006-mask-before-arithmetic-b2.patch").write_text("".join(diff), encoding="utf-8", newline="\n")
