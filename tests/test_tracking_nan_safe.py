import numpy as np
import torch

from hift_research.inference import configure_upstream


class _FakeModel:
    """Constant heads: one inverted-box cell with the top score, one good cell nearby."""
    def __init__(self, inverted, good):
        self.loc = torch.full((1, 4, 11, 11), 0.2)
        # width -198 px, height +420 px: (w+pad)*(h+pad) < 0 -> NaN scale penalty upstream
        self.loc[0, :2, inverted // 11, inverted % 11] = -0.6
        self.loc[0, 2:, inverted // 11, inverted % 11] = 0.9
        self.cls1 = torch.zeros(1, 2, 11, 11)
        self.cls1[0, 1].view(-1)[good] = 3.0
        self.cls1[0, 1].view(-1)[inverted] = 6.0
        self.cls2 = torch.zeros(1, 1, 11, 11)

    def eval(self):
        return self

    def template(self, z):
        pass

    def track(self, x):
        return {"loc": self.loc, "cls1": self.cls1, "cls2": self.cls2}


def _run(tracker_factory):
    from pathlib import Path
    cfg = configure_upstream(Path(__file__).resolve().parents[1])
    cfg.CUDA = False  # CPU-only fake model
    image = np.full((300, 300, 3), 127, dtype=np.uint8)
    tracker = tracker_factory(_FakeModel(inverted=0, good=60))
    tracker.init(image, [130, 130, 40, 40])
    return tracker, tracker.track(image)


def test_upstream_selects_nonfinite_cell_and_nan_safe_does_not():
    from pathlib import Path
    configure_upstream(Path(__file__).resolve().parents[1]).CUDA = False
    from pysot.tracker.hift_tracker import HiFTTracker
    from hift_research.tracking_nan_safe import make_nan_safe_tracker
    with np.errstate(invalid="ignore"):
        _, upstream = _run(HiFTTracker)
    tracker, safe = _run(make_nan_safe_tracker)
    assert type(tracker).first_nan_would_have_won == 1 and type(tracker).nonfinite_cells >= 1
    assert np.isfinite(safe["bbox"]).all() and safe["bbox"][2] > 10
    assert not np.allclose(upstream["bbox"], safe["bbox"])
