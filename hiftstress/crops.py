"""Cut template / search crops exactly the way HiFT does, from ground-truth boxes.
Used for int8 calibration and for the latency benchmark (real image content)."""
import numpy as np

from .tracker import HookedTracker, cfg


def gt_crops(seqs, frames_per_seq=2, device="cpu"):
    helper = HookedTracker.__new__(HookedTracker)  # only need get_subwindow
    helper.device = device
    zs, xs = [], []
    for seq in seqs:
        img0 = seq.image(0)
        avg = np.mean(img0, axis=(0, 1))
        x, y, w, h = seq.gt[0]
        size = np.array([w, h])
        s_z = round(np.sqrt(np.prod(size + cfg.TRACK.CONTEXT_AMOUNT * size.sum())))
        centre = np.array([x + (w - 1) / 2, y + (h - 1) / 2])
        zs.append(helper.get_subwindow(img0, centre, cfg.TRACK.EXEMPLAR_SIZE, s_z, avg))
        s_x = round(s_z * cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        for i in range(1, min(len(seq), frames_per_seq + 1)):
            bx, by, bw, bh = seq.gt[i]
            if not np.all(np.isfinite(seq.gt[i])) or bw <= 0 or bh <= 0:
                continue
            c = np.array([bx + (bw - 1) / 2, by + (bh - 1) / 2])
            xs.append(helper.get_subwindow(seq.image(i), c, cfg.TRACK.INSTANCE_SIZE, s_x, avg))
    return zs, xs


def gt_pairs(seqs, frames_per_seq=2, device="cpu"):
    """(template crop, search crop) pairs from the same sequence, for head calibration."""
    pairs = []
    for seq in seqs:
        zs, xs = gt_crops([seq], frames_per_seq, device)
        pairs += [(zs[0], x) for x in xs]
    return pairs
