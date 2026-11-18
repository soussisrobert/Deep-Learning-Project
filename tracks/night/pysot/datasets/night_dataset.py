"""Training / validation dataset for the night fine-tuning (new file; pysot/datasets/dataset.py is not changed).

Reads the crop511 data written by tools/crop_dataset.py (training_dataset/night_ft/<dataset>/). Differences from TrkDataset:
  * the mix of datasets per epoch is set by cfg.DATASET.MIX (fractions of sampled pairs), epoch length by cfg.DATASET.VIDEOS_PER_EPOCH
    (pairs per epoch), and an epoch's pairs are drawn from a seed that depends on the epoch, so a resumed run does not repeat;
  * night augmentation (pysot/datasets/night_aug.py) after the crop: on-the-fly darkening for UAV123 / UAVDT, random gamma
    enhancement with ONE value shared by template and search;
  * real negative pairs (all search cells labelled negative, no localisation loss). The old NEG branch in TrkDataset labels
    its pair as a positive at the search crop's own box (anchor_target.get is called without a negative flag); here
      - cfg.DATASET.NEG: template from one target, search crop from a different video;
      - cfg.DATASET.NIGHT.HARDNEG (UAVDT only): template of object A, search crop centred on another object B of the same
        class from the same sequence, valid only if A's box in that frame lies completely outside the 287 search window
        after the shift/scale augmentation (else tried again, finally a normal positive pair is used).
"""
import os
import json
import logging

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.core.config import cfg
from pysot.datasets.anchortarget import AnchorTarget
from pysot.datasets.dataset import SubDataset
from pysot.datasets.night_aug import NightAugmentation, DarkenSpec, darken, enhance_gamma
from pysot.utils.bbox import Corner, center2corner, Center

logger = logging.getLogger('global')
cv2.setNumThreads(0)

# darkening profiles (ranges of gamma and brightness scale; noise / cast / JPEG ranges are the DarkenSpec defaults)
PROFILES = {
    'UAV123': DarkenSpec(p=0.6, gamma=(1.5, 3.0), scale=(0.3, 1.0)),
    'UAVDT_day': DarkenSpec(p=0.8, gamma=(2.0, 3.5), scale=(0.25, 0.8)),      # daylight sequences: extra darkening
    'UAVDT_night': DarkenSpec(p=0.6, gamma=(1.3, 2.2), scale=(0.4, 1.0)),     # already dusk-level, milder
}


def kind_of(name):
    n = name.upper()
    return 'UAV123' if n.startswith('UAV123') else ('UAVDT' if n.startswith('UAVDT') else 'UAVDARK')


class NightTrkDataset(Dataset):
    def __init__(self, split='train', n_epochs=1, start_epoch=0, seed=0, val_gamma=None, n_val=2000):
        super(NightTrkDataset, self).__init__()
        self.split = split
        self.train = split == 'train'
        self.val_gamma = val_gamma
        self.seed = seed
        names = list(cfg.DATASET.NAMES) if self.train else ['UAVDARK_VAL']
        mix = list(cfg.DATASET.MIX) if self.train else [1.0]
        self.pairs_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH if self.train else n_val
        self.subs, self.kinds, self.meta = [], [], {}
        start = 0
        for name in names:
            c = getattr(cfg.DATASET, name)
            sub = SubDataset(name, c.ROOT, c.ANNO, c.FRAME_RANGE, -1, start)
            start += sub.num
            self.subs.append(sub)
            self.kinds.append(kind_of(name))
            mp = os.path.join(os.path.dirname(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../', c.ANNO)), 'meta.json')
            if os.path.exists(mp) and kind_of(name) == 'UAVDT':
                self.meta[len(self.subs) - 1] = json.load(open(mp))
                if getattr(c, 'WEATHER', ''):   # e.g. 'daylight': keep only the tracks of that weather label (control run D)
                    m = self.meta[len(self.subs) - 1]
                    for vid in list(sub.labels.keys()):
                        for trk in list(sub.labels[vid].keys()):
                            if m.get(vid, {}).get(trk, {}).get('weather') != c.WEATHER:
                                del sub.labels[vid][trk]
                        if not sub.labels[vid]:
                            del sub.labels[vid]
                    sub.videos = list(sub.labels.keys())
                    sub.num = len(sub.videos)
                    logger.info('{}: kept only {} tracks ({} videos)'.format(name, c.WEATHER, sub.num))
        self.mix = mix
        self.anchor_target = AnchorTarget()
        self.template_aug = NightAugmentation(cfg.DATASET.TEMPLATE.SHIFT, cfg.DATASET.TEMPLATE.SCALE, cfg.DATASET.TEMPLATE.BLUR,
                                              cfg.DATASET.TEMPLATE.FLIP, cfg.DATASET.TEMPLATE.COLOR)
        self.search_aug = NightAugmentation(cfg.DATASET.SEARCH.SHIFT, cfg.DATASET.SEARCH.SCALE, cfg.DATASET.SEARCH.BLUR,
                                            cfg.DATASET.SEARCH.FLIP, cfg.DATASET.SEARCH.COLOR)
        night = cfg.DATASET.NIGHT
        self.enh_p, self.enh_gamma = night.ENH_P, night.ENH_GAMMA
        self.hardneg_p, self.neg_p = night.HARDNEG, cfg.DATASET.NEG
        self.darken_on = bool(night.DARKEN) and self.train
        self.epochs = list(range(start_epoch, start_epoch + n_epochs))
        self.picks = []
        for e in self.epochs:
            self.picks += self._epoch_picks(e)
        logger.info('{} dataset: {} pairs ({} epochs x {}), mix {}'.format(split, len(self.picks), n_epochs, self.pairs_per_epoch, self.mix))

    def _epoch_picks(self, epoch):
        rng = np.random.RandomState((self.seed * 100003 + epoch * 101 + (0 if self.train else 7)) % (2 ** 32))
        out = []
        for j, sub in enumerate(self.subs):
            n = int(round(self.pairs_per_epoch * self.mix[j]))
            vids = []
            while len(vids) < n:
                vids += list(rng.permutation(sub.num))
            out += [(j, int(v)) for v in vids[:n]]
        rng.shuffle(out)
        return out

    def __len__(self):
        return len(self.picks)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _get_bbox(image, shape):
        imh, imw = image.shape[:2]
        w, h = (shape[2] - shape[0], shape[3] - shape[1]) if len(shape) == 4 else shape
        wc_z = w + 0.5 * (w + h)
        hc_z = h + 0.5 * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = cfg.TRAIN.EXEMPLAR_SIZE / s_z
        return center2corner(Center(imw // 2, imh // 2, w * scale_z, h * scale_z))

    def _neg_labels(self, rng):
        size = cfg.TRAIN.OUTPUT_SIZE
        l1 = -np.ones((1, size, size))
        idx = rng.choice(size * size, 40, replace=False)
        l1.reshape(-1)[idx] = 0                      # 40 negative cells, the rest ignored (a positive pair uses about 2.5 x its positives)
        return l1, np.zeros((4, size, size), dtype=np.float32), np.zeros((1, size, size)), np.zeros((1, size, size))

    def _darken_profile(self, path):
        """profile by the crop's own dataset (from its path), so a negative pair's search crop from another dataset is handled"""
        p = path.lower()
        if '/uav123/' in p:
            return PROFILES['UAV123']
        if '/uavdt/' in p:
            video, track = path.split('/')[-2], path.split('/')[-1].split('.')[1]
            w = 'daylight'
            for m in self.meta.values():
                if video in m and track in m[video]:
                    w = m[video][track].get('weather', 'daylight')
            return PROFILES['UAVDT_night' if w == 'night' else 'UAVDT_day']
        return None

    def _distractor(self, j, v, rng):
        """-> (template (path, anno), search (path, anno), A's box at the search frame in B's crop511 coordinates) or None"""
        sub = self.subs[j]
        vid = sub.videos[v]
        tracks = list(sub.labels[vid].keys())
        meta = self.meta.get(j, {}).get(vid, {})
        if len(tracks) < 2:
            return None
        for _ in range(10):
            a, b = rng.choice(tracks, 2, replace=False)
            if meta.get(a, {}).get('class') != meta.get(b, {}).get('class'):
                continue
            common = sorted(set(sub.labels[vid][a]['frames']) & set(sub.labels[vid][b]['frames']))
            if not common:
                continue
            f2 = int(rng.choice(common))
            fa = sub.labels[vid][a]['frames']
            near = [f for f in fa if abs(f - f2) <= sub.frame_range]
            f1 = int(rng.choice(near))
            box_a = sub.labels[vid][a]['{:06d}'.format(f2)]
            box_b = sub.labels[vid][b]['{:06d}'.format(f2)]
            bw, bh = box_b[2] - box_b[0], box_b[3] - box_b[1]
            s_z = np.sqrt((bw + 0.5 * (bw + bh)) * (bh + 0.5 * (bw + bh)))
            k = 510.0 * 127.0 / (511.0 * s_z)            # original pixels -> crop511 pixels
            cx, cy = (box_b[0] + box_b[2]) / 2., (box_b[1] + box_b[3]) / 2.
            ab = Corner((box_a[0] - cx) * k + 255.0, (box_a[1] - cy) * k + 255.0, (box_a[2] - cx) * k + 255.0, (box_a[3] - cy) * k + 255.0)
            return sub.get_image_anno(vid, a, f1), sub.get_image_anno(vid, b, f2), ab, (vid, a, b)
        return None

    # ------------------------------------------------------------------ item
    def __getitem__(self, index):
        return self.make(index)[0]

    def make(self, index, night=True, force_pair=None):
        """-> (batch item, debug info, (template, search) crops before darkening / enhancement)
        night=False: no darkening / enhancement. force_pair in (None, 'pos', 'neg', 'hard') is for the sample images."""
        j, v = self.picks[index]
        if self.train:
            rng = np.random.default_rng(np.random.randint(0, 2 ** 31 - 1))
            u = np.random.random()
        else:
            np.random.seed(10000 + index)
            rng = np.random.default_rng(10000 + index)
            u = 1.0
        sub, kind = self.subs[j], self.kinds[j]
        ptype, dbg = 'pos', {'dataset': sub.name}
        pair = None
        if force_pair == 'hard' or (force_pair is None and kind == 'UAVDT' and self.hardneg_p > 0 and u < self.hardneg_p):
            pair = self._distractor(j, v, rng)
            if pair is not None:
                ptype = 'hard'
        if pair is None and (force_pair == 'neg' or (force_pair is None and self.neg_p > 0 and self.hardneg_p <= u < self.hardneg_p + self.neg_p)):
            template = sub.get_random_target(v)
            other = int(rng.choice([k for k in range(len(self.subs))]))
            sub2 = self.subs[other]
            for _ in range(5):
                search = sub2.get_random_target()
                if sub2 is not sub or search[0].split('/')[-2] != template[0].split('/')[-2]:
                    break
            ptype = 'neg'
            pair = (template, search, None, None)
        if pair is None:
            template, search = sub.get_positive_pair(v)
            pair = (template, search, None, None)
        template, search, a_box, ids = pair
        timg = cv2.imread(template[0])
        simg = cv2.imread(search[0])
        tbox = self._get_bbox(timg, template[1])
        sbox = self._get_bbox(simg, search[1])
        t_crop, _, _ = self.template_aug.crop(timg, tbox, cfg.TRAIN.EXEMPLAR_SIZE)
        extra = [a_box] if a_box is not None else []
        ok = True
        for attempt in range(8):
            s_crop, bbox, ex = self.search_aug.crop(simg, sbox, cfg.TRAIN.SEARCH_SIZE, extra)
            if a_box is None:
                break
            a = ex[0]
            inside = a.x2 > 0 and a.x1 < cfg.TRAIN.SEARCH_SIZE and a.y2 > 0 and a.y1 < cfg.TRAIN.SEARCH_SIZE
            if not inside:
                break
        else:
            ok = False
        if a_box is not None and not ok:           # no valid distractor window found: fall back to a positive pair
            template, search = sub.get_positive_pair(v)
            timg, simg = cv2.imread(template[0]), cv2.imread(search[0])
            tbox, sbox = self._get_bbox(timg, template[1]), self._get_bbox(simg, search[1])
            t_crop, _, _ = self.template_aug.crop(timg, tbox, cfg.TRAIN.EXEMPLAR_SIZE)
            s_crop, bbox, _ = self.search_aug.crop(simg, sbox, cfg.TRAIN.SEARCH_SIZE)
            ptype = 'pos'
        dbg['pair'] = ptype
        raw = (t_crop.copy(), s_crop.copy())
        if night and self.darken_on:
            for which in (0, 1):
                prof = self._darken_profile((template if which == 0 else search)[0])
                if prof is not None and rng.random() < prof.p:
                    img, d = darken(t_crop if which == 0 else s_crop, prof, rng)
                    if which == 0:
                        t_crop = img
                    else:
                        s_crop = img
                    dbg['darken_' + 'ts'[which]] = d
        gamma = None
        if night and self.train and self.enh_p > 0 and rng.random() < self.enh_p:
            gamma = float(rng.uniform(*self.enh_gamma))
        elif not self.train and self.val_gamma:
            gamma = float(self.val_gamma)
        if gamma is not None:
            t_crop, s_crop = enhance_gamma(t_crop, gamma), enhance_gamma(s_crop, gamma)
            dbg['enh_gamma'] = round(gamma, 2)
        if ptype in ('neg', 'hard'):
            l1, lx, l2, wx = self._neg_labels(rng)
            bb = np.zeros(4)
        else:
            l1, lx, l2, wx = self.anchor_target.get(bbox, cfg.TRAIN.OUTPUT_SIZE)
            bb = np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        out = {'template': t_crop.transpose((2, 0, 1)).astype(np.float32), 'search': s_crop.transpose((2, 0, 1)).astype(np.float32),
               'bbox': bb, 'label_cls1': l1, 'labelxff': lx, 'labelcls2': l2, 'weightxff': wx}
        return out, dbg, raw
