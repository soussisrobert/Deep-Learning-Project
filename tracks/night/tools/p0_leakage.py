"""Phase 0 leakage check. Replaces the 64-bit dHash of the task text: dHash collides on dark / vignetted frames (entry 57),
so this uses thumbnail correlation + a high-pass check, validated with UAV123 as negative control (0 matches).

Queries: first frame and every 50th frame of every sequence of DarkTrack2021, UAV123 (frame folders), UAVDT-M.
Reference: ALL NAT2021 + NAT2021L frames (194,379; the task asks for every 10th, all is stronger and the thumbnails are cached).
MATCH = 24x24 z-scored thumbnail correlation >= 0.95 and high-pass (160x90, minus Gaussian sigma 3) correlation >= 0.5, same candidate.
Also: within DarkTrack2021, test-split frames against train+val frames; within UAVDT, official test vs train sequences.
Writes tools/p0/leak_*.csv and side-by-side images in tools/p0/pairs/.
"""
import os
import sys
import glob
import numpy as np
import cv2
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'tools'))
from overlap_check import thumb, hp, NAT_ROOTS

os.makedirs('tools/p0/pairs', exist_ok=True)
LO, HI = 0.95, 0.5
DT = 'training_dataset/DarkTrack2021/extracted/data_seq'
U123 = 'training_dataset/UAV123/extracted/UAV123/data_seq/UAV123'
BM = glob.glob('training_dataset/UAVDT/extracted/benchmark_M/*')[0]
AT = glob.glob('training_dataset/UAVDT/extracted/attributes/*')[0]

nat_files = []
for r in NAT_ROOTS:
    for s in sorted(os.listdir(r)):
        for f in sorted(glob.glob(os.path.join(r, s, '*.jpg'))):
            nat_files.append((r.split('/')[1], s, f))
nat = np.load('tools/splits/nat_thumb24.npy')
assert len(nat) == len(nat_files), 'thumbnail cache does not match the NAT file list'
nat = nat.astype(np.float32)
print('NAT reference frames', len(nat_files), flush=True)

split_of = {}
for k in ('train', 'val', 'test'):
    for l in open('tools/splits/DarkTrack2021_{}.txt'.format(k)):
        if l.strip():
            split_of[l.strip()] = k
ut = {}
for part in ('train', 'test'):
    for f in glob.glob('{}/{}/*_attr.txt'.format(AT, part)):
        ut[os.path.basename(f).replace('_attr.txt', '').strip()] = part


def queries(root, names, label, part_of):
    out = []
    for s in names:
        fr = sorted(glob.glob(os.path.join(root, s, '*.jpg')))
        idx = sorted(set([0] + list(range(0, len(fr), 50))))
        out += [(label, s, part_of(s), i, fr[i]) for i in idx]
    return out


Q = queries(DT, sorted(os.listdir(DT)), 'DarkTrack2021', lambda s: split_of[s])
Q += queries(U123, sorted(d for d in os.listdir(U123) if os.path.isdir(os.path.join(U123, d))), 'UAV123', lambda s: 'all')
Q += queries(BM, sorted(d for d in os.listdir(BM) if os.path.isdir(os.path.join(BM, d))), 'UAVDT', lambda s: ut.get(s, '?'))
print('query frames', len(Q), flush=True)
T = np.stack([thumb(q[4]) for q in Q])


def save_pair(a, b, name):
    ia, ib = cv2.imread(a), cv2.imread(b)
    ia = cv2.resize(ia, (480, 270)); ib = cv2.resize(ib, (480, 270))
    cv2.imwrite('tools/p0/pairs/' + name, np.hstack([ia, ib]))


# --- against NAT
rows = []
saved = {'DarkTrack2021': 0, 'UAV123': 0, 'UAVDT': 0}
for c0 in range(0, len(Q), 500):
    S = T[c0:c0 + 500] @ nat.T
    for k in range(S.shape[0]):
        q = Q[c0 + k]
        cand = np.argsort(-S[k])[:5]
        best = (-1.0, -1.0, int(cand[0]))
        if S[k, cand[0]] >= 0.9:
            qh = hp(q[4])
            for j in cand:
                h = float(qh @ hp(nat_files[j][2]))
                if h > best[1]:
                    best = (float(S[k, j]), h, int(j))
        low, h, j = best
        match = low >= LO and h >= HI
        rows.append((q[0], q[1], q[2], q[3], round(float(S[k].max()), 4), round(h, 4), int(match), nat_files[j][0], nat_files[j][1], os.path.basename(nat_files[j][2])))
        if match and saved[q[0]] < 6 and q[3] == 0:
            save_pair(q[4], nat_files[j][2], 'nat_{}_{}_vs_{}_{}.jpg'.format(q[0], q[1], nat_files[j][1], os.path.basename(nat_files[j][2])[:-4]))
            saved[q[0]] += 1
        if (not match) and q[0] == 'DarkTrack2021' and q[3] == 0 and saved.setdefault('neg', 0) < 3:
            save_pair(q[4], nat_files[j][2], 'nat_NONMATCH_{}_vs_{}_{}.jpg'.format(q[1], nat_files[j][1], os.path.basename(nat_files[j][2])[:-4]))
            saved['neg'] += 1
with open('tools/p0/leak_nat_all.csv', 'w') as f:
    f.write('dataset,seq,split,frame_idx,best_lowres_corr,hp_corr,match,nat_set,nat_seq,nat_frame\n')
    for r in rows:
        f.write(','.join(map(str, r)) + '\n')
print('\n== frames matching NAT2021/NAT2021L (per dataset and split): matched queries / queries; sequences with >=1 match / sequences')
agg = {}
for r in rows:
    a = agg.setdefault((r[0], r[2]), {'q': 0, 'm': 0, 'seqs': set(), 'mseqs': set()})
    a['q'] += 1; a['m'] += r[6]; a['seqs'].add(r[1])
    if r[6]:
        a['mseqs'].add(r[1])
for k, a in sorted(agg.items()):
    print('   {:14s} {:6s} frames {:4d}/{:4d}   sequences {:3d}/{:3d}'.format(k[0], k[1], a['m'], a['q'], len(a['mseqs']), len(a['seqs'])))
hpv = np.array([r[5] for r in rows if r[0] == 'UAV123'])
print('negative control UAV123: max hp corr {:.3f}, matched {}'.format(hpv.max(), sum(r[6] for r in rows if r[0] == 'UAV123')))
for ds in ('DarkTrack2021', 'UAVDT'):
    mseqs = sorted({r[1] for r in rows if r[0] == ds and r[6]})
    print('{} sequences with a NAT match ({}): {}'.format(ds, len(mseqs), mseqs))
    allseq = sorted({r[1] for r in rows if r[0] == ds})
    print('{} sequences WITHOUT any match among the sampled frames ({}): {}'.format(ds, len(allseq) - len(mseqs), [s for s in allseq if s not in mseqs]))

# --- within-dataset: test vs train+val
def within(label, test_of, ref_of, tag):
    qi = [i for i, q in enumerate(Q) if q[0] == label and q[2] in test_of]
    ri = [i for i, q in enumerate(Q) if q[0] == label and q[2] in ref_of]
    S = T[qi] @ T[ri].T
    pairs, n_match = {}, 0
    for a in range(S.shape[0]):
        for b in np.where(S[a] >= LO)[0]:
            qa, rb = Q[qi[a]], Q[ri[b]]
            if qa[1] == rb[1]:
                continue
            h = float(hp(qa[4]) @ hp(rb[4]))
            if h >= HI:
                pairs.setdefault((qa[1], rb[1]), []).append(h)
                if len(pairs) <= 4 and len(pairs[(qa[1], rb[1])]) == 1:
                    save_pair(qa[4], rb[4], 'within_{}_{}_vs_{}.jpg'.format(tag, qa[1], rb[1]))
    print('\n== {}: {} test-side sequences vs {} reference-side; sequence pairs with matching frames: {}'.format(
        tag, len({Q[i][1] for i in qi}), len({Q[i][1] for i in ri}), len(pairs)))
    for (a, b), hs in sorted(pairs.items())[:30]:
        print('   {} ~ {}  ({} frame pairs, max hp corr {:.3f})'.format(a, b, len(hs), max(hs)))
    return pairs


within('DarkTrack2021', {'test'}, {'train', 'val'}, 'DarkTrack_test_vs_trainval')
within('UAVDT', {'test'}, {'train'}, 'UAVDT_test_vs_train')
