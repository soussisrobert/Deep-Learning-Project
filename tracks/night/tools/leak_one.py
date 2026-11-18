"""Same-frame check of one extracted dataset against ALL NAT2021 + NAT2021L frames (method of tools/p0_leakage.py).

Finds every directory that holds jpg files (sequence = directory name), queries the first frame and every 50th frame of each,
MATCH = 24x24 z-scored thumbnail correlation >= 0.95 and high-pass (160x90) correlation >= 0.5. Also checks, inside the dataset,
whether sequences share footage with each other (pairs of different sequences with matching frames).
usage: python -I tools/leak_one.py <label> <root>
writes tools/p0/leak_<label>.csv, tools/p0/leak_<label>_within.csv, pairs in tools/p0/pairs/
"""
import os
import sys
import glob
import numpy as np
import cv2
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'tools'))
from overlap_check import thumb, hp, NAT_ROOTS

label, root = sys.argv[1], sys.argv[2]
LO, HI = 0.95, 0.5
os.makedirs('tools/p0/pairs', exist_ok=True)
nat_files = []
for r in NAT_ROOTS:
    for s in sorted(os.listdir(r)):
        for f in sorted(glob.glob(os.path.join(r, s, '*.jpg'))):
            nat_files.append((r.split('/')[1], s, f))
nat = np.load('tools/splits/nat_thumb24.npy')
assert len(nat) == len(nat_files)
nat = nat.astype(np.float32)

seqs = {}
for d, _, files in os.walk(root):
    j = sorted(f for f in files if f.lower().endswith('.jpg'))
    if j:
        seqs[d] = [os.path.join(d, f) for f in j]
print('{}: {} sequence folders, {} frames'.format(label, len(seqs), sum(len(v) for v in seqs.values())), flush=True)
Q = []
for d, fr in sorted(seqs.items()):
    for i in sorted(set([0] + list(range(0, len(fr), 50)))):
        Q.append((os.path.basename(d), i, fr[i]))
print('query frames', len(Q), flush=True)
T = np.stack([thumb(q[2]) for q in Q])


def save_pair(a, b, name):
    cv2.imwrite('tools/p0/pairs/' + name, np.hstack([cv2.resize(cv2.imread(a), (480, 270)), cv2.resize(cv2.imread(b), (480, 270))]))


rows, saved = [], 0
for c0 in range(0, len(Q), 500):
    S = T[c0:c0 + 500] @ nat.T
    for k in range(S.shape[0]):
        q = Q[c0 + k]
        cand = np.argsort(-S[k])[:5]
        best = (-1.0, -1.0, int(cand[0]))
        if S[k, cand[0]] >= 0.9:
            qh = hp(q[2])
            for j in cand:
                h = float(qh @ hp(nat_files[j][2]))
                if h > best[1]:
                    best = (float(S[k, j]), h, int(j))
        low, h, j = best
        m = low >= LO and h >= HI
        rows.append((q[0], q[1], round(float(S[k].max()), 4), round(h, 4), int(m), nat_files[j][0], nat_files[j][1], os.path.basename(nat_files[j][2])))
        if m and saved < 8 and q[1] == 0:
            save_pair(q[2], nat_files[j][2], 'nat_{}_{}_vs_{}_{}.jpg'.format(label, q[0], nat_files[j][1], os.path.basename(nat_files[j][2])[:-4]))
            saved += 1
with open('tools/p0/leak_{}.csv'.format(label), 'w') as f:
    f.write('seq,frame_idx,best_lowres_corr,hp_corr,match,nat_set,nat_seq,nat_frame\n')
    for r in rows:
        f.write(','.join(map(str, r)) + '\n')
names = sorted({r[0] for r in rows})
ms = sorted({r[0] for r in rows if r[4]})
print('frames matching NAT: {}/{}; sequences with a match: {}/{}'.format(sum(r[4] for r in rows), len(rows), len(ms), len(names)))
print('matching sequences:', ms)
print('sequences without any match:', [n for n in names if n not in ms])
print('highest lowres corr among non-matches: {:.3f}'.format(max([r[2] for r in rows if not r[4]] or [0])))

# within the dataset: different sequences sharing frames
S = T @ T.T
pairs = {}
for a in range(len(Q)):
    for b in np.where(S[a] >= LO)[0]:
        if b <= a or Q[a][0] == Q[b][0]:
            continue
        h = float(hp(Q[a][2]) @ hp(Q[b][2]))
        if h >= HI:
            pairs.setdefault((Q[a][0], Q[b][0]), []).append(h)
with open('tools/p0/leak_{}_within.csv'.format(label), 'w') as f:
    f.write('seq_a,seq_b,frame_pairs,max_hp_corr\n')
    for (a, b), hs in sorted(pairs.items()):
        f.write('{},{},{},{:.3f}\n'.format(a, b, len(hs), max(hs)))
print('within {}: {} sequence pairs share footage'.format(label, len(pairs)))
for (a, b), hs in sorted(pairs.items())[:40]:
    print('   {} ~ {} ({} frame pairs, max hp {:.3f})'.format(a, b, len(hs), max(hs)))
