"""Diagnose the NAT dHash cache: how many hashes are zero / identical, and what the zero-hash frames look like."""
import numpy as np
import cv2
h = np.load('tools/splits/nat_dhash.npy')
print('n', len(h), 'unique', len(np.unique(h)), 'zero', int((h == 0).sum()))
u, c = np.unique(h, return_counts=True)
o = np.argsort(-c)[:5]
print('most common hashes:', [(hex(int(u[i])), int(c[i])) for i in o])
im = cv2.imread('test_dataset/NAT2021/data_seq/N01001/000001.jpg', cv2.IMREAD_REDUCED_GRAYSCALE_2)
print('sample read', None if im is None else (im.shape, im.mean()))
t = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
print('thumb\n', t)
