"""List what is inside the three UAVDT zips without extracting them (reads the zip directories and the small text files)."""
import os
import sys
import zipfile
from collections import Counter

ROOT = 'training_dataset/UAVDT'
ZIPS = ['attributes/M_attr.zip', 'toolkit/UAV-benchmark-MOTD_v1.0.zip', 'benchmark_M/UAV-benchmark-M.zip']

for z in ZIPS:
    path = os.path.join(ROOT, z)
    print('=' * 100)
    print(path, '{:.2f} GB'.format(os.path.getsize(path) / 1e9))
    zf = zipfile.ZipFile(path)
    infos = zf.infolist()
    files = [i for i in infos if not i.is_dir()]
    print('entries: {}, files: {}, uncompressed {:.2f} GB'.format(len(infos), len(files), sum(i.file_size for i in files) / 1e9))
    print('extensions:', dict(Counter(os.path.splitext(i.filename)[1].lower() for i in files).most_common(10)))
    tops = Counter('/'.join(i.filename.split('/')[:2]) for i in files)
    print('first two path levels ({} distinct), first 15:'.format(len(tops)))
    for k, v in list(tops.items())[:15]:
        print('   {:60s} {} files'.format(k, v))
    txt = [i for i in files if i.filename.lower().endswith(('.txt', '.csv', '.md', '.m', '.json'))]
    print('text-like files: {}'.format(len(txt)))
    for i in txt[:12]:
        print('   {}  ({} bytes)'.format(i.filename, i.file_size))
    if z.startswith('attributes'):
        for i in txt[:6]:
            data = zf.read(i).decode('utf-8', 'replace')
            print('-' * 40, i.filename)
            print('\n'.join(data.splitlines()[:25]))
    if z.startswith('toolkit'):
        for i in txt:
            if any(k in i.filename.lower() for k in ('split', 'train', 'test', 'readme', 'list')):
                data = zf.read(i).decode('utf-8', 'replace')
                print('-' * 40, i.filename)
                print('\n'.join(data.splitlines()[:12]))
    if z.startswith('benchmark_M'):
        gt = [i for i in files if 'gt' in os.path.basename(i.filename).lower() and i.filename.endswith('.txt')]
        print('gt-like files: {}'.format(len(gt)))
        for i in gt[:3]:
            data = zf.read(i).decode('utf-8', 'replace')
            print('-' * 40, i.filename, '({} lines)'.format(len(data.splitlines())))
            print('\n'.join(data.splitlines()[:6]))
        seq = Counter(i.filename.split('/')[1] if i.filename.count('/') >= 2 else '' for i in files if i.filename.lower().endswith('.jpg'))
        print('jpg per first-level folder (first 10):', dict(list(seq.items())[:10]))
        print('number of folders holding jpg frames:', len(seq))
