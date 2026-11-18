"""Extract one zip into a new, empty directory and check the result (member count, sizes, CRC via zipfile).

usage: python -I tools/extract_zip_checked.py <zip> <dest_dir>
"""
import os
import sys
import time
import zipfile

src, dest = sys.argv[1], sys.argv[2]
os.makedirs(dest, exist_ok=True)
if os.listdir(dest):
    sys.exit('destination {} is not empty, refusing to extract into it'.format(dest))

zf = zipfile.ZipFile(src)
infos = zf.infolist()
t0 = time.time()
for n, info in enumerate(infos, 1):
    zf.extract(info, dest)  # zipfile strips absolute paths and '..', and checks the CRC while reading
    if n % 10000 == 0:
        print('{}/{} entries, {:.0f} s'.format(n, len(infos), time.time() - t0), flush=True)

files = [i for i in infos if not i.is_dir()]
missing, wrong = 0, 0
for i in files:
    p = os.path.join(dest, i.filename)
    if not os.path.isfile(p):
        missing += 1
    elif os.path.getsize(p) != i.file_size:
        wrong += 1
print('extracted {} entries ({} files) in {:.0f} s; missing {}, wrong size {}'.format(
    len(infos), len(files), time.time() - t0, missing, wrong))
if missing or wrong:
    sys.exit(1)
