"""Extract a zip that was split into raw parts (name.zip.001, .002, ...) without concatenating them first.

The parts are presented to zipfile as one seekable file. zipfile checks the CRC of every member while it is read,
so a damaged part shows up as an error.
"""
import io
import os
import sys
import time
import argparse
import zipfile
from collections import Counter


class MultiFile(io.RawIOBase):
    def __init__(self, paths):
        self.paths = paths
        self.files = [open(p, 'rb') for p in paths]
        self.sizes = [os.path.getsize(p) for p in paths]
        self.starts = [sum(self.sizes[:i]) for i in range(len(paths))]
        self.total = sum(self.sizes)
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence == 0:
            self.pos = offset
        elif whence == 1:
            self.pos += offset
        else:
            self.pos = self.total + offset
        self.pos = max(0, min(self.pos, self.total))
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.total - self.pos
        out = []
        while n > 0 and self.pos < self.total:
            i = max(k for k in range(len(self.paths)) if self.starts[k] <= self.pos)
            f = self.files[i]
            f.seek(self.pos - self.starts[i])
            chunk = f.read(min(n, self.sizes[i] - (self.pos - self.starts[i])))
            if not chunk:
                break
            out.append(chunk)
            self.pos += len(chunk)
            n -= len(chunk)
        return b''.join(out)

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parts', nargs='+', required=True, help='the parts in order (.001, .002, ...)')
    parser.add_argument('--out', required=True)
    parser.add_argument('--max_home_mb', type=int, default=0, help='refuse to extract if home would exceed this')
    args = parser.parse_args()

    mf = MultiFile(args.parts)
    print('parts: {}, total {:.2f} GB'.format(len(args.parts), mf.total / 1e9), flush=True)
    zf = zipfile.ZipFile(mf)
    infos = zf.infolist()
    print('entries: {}, uncompressed {:.2f} GB'.format(len(infos), sum(i.file_size for i in infos) / 1e9), flush=True)
    print('compression types:', dict(Counter(i.compress_type for i in infos)), '(0 stored, 8 deflated)', flush=True)
    tops = Counter(i.filename.split('/')[0] for i in infos)
    print('top level names (first 10):', dict(list(tops.items())[:10]), flush=True)
    print('file extensions:', dict(Counter(os.path.splitext(i.filename)[1].lower() for i in infos if not i.is_dir()).most_common(8)), flush=True)

    if args.max_home_mb:
        need = sum(i.file_size for i in infos) / 1e6
        used = int(os.popen('du -sm ' + os.path.expanduser('~')).read().split()[0])
        print('home {} MB + {:.0f} MB to extract, limit {} MB'.format(used, need, args.max_home_mb), flush=True)
        if used + need > args.max_home_mb:
            sys.exit('would exceed the limit, not extracting')

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for k, info in enumerate(infos):
        zf.extract(info, args.out)
        if k % 5000 == 0:
            print('{}/{} files, {:.0f} s'.format(k, len(infos), time.time() - t0), flush=True)
    print('extracted {} entries to {} in {:.0f} s'.format(len(infos), args.out, time.time() - t0), flush=True)
