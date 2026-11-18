"""List what is inside Dataset_UAV123.zip without extracting (reads the zip directory and a few small text files)."""
import os
import zipfile
from collections import Counter

PATH = 'training_dataset/UAV123/Dataset_UAV123.zip'
zf = zipfile.ZipFile(PATH)
infos = zf.infolist()
files = [i for i in infos if not i.is_dir()]
print(PATH, '{:.2f} GB'.format(os.path.getsize(PATH) / 1e9))
print('entries {}, files {}, uncompressed {:.2f} GB'.format(len(infos), len(files), sum(i.file_size for i in files) / 1e9))
print('extensions:', dict(Counter(os.path.splitext(i.filename)[1].lower() for i in files).most_common(10)))
print('compression types:', dict(Counter(i.compress_type for i in files)))
for depth in (1, 2, 3):
    c = Counter('/'.join(i.filename.split('/')[:depth]) for i in files)
    print('\npath prefixes at depth {} ({} distinct), first 12:'.format(depth, len(c)))
    for k, v in list(c.items())[:12]:
        print('   {:70s} {}'.format(k, v))

txt = [i for i in files if i.filename.lower().endswith('.txt')]
print('\ntxt files: {}, first 8:'.format(len(txt)))
for i in txt[:8]:
    print('   {} ({} bytes)'.format(i.filename, i.file_size))
for i in txt[:2]:
    data = zf.read(i).decode('utf-8', 'replace')
    print('-' * 30, i.filename, '({} lines)'.format(len(data.splitlines())))
    print('\n'.join(data.splitlines()[:5]))
others = [i for i in files if i.filename.lower().endswith(('.mat', '.m', '.pdf', '.md'))]
print('\nother non-image files:', [i.filename for i in others[:10]])

# frames per sequence folder (folder that holds jpgs)
jpg = Counter(os.path.dirname(i.filename) for i in files if i.filename.lower().endswith('.jpg'))
print('\nfolders with jpg: {}, total jpg {}'.format(len(jpg), sum(jpg.values())))
vals = sorted(jpg.values())
print('frames per folder: min {} median {} max {}'.format(vals[0], vals[len(vals) // 2], vals[-1]))
print('first 10:', list(jpg.items())[:10])
