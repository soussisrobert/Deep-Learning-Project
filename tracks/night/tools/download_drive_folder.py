"""List a public Google Drive folder, then download its files one by one into a new directory.

usage: python -I tools/download_drive_folder.py <folder_url> <dest_dir> [max_home_mb]
Stops before a file if the home directory is above max_home_mb (default 75000). Does not extract anything.
"""
import os
import sys
import subprocess
import gdown

url, dest = sys.argv[1], sys.argv[2]
max_mb = int(sys.argv[3]) if len(sys.argv) > 3 else 75000
os.makedirs(dest, exist_ok=True)

items = gdown.download_folder(url=url, output=dest, quiet=True, use_cookies=False, skip_download=True)
if not items:
    sys.exit('no files listed (folder empty, not public, or too many accesses)')
print('listed {} files:'.format(len(items)))
for it in items:
    print('   {}  (id {})'.format(it.path, it.id))
sys.stdout.flush()

only = sys.argv[4:]  # optional: download only files whose name contains one of these strings
for it in items:
    if only and not any(o in it.path for o in only):
        continue
    used = int(subprocess.check_output(['du', '-sm', os.path.expanduser('~')]).split()[0])
    print('home usage before {}: {} MB'.format(it.path, used), flush=True)
    if used > max_mb:
        sys.exit('home above {} MB, stopping before {}'.format(max_mb, it.path))
    out = it.local_path
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print('   exists, skipping', flush=True)
        continue
    res = gdown.download(id=it.id, output=out + '.part', quiet=True, use_cookies=False)
    if res is None:
        print('   REFUSED by Drive: {}'.format(it.path), flush=True)
        continue
    os.rename(out + '.part', out)
    print('   done {} bytes'.format(os.path.getsize(out)), flush=True)

print('\nfinal listing:')
for root, _, files in os.walk(dest):
    for f in sorted(files):
        p = os.path.join(root, f)
        print('   {} {:.1f} MB'.format(os.path.relpath(p, dest), os.path.getsize(p) / 1e6))
