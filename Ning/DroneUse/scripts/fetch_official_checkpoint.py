"""Bounded download from the exact Google Drive ID linked in the pinned upstream README.

No pickle execution, login circumvention, alternate weights, or silent overwrite.
"""
import argparse
import hashlib
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

URL = 'https://drive.usercontent.google.com/download?id=1X8vScXvZ1QohbqzE9EAheQ_yaJCTslSA&export=download&confirm=t'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('artifacts/checkpoints/general_model.pth'))
    parser.add_argument('--max-mib', type=int, default=256)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {'source_url': URL, 'upstream_readme_line': 23, 'sampled_utc': datetime.now(timezone.utc).isoformat(), 'official_hash_available': False}
    if args.output.exists():
        raise FileExistsError(args.output)
    try:
        with urllib.request.urlopen(URL, timeout=30) as response:
            content_type = response.headers.get('Content-Type', '')
            record.update({'http_status': response.status, 'content_type': content_type})
            if 'text/' in content_type:
                raise ValueError('Received text instead of checkpoint; possible access/login/confirmation page')
            total, digest = 0, hashlib.sha256()
            with args.output.open('xb') as stream:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > args.max_mib * 1024 * 1024:
                        raise ValueError('Download exceeds bounded initialization budget')
                    stream.write(block)
                    digest.update(block)
            record.update({'status': 'downloaded_unverified_until_strict_model_load', 'bytes': total, 'sha256': digest.hexdigest()})
    except (OSError, ValueError, urllib.error.URLError) as exc:
        record.update({'status': 'blocked', 'error': repr(exc), 'partial_file_may_exist': args.output.exists()})
    receipt = args.output.with_suffix('.acquisition.json')
    with receipt.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    print(json.dumps(record, indent=2))
    return 0 if record['status'].startswith('downloaded') else 2


if __name__ == '__main__':
    raise SystemExit(main())
