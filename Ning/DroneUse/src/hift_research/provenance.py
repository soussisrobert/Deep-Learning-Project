"""Content identities and append-only run directories."""
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from hift_research import UPSTREAM_COMMIT


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def git_output(root, *args):
    result = subprocess.run(['git', '-C', str(root), *args], capture_output=True, check=True)
    return result.stdout


def source_identity(root):
    root = Path(root)
    actual = git_output(root / 'third_party/HiFT', 'rev-parse', 'HEAD').decode().strip()
    if actual != UPSTREAM_COMMIT:
        raise ValueError(f'Upstream mismatch: {actual}')
    if git_output(root / 'third_party/HiFT', 'status', '--porcelain', '--untracked-files=no').strip():
        raise ValueError('Upstream has tracked changes; B0 identity cannot be established')
    try:
        commit = git_output(root, 'rev-parse', 'HEAD').decode().strip()
    except subprocess.CalledProcessError:
        commit = None
    status = git_output(root, 'status', '--porcelain').decode()
    # Diff alone misses new files. Hash project source and docs, excluding generated/ignored data.
    paths = git_output(root, 'ls-files', '--cached', '--others', '--exclude-standard', '-z').decode().split('\0')
    scopes = {'src', 'configs', 'scripts', 'tests', 'requirements', 'docs', 'reports', 'experiments', 'patches', '.cursor'}
    root_files = {'README.md', 'AGENTS.md', 'pyproject.toml', '.gitignore', '.gitattributes', '.gitmodules'}
    # Include all tracked files plus declared project scopes, not transient pip temp files.
    tracked = set(git_output(root, 'ls-files', '--cached', '-z').decode().split('\0'))
    candidates = [p for p in sorted(set(paths)) if p and (p in tracked or p in root_files or p.split('/')[0] in scopes)]
    identities = {p: sha256(root / p) for p in candidates if (root / p).is_file()}
    return {'git_commit': commit, 'git_status': status, 'source_files_sha256': identities,
            'source_tree_sha256': hashlib.sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest(),
            'dirty_diff_sha256': hashlib.sha256(git_output(root, 'diff', 'HEAD') if commit else git_output(root, 'diff', '--cached')).hexdigest(),
            'upstream_commit': actual}


def new_run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    return output


def runtime_identity():
    import torch
    return {'utc': datetime.now(timezone.utc).isoformat(), 'python': sys.version,
            'platform': platform.platform(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
