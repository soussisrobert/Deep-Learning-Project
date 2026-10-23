#!/usr/bin/env bash
set -euo pipefail
git submodule update --init --recursive
actual=$(git -C third_party/HiFT rev-parse HEAD)
expected=992aa7c1605360d607af84cb804b93cd8ee81caf
if [ "$actual" != "$expected" ]; then
  printf 'Wrong upstream: %s\nExpected: %s\n' "$actual" "$expected" >&2
  exit 1
fi
test -z "$(git -C third_party/HiFT status --porcelain --untracked-files=no)"
printf 'Verified immutable upstream %s\n' "$actual"
