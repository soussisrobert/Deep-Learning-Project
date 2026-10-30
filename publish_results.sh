#!/bin/bash
# Publish all rounds (code, plans, analyses, RESULTS.md) to the group repo, branch "hift-stress",
# with every commit shown as your GitHub account. Press Enter to accept each step.
#   cd ~/Documents/Projects/hift-stress && bash publish_results.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO=soussisrobert/Deep-Learning-Project
URL=${PUBLISH_URL:-https://github.com/$REPO.git}   # override only for testing
BRANCH=hift-stress
ask() { read -r -p "$1 [Enter = yes, n = no] " r; [[ -z "$r" || "$r" =~ ^[Yy] ]]; }
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1/6  Checking the folder"
[[ -d .git ]] || { echo "Not a git repo: run this inside the hift-stress folder."; exit 1; }
rm -f .git/*.lock .git/refs/heads/*.lock 2>/dev/null || true
for p in weights/general_model.pth data results runs; do
  if git ls-files --error-unmatch "$p" >/dev/null 2>&1; then echo "STOP: $p is tracked by git (must not be pushed)."; exit 1; fi
done
BIG=$(git ls-files -o -m --exclude-standard | while read -r f; do [[ -f "$f" && $(wc -c <"$f") -gt 5000000 ]] && echo "$f"; done || true)
[[ -z "$BIG" ]] || { echo "STOP: files over 5 MB would be added:"; echo "$BIG"; exit 1; }
echo "OK: no weights, data, raw results or big files."

say "2/6  Signing in to GitHub"
command -v gh >/dev/null || { echo "Install GitHub's tool first: brew install gh"; exit 1; }
gh auth status -h github.com >/dev/null 2>&1 || gh auth login -h github.com -p https -w
gh auth setup-git >/dev/null 2>&1 || true
LOGIN=$(gh api user -q .login); ID=$(gh api user -q .id); NAME=$(gh api user -q '.name // .login')
EMAIL="${ID}+${LOGIN}@users.noreply.github.com"
echo "Signed in as $LOGIN. Commits will show as: $NAME ($LOGIN)"
[[ $(gh api "repos/$REPO" -q '.permissions.push' 2>/dev/null || echo none) == "true" ]] || {
  echo "$LOGIN cannot push to $REPO yet. Ask Robert to add '$LOGIN' as a collaborator (Write), accept, re-run."; exit 1; }
git config user.name "$NAME"; git config user.email "$EMAIL"
if git remote get-url group >/dev/null 2>&1; then git remote set-url group "$URL"; else git remote add group "$URL"; fi

say "3/6  Checking GitHub for changes made by someone else"
if git ls-remote --exit-code --heads group "$BRANCH" >/dev/null 2>&1; then
  git fetch -q group "$BRANCH"
  if ! git merge-base --is-ancestor "group/$BRANCH" HEAD; then
    echo "STOP: branch '$BRANCH' on GitHub has commits you don't have (a teammate pushed?)."
    echo "Send Manzi's helper this message and the output of: git log --oneline group/$BRANCH -5"; exit 1
  fi
  echo "OK: GitHub has nothing you don't."
fi

say "4/6  Saving rounds 3-5 and the results summary as one commit"
printf 'fix_author.sh\nmerge_into_main.sh\n' >> .git/info/exclude   # old one-off helpers stay local
git add -A
if git diff --cached --quiet; then echo "Nothing new to commit."
else
  git commit -q -F - <<'MSG'
Rounds 3-5 and an all-results summary (RESULTS.md)

- RESULTS.md: one-page summary of every test, for the team.
- analysis/all-rounds: every run in one table (103 runs), per-video success, overview figure.
- Round 3-5 analyses (REPORT.md, figures, independent scripts and outputs).
- Code: torch JPEG and adaptive attacks through JPEG (+EOT), image corruptions incl. synthetic night,
  per-frame confidence logging, int8 backbone with layer1 kept fp32, calibration options.
- Plans, gates and agent runbooks for rounds 3-5.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014FfrHMbvimu3kapk9Guadj
MSG
  echo "Committed: $(git log --oneline -1)"
fi

say "5/6  Showing every commit as $LOGIN"
FORCE=""
OTHER=$(git log --format='%ae' | grep -cv "^$EMAIL$" || true)
if [[ "$OTHER" != "0" ]]; then
  echo "Some earlier commits show another name (e.g. code4tesla). Re-labelling rewrites the branch on GitHub."
  if ask "Re-label all commits as $LOGIN?"; then
    git rebase -q -r --root --exec 'git commit -q --amend --no-edit --reset-author'
    FORCE="--force-with-lease"
  fi
fi
git log --format='   %h %an  %s' -10

say "6/6  Pushing to $REPO, branch '$BRANCH'"
ask "Go ahead?" || { echo "Cancelled (your commits are saved locally)."; exit 0; }
git push $FORCE -u group "HEAD:$BRANCH"
echo
echo "Done. Team summary: https://github.com/$REPO/blob/$BRANCH/RESULTS.md"
echo "To put it into main: bash merge_into_main.sh (opens a pull request)."
