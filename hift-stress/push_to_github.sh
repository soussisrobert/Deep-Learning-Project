#!/bin/bash
# Push hift-stress to the group repo as its own branch ("hift-stress"), so main is never touched.
#   cd ~/Documents/Projects/hift-stress && bash push_to_github.sh
set -euo pipefail
cd "$(dirname "$0")"
REPO=soussisrobert/Deep-Learning-Project
URL=https://github.com/$REPO.git
BRANCH=${BRANCH:-hift-stress}

ask() { read -r -p "$1 [Enter = yes, n = no] " r; [[ -z "$r" || "$r" =~ ^[Yy] ]]; }
say() { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "1/5  Checking the folder"
[[ -d .git ]] || { echo "Not a git repo: run this inside the hift-stress folder."; exit 1; }
rm -f .git/*.lock .git/refs/heads/*.lock 2>/dev/null || true   # leftovers from the sandbox sync
for p in weights/general_model.pth data results runs; do
  if git ls-files --error-unmatch "$p" >/dev/null 2>&1; then echo "STOP: $p is tracked by git (must not be pushed)."; exit 1; fi
done
echo "OK: no weights, data or results will be uploaded ($(git ls-files | wc -l | tr -d ' ') files tracked)."

say "2/5  Saving the latest changes as a commit"
git add -A
if git diff --cached --quiet; then echo "Nothing new to commit."
else
  git -c user.name="$(git config user.name || echo Sandhy)" -c user.email="$(git config user.email || echo sishimwe15@gmail.com)" \
    commit -q -m "Add TC2 agent runbook for round 2; rescore job falls back to KAUST attribute files"
  echo "Committed: $(git log --oneline -1)"
fi

say "3/5  Signing in to GitHub"
if ! command -v gh >/dev/null; then
  if command -v brew >/dev/null && ask "GitHub's command-line tool (gh) is not installed. Install it with Homebrew?"; then
    brew install gh
  else
    echo "Install gh (https://cli.github.com) or Homebrew first, then re-run."; exit 1
  fi
fi
if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "A browser window will open. Choose: GitHub.com -> HTTPS -> Yes (authenticate git) -> Login with a web browser."
  gh auth login -h github.com -p https -w
fi
gh auth setup-git >/dev/null 2>&1 || true
ME=$(gh api user -q .login)
echo "Signed in as: $ME"

say "4/5  Checking access to $REPO"
PERM=$(gh api "repos/$REPO" -q '.permissions.push' 2>/dev/null || echo none)
if [[ "$PERM" != "true" ]]; then
  echo "Your account ($ME) cannot push to $REPO yet."
  echo "Ask Robert: GitHub -> $REPO -> Settings -> Collaborators -> Add people -> '$ME' (Write)."
  echo "Accept the invite (email or github.com/notifications), then re-run this script."
  exit 1
fi
echo "OK: you have write access."
if git remote get-url group >/dev/null 2>&1; then git remote set-url group "$URL"; else git remote add group "$URL"; fi
if git ls-remote --exit-code --heads group "$BRANCH" >/dev/null 2>&1; then
  echo "Note: branch '$BRANCH' already exists on GitHub; the push only works if it adds to it (no overwriting)."
fi

say "5/5  Pushing"
echo "Will push $(git rev-list --count HEAD) commits from your '$(git branch --show-current)' to $REPO, branch '$BRANCH'."
ask "Go ahead?" || { echo "Cancelled."; exit 0; }
git push -u group "HEAD:$BRANCH"
echo
echo "Done: https://github.com/$REPO/tree/$BRANCH"
echo "To merge it into main later: https://github.com/$REPO/compare/main...$BRANCH (open a pull request)."
