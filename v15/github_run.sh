#!/usr/bin/env bash
# Plan v15, deviation D6: Stage C runs on GitHub-hosted runners (workflow .github/workflows/v15.yml).
# Run from Git Bash, from anywhere:
#   bash v15/github_run.sh push              push the local commits (code, plan, deviations) to GitHub
#   bash v15/github_run.sh test              start the 10-minute code check on GitHub (stage g0)
#   bash v15/github_run.sh stagec            start Stage C with M = 100 (v15b): G0, screening, S2, 104 generators, analysis
#   bash v15/github_run.sh status            list the latest runs of the workflow
#   bash v15/github_run.sh download RUN_ID   download every artifact of that run into results_v15_github/RUN_ID/
# All commands except 'push' need the GitHub CLI logged in: run `gh auth login` yourself once.
set -euo pipefail
REPO=rajaiimad/synthetic-label-multiverse-vr
M=100                      # fixed by v15b: results_v15/analysis_X.json -> stage_x.reestimation.M_chosen
cd "$(dirname "$0")/.."

need_gh() { gh auth status >/dev/null 2>&1 || { echo "GitHub CLI is not logged in: run 'gh auth login' first."; exit 1; }; }
latest() { sleep 8; gh run list --repo "$REPO" --workflow v15.yml --limit 3; }

case "${1:-}" in
  push)
    git push origin main ;;
  test)
    need_gh
    gh workflow run v15.yml --repo "$REPO" --ref main -f stage=g0
    latest ;;
  stagec)
    need_gh
    python -c "import json,sys; m=json.load(open('results_v15/analysis_X.json'))['stage_x']['reestimation']['M_chosen']; sys.exit(0 if m == $M else f'M in analysis_X.json is {m}, not $M')"
    [ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "Uncommitted changes: commit them first."; exit 1; }
    [ "$(git rev-parse HEAD)" = "$(git rev-parse origin/main)" ] || { echo "Local commits are not on GitHub: run 'bash v15/github_run.sh push' first."; exit 1; }
    gh workflow run v15.yml --repo "$REPO" --ref main -f stage=C -f M="$M"
    latest ;;
  status)
    need_gh
    gh run list --repo "$REPO" --workflow v15.yml --limit 5 ;;
  download)
    need_gh
    [ -n "${2:-}" ] || { echo "Usage: bash v15/github_run.sh download RUN_ID   (RUN_ID from 'status')"; exit 1; }
    gh run download "$2" --repo "$REPO" -D "results_v15_github/$2"
    echo "Downloaded to results_v15_github/$2" ;;
  *)
    sed -n 2,9p "$0" ;;
esac
