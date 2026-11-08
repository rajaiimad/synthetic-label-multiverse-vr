#!/usr/bin/env bash
# Prepare (and optionally push) the public repository for analysis plan v15:
#   https://github.com/rajaiimad/synthetic-label-multiverse-vr
#
# Usage (Git Bash):
#   bash make_v15_repo.sh          # build the curated folder and a local commit, print a review summary (no upload)
#   bash make_v15_repo.sh --push   # same, then push to GitHub (only if the remote is still empty)
#
# What is uploaded: the benchmark and training code, the frozen backbone model (52 KB), the v2 gain constant,
# the plan-v7 prior definition, and the v15 draft plan with its planning simulations.
# What is NOT uploaded: corpora and outputs (about 3 GB; regenerated from seeds), manuscripts, reviews, reports,
# the technical report and earlier plans. Nothing in the source folders is modified.
set -euo pipefail

J7="$(cd "$(dirname "$0")/.." && pwd)"
REL="$J7/github_release"
DST="$J7/repo_synthetic-label-multiverse-vr"
REMOTE="https://github.com/rajaiimad/synthetic-label-multiverse-vr.git"
PUSH="${1:-}"

push_only() {
  cd "$DST"
  if [ -n "$(git status --porcelain)" ]; then echo "ABORT PUSH: uncommitted changes in $DST."; exit 1; fi
  if [ -n "$(git ls-remote origin 2>/dev/null)" ]; then
    echo "ABORT PUSH: the remote repository is not empty; nothing pushed."; exit 1
  fi
  git push -u origin main
  echo "== pushed to $REMOTE"
  exit 0
}

if [ -e "$DST" ] && [ -n "$(ls -A "$DST" 2>/dev/null)" ]; then
  if [ "$PUSH" = "--push" ] && [ -d "$DST/.git" ]; then
    echo "== folder already prepared; pushing it as reviewed"; push_only
  fi
  echo "ABORT: $DST already exists and is not empty (nothing overwritten)."; exit 1
fi
mkdir -p "$DST"

echo "== copying curated files"
# benchmark and pipeline code (release copy: self-contained paths)
mkdir -p "$DST/rigorous_rerun"
cp "$REL"/rigorous_rerun/*.py "$DST/rigorous_rerun/"
cp "$REL/rigorous_rerun/PREREGISTRATION_v7.md" "$DST/rigorous_rerun/"          # defines the generator prior used by v15
mkdir -p "$DST/rigorous_rerun/outputs_v2/models"
cp "$J7/rigorous_rerun/outputs_v2/models/backbone_calib_v3.pkl" "$DST/rigorous_rerun/outputs_v2/models/"
# generator package and the v2 gain constant
cp -r "$REL/coper_vr" "$DST/coper_vr"
mkdir -p "$DST/data/v2_generator"
cp "$REL/data/v2_generator/gain.json" "$DST/data/v2_generator/"
# analysis helpers (committed versions)
mkdir -p "$DST/option_b/code"
cp "$REL"/option_b/code/*.py "$DST/option_b/code/"
# plan v15 and planning simulations
mkdir -p "$DST/redesign_v15"
for f in ANALYSIS_PLAN_v15_DRAFT.md README.md proxy_variance_components.py proxy_components.json \
         power_sim_v15.py power_sim_v15.json power_sim_v15_design.py power_sim_v15_design.json \
         power_sim_v15_alpha.py power_sim_v15_alpha.json power_sim_v15_proxyrandom.py power_sim_v15_proxyrandom.json \
         make_v15_repo.sh; do
  cp "$J7/redesign_v15/$f" "$DST/redesign_v15/"
done
cp "$REL/LICENSE" "$DST/LICENSE"
find "$DST" -name "__pycache__" -type d -prune -exec rm -rf {} +
find "$DST" -name "*.pyc" -delete

echo "== writing README, .gitignore, requirements.txt"
cat > "$DST/.gitignore" <<'EOF'
__pycache__/
*.pyc
# generated corpora, predictions and outputs (regenerated from seeds; results are archived separately)
rigorous_rerun/outputs/
rigorous_rerun/outputs_v2/*
!rigorous_rerun/outputs_v2/models/
rigorous_rerun/outputs_v2/models/*
!rigorous_rerun/outputs_v2/models/backbone_calib_v3.pkl
results_v15/
*.npz
*.npy
EOF
cat > "$DST/requirements.txt" <<'EOF'
# Pinned to the versions of the development machine (Python 3.13.0). CI installs the CPU build of torch.
numpy==2.2.1
scipy==1.15.0
scikit-learn==1.6.0
pandas==2.2.3
torch==2.11.0
EOF
cat > "$DST/README.md" <<'EOF'
# synthetic-label-multiverse-vr

How do a benchmark designer's label choices (the harm generator, and whether positives are all unsafe frames or only
attack-caused ones) change which detector wins, for a policy-compliant avatar attack in multi-user VR?

**Status: analysis plan v15 is a DRAFT under author review. No experiment of plan v15 has been run.**
When the plan is approved it is frozen with a git tag (`v15-frozen`) *before* any exploratory (Stage X) or
confirmatory (Stage C) computation; every later change is recorded as a deviation. The analysis code and the GitHub
Actions workflow are added with the frozen plan.

| Folder | Content |
|---|---|
| `redesign_v15/` | Draft plan (`ANALYSIS_PLAN_v15_DRAFT.md`) and the planning simulations (power, size, coverage) |
| `rigorous_rerun/` | Benchmark generator and training pipeline (XR-PERIL-Multi); `PREREGISTRATION_v7.md` defines the generator prior |
| `coper_vr/` | Generator package (channels, poses, user tracks) |
| `option_b/code/` | Analysis helpers (tie-aware PR-AUC, bootstrap, equal-budget tuning) |
| `data/v2_generator/` | Calibrated gain of the base generator |

Corpora are not stored: every corpus is regenerated from its seed range. Requirements: `requirements.txt`
(set `XRP_GEN=v2`).

Licence: MIT (see `LICENSE`).
EOF

echo "== safety checks"
big=$(find "$DST" -type f -size +50M)
if [ -n "$big" ]; then echo "ABORT: files larger than 50 MB:"; echo "$big"; exit 1; fi
if grep -rniE "password|secret|api[_-]?key|BEGIN [A-Z ]*PRIVATE KEY|ghp_[A-Za-z0-9]|github_pat_|sk-[A-Za-z0-9]{20}" \
     --include="*.py" --include="*.md" --include="*.json" --include="*.sh" --include="*.txt" "$DST" \
     | grep -v "make_v15_repo.sh"; then
  echo "ABORT: possible secret found above."; exit 1
fi
echo "   no file > 50 MB; no secret patterns"

echo "== local git repository"
cd "$DST"
git init -q -b main
git config user.name "Rajai Darabee"
git config user.email "ri.drabee@gmail.com"
git add -A
git commit -q -m "Initial import: draft analysis plan v15, planning simulations, benchmark and training code (no corpora, no results)"
git remote add origin "$REMOTE"

echo "== review summary"
echo "   folder : $DST"
echo "   files  : $(git ls-files | wc -l)"
echo "   size   : $(du -sh . | cut -f1)"
echo "   commit : $(git log --oneline -1)"
git ls-files | sed 's|/[^/]*$||' | sort | uniq -c

if [ "$PUSH" = "--push" ]; then
  push_only
else
  echo "== not pushed. Review the folder, then run:  bash \"$J7/redesign_v15/make_v15_repo.sh\" --push"
fi
