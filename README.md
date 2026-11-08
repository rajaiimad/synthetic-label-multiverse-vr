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
