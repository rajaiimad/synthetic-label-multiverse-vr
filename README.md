# synthetic-label-multiverse-vr

How do a benchmark designer's label choices (the harm generator, and whether positives are all unsafe frames or only
attack-caused ones) change which detector wins, for a policy-compliant avatar attack in multi-user VR?

**Status: analysis plan v15 is approved and frozen (`redesign_v15/ANALYSIS_PLAN_v15.md`, git tag `v15-frozen`).**
The tag is set before any exploratory (Stage X) or confirmatory (Stage C) computation; every later change is recorded
as a deviation. Code: `v15/`; runs: `.github/workflows/v15.yml` (Actions -> v15 -> Run workflow).

| Folder | Content |
|---|---|
| `redesign_v15/` | Frozen plan (`ANALYSIS_PLAN_v15.md`), the earlier draft, and the planning simulations (power, size, coverage) |
| `v15/` | Generator with the counterfactual targets, gate G0, training, screening and analysis code |
| `rigorous_rerun/` | Benchmark generator and training pipeline (XR-PERIL-Multi); `PREREGISTRATION_v7.md` defines the generator prior |
| `coper_vr/` | Generator package (channels, poses, user tracks) |
| `option_b/code/` | Analysis helpers (tie-aware PR-AUC, bootstrap, equal-budget tuning) |
| `data/v2_generator/` | Calibrated gain of the base generator |

Corpora are not stored: every corpus is regenerated from its seed range. Requirements: `requirements.txt`
(set `XRP_GEN=v2`).

Licence: MIT (see `LICENSE`).
