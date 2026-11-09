# v15b: Stage X record (gates G1–G2 and M)

**Status.** Written on 2026-11-10, after Stage X and before any Stage C computation. This file is a human-readable copy of what the pre-registered `x_aggregate` step computed automatically (`results_v15/analysis_X.json`, key `stage_x`; run log `results_v15/local_run/run_local.log`, 06:03:23). Nothing in it was chosen by hand. The only quantity Stage X fixes is M (plan Sections 6.1 and 7). Hypotheses, estimands, tests and decision rules are unchanged.

## 1. Run

- **Chain.** `v15/run_local.py --workers 3 --stop-after X` (deviation D1), started 2026-11-10 02:02:35 at commit a9e460d. Each worker used 5 threads, with Python 3.13.0.
- **Runs.** All 13 Stage X generators were trained and summarised without failure, and the aggregate ran at 06:03:23.
- **Per generator.** 60 dev cohorts and 60 test cohorts (seeds 60000–60059 and 61000–61059), 5 training seeds, both targets, both detectors, and the full grids.

## 2. Gates

| Generator | Aggregation | Ceiling U | Ceiling A | Prevalence U | Prevalence A | Lift A, MPNN | Lift A, Raw-HGB |
|---|---|---|---|---|---|---|---|
| and | sum | 0.530 | 0.412 | 0.0073 | 0.0045 | 35.7 | 43.1 |
| cutoff | sum | 0.546 | 0.517 | 0.0169 | 0.0128 | 21.2 | 20.4 |
| d2 | sum | 0.801 | 0.492 | 0.0064 | 0.0019 | 33.7 | 85.6 |
| fovfree | sum | 0.493 | 0.261 | 0.0027 | 0.0006 | 13.1 | 17.2 |
| rand00 | max | 0.201 | 0.231 | 0.0157 | 0.0084 | 14.4 | 13.5 |
| rand02 | sum | 0.626 | 0.526 | 0.0119 | 0.0075 | 33.2 | 41.3 |
| rand03 | sum | 0.748 | 0.734 | 0.0103 | 0.0080 | 44.1 | 43.6 |
| rand04 | sum | 0.550 | 0.533 | 0.0122 | 0.0089 | 27.4 | 29.1 |
| rand05 | sum | 0.899 | 0.898 | 0.0208 | 0.0170 | 32.4 | 36.1 |
| rand06 | sum | 0.190 | 0.128 | 0.0035 | 0.0016 | 37.8 | 24.1 |
| rand07 | sum | 0.138 | 0.135 | 0.0075 | 0.0037 | 19.0 | 13.1 |
| rand11 | max | 0.291 | 0.368 | 0.0122 | 0.0075 | 22.5 | 26.9 |
| sat | sum | 0.490 | 0.373 | 0.0084 | 0.0052 | 32.6 | 34.8 |

The ceilings are the observable (attainable) oracle PR-AUCs on the test corpus. Lift is a detector's PR-AUC under A divided by the prevalence of A.

- **G1 (detectability): passed, exactly at the threshold.**
  - 6 of the 8 prior-drawn generators have both ceilings ≥ 0.15; the requirement is ≥ 6.
  - rand06 fails on A (0.128).
  - rand07 fails on both U (0.138) and A (0.135).
- **G2 (learnability): passed.**
  - Median lift under A: MPNN 32.4, Raw-HGB 29.1; the requirement is ≥ 3 for both.
- **Rule baseline (in-band blindness, plan Section 5).** On all 13 generators and both targets, the rule's PR-AUC equals the prevalence (lift 1.00). The excess-based rule cannot see the in-band attacker, as designed.

## 3. Sample-size re-estimation (plan Section 7)

**Inputs** (the 8 prior-drawn generators, log scale):

| μ_U | τ_U | mean δ | μ_δ shrunk by 25% | τ_δ | ρ_G | ρ_E | f_A |
|---|---|---|---|---|---|---|---|
| 0.025 | 0.168 | 0.055 | 0.042 | 0.331 | −0.73 | 0.32 | 1.46 |

**H2 power by M** (`analyse_v15.simulate_power`, 1000 simulated studies):

| M = 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|
| 0.097 | 0.122 | 0.109 | 0.133 | 0.140 |

No M reaches 0.80, so **M = 100** by rule step 3. As that rule requires, the paper states that **H2 is powered at about 0.14**.

**Why the power is low.** The heterogeneity of the target shift across generators (τ_δ 0.33) is much larger than in the above-band proxies used for planning (about 0.07). The estimated mean shift (0.04 after shrinkage) is small relative to it. The planning simulation's "weak-effect" scenario did not cover this case: the shift varies strongly between generators but has a small mean.

**Pre-registered reading that applies if H2 is not rejected** (plan Section 9, row 2): "no detectable systematic target shift". The 97.5% t-interval of μ_δ is then reported as the bound on any systematic shift. This reading was fixed before Stage X; it is restated here, not chosen now.

## 4. Exploratory planning check (not part of v15b, no decision depends on it)

`redesign_v15/power_check_stage_x.py` repeats the simulation of `simulate_power` with the same Stage X inputs and noise pairs. It also applies the plan's H1 analysis (DerSimonian–Laird, π = Φ(μ/τ), 95% percentile bootstrap over generators with B = 500, inside (0.10, 0.90)). It uses 1000 simulated studies, so the Monte Carlo SE is about 0.01. Results:

| M | H1 power | H2 power | Expected half-width of the 97.5% interval of μ_δ (log scale) |
|---|---|---|---|
| 60 | 1.00 | 0.09 | 0.103 |
| 80 | 1.00 | 0.11 | 0.088 |
| 100 | 1.00 | 0.13 | 0.078 |

The point estimate from Stage X is π̂_U = Φ(0.025 / 0.168) = 0.56. These values rest on 8 generators and are only as good as those estimates.

## 5. Provenance of the Stage X outputs

1. **Commit field.** `meta.json` of every Stage X generator records `git_commit = a9e460d`. `run_local.py` sets this value once, at start.
2. **Code actually executed.**
   - The training jobs of cutoff, d2 and fovfree started at 02:02 and ran the a9e460d files.
   - The other 10 training jobs started between 02:37 and 05:20. They ran the files of ddf5cdf (deviations D2 and D3, committed at 02:08 and 02:23). This is visible in their `meta.json`, which contains `ceiling_A_full` (D3); the first three do not.
   - All 13 summaries and the aggregate ran the ddf5cdf files.
3. **Effect on Stage X.** The D2/D3 diff of `train_v15.py` only adds code: one line was extended to store the extra ceiling. In `analyse_v15.py`, the covariance is unchanged whenever every replicate is finite. The Stage X computation is therefore the same under both commits.
4. **Café-run predictions.** `X/cutoff/preds_U_mpnn.npy` was written by the interrupted run of 2026-11-09 (17:19, commit 4f3cb22) and was reused by the resumed run, which skips predictions that already exist.
   - Between 4f3cb22 and a9e460d only `run_local.py` changed.
   - The regenerated cutoff corpus had identical ceilings (U 0.5460, A 0.5168, A0 0.5278).
5. **Stage C.** Stage C will start from one commit, and no code will be committed while it runs, so its `git_commit` field will be exact.
